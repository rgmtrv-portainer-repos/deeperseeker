import asyncio
import json
import logging
import os
import re
import secrets
import time
import sys
import uuid
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import deepseek_tokenizer
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

load_dotenv()

API_KEY = os.getenv("DEEPSEEKER_API_KEY", "dseeker")
ADMIN_USER = os.getenv("DEEPSEEKER_ADMIN_USER", "admin")
ADMIN_PASSWORD = os.getenv("DEEPSEEKER_ADMIN_PASSWORD", "admin")

# --- Verbose logging configuration -----------------------------------------
# Set LOG_LEVEL=DEBUG (or export DEEPSEEKER_VERBOSE=1) to get full tracing of
# every request, session lookup, token pick, and streaming event.
LOG_LEVEL = os.getenv("LOG_LEVEL", "").upper()
if not LOG_LEVEL:
    LOG_LEVEL = "DEBUG" if os.getenv("DEEPSEEKER_VERBOSE", "").lower() in ("1", "true", "yes") else "INFO"

security = HTTPBasic()


from functions import (
    add_token,
    count_tokens,
    create_new_chat,
    delete_token,
    find_session,
    get_auth_token,
    get_token,
    get_tokens,
    init_db,
    mark_limited,
    mark_active,
    parse_tools,
    pick_token,
    save_session,
    delete_session,
    send_message,
    StreamToolParser,
    upload_file,
    get_file_content,
)
from plugin_helper import build_prompt, extract_and_upload_files, generate_signature, generate_signature_sync


logger = logging.getLogger("uvicorn.error")


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s"
))

logger.addHandler(handler)
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
logger.info("Logger initialized at level %s", LOG_LEVEL)


def _short(val, n=200):
    """Truncate a value for safe/readable logging."""
    try:
        s = val if isinstance(val, str) else json.dumps(val, default=str)
    except Exception:
        s = str(val)
    return s if len(s) <= n else s[:n] + f"...<+{len(s) - n} chars>"


def _redact_key(key):
    """Never log full API keys / tokens."""
    if not key:
        return "<empty>"
    key = str(key)
    if len(key) <= 8:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


def _mask_scope(scope):
    return _redact_key(scope) if scope else "<none>"


def count_tok(text):
    n = len(deepseek_tokenizer.ds_token.encode(text))
    logger.debug("count_tok: %d tokens for text of length %d", n, len(text) if text else 0)
    return n


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup: initializing database")
    init_db()
    logger.info("Database initialized, application ready")
    yield
    logger.info("Application shutdown")


app = FastAPI(title="DeeperSeeker", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    start = time.time()
    cl = request.headers.get("content-length", "")
    logger.debug("Incoming %s %s (content-length=%s)", request.method, request.url.path, cl or "unknown")
    if cl.isdigit() and int(cl) > 32 * 1024 * 1024:
        logger.warning("Rejecting %s %s: body too large (%s bytes)", request.method, request.url.path, cl)
        return JSONResponse({"error": "Request body too large"}, status_code=413)
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled exception while processing %s %s", request.method, request.url.path)
        raise
    duration_ms = (time.time() - start) * 1000
    logger.info("%s %s -> %s (%.1f ms)", request.method, request.url.path, response.status_code, duration_ms)
    return response


SESSIONS = {}
SESSION_TTL = 7 * 24 * 3600
_sig_locks = {}
_login_fails = {"count": 0, "locked_until": 0}


def get_current_admin(request: Request):
    sid = request.cookies.get("session_id")
    if not sid or sid not in SESSIONS or time.time() - SESSIONS[sid] > SESSION_TTL:
        logger.warning("Admin auth failed: missing/expired/unknown session (sid=%s)", _redact_key(sid))
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    SESSIONS[sid] = time.time()
    origin = request.headers.get("origin", "")
    if origin:
        parsed = urlparse(origin).netloc
        if parsed and parsed != request.headers.get("host", ""):
            logger.warning("Admin auth CSRF check failed: origin=%s host=%s", parsed, request.headers.get("host", ""))
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
    logger.debug("Admin session validated (sid=%s)", _redact_key(sid))
    return "admin"


def get_api_key(request: Request):
    auth = request.headers.get("authorization", "")
    api_key_header = request.headers.get("x-api-key", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    elif auth:
        return auth
    return api_key_header


def check_key(request: Request):
    key = get_api_key(request)
    ok = secrets.compare_digest(key.encode("utf-8"), API_KEY.encode("utf-8"))
    if ok:
        logger.debug("API key check passed (key=%s)", _redact_key(key))
    else:
        logger.warning("API key check FAILED (key=%s) from %s", _redact_key(key), request.client.host if request.client else "unknown")
    return ok


async def handle_chat(messages, model, thinking=False, search=False, stream=False, tools=None, is_anthropic=False, req_model=None, scope="", _retried=False):
    logger.info(
        "handle_chat start: model=%s thinking=%s search=%s stream=%s tools=%d msgs=%d anthropic=%s scope=%s retried=%s",
        model, thinking, search, stream, len(tools or []), len(messages), is_anthropic, _mask_scope(scope), _retried,
    )
    logger.debug("handle_chat messages preview: %s", _short(messages, 500))

    auth_token = get_auth_token()
    if not auth_token:
        logger.error("handle_chat: no auth token available, add via dashboard")
        return JSONResponse({"error": "No auth token. Add via dashboard."}, status_code=401)

    sig = await generate_signature(messages, model, scope)
    logger.debug("Generated signature=%s for %d messages", sig, len(messages))
    sess = find_session(sig)
    if sess:
        logger.info("Session cache HIT on full signature (sig=%s)", sig)
    else:
        logger.debug("Session cache MISS on full signature, trying prefix fallback")
        for i in range(len(messages) - 1, 0, -1):
            fallback_sig = generate_signature_sync(messages[:i], model, scope)
            sess = find_session(fallback_sig)
            if sess:
                logger.info("Session cache HIT on prefix fallback (i=%d, sig=%s)", i, fallback_sig)
                break
        if not sess:
            logger.debug("No session found via prefix fallback either")

    if sess:

        token_id = sess["token_id"]
        session_id = sess["session_id"]
        parent_message_id = sess["parent_message_id"]
        logger.debug("Resolved existing session: token_id=%s session_id=%s parent_message_id=%s", token_id, session_id, parent_message_id)
        tok = get_token(token_id)
        if not tok or tok["status"] == "RATE_LIMITED":
            logger.warning("Token %s unavailable or rate-limited, attempting to switch tokens", token_id)
            new_token_id = pick_token()
            if new_token_id and (not tok or new_token_id != token_id):
                logger.info("Switching from token %s to new token %s", token_id, new_token_id)
                new_tok = get_token(new_token_id)
                if new_tok:
                    new_session_id = await create_new_chat(new_tok["token"])
                    logger.info("Created new upstream chat session %s on token %s", new_session_id, new_token_id)
                    prompt = await build_prompt(messages, tools or [], model, is_first_message=True)

                    file_ids = await extract_and_upload_files(messages, new_tok["token"])
                    if file_ids:
                        logger.debug("Extracted/uploaded %d file(s) for new session", len(file_ids))
                    gen = send_message(new_session_id, new_tok["token"], prompt, 0, thinking, search, None if model == "instant" else model, file_ids)
                    if stream:
                        logger.info("Dispatching STREAMING response (token-switch path, anthropic=%s)", is_anthropic)
                        if is_anthropic:
                            return StreamingResponse(stream_anthropic_response(gen, model, messages, new_token_id, new_session_id, sig, tools, req_model, 0, scope), media_type="text/event-stream")
                        return StreamingResponse(stream_response(gen, model, messages, new_token_id, new_session_id, sig, tools, 0, scope), media_type="text/event-stream")
                    else:
                        resp_text = await collect_response(gen)
                        logger.debug("Collected non-streaming response, length=%d chars", len(resp_text))
                        mark_active(new_token_id)

                        parsed_tools, clean_text = parse_tools(resp_text)
                        clean_text = re.sub(r"<think>.*?</think>", "", clean_text, flags=re.DOTALL).strip()
                        clean_text = re.sub(r"</?(?:tool_calls?|invoke|function_call|parameter)[^>]*>", "", clean_text, flags=re.IGNORECASE).strip()
                        if parsed_tools:
                            logger.info("Parsed %d tool call(s) from response", len(parsed_tools))
                        next_messages = messages.copy()
                        ast_msg = {"role": "assistant"}
                        if parsed_tools:
                            ast_msg["tool_calls"] = parsed_tools
                        else:
                            ast_msg["content"] = clean_text
                        next_messages.append(ast_msg)
                        next_sig = await generate_signature(next_messages, model, scope)

                        save_session(sig, new_token_id, new_session_id, 2)
                        save_session(next_sig, new_token_id, new_session_id, 2)
                        logger.debug("Saved sessions for sig=%s and next_sig=%s", sig, next_sig)
                        return format_response(resp_text, model, messages, tools)
    else:
        logger.debug("No existing session found; acquiring creation lock for sig=%s", sig)
        create_lock = _sig_locks.setdefault(sig, asyncio.Lock())
        async with create_lock:
            sess = find_session(sig)
            if not sess:
                token_id = pick_token()
                if not token_id:
                    logger.error("No tokens available to service request (sig=%s)", sig)
                    return JSONResponse({"error": "No tokens available"}, status_code=503)
                tok = get_token(token_id)
                if not tok:
                    logger.error("Picked token_id=%s but token record not found", token_id)
                    return JSONResponse({"error": "Token not found"}, status_code=503)
                session_id = await create_new_chat(tok["token"])
                logger.info("Created new upstream chat session=%s using token_id=%s", session_id, token_id)
                save_session(sig, token_id, session_id, 0)
                parent_message_id = 0
            else:
                token_id = sess["token_id"]
                session_id = sess["session_id"]
                parent_message_id = sess["parent_message_id"]
                logger.debug("Session appeared after acquiring lock: token_id=%s session_id=%s", token_id, session_id)

    tok = get_token(token_id)
    if not tok:
        logger.error("Token %s expired/missing before send_message", token_id)
        return JSONResponse({"error": "Token expired"}, status_code=503)

    is_first = parent_message_id == 0
    logger.debug("Preparing message: token_id=%s session_id=%s parent_message_id=%s is_first=%s", token_id, session_id, parent_message_id, is_first)
    file_ids = await extract_and_upload_files(messages, tok["token"], last_user_only=not is_first)
    if file_ids:
        logger.info("Attached %d file id(s) to prompt: %s", len(file_ids), file_ids)
    prompt = await build_prompt(messages, tools or [], model, is_first)
    logger.debug("Built prompt, length=%d chars", len(prompt) if isinstance(prompt, str) else -1)

    try:
        gen = send_message(session_id, tok["token"], prompt, parent_message_id, thinking, search, None if model == "instant" else model, file_ids)
        if stream:
            logger.info("Dispatching STREAMING response (anthropic=%s, token_id=%s, session_id=%s)", is_anthropic, token_id, session_id)
            if is_anthropic:
                return StreamingResponse(stream_anthropic_response(gen, model, messages, token_id, session_id, sig, tools, req_model, parent_message_id, scope), media_type="text/event-stream")
            return StreamingResponse(stream_response(gen, model, messages, token_id, session_id, sig, tools, parent_message_id, scope), media_type="text/event-stream")
        else:
            resp_text = await collect_response(gen)
            logger.debug("Collected non-streaming response, length=%d chars", len(resp_text))
            mark_active(token_id)

            parsed_tools, clean_text = parse_tools(resp_text)
            clean_text = re.sub(r"<think>.*?</think>", "", clean_text, flags=re.DOTALL).strip()
            clean_text = re.sub(r"</?(?:tool_calls?|invoke|function_call|parameter)[^>]*>", "", clean_text, flags=re.IGNORECASE).strip()
            if parsed_tools:
                logger.info("Parsed %d tool call(s) from response", len(parsed_tools))
            next_messages = messages.copy()
            ast_msg = {"role": "assistant"}
            if parsed_tools:
                ast_msg["tool_calls"] = parsed_tools
            else:
                ast_msg["content"] = clean_text
            next_messages.append(ast_msg)
            next_sig = await generate_signature(next_messages, model, scope)

            save_session(sig, token_id, session_id, parent_message_id + 2)
            save_session(next_sig, token_id, session_id, parent_message_id + 2)
            logger.debug("Saved sessions for sig=%s and next_sig=%s (parent_message_id=%d)", sig, next_sig, parent_message_id + 2)
            logger.info("handle_chat complete (non-streaming): out_len=%d", len(clean_text))
            return format_response(resp_text, model, messages, tools)
    except Exception as e:
        logger.exception("handle_chat: send_message failed (token_id=%s, session_id=%s)", token_id, session_id)
        delete_session(sig)
        m = re.match(r"HTTP (\d{3}):", str(e))
        code = int(m.group(1)) if m else None
        if code in (401, 403, 429):
            logger.warning("Marking token_id=%s as RATE_LIMITED due to HTTP %s", token_id, code)
            mark_limited(token_id)
        if _retried or code not in (401, 403, 429):
            logger.error("handle_chat: giving up (retried=%s, code=%s)", _retried, code)
            raise
        logger.info("handle_chat: retrying once after recoverable error (code=%s)", code)
        return await handle_chat(messages, model, thinking, search, stream, tools, is_anthropic, req_model, scope, _retried=True)


async def collect_response(gen):
    text = ""
    chunk_count = 0
    async for chunk in gen:
        text += chunk
        chunk_count += 1
    logger.debug("collect_response: assembled %d chunks, total length=%d", chunk_count, len(text))
    return text


def _messages_text(messages):
    parts = []
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, list):
            parts.append(" ".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"))
        else:
            parts.append(str(c))
    return "\n".join(parts)


async def _hold_think_tags(gen):
    carry = ""
    async for chunk in gen:
        chunk = carry + chunk
        carry = ""
        hold = 0
        for tag in ("<think>", "</think>"):
            for i in range(1, len(tag)):
                if chunk.endswith(tag[:i]):
                    hold = max(hold, i)
        if hold:
            carry = chunk[-hold:]
            chunk = chunk[:-hold]
        if chunk:
            yield chunk
    if carry:
        yield carry


async def stream_response(gen, model, messages, token_id, session_id, sig, tools, parent_message_id=0, scope=""):
    logger.info("stream_response: starting OpenAI-style SSE stream (token_id=%s, session_id=%s)", token_id, session_id)
    parser = StreamToolParser()
    full_text = ""
    is_thinking = False
    aborted = False
    failed = False
    chunk_count = 0
    event_count = 0
    try:
        async for chunk in _hold_think_tags(gen):
            if not chunk:
                continue
            chunk_count += 1
            full_text += chunk

            if "<think>" in chunk:
                is_thinking = True
                logger.debug("stream_response: entering <think> block at chunk #%d", chunk_count)
                chunk = chunk.replace("<think>", "").lstrip("\n")

            end_thinking = False
            if "</think>" in chunk:
                is_thinking = False
                end_thinking = True
                logger.debug("stream_response: exiting <think> block at chunk #%d", chunk_count)
                parts = chunk.split("</think>")
                think_part = parts[0]
                chunk = parts[1].lstrip("\n") if len(parts) > 1 else ""
                if think_part:
                    event_count += 1
                    yield f"data: {json.dumps({'choices': [{'delta': {'reasoning_content': think_part}}]})}\n\n"

            if is_thinking and chunk:
                event_count += 1
                yield f"data: {json.dumps({'choices': [{'delta': {'reasoning_content': chunk}}]})}\n\n"
                continue

            if end_thinking and not chunk:
                continue

            for r in parser.feed(chunk):
                if "text" in r:
                    event_count += 1
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': r['text']}}]})}\n\n"
        mark_active(token_id)
        logger.debug("stream_response: upstream generator exhausted after %d chunks, %d events emitted", chunk_count, event_count)
    except (asyncio.CancelledError, GeneratorExit):
        aborted = True
        failed = True
        logger.warning("stream_response: client disconnected / stream cancelled (token_id=%s, session_id=%s)", token_id, session_id)
        raise
    except Exception as e:
        failed = True
        m = re.match(r"HTTP (\d{3}):", str(e))
        code = int(m.group(1)) if m else None
        if code in (401, 403, 429):
            logger.warning("stream_response: marking token_id=%s RATE_LIMITED due to HTTP %s", token_id, code)
            mark_limited(token_id)
        logger.exception("stream_response failed")
        try:
            yield f"data: {json.dumps({'error': {'message': str(e)[:300]}})}\n\n"
        except Exception:
            pass
    finally:
        parsed_tools, clean_text = parse_tools(full_text)
        clean_text = re.sub(r"<think>.*?</think>", "", clean_text, flags=re.DOTALL).strip()
        clean_text = re.sub(r"</?(?:tool_calls?|invoke|function_call|parameter)[^>]*>", "", clean_text, flags=re.IGNORECASE).strip()

        if not failed:
            next_messages = messages.copy()
            ast_msg = {"role": "assistant"}
            if parsed_tools:
                ast_msg["tool_calls"] = parsed_tools
                logger.info("stream_response: parsed %d tool call(s)", len(parsed_tools))
            else:
                ast_msg["content"] = clean_text
            next_messages.append(ast_msg)
            next_sig = generate_signature_sync(next_messages, model, scope)
            save_session(sig, token_id, session_id, parent_message_id + 2)
            save_session(next_sig, token_id, session_id, parent_message_id + 2)
            logger.debug("stream_response: saved sessions sig=%s next_sig=%s", sig, next_sig)

        if not aborted and not failed:
            try:
                if not parsed_tools:
                    for r in parser.flush():
                        if "text" in r:
                            yield f"data: {json.dumps({'choices': [{'delta': {'content': r['text']}}]})}\n\n"

                if parsed_tools:
                    for i, tc in enumerate(parsed_tools):
                        delta_tc = {"index": i, "id": tc["id"], "type": "function",
                                    "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}
                        yield f"data: {json.dumps({'choices': [{'delta': {'tool_calls': [delta_tc]}}]})}\n\n"
                    yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'tool_calls'}]})}\n\n"
                else:
                    yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                yield "data: [DONE]\n\n"
                logger.info("stream_response: completed successfully (chunks=%d, events=%d, out_len=%d)", chunk_count, event_count, len(clean_text))
            except asyncio.CancelledError:
                logger.warning("stream_response: cancelled while emitting tail events")


async def stream_anthropic_response(gen, model, messages, token_id, session_id, sig, tools, req_model=None, parent_message_id=0, scope=""):
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    logger.info("stream_anthropic_response: starting Anthropic-style SSE stream (msg_id=%s, token_id=%s, session_id=%s)", msg_id, token_id, session_id)
    in_tokens = count_tok(_messages_text(messages))
    model_name = req_model if req_model else model
    start_evt = f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': model_name, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': in_tokens, 'output_tokens': 1}}})}\n\n"
    yield start_evt

    parser = StreamToolParser()
    full_text = ""
    text_block_started = False
    block_index = 0
    aborted = False
    failed = False
    chunk_count = 0

    try:
        is_thinking = False
        async for chunk in _hold_think_tags(gen):
            if not chunk:
                continue
            chunk_count += 1
            full_text += chunk

            if "<think>" in chunk:
                is_thinking = True
                logger.debug("stream_anthropic_response: entering thinking block at index=%d", block_index)
                chunk = chunk.replace("<think>", "").lstrip("\n")
                start_block = f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'thinking'}})}\n\n"
                yield start_block

            end_thinking = False
            if "</think>" in chunk:
                is_thinking = False
                end_thinking = True
                logger.debug("stream_anthropic_response: exiting thinking block at index=%d", block_index)
                parts = chunk.split("</think>")
                think_part = parts[0]
                chunk = parts[1].lstrip("\n") if len(parts) > 1 else ""
                if think_part:
                    delta_evt = f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'thinking_delta', 'thinking': think_part}})}\n\n"
                    yield delta_evt

            if is_thinking and chunk:
                delta_evt = f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'thinking_delta', 'thinking': chunk}})}\n\n"
                yield delta_evt
                continue

            if end_thinking:
                stop_evt = f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                yield stop_evt
                block_index += 1
                if not chunk:
                    continue

            for r in parser.feed(chunk):
                if "text" in r:
                    if not text_block_started:
                        logger.debug("stream_anthropic_response: starting text block at index=%d", block_index)
                        start_block = f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                        yield start_block
                        text_block_started = True
                    delta_evt = f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': r['text']}})}\n\n"
                    yield delta_evt
        mark_active(token_id)
        logger.debug("stream_anthropic_response: upstream generator exhausted after %d chunks", chunk_count)
    except (asyncio.CancelledError, GeneratorExit):
        aborted = True
        failed = True
        logger.warning("stream_anthropic_response: client disconnected / stream cancelled (token_id=%s, session_id=%s)", token_id, session_id)
        raise
    except Exception as e:
        failed = True
        m = re.match(r"HTTP (\d{3}):", str(e))
        code = int(m.group(1)) if m else None
        if code in (401, 403, 429):
            logger.warning("stream_anthropic_response: marking token_id=%s RATE_LIMITED due to HTTP %s", token_id, code)
            mark_limited(token_id)
        logger.exception("stream_anthropic_response failed")
        try:
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': str(e)[:300]}})}\n\n"
        except Exception:
            pass
    finally:
        parsed_tools, clean_text = parse_tools(full_text)
        out_tokens = count_tok(full_text)

        clean_text = re.sub(r"<think>.*?</think>", "", clean_text, flags=re.DOTALL).strip()
        clean_text = re.sub(r"</?(?:tool_calls?|invoke|function_call|parameter)[^>]*>", "", clean_text, flags=re.IGNORECASE).strip()

        if not failed:
            next_messages = messages.copy()
            ast_msg = {"role": "assistant"}
            if parsed_tools:
                ast_msg["tool_calls"] = parsed_tools
                logger.info("stream_anthropic_response: parsed %d tool call(s)", len(parsed_tools))
            else:
                ast_msg["content"] = clean_text
            next_messages.append(ast_msg)
            next_sig = generate_signature_sync(next_messages, model, scope)
            save_session(sig, token_id, session_id, parent_message_id + 2)
            save_session(next_sig, token_id, session_id, parent_message_id + 2)
            logger.debug("stream_anthropic_response: saved sessions sig=%s next_sig=%s", sig, next_sig)

        def _tb(text):
            return (f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index_local[0], 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                    f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index_local[0], 'delta': {'type': 'text_delta', 'text': text}})}\n\n"
                    f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index_local[0]})}\n\n")

        block_index_local = [block_index]
        tail_events = ""
        if is_thinking:
            tail_events += f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index_local[0]})}\n\n"
            block_index_local[0] += 1

        flushed_text = ""
        if not parsed_tools:
            for r in parser.flush():
                if "text" in r:
                    flushed_text += r["text"]

        if not text_block_started and not parsed_tools and (clean_text or flushed_text):
            tail_events += _tb(clean_text or flushed_text)
        elif text_block_started and not parsed_tools:
            tail_events += f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index_local[0]})}\n\n"

        if parsed_tools:
            for tc in parsed_tools:
                tool_input = json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"]
                json_str = json.dumps(tool_input)
                tail_events += f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index_local[0], 'content_block': {'type': 'tool_use', 'id': tc['id'], 'name': tc['function']['name'], 'input': {}}})}\n\n"
                tail_events += f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index_local[0], 'delta': {'type': 'input_json_delta', 'partial_json': json_str}})}\n\n"
                tail_events += f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index_local[0]})}\n\n"
                block_index_local[0] += 1
            tail_events += f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'tool_use', 'stop_sequence': None}, 'usage': {'output_tokens': out_tokens}})}\n\n"
        else:
            tail_events += f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': out_tokens}})}\n\n"
        tail_events += f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

        if not aborted and not failed:
            try:
                for evt in tail_events.split("\n\n"):
                    if evt.strip():
                        yield evt + "\n\n"
                logger.info(
                    "stream_anthropic_response: completed successfully (msg_id=%s, chunks=%d, in_tokens=%d, out_tokens=%d)",
                    msg_id, chunk_count, in_tokens, out_tokens,
                )
            except asyncio.CancelledError:
                logger.warning("stream_anthropic_response: cancelled while emitting tail events")


def format_response(text, model, messages, tools=None):
    from functions import DEEPSEEK_TARIFFS
    parsed_tools, clean_text = parse_tools(text)

    reasoning = None
    match = re.search(r"<think>\s*(.*?)\s*</think>\s*", text, flags=re.DOTALL)
    if match:
        reasoning = match.group(1).strip()
        logger.debug("format_response: extracted reasoning block, length=%d chars", len(reasoning))
    clean_text = re.sub(r"<think>.*?</think>", "", clean_text, flags=re.DOTALL).strip()
    clean_text = re.sub(r"</?(?:tool_calls?|invoke|function_call|parameter)[^>]*>", "", clean_text, flags=re.IGNORECASE).strip()

    in_tokens = count_tok(_messages_text(messages))
    out_tokens = count_tok(text)
    tariff_key = "deepseek-v4-pro" if model == "expert" else "deepseek-v4-flash"
    tariff = DEEPSEEK_TARIFFS[tariff_key]
    cost = (in_tokens / 1_000_000 * tariff["cache_miss_input"]) + (out_tokens / 1_000_000 * tariff["output_generation"])
    logger.info("format_response: model=%s in_tokens=%d out_tokens=%d cost=$%.6f", model, in_tokens, out_tokens, cost)

    msg_dict = {
        "role": "assistant",
        "content": clean_text if not parsed_tools else None,
        "tool_calls": parsed_tools if parsed_tools else None,
    }
    if reasoning:
        msg_dict["reasoning_content"] = reasoning

    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": msg_dict,
            "finish_reason": "tool_calls" if parsed_tools else "stop",
        }],
        "usage": {
            "prompt_tokens": in_tokens,
            "completion_tokens": out_tokens,
            "total_tokens": in_tokens + out_tokens,
            "cost": round(cost, 6),
        },
    }


format_openai_response = format_response


def format_anthropic_response(result, model):
    choice = result["choices"][0]
    msg = choice["message"]
    ant_content = []

    if msg.get("reasoning_content"):
        ant_content.append({"type": "thinking", "thinking": msg["reasoning_content"]})

    if msg.get("content"):
        ant_content.append({"type": "text", "text": msg["content"]})

    if msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            args = tc["function"]["arguments"]
            tool_input = json.loads(args) if isinstance(args, str) else args
            ant_content.append({
                "type": "tool_use",
                "id": tc["id"],
                "name": tc["function"]["name"],
                "input": tool_input,
            })
    usage = result.get("usage", {})
    msg_id = result["id"]
    if not msg_id.startswith("msg_"):
        msg_id = f"msg_{msg_id.replace('chatcmpl-', '')}"
    logger.debug("format_anthropic_response: converted %s -> %s (%d content blocks)", result["id"], msg_id, len(ant_content))
    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": ant_content,
        "model": model,
        "stop_reason": "tool_use" if msg.get("tool_calls") else "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }



@app.post("/v1/files")
@app.post("/v1/files/upload")
async def files_upload(request: Request):
    if not check_key(request):
        return JSONResponse({"error": "Invalid API key"}, status_code=401)
    logger.info("files_upload: request received (path=%s)", request.url.path)
    tok_id = pick_token()
    if not tok_id:
        logger.error("files_upload: no tokens available")
        return JSONResponse({"error": "No tokens available"}, status_code=503)
    tok = get_token(tok_id)
    form = await request.form()
    file_obj = form.get("file")
    if not file_obj:
        logger.warning("files_upload: no file provided in form")
        return JSONResponse({"error": "No file provided"}, status_code=400)
    file_bytes = await file_obj.read(25 * 1024 * 1024 + 1)
    if len(file_bytes) > 25 * 1024 * 1024:
        logger.warning("files_upload: file too large (%d bytes)", len(file_bytes))
        return JSONResponse({"error": "File too large"}, status_code=413)
    filename = getattr(file_obj, "filename", "file.bin")
    content_type = getattr(file_obj, "content_type", "application/octet-stream")
    logger.info("files_upload: uploading '%s' (%s, %d bytes) via token_id=%s", filename, content_type, len(file_bytes), tok_id)
    file_info = None
    async for status, data in upload_file(file_bytes, filename, content_type, tok["token"]):
        logger.debug("files_upload: upload_file status=%s", status)
        if status == "success":
            file_info = data
            break
    if not file_info:
        logger.error("files_upload: upload failed for '%s'", filename)
        return JSONResponse({"error": "Upload failed"}, status_code=500)
    logger.info("files_upload: upload succeeded, file_id=%s size=%s", file_info.get("file_id"), file_info.get("size"))

    if request.url.path.startswith("/v1/files/upload"):
        return {
            "id": file_info["file_id"],
            "type": "file",
            "filename": filename,
            "size": file_info["size"],
            "created_at": file_info["anthropic_timestamp"],
        }
    return {
        "id": file_info["file_id"],
        "object": "file",
        "bytes": file_info["size"],
        "created_at": file_info["openai_timestamp"],
        "filename": filename,
        "purpose": "answers",
    }


@app.get("/v1/files/{file_id}/content")
@app.get("/v1/files/{file_id}")
async def files_content(file_id: str, request: Request):
    if not check_key(request):
        return JSONResponse({"error": "Invalid API key"}, status_code=401)
    logger.info("files_content: fetching file_id=%s", file_id)
    tok_id = pick_token()
    if not tok_id:
        logger.error("files_content: no tokens available")
        return JSONResponse({"error": "No tokens available"}, status_code=503)
    tok = get_token(tok_id)
    gen = get_file_content(tok["token"], file_id)
    try:
        mime = await gen.__anext__()
    except StopAsyncIteration:
        logger.warning("files_content: file_id=%s not found", file_id)
        return JSONResponse({"error": "File not found"}, status_code=404)
    except Exception:
        logger.exception("files_content: fetch failed for file_id=%s", file_id)
        return JSONResponse({"error": "File fetch failed"}, status_code=502)
    logger.debug("files_content: streaming file_id=%s with mime=%s", file_id, mime)
    async def stream_chunks():
        n = 0
        async for chunk in gen:
            n += 1
            yield chunk
        logger.debug("files_content: finished streaming file_id=%s (%d chunks)", file_id, n)
    return StreamingResponse(stream_chunks(), media_type=mime or "application/octet-stream")


def is_thinking_enabled(body, request=None):
    effort = body.get("effort")
    if effort is not None:
        e_str = str(effort).strip().lower()
        if e_str in ["medium", "high", "max", "ultra", "extreme", "enabled", "adaptive", "on"]:
            logger.debug("is_thinking_enabled: True via body.effort=%s", e_str)
            return True
        if e_str in ["low", "none", "off", "disable", "disabled", "false"]:
            logger.debug("is_thinking_enabled: False via body.effort=%s", e_str)
            return False

    out_cfg = body.get("output_config")
    if isinstance(out_cfg, dict):
        out_effort = out_cfg.get("effort") or out_cfg.get("reasoning_effort")
        if out_effort is not None:
            e_str = str(out_effort).strip().lower()
            if e_str in ["medium", "high", "max", "ultra", "extreme", "enabled", "adaptive", "on"]:
                logger.debug("is_thinking_enabled: True via output_config.effort=%s", e_str)
                return True
            if e_str in ["low", "none", "off", "disable", "disabled", "false"]:
                logger.debug("is_thinking_enabled: False via output_config.effort=%s", e_str)
                return False

    thinking_val = body.get("thinking")
    if isinstance(thinking_val, dict):
        t_type = str(thinking_val.get("type", "")).strip().lower()
        if t_type in ["enabled", "adaptive", "true"]:
            logger.debug("is_thinking_enabled: True via thinking.type=%s", t_type)
            return True
        if t_type == "disabled":
            logger.debug("is_thinking_enabled: False via thinking.type=disabled")
            return False
        budget = thinking_val.get("budget_tokens", 0)
        if isinstance(budget, (int, float)) and budget > 0:
            logger.debug("is_thinking_enabled: True via thinking.budget_tokens=%s", budget)
            return True
        t_effort = thinking_val.get("effort") or thinking_val.get("reasoning_effort") or thinking_val.get("level")
        if t_effort is not None:
            e_str = str(t_effort).strip().lower()
            if e_str in ["medium", "high", "max", "ultra", "extreme", "enabled", "adaptive", "on"]:
                logger.debug("is_thinking_enabled: True via thinking.effort=%s", e_str)
                return True
            if e_str in ["low", "none", "off", "disable", "disabled", "false"]:
                logger.debug("is_thinking_enabled: False via thinking.effort=%s", e_str)
                return False
    elif isinstance(thinking_val, str):
        t_str = thinking_val.strip().lower()
        if t_str in ["medium", "high", "max", "ultra", "extreme", "true", "enabled", "adaptive", "on"]:
            logger.debug("is_thinking_enabled: True via thinking(str)=%s", t_str)
            return True
        if t_str in ["low", "none", "off", "disable", "disabled", "false"]:
            logger.debug("is_thinking_enabled: False via thinking(str)=%s", t_str)
            return False
    elif isinstance(thinking_val, bool):
        logger.debug("is_thinking_enabled: %s via thinking(bool)", thinking_val)
        return thinking_val

    reasoning_effort = body.get("reasoning_effort")
    if reasoning_effort is not None:
        effort_str = str(reasoning_effort).strip().lower()
        if effort_str in ["medium", "high", "max", "ultra", "extreme"]:
            logger.debug("is_thinking_enabled: True via reasoning_effort=%s", effort_str)
            return True
        if effort_str in ["low", "none", "off", "disable", "disabled"]:
            logger.debug("is_thinking_enabled: False via reasoning_effort=%s", effort_str)
            return False

    if request:
        req_effort = request.headers.get("anthropic-thinking") or request.headers.get("x-anthropic-thinking") or request.headers.get("effort") or request.headers.get("x-effort")
        if req_effort:
            e_str = str(req_effort).strip().lower()
            if e_str in ["medium", "high", "max", "ultra", "extreme", "enabled", "adaptive", "on"]:
                logger.debug("is_thinking_enabled: True via header=%s", e_str)
                return True
    logger.debug("is_thinking_enabled: defaulting to False")
    return False


def resolve_model(model_raw):
    if not model_raw or not isinstance(model_raw, str):
        logger.debug("resolve_model: no/invalid model_raw=%r, defaulting to 'expert'", model_raw)
        return "expert"
    m = model_raw.lower()
    if "instant" in m or "haiku" in m or "flash" in m:
        resolved = "instant"
    elif "vision" in m:
        resolved = "vision"
    else:
        resolved = "expert"
    logger.debug("resolve_model: %r -> %s", model_raw, resolved)
    return resolved


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if not check_key(request):
        return JSONResponse({"error": "Invalid API key"}, status_code=401)
    body = await request.json()
    messages = body.get("messages", [])
    model = resolve_model(body.get("model", "expert"))
    thinking = is_thinking_enabled(body, request)
    search = body.get("search", False)
    stream = body.get("stream", False)
    tools = body.get("tools", None)
    logger.info(
        "chat_completions: model=%s(%s) msgs=%d thinking=%s search=%s stream=%s tools=%s",
        body.get("model"), model, len(messages), thinking, search, stream, len(tools) if tools else 0,
    )
    return await handle_chat(messages, model, thinking, search, stream, tools, scope=get_api_key(request))


@app.post("/v1/responses")
async def openai_responses(request: Request):
    if not check_key(request):
        return JSONResponse({"error": "Invalid API key"}, status_code=401)
    body = await request.json()
    model = resolve_model(body.get("model", "expert"))
    inputs = body.get("input", [])
    if isinstance(inputs, str):
        inputs = [inputs]
    elif isinstance(inputs, dict):
        inputs = [inputs]
    logger.info("openai_responses: model=%s(%s) input_items=%d", body.get("model"), model, len(inputs))

    messages = []
    for item in inputs:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue

        role = item.get("role", "user")
        content = item.get("content", [])
        msg_content = []
        if isinstance(content, str):
            msg_content = content
        else:
            for c in content:
                if c.get("type") == "input_text":
                    msg_content.append({"type": "text", "text": c.get("text")})
                elif c.get("type") == "input_file":
                    msg_content.append({"type": "file", "file_id": c.get("file_id")})
                else:
                    msg_content.append(c)
        messages.append({"role": role, "content": msg_content})

    thinking = is_thinking_enabled(body, request)
    search = body.get("search", False)
    stream = body.get("stream", False)
    tools = body.get("tools", None)
    logger.debug("openai_responses: converted to %d chat-style messages, thinking=%s search=%s stream=%s", len(messages), thinking, search, stream)

    result = await handle_chat(messages, model, thinking, search, stream, tools, scope=get_api_key(request))

    if stream:
        return result

    if isinstance(result, dict) and "choices" in result:
        message = result["choices"][0]["message"]
        out_content = []
        if message.get("content"):
            out_content.append({"type": "text", "text": message["content"]})
        if message.get("tool_calls"):
            out_content.extend([{"type": "tool_call", "id": tc["id"], "name": tc["function"]["name"], "arguments": tc["function"]["arguments"]} for tc in message["tool_calls"]])

        msg_output = {
            "type": "message",
            "role": "assistant",
            "content": out_content
        }
        if message.get("reasoning_content"):
            msg_output["reasoning_content"] = message["reasoning_content"]

        logger.debug("openai_responses: returning response id=%s with %d content item(s)", result["id"], len(out_content))
        return {
            "id": result["id"],
            "object": "response",
            "model": result["model"],
            "output": [msg_output],
            "usage": result.get("usage", {})
        }
    return result


@app.post("/v1/messages")
@app.post("/messages")
async def anthropic_messages(request: Request):
    if not check_key(request):
        return JSONResponse({"error": "Invalid API key"}, status_code=401)
    body = await request.json()
    system = body.get("system", "")

    messages = body.get("messages", [])
    model = resolve_model(body.get("model", "expert"))

    thinking = is_thinking_enabled(body, request)
    stream = body.get("stream", False)
    tools = body.get("tools", [])
    logger.info(
        "anthropic_messages: requested_model=%s -> %s msgs=%d thinking=%s stream=%s tools=%d has_system=%s",
        body.get("model"), model, len(messages), thinking, stream, len(tools), bool(system),
    )

    openai_msgs = []
    if system:
        if isinstance(system, list):
            system_str = " ".join(c.get("text", "") for c in system if isinstance(c, dict) and c.get("type") == "text")
        else:
            system_str = str(system)
        if system_str:
            openai_msgs.append({"role": "system", "content": system_str})
            logger.debug("anthropic_messages: system prompt length=%d chars", len(system_str))

    for m in messages:
        content = m.get("content", "")
        if isinstance(content, list):
            parts = []
            image_parts = []
            for c in content:
                if isinstance(c, dict):
                    if c.get("type") == "text":
                        parts.append(c.get("text", ""))
                    elif c.get("type") == "image":
                        image_parts.append(c)
                    elif c.get("type") == "tool_use":
                        parts.append(f"{json.dumps({'name': c.get('name'), 'arguments': c.get('input', {})})}")
                    elif c.get("type") == "tool_result":
                        res_content = c.get("content", "")
                        if isinstance(res_content, list):
                            for item in res_content:
                                if isinstance(item, dict) and item.get("type") == "image":
                                    image_parts.append(item)
                            res_content = " ".join(item.get("text", "") for item in res_content if isinstance(item, dict) and item.get("type") == "text")
                        parts.append(f"[Tool Result for {c.get('tool_use_id', 'tool')}]: {res_content}")
            if image_parts:
                logger.debug("anthropic_messages: message contains %d image part(s)", len(image_parts))
                content = [{"type": "text", "text": s} for s in parts if s] + image_parts
            else:
                content = "\n".join(parts)
        if m.get("role") == "system":
            if content:
                openai_msgs.append({"role": "system", "content": content})
            continue
        if m["role"] == "assistant" and (not content or content.strip() == "(no content)"):
            logger.debug("anthropic_messages: skipping empty assistant message")
            continue
        openai_msgs.append({"role": m["role"], "content": content})

    openai_tools = []
    for t in tools:
        if t.get("type") == "function":
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", "NO DESCRIPTION"),
                    "parameters": t.get("input_schema", {}),
                },
            })
        elif "name" in t:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", t.get("parameters", {})),
                },
            })
    if openai_tools:
        logger.debug("anthropic_messages: converted %d tool definitions: %s", len(openai_tools), [t["function"]["name"] for t in openai_tools])

    output_config = body.get("output_config")
    if isinstance(output_config, dict) and output_config.get("format", {}).get("type") == "json_schema":
        json_schema = output_config["format"].get("schema")
        if json_schema:
            logger.debug("anthropic_messages: injecting json_schema constraint into system prompt")
            openai_msgs.insert(0, {"role": "system", "content": f"You MUST return valid JSON adhering strictly to this JSON Schema:\n{json.dumps(json_schema)}"})

    req_model = body.get("model")
    if stream:
        return await handle_chat(openai_msgs, model, thinking, False, True, openai_tools or None, is_anthropic=True, req_model=req_model, scope=get_api_key(request))

    result = await handle_chat(openai_msgs, model, thinking, False, False, openai_tools or None, is_anthropic=True, req_model=req_model, scope=get_api_key(request))
    if not isinstance(result, dict) or "choices" not in result:
        return result
    return format_anthropic_response(result, req_model)


@app.get("/v1/models")
@app.get("/models")
async def list_models(request: Request):
    if not check_key(request):
        return JSONResponse({"error": "Invalid API key"}, status_code=401)
    logger.debug("list_models: request received")

    base_models = [
        {
            "id": "instant",
            "object": "model",
            "type": "model",
            "name": "instant",
            "display_name": "Instant",
            "created": 1785456000,
            "created_at": "2026-07-31T00:00:00Z",
            "owned_by": "deeperseeker",
            "capabilities": {
                "batch": {"supported": True},
                "structured_outputs": {"supported": True},
                "thinking": {
                    "supported": True,
                    "types": {
                        "enabled": {"supported": True},
                        "adaptive": {"supported": True}
                    }
                },
                "effort": {
                    "supported": True,
                    "low": {"supported": True},
                    "medium": {"supported": True}
                },
                "context_management": {
                    "clear_thinking_20251015": {"supported": True},
                    "compact_20260112": {"supported": True},
                    "supported": True
                }
            }
        },
        {
            "id": "expert",
            "object": "model",
            "type": "model",
            "name": "expert",
            "display_name": "Expert",
            "created": 1788134400,
            "created_at": "2026-08-31T00:00:00Z",
            "owned_by": "deeperseeker",
            "capabilities": {
                "batch": {"supported": True},
                "code_execution": {"supported": True},
                "structured_outputs": {"supported": True},
                "thinking": {
                    "supported": True,
                    "types": {
                        "enabled": {"supported": True},
                        "adaptive": {"supported": True}
                    }
                },
                "effort": {
                    "supported": True,
                    "low": {"supported": True},
                    "medium": {"supported": True}
                },
                "context_management": {
                    "clear_thinking_20251015": {"supported": True},
                    "compact_20260112": {"supported": True},
                    "supported": True
                }
            }
        },
        {
            "id": "vision",
            "object": "model",
            "type": "model",
            "name": "vision",
            "display_name": "Vision",
            "created": 1785456000,
            "created_at": "2026-07-31T00:00:00Z",
            "owned_by": "deeperseeker",
            "capabilities": {
                "batch": {"supported": True},
                "image_input": {"supported": True},
                "pdf_input": {"supported": True},
                "structured_outputs": {"supported": True},
                "thinking": {
                    "supported": True,
                    "types": {
                        "enabled": {"supported": True},
                        "adaptive": {"supported": True}
                    }
                },
                "effort": {
                    "supported": True,
                    "low": {"supported": True},
                    "medium": {"supported": True}
                },
                "context_management": {
                    "clear_thinking_20251015": {"supported": True},
                    "compact_20260112": {"supported": True},
                    "supported": True
                }
            }
        }
    ]

    claude_aliases = []
    for m in base_models:
        alias = dict(m)
        alias["id"] = f"anthropic/claude-{m['id']}"
        alias["name"] = f"anthropic/claude-{m['name']}"
        alias["display_name"] = f"Claude {m['display_name']}"
        claude_aliases.append(alias)

    all_models = base_models + claude_aliases
    logger.debug("list_models: returning %d model entries", len(all_models))

    return {
        "object": "list",
        "data": all_models,
        "has_more": False,
        "first_id": all_models[0]["id"],
        "last_id": all_models[-1]["id"]
    }


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    logger.debug("login_page: rendering login form")
    return templates.TemplateResponse(request, "login.html", {"error": None})


@app.post("/login", response_class=HTMLResponse)
async def login_submit(request: Request):
    form = await request.form()
    username = form.get("username", "")
    password = form.get("password", "")
    client_host = request.client.host if request.client else "unknown"
    if time.time() < _login_fails["locked_until"]:
        logger.warning("login_submit: attempt blocked by lockout (client=%s)", client_host)
        return templates.TemplateResponse(request, "login.html", {"error": "Too many attempts. Try again later."})
    if secrets.compare_digest(username.encode("utf-8"), ADMIN_USER.encode("utf-8")) and secrets.compare_digest(password.encode("utf-8"), ADMIN_PASSWORD.encode("utf-8")):
        logger.info("login_submit: successful login for user=%s (client=%s)", username, client_host)
        _login_fails["count"] = 0
        sid = str(uuid.uuid4())
        SESSIONS[sid] = time.time()
        resp = HTMLResponse("<meta http-equiv='refresh' content='0;url=/dashboard'>")
        resp.set_cookie("session_id", sid, httponly=True, samesite="lax")
        return resp
    _login_fails["count"] += 1
    logger.warning("login_submit: failed login attempt #%d for user=%r (client=%s)", _login_fails["count"], username, client_host)
    if _login_fails["count"] >= 5:
        _login_fails["locked_until"] = time.time() + 300
        _login_fails["count"] = 0
        logger.warning("login_submit: lockout triggered for 300s (client=%s)", client_host)
    return templates.TemplateResponse(request, "login.html", {"error": "Invalid username or password"})


@app.get("/logout")
async def logout(request: Request):
    sid = request.cookies.get("session_id")
    logger.info("logout: clearing session sid=%s", _redact_key(sid))
    SESSIONS.pop(sid, None)
    resp = HTMLResponse("<meta http-equiv='refresh' content='0;url=/login'>")
    resp.delete_cookie("session_id")
    return resp


@app.get("/dashboard")
async def dashboard(request: Request):
    try:
        get_current_admin(request)
    except HTTPException:
        logger.debug("dashboard: unauthenticated access, redirecting to login")
        return HTMLResponse("<meta http-equiv='refresh' content='0;url=/login'>")
    tokens = get_tokens()
    logger.debug("dashboard: rendering with %d token(s)", len(tokens))
    return templates.TemplateResponse(request, "dashboard.html", {"tokens": tokens})


@app.post("/tokens/add")
async def tokens_add(request: Request):
    try:
        get_current_admin(request)
    except HTTPException:
        return HTMLResponse("<meta http-equiv='refresh' content='0;url=/login'>")
    form = await request.form()
    auth_token = form.get("auth_token", "").strip().strip("'\"")
    alias = form.get("alias", "").strip() or None
    if auth_token:
        logger.info("tokens_add: adding new token (alias=%s, token=%s)", alias, _redact_key(auth_token))
        add_token(auth_token, alias)
    else:
        logger.warning("tokens_add: empty auth_token submitted, ignoring")
    return HTMLResponse("<meta http-equiv='refresh' content='0;url=/dashboard'>")


@app.post("/tokens/{token_id}/delete")
async def tokens_delete(token_id: int, request: Request):
    try:
        get_current_admin(request)
    except HTTPException:
        return HTMLResponse("<meta http-equiv='refresh' content='0;url=/login'>")
    logger.info("tokens_delete: deleting token_id=%s", token_id)
    delete_token(token_id)
    return HTMLResponse("<meta http-equiv='refresh' content='0;url=/dashboard'>")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return await dashboard(request)


@app.get("/health")
async def health(request: Request):
    active = sum(1 for t in get_tokens() if t["status"] == "ACTIVE")
    cookies_valid = False
    try:
        with open("aws_cookies_deepseek.json") as f:
            c = json.load(f)
        exp = c.get("expiry")
        cookies_valid = bool(exp and exp > time.time())
    except Exception:
        logger.debug("health: could not read/validate aws_cookies_deepseek.json", exc_info=True)
        cookies_valid = False
    ok = active > 0 and cookies_valid
    logger.info("health: active_tokens=%d cookies_valid=%s -> %s", active, cookies_valid, "ok" if ok else "degraded")
    data = {"status": "ok" if ok else "degraded"}
    if check_key(request):
        data["active_tokens"] = active
        data["cookies_valid"] = cookies_valid
    return JSONResponse(data, status_code=200 if ok else 503)


if __name__ == "__main__":
    logger.info("Starting server on %s:%s", os.getenv("HOST", "0.0.0.0"), os.getenv("PORT", "4000"))
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "4000")))
