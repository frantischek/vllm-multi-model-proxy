import hmac
import io
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Dict, Optional, Tuple

import httpx
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image
from pydantic import BaseModel

APP_TITLE = "vLLM Multi-Model Proxy"

# -------------------------
# Config (env-driven)
# -------------------------
INTERNAL_TOKEN = os.getenv("INTERNAL_TOKEN", "")

# Backward-compatible defaults
VLLM_URL = os.getenv("VLLM_URL", "http://127.0.0.1:8000/v1/chat/completions")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")

# Optional task-specific defaults (single-target mode)
VLLM_CHAT_URL = os.getenv("VLLM_CHAT_URL", VLLM_URL)
VLLM_CHAT_MODEL = os.getenv("VLLM_CHAT_MODEL", VLLM_MODEL)
VLLM_CAPTION_MODEL = os.getenv("VLLM_CAPTION_MODEL", VLLM_CHAT_MODEL)
VLLM_IMAGE_URL = os.getenv("VLLM_IMAGE_URL", "")
VLLM_IMAGE_MODEL = os.getenv("VLLM_IMAGE_MODEL", VLLM_CHAT_MODEL)
VLLM_TTS_URL = os.getenv("VLLM_TTS_URL", "")
VLLM_TTS_MODEL = os.getenv("VLLM_TTS_MODEL", VLLM_CHAT_MODEL)

VLLM_TARGETS_JSON = os.getenv("VLLM_TARGETS_JSON", "")
VLLM_DEFAULT_TARGET = os.getenv("VLLM_DEFAULT_TARGET", "").strip()

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

MAX_RESIZE = int(os.getenv("MAX_RESIZE", "1536"))  # longest edge
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "85"))

# Serve resized images locally to vLLM (avoid data: URLs)
DEFAULT_TMP_BASE = Path("/dev/shm") if Path("/dev/shm").is_dir() else Path(gettempdir())
DEFAULT_TMP_DIR = DEFAULT_TMP_BASE / "vllm-multi-model-proxy"
PROXY_TMP_DIR = Path(os.getenv("PROXY_TMP_DIR", str(DEFAULT_TMP_DIR)))
PROXY_TMP_DIR.mkdir(parents=True, exist_ok=True)
PROXY_PUBLIC_BASE = os.getenv("PROXY_PUBLIC_BASE", "http://127.0.0.1:9000")

# Prompt: keep it strict and stable
CAPTION_PROMPT = os.getenv(
    "CAPTION_PROMPT",
    "Beschreibe das Bild in genau 1 deutschen Satz. "
    "Nenne die wichtigsten Objekte (mind. 2), den Raum/Ort und auffällige Details. "
    "Keine Floskeln."
)

# vLLM call defaults
VLLM_TEMPERATURE = float(os.getenv("VLLM_TEMPERATURE", "0.2"))
VLLM_MAX_TOKENS = int(os.getenv("VLLM_MAX_TOKENS", "80"))
VLLM_TIMEOUT_S = int(os.getenv("VLLM_TIMEOUT_S", "240"))

ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_-]+")
SAFE_ID_FULL_RE = re.compile(r"^[A-Za-z0-9_-]{1,96}$")
TARGET_ALIAS_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


class ChatRequest(BaseModel):
    messages: list[Dict[str, Any]]
    target: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False


class ImageGenerateRequest(BaseModel):
    prompt: str
    target: Optional[str] = None
    n: Optional[int] = None
    size: Optional[str] = None
    quality: Optional[str] = None
    response_format: Optional[str] = None


class TTSRequest(BaseModel):
    input: str
    target: Optional[str] = None
    voice: Optional[str] = None
    response_format: Optional[str] = None
    speed: Optional[float] = None


# -------------------------
# Logging
# -------------------------
logger = logging.getLogger("vllm-multi-model-proxy")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s request_id=%(request_id)s msg=%(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title=APP_TITLE)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    rid = _get_request_id(request.headers.get("x-request-id"))
    request.state.request_id = rid
    t0 = time.time()
    resp = await call_next(request)
    dt = int((time.time() - t0) * 1000)
    logger.info(
        f"{request.method} {request.url.path} status={resp.status_code} ms={dt} client={request.client.host if request.client else 'na'}",
        extra={"request_id": rid},
    )
    return resp


def _require_internal_token(token: Optional[str]) -> None:
    if not INTERNAL_TOKEN:
        raise HTTPException(status_code=500, detail="Server is misconfigured")
    if not token or not hmac.compare_digest(token, INTERNAL_TOKEN):
        raise HTTPException(status_code=401, detail="Unauthorized")


def _normalize_target_config(alias: str, cfg: Dict[str, Any]) -> Dict[str, str]:
    legacy_url = cfg.get("url")
    legacy_model = cfg.get("model")

    chat_url = cfg.get("chat_url", legacy_url if legacy_url is not None else VLLM_CHAT_URL)
    chat_model = cfg.get("chat_model", legacy_model if legacy_model is not None else VLLM_CHAT_MODEL)
    caption_model = cfg.get("caption_model", chat_model)

    image_url = cfg.get("image_url", VLLM_IMAGE_URL)
    image_model = cfg.get("image_model", VLLM_IMAGE_MODEL or chat_model)
    tts_url = cfg.get("tts_url", VLLM_TTS_URL)
    tts_model = cfg.get("tts_model", VLLM_TTS_MODEL or chat_model)

    if not isinstance(chat_url, str) or not chat_url.strip():
        raise RuntimeError(f"Target '{alias}' has invalid or empty 'chat_url'")
    if not isinstance(chat_model, str) or not chat_model.strip():
        raise RuntimeError(f"Target '{alias}' has invalid or empty 'chat_model'")
    if not isinstance(caption_model, str) or not caption_model.strip():
        raise RuntimeError(f"Target '{alias}' has invalid or empty 'caption_model'")

    if image_url is None:
        image_url = ""
    if not isinstance(image_url, str):
        raise RuntimeError(f"Target '{alias}' has invalid 'image_url'")
    if not isinstance(image_model, str) or not image_model.strip():
        raise RuntimeError(f"Target '{alias}' has invalid or empty 'image_model'")

    if tts_url is None:
        tts_url = ""
    if not isinstance(tts_url, str):
        raise RuntimeError(f"Target '{alias}' has invalid 'tts_url'")
    if not isinstance(tts_model, str) or not tts_model.strip():
        raise RuntimeError(f"Target '{alias}' has invalid or empty 'tts_model'")

    return {
        "chat_url": chat_url.strip(),
        "chat_model": chat_model.strip(),
        "caption_model": caption_model.strip(),
        "image_url": image_url.strip(),
        "image_model": image_model.strip(),
        "tts_url": tts_url.strip(),
        "tts_model": tts_model.strip(),
    }


def _parse_vllm_targets(raw_json: str) -> Dict[str, Dict[str, str]]:
    raw = (raw_json or "").strip()
    if not raw:
        return {"default": _normalize_target_config("default", {})}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("VLLM_TARGETS_JSON must be valid JSON") from exc

    if not isinstance(parsed, dict) or not parsed:
        raise RuntimeError("VLLM_TARGETS_JSON must be a non-empty JSON object")

    targets: Dict[str, Dict[str, str]] = {}
    for alias, cfg in parsed.items():
        if not isinstance(alias, str) or not TARGET_ALIAS_RE.fullmatch(alias):
            raise RuntimeError(f"Invalid target alias: {alias!r}")
        if not isinstance(cfg, dict):
            raise RuntimeError(
                f"Target '{alias}' must be an object with chat_* / image_* / tts_* fields"
            )
        targets[alias] = _normalize_target_config(alias, cfg)

    return targets


def _resolve_default_target_alias(targets: Dict[str, Dict[str, str]], configured_default: str) -> str:
    if configured_default:
        if configured_default not in targets:
            raise RuntimeError("VLLM_DEFAULT_TARGET is not present in configured targets")
        return configured_default
    if "default" in targets:
        return "default"
    return next(iter(targets))


VLLM_TARGETS = _parse_vllm_targets(VLLM_TARGETS_JSON)
DEFAULT_VLLM_TARGET = _resolve_default_target_alias(VLLM_TARGETS, VLLM_DEFAULT_TARGET)


def _sanitize_identifier(value: str, fallback: str, max_len: int = 64) -> str:
    clean = SAFE_ID_RE.sub("-", value.strip())[:max_len].strip("-")
    if not clean:
        return fallback
    return clean


def _get_request_id(x_request_id: Optional[str]) -> str:
    return _sanitize_identifier(x_request_id or "", fallback=uuid.uuid4().hex, max_len=64)


def _new_image_id(request_id: str) -> str:
    suffix = uuid.uuid4().hex[:12]
    return _sanitize_identifier(f"{request_id}-{suffix}", fallback=uuid.uuid4().hex, max_len=96)


def _resolve_vllm_target(requested_target: Optional[str]) -> Tuple[str, Dict[str, str]]:
    alias = (requested_target or "").strip()
    if not alias:
        alias = DEFAULT_VLLM_TARGET
    if not TARGET_ALIAS_RE.fullmatch(alias) or alias not in VLLM_TARGETS:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Unknown vLLM target",
                "target": alias,
                "available_targets": sorted(VLLM_TARGETS.keys()),
            },
        )
    return alias, VLLM_TARGETS[alias]


async def _read_limited(upload: UploadFile) -> bytes:
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_UPLOAD_MB} MB)")
    return data


def _resize_and_write_jpeg(raw: bytes, image_id: str) -> Tuple[Path, int, int]:
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    w, h = img.size

    # Resize longest edge to MAX_RESIZE (only downscale)
    scale = min(MAX_RESIZE / max(w, h), 1.0)
    if scale < 1.0:
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
        w, h = img.size

    out_path = (PROXY_TMP_DIR / f"{image_id}.jpg").resolve()
    if not out_path.is_relative_to(PROXY_TMP_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Invalid image id")
    # IMPORTANT: keep JPEG simple (no optimize=True) for max compatibility
    img.save(out_path, format="JPEG", quality=JPEG_QUALITY)

    return out_path, w, h


def _cleanup_tmp_image(path: Optional[Path], request_id: str) -> None:
    if not path:
        return
    try:
        path.unlink(missing_ok=True)
    except Exception:
        logger.warning("tmp image cleanup failed", extra={"request_id": request_id})


def _error_response(status: int, request_id: str, code: str, message: str, detail=None):
    body = {
        "error": {
            "code": code,
            "message": message,
            "detail": detail,
            "request_id": request_id,
        }
    }
    return JSONResponse(status_code=status, content=body)


async def _call_upstream(
    url: str, payload: Dict[str, Any], request_id: str
) -> Tuple[Optional[httpx.Response], int, Optional[JSONResponse]]:
    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=VLLM_TIMEOUT_S) as client:
            resp = await client.post(url, json=payload, headers={"X-Request-Id": request_id})
    except httpx.TimeoutException:
        dt_ms = int((time.time() - t0) * 1000)
        logger.warning(f"upstream timeout url={url}", extra={"request_id": request_id})
        return (
            None,
            dt_ms,
            _error_response(
                status=504,
                request_id=request_id,
                code="upstream_timeout",
                message="Upstream request timed out",
                detail={"latency_ms": dt_ms},
            ),
        )
    except Exception as e:
        dt_ms = int((time.time() - t0) * 1000)
        logger.exception(f"upstream error url={url}", extra={"request_id": request_id})
        return (
            None,
            dt_ms,
            _error_response(
                status=502,
                request_id=request_id,
                code="upstream_error",
                message="Upstream request failed",
                detail={"latency_ms": dt_ms},
            ),
        )

    dt_ms = int((time.time() - t0) * 1000)
    return resp, dt_ms, None


def _upstream_bad_status(resp: httpx.Response, request_id: str, dt_ms: int, task: str) -> JSONResponse:
    logger.warning(
        f"upstream bad status task={task} status={resp.status_code} body={resp.text[:2000]}",
        extra={"request_id": request_id},
    )
    return _error_response(
        status=502,
        request_id=request_id,
        code="upstream_bad_response",
        message=f"{task} upstream returned an error",
        detail={
            "upstream_status": resp.status_code,
            "latency_ms": dt_ms,
        },
    )


def _upstream_json_or_error(
    resp: httpx.Response, request_id: str, dt_ms: int, task: str
) -> Tuple[Optional[Dict[str, Any]], Optional[JSONResponse]]:
    try:
        body = resp.json()
    except Exception:
        logger.warning(
            f"upstream invalid JSON body={resp.text[:2000]}",
            extra={"request_id": request_id},
        )
        return None, _error_response(
            status=502,
            request_id=request_id,
            code="upstream_invalid_json",
            message=f"{task} upstream returned invalid JSON",
            detail={"latency_ms": dt_ms},
        )
    if not isinstance(body, dict):
        return None, _error_response(
            status=502,
            request_id=request_id,
            code="upstream_invalid_json_shape",
            message=f"{task} upstream JSON must be an object",
            detail={"latency_ms": dt_ms},
        )
    return body, None


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex)
    detail = exc.detail
    return _error_response(
        status=exc.status_code,
        request_id=request_id,
        code="http_error",
        message=str(detail) if not isinstance(detail, dict) else "Request failed",
        detail=detail if isinstance(detail, dict) else None,
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex)
    logger.exception("unhandled exception", extra={"request_id": request_id})
    return _error_response(
        status=500,
        request_id=request_id,
        code="internal_error",
        message="Internal server error",
    )


@app.get("/health")
async def health():
    if not INTERNAL_TOKEN:
        return {"ok": False, "service": APP_TITLE, "error": "INTERNAL_TOKEN not set"}
    return {
        "ok": True,
        "service": APP_TITLE,
        "default_target": DEFAULT_VLLM_TARGET,
        "target_count": len(VLLM_TARGETS),
        "capabilities": ["caption", "chat", "image_generate", "tts"],
    }


@app.get("/internal/models")
async def list_models(
    request: Request,
    x_internal_token: Optional[str] = Header(default=None),
):
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex)
    request.state.request_id = request_id
    _require_internal_token(x_internal_token)
    return {
        "request_id": request_id,
        "default_target": DEFAULT_VLLM_TARGET,
        "targets": [
            {
                "alias": alias,
                "chat_model": cfg["chat_model"],
                "caption_model": cfg["caption_model"],
                "image_model": cfg["image_model"],
                "tts_model": cfg["tts_model"],
                "chat_url": cfg["chat_url"],
                "image_url": cfg["image_url"],
                "tts_url": cfg["tts_url"],
                "capabilities": {
                    "caption": True,
                    "chat": True,
                    "image_generate": bool(cfg["image_url"]),
                    "tts": bool(cfg["tts_url"]),
                },
            }
            for alias, cfg in sorted(VLLM_TARGETS.items())
        ],
    }


@app.get("/_img/{rid}.jpg")
async def get_tmp_image(request: Request, rid: str):
    # vLLM should fetch via localhost; block non-local clients
    ip = request.client.host if request.client else "unknown"
    if ip not in ("127.0.0.1", "::1"):
        raise HTTPException(status_code=403, detail="Forbidden")
    if not SAFE_ID_FULL_RE.fullmatch(rid):
        raise HTTPException(status_code=404, detail="Not found")

    p = (PROXY_TMP_DIR / f"{rid}.jpg").resolve()
    if not p.is_relative_to(PROXY_TMP_DIR.resolve()):
        raise HTTPException(status_code=403, detail="Forbidden")
    if not p.exists():
        raise HTTPException(status_code=404, detail="Not found")

    return FileResponse(str(p), media_type="image/jpeg")


@app.post("/internal/caption")
async def caption(
    request: Request,
    image: UploadFile = File(...),
    target: Optional[str] = Form(default=None),
    x_internal_token: Optional[str] = Header(default=None),
    x_request_id: Optional[str] = Header(default=None),
    x_vllm_target: Optional[str] = Header(default=None),
):
    request_id = _sanitize_identifier(
        x_request_id or getattr(request.state, "request_id", ""),
        fallback=uuid.uuid4().hex,
        max_len=64,
    )
    request.state.request_id = request_id

    _require_internal_token(x_internal_token)
    client_ip = request.client.host if request.client else "unknown"

    if image.content_type not in ALLOWED_MIME:
        raise HTTPException(status_code=415, detail=f"Unsupported image type: {image.content_type}")

    requested_target = target if target is not None else x_vllm_target
    target_alias, target_cfg = _resolve_vllm_target(requested_target)

    raw = await _read_limited(image)

    # decode + resize + write
    image_id = _new_image_id(request_id)
    image_path: Optional[Path] = None
    image_path, w, h = _resize_and_write_jpeg(raw, image_id)
    image_url = f"{PROXY_PUBLIC_BASE.rstrip('/')}/_img/{image_id}.jpg"

    payload = {
        "model": target_cfg["caption_model"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": CAPTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "temperature": VLLM_TEMPERATURE,
        "max_tokens": VLLM_MAX_TOKENS,
    }

    try:
        resp, dt_ms, err = await _call_upstream(target_cfg["chat_url"], payload, request_id)
        if err is not None:
            return err
        if resp is None:
            return _error_response(502, request_id, "upstream_error", "Unexpected upstream state")

        if resp.status_code != 200:
            return _upstream_bad_status(resp, request_id, dt_ms, "caption")

        body, parse_err = _upstream_json_or_error(resp, request_id, dt_ms, "caption")
        if parse_err is not None:
            return parse_err
        if body is None:
            return _error_response(502, request_id, "upstream_invalid_json", "Missing body")

        try:
            caption_text = body["choices"][0]["message"]["content"].strip()
        except Exception:
            logger.warning(
                f"bad caption response shape body={json.dumps(body)[:2000]}",
                extra={"request_id": request_id},
            )
            return _error_response(
                status=502,
                request_id=request_id,
                code="bad_model_response",
                message="Unexpected caption response format",
                detail={"latency_ms": dt_ms},
            )

        logger.info(
            f"ok task=caption status=200 latency_ms={dt_ms} client_ip={client_ip} target={target_alias} resized={w}x{h}",
            extra={"request_id": request_id},
        )

        return {
            "request_id": request_id,
            "task": "caption",
            "target": target_alias,
            "model": target_cfg["caption_model"],
            "caption": caption_text,
            "latency_ms": dt_ms,
            "image_after_resize": {"width": w, "height": h, "max_edge": MAX_RESIZE},
        }
    finally:
        _cleanup_tmp_image(image_path, request_id)


@app.post("/internal/chat")
async def chat(
    request: Request,
    payload: ChatRequest,
    x_internal_token: Optional[str] = Header(default=None),
    x_request_id: Optional[str] = Header(default=None),
    x_vllm_target: Optional[str] = Header(default=None),
):
    request_id = _sanitize_identifier(
        x_request_id or getattr(request.state, "request_id", ""),
        fallback=uuid.uuid4().hex,
        max_len=64,
    )
    request.state.request_id = request_id
    _require_internal_token(x_internal_token)

    if payload.stream:
        raise HTTPException(status_code=400, detail="Streaming is not supported by this proxy endpoint")
    if not payload.messages:
        raise HTTPException(status_code=422, detail="messages must not be empty")

    requested_target = payload.target if payload.target is not None else x_vllm_target
    target_alias, target_cfg = _resolve_vllm_target(requested_target)

    messages = list(payload.messages)
    if payload.system_prompt:
        messages = [{"role": "system", "content": payload.system_prompt}] + messages

    upstream_payload: Dict[str, Any] = {
        "model": target_cfg["chat_model"],
        "messages": messages,
        "stream": False,
        "temperature": payload.temperature if payload.temperature is not None else VLLM_TEMPERATURE,
        "max_tokens": payload.max_tokens if payload.max_tokens is not None else VLLM_MAX_TOKENS,
    }

    resp, dt_ms, err = await _call_upstream(target_cfg["chat_url"], upstream_payload, request_id)
    if err is not None:
        return err
    if resp is None:
        return _error_response(502, request_id, "upstream_error", "Unexpected upstream state")

    if resp.status_code != 200:
        return _upstream_bad_status(resp, request_id, dt_ms, "chat")

    body, parse_err = _upstream_json_or_error(resp, request_id, dt_ms, "chat")
    if parse_err is not None:
        return parse_err
    if body is None:
        return _error_response(502, request_id, "upstream_invalid_json", "Missing body")

    text_preview = None
    try:
        text_preview = body["choices"][0]["message"]["content"]
    except Exception:
        text_preview = None

    logger.info(
        f"ok task=chat status=200 latency_ms={dt_ms} target={target_alias}",
        extra={"request_id": request_id},
    )

    return {
        "request_id": request_id,
        "task": "chat",
        "target": target_alias,
        "model": target_cfg["chat_model"],
        "latency_ms": dt_ms,
        "text": text_preview,
        "result": body,
    }


@app.post("/internal/image-generate")
async def image_generate(
    request: Request,
    payload: ImageGenerateRequest,
    x_internal_token: Optional[str] = Header(default=None),
    x_request_id: Optional[str] = Header(default=None),
    x_vllm_target: Optional[str] = Header(default=None),
):
    request_id = _sanitize_identifier(
        x_request_id or getattr(request.state, "request_id", ""),
        fallback=uuid.uuid4().hex,
        max_len=64,
    )
    request.state.request_id = request_id
    _require_internal_token(x_internal_token)

    requested_target = payload.target if payload.target is not None else x_vllm_target
    target_alias, target_cfg = _resolve_vllm_target(requested_target)

    if not target_cfg["image_url"]:
        raise HTTPException(
            status_code=400,
            detail=f"Target '{target_alias}' does not have image generation configured (missing image_url)",
        )

    upstream_payload: Dict[str, Any] = {
        "model": target_cfg["image_model"],
        "prompt": payload.prompt,
    }
    if payload.n is not None:
        upstream_payload["n"] = payload.n
    if payload.size is not None:
        upstream_payload["size"] = payload.size
    if payload.quality is not None:
        upstream_payload["quality"] = payload.quality
    if payload.response_format is not None:
        upstream_payload["response_format"] = payload.response_format

    resp, dt_ms, err = await _call_upstream(target_cfg["image_url"], upstream_payload, request_id)
    if err is not None:
        return err
    if resp is None:
        return _error_response(502, request_id, "upstream_error", "Unexpected upstream state")

    if resp.status_code != 200:
        return _upstream_bad_status(resp, request_id, dt_ms, "image_generate")

    body, parse_err = _upstream_json_or_error(resp, request_id, dt_ms, "image_generate")
    if parse_err is not None:
        return parse_err
    if body is None:
        return _error_response(502, request_id, "upstream_invalid_json", "Missing body")

    logger.info(
        f"ok task=image_generate status=200 latency_ms={dt_ms} target={target_alias}",
        extra={"request_id": request_id},
    )

    return {
        "request_id": request_id,
        "task": "image_generate",
        "target": target_alias,
        "model": target_cfg["image_model"],
        "latency_ms": dt_ms,
        "result": body,
    }


@app.post("/internal/tts")
async def tts(
    request: Request,
    payload: TTSRequest,
    x_internal_token: Optional[str] = Header(default=None),
    x_request_id: Optional[str] = Header(default=None),
    x_vllm_target: Optional[str] = Header(default=None),
):
    request_id = _sanitize_identifier(
        x_request_id or getattr(request.state, "request_id", ""),
        fallback=uuid.uuid4().hex,
        max_len=64,
    )
    request.state.request_id = request_id
    _require_internal_token(x_internal_token)

    requested_target = payload.target if payload.target is not None else x_vllm_target
    target_alias, target_cfg = _resolve_vllm_target(requested_target)

    if not target_cfg["tts_url"]:
        raise HTTPException(
            status_code=400,
            detail=f"Target '{target_alias}' does not have TTS configured (missing tts_url)",
        )

    upstream_payload: Dict[str, Any] = {
        "model": target_cfg["tts_model"],
        "input": payload.input,
    }
    if payload.voice is not None:
        upstream_payload["voice"] = payload.voice
    if payload.response_format is not None:
        upstream_payload["response_format"] = payload.response_format
    if payload.speed is not None:
        upstream_payload["speed"] = payload.speed

    resp, dt_ms, err = await _call_upstream(target_cfg["tts_url"], upstream_payload, request_id)
    if err is not None:
        return err
    if resp is None:
        return _error_response(502, request_id, "upstream_error", "Unexpected upstream state")

    if resp.status_code != 200:
        return _upstream_bad_status(resp, request_id, dt_ms, "tts")

    content_type = (resp.headers.get("content-type") or "").lower()

    logger.info(
        f"ok task=tts status=200 latency_ms={dt_ms} target={target_alias}",
        extra={"request_id": request_id},
    )

    if "application/json" in content_type:
        body, parse_err = _upstream_json_or_error(resp, request_id, dt_ms, "tts")
        if parse_err is not None:
            return parse_err
        if body is None:
            return _error_response(502, request_id, "upstream_invalid_json", "Missing body")
        return {
            "request_id": request_id,
            "task": "tts",
            "target": target_alias,
            "model": target_cfg["tts_model"],
            "latency_ms": dt_ms,
            "result": body,
        }

    media_type = resp.headers.get("content-type", "application/octet-stream")
    return Response(
        content=resp.content,
        media_type=media_type,
        headers={
            "X-Request-Id": request_id,
            "X-VLLM-Target": target_alias,
            "X-VLLM-Model": target_cfg["tts_model"],
        },
    )
