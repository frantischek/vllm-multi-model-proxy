import io
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from tempfile import gettempdir
from typing import Dict, Optional, Tuple

import httpx
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

APP_TITLE = "vLLM Multi-Model Proxy"

# -------------------------
# Config (env-driven)
# -------------------------
INTERNAL_TOKEN = os.getenv("INTERNAL_TOKEN", "")
VLLM_URL = os.getenv("VLLM_URL", "http://127.0.0.1:8000/v1/chat/completions")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
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
    if not token or token != INTERNAL_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _parse_vllm_targets(raw_json: str, fallback_url: str, fallback_model: str) -> Dict[str, Dict[str, str]]:
    raw = (raw_json or "").strip()
    if not raw:
        return {"default": {"url": fallback_url, "model": fallback_model}}

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
            raise RuntimeError(f"Target '{alias}' must be an object with 'url' and 'model'")

        url = cfg.get("url")
        model = cfg.get("model")
        if not isinstance(url, str) or not url.strip():
            raise RuntimeError(f"Target '{alias}' has invalid or empty 'url'")
        if not isinstance(model, str) or not model.strip():
            raise RuntimeError(f"Target '{alias}' has invalid or empty 'model'")

        targets[alias] = {"url": url.strip(), "model": model.strip()}

    return targets


def _resolve_default_target_alias(targets: Dict[str, Dict[str, str]], configured_default: str) -> str:
    if configured_default:
        if configured_default not in targets:
            raise RuntimeError("VLLM_DEFAULT_TARGET is not present in configured targets")
        return configured_default
    if "default" in targets:
        return "default"
    return next(iter(targets))


VLLM_TARGETS = _parse_vllm_targets(VLLM_TARGETS_JSON, VLLM_URL, VLLM_MODEL)
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


def _resolve_vllm_target(requested_target: Optional[str]) -> Tuple[str, str, str]:
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
    target_cfg = VLLM_TARGETS[alias]
    return alias, target_cfg["url"], target_cfg["model"]


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

    out_path = PROXY_TMP_DIR / f"{image_id}.jpg"
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
            {"alias": alias, "model": cfg["model"], "url": cfg["url"]}
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

    p = PROXY_TMP_DIR / f"{rid}.jpg"
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
    target_alias, target_url, target_model = _resolve_vllm_target(requested_target)

    raw = await _read_limited(image)

    # decode + resize + write
    image_id = _new_image_id(request_id)
    image_path: Optional[Path] = None
    image_path, w, h = _resize_and_write_jpeg(raw, image_id)
    image_url = f"{PROXY_PUBLIC_BASE.rstrip('/')}/_img/{image_id}.jpg"

    payload = {
        "model": target_model,
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

    t0 = time.time()
    try:
        try:
            async with httpx.AsyncClient(timeout=VLLM_TIMEOUT_S) as client:
                resp = await client.post(target_url, json=payload, headers={"X-Request-Id": request_id})
        except httpx.TimeoutException:
            dt_ms = int((time.time() - t0) * 1000)
            logger.warning("vllm timeout", extra={"request_id": request_id})
            return _error_response(
                status=504,
                request_id=request_id,
                code="upstream_timeout",
                message="vLLM request timed out",
                detail={"latency_ms": dt_ms},
            )
        except Exception as e:
            dt_ms = int((time.time() - t0) * 1000)
            logger.exception("upstream error", extra={"request_id": request_id})
            return _error_response(
                status=502,
                request_id=request_id,
                code="upstream_error",
                message="vLLM request failed",
                detail={"exception": str(e), "latency_ms": dt_ms},
            )

        dt_ms = int((time.time() - t0) * 1000)

        if resp.status_code != 200:
            body_preview = resp.text[:2000]
            logger.warning(f"vllm non-200 status={resp.status_code}", extra={"request_id": request_id})
            return _error_response(
                status=502,
                request_id=request_id,
                code="upstream_bad_response",
                message="vLLM returned an error",
                detail={
                    "vllm_status": resp.status_code,
                    "body_preview": body_preview,
                    "latency_ms": dt_ms,
                },
            )

        try:
            j = resp.json()
        except Exception:
            logger.warning("vllm invalid json", extra={"request_id": request_id})
            return _error_response(
                status=502,
                request_id=request_id,
                code="upstream_invalid_json",
                message="vLLM returned invalid JSON",
                detail={"latency_ms": dt_ms, "body_preview": resp.text[:2000]},
            )

        try:
            caption_text = j["choices"][0]["message"]["content"].strip()
        except Exception:
            logger.warning("bad model response shape", extra={"request_id": request_id})
            return _error_response(
                status=502,
                request_id=request_id,
                code="bad_model_response",
                message="Unexpected vLLM response format",
                detail={"latency_ms": dt_ms, "body_preview": json.dumps(j)[:2000]},
            )

        logger.info(
            f"ok status=200 latency_ms={dt_ms} client_ip={client_ip} target={target_alias} resized={w}x{h}",
            extra={"request_id": request_id},
        )

        return {
            "request_id": request_id,
            "caption": caption_text,
            "latency_ms": dt_ms,
            "target": target_alias,
            "model": target_model,
            "image_after_resize": {"width": w, "height": h, "max_edge": MAX_RESIZE},
        }
    finally:
        _cleanup_tmp_image(image_path, request_id)
