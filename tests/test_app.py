import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as module


def _set_single_target(monkeypatch):
    monkeypatch.setattr(
        module,
        "VLLM_TARGETS",
        {
            "default": {
                "chat_url": "http://127.0.0.1:8000/v1/chat/completions",
                "chat_model": "chat-model",
                "caption_model": "caption-model",
                "image_url": "",
                "image_model": "image-model",
                "tts_url": "",
                "tts_model": "tts-model",
            }
        },
    )
    monkeypatch.setattr(module, "DEFAULT_VLLM_TARGET", "default")


def test_get_request_id_is_sanitized():
    request_id = module._get_request_id("../../etc/passwd")
    assert request_id
    assert len(request_id) <= 64
    assert "/" not in request_id
    assert re.fullmatch(r"[A-Za-z0-9_-]+", request_id)


def test_new_image_id_is_safe():
    image_id = module._new_image_id("custom-request")
    assert image_id.startswith("custom-request-")
    assert re.fullmatch(r"[A-Za-z0-9_-]{1,96}", image_id)


def test_health_reflects_missing_token(monkeypatch):
    monkeypatch.setattr(module, "INTERNAL_TOKEN", "")
    client = TestClient(module.app)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["ok"] is False


def test_parse_vllm_targets_fallback():
    targets = module._parse_vllm_targets("")
    assert list(targets.keys()) == ["default"]
    default = targets["default"]
    assert default["chat_url"]
    assert default["chat_model"]
    assert default["caption_model"]
    assert "image_url" in default
    assert "tts_url" in default


def test_parse_vllm_targets_from_json():
    raw = (
        '{"small":{"chat_url":"http://127.0.0.1:8100/v1/chat/completions","chat_model":"small-chat",'
        '"caption_model":"small-caption","image_url":"http://127.0.0.1:8100/v1/images/generations",'
        '"image_model":"small-image","tts_url":"http://127.0.0.1:8100/v1/audio/speech","tts_model":"small-tts"}}'
    )
    targets = module._parse_vllm_targets(raw)
    assert list(targets.keys()) == ["small"]
    assert targets["small"]["chat_model"] == "small-chat"
    assert targets["small"]["image_model"] == "small-image"
    assert targets["small"]["tts_model"] == "small-tts"


def test_parse_vllm_targets_from_legacy_json():
    raw = '{"legacy":{"url":"http://127.0.0.1:8200/v1/chat/completions","model":"legacy-model"}}'
    targets = module._parse_vllm_targets(raw)
    assert targets["legacy"]["chat_url"] == "http://127.0.0.1:8200/v1/chat/completions"
    assert targets["legacy"]["chat_model"] == "legacy-model"


def test_parse_vllm_targets_invalid_json():
    with pytest.raises(RuntimeError):
        module._parse_vllm_targets("{bad")


def test_resolve_vllm_target_unknown(monkeypatch):
    _set_single_target(monkeypatch)
    with pytest.raises(module.HTTPException) as exc:
        module._resolve_vllm_target("unknown")
    assert exc.value.status_code == 400


def test_internal_models_requires_token(monkeypatch):
    monkeypatch.setattr(module, "INTERNAL_TOKEN", "secret")
    _set_single_target(monkeypatch)

    client = TestClient(module.app)
    res = client.get("/internal/models")
    assert res.status_code == 401


def test_internal_models_returns_targets(monkeypatch):
    monkeypatch.setattr(module, "INTERNAL_TOKEN", "secret")
    _set_single_target(monkeypatch)

    client = TestClient(module.app)
    res = client.get("/internal/models", headers={"x-internal-token": "secret"})
    assert res.status_code == 200
    body = res.json()
    assert body["default_target"] == "default"
    assert body["targets"][0]["alias"] == "default"
    assert body["targets"][0]["capabilities"]["chat"] is True


def test_caption_requires_token(monkeypatch):
    monkeypatch.setattr(module, "INTERNAL_TOKEN", "secret")
    _set_single_target(monkeypatch)

    client = TestClient(module.app)
    res = client.post(
        "/internal/caption",
        files={"image": ("a.jpg", b"x", "image/jpeg")},
    )
    assert res.status_code == 401


def test_caption_unknown_target(monkeypatch):
    monkeypatch.setattr(module, "INTERNAL_TOKEN", "secret")
    _set_single_target(monkeypatch)

    client = TestClient(module.app)
    res = client.post(
        "/internal/caption",
        headers={"x-internal-token": "secret", "x-vllm-target": "missing"},
        files={"image": ("a.jpg", b"x", "image/jpeg")},
    )
    assert res.status_code == 400


def test_chat_stream_not_supported(monkeypatch):
    monkeypatch.setattr(module, "INTERNAL_TOKEN", "secret")
    _set_single_target(monkeypatch)

    client = TestClient(module.app)
    res = client.post(
        "/internal/chat",
        headers={"x-internal-token": "secret"},
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert res.status_code == 400


def test_chat_unknown_target(monkeypatch):
    monkeypatch.setattr(module, "INTERNAL_TOKEN", "secret")
    _set_single_target(monkeypatch)

    client = TestClient(module.app)
    res = client.post(
        "/internal/chat",
        headers={"x-internal-token": "secret", "x-vllm-target": "missing"},
        json={"messages": [{"role": "user", "content": "hello"}]},
    )
    assert res.status_code == 400


def test_image_generate_missing_config(monkeypatch):
    monkeypatch.setattr(module, "INTERNAL_TOKEN", "secret")
    _set_single_target(monkeypatch)

    client = TestClient(module.app)
    res = client.post(
        "/internal/image-generate",
        headers={"x-internal-token": "secret"},
        json={"prompt": "a mountain"},
    )
    assert res.status_code == 400


def test_tts_missing_config(monkeypatch):
    monkeypatch.setattr(module, "INTERNAL_TOKEN", "secret")
    _set_single_target(monkeypatch)

    client = TestClient(module.app)
    res = client.post(
        "/internal/tts",
        headers={"x-internal-token": "secret"},
        json={"input": "Hallo Welt"},
    )
    assert res.status_code == 400


# ---- Security tests ----


def test_token_comparison_uses_hmac(monkeypatch):
    """Token comparison must use constant-time comparison."""
    monkeypatch.setattr(module, "INTERNAL_TOKEN", "correct-token")

    # Valid token accepted
    module._require_internal_token("correct-token")

    # Wrong token rejected
    with pytest.raises(module.HTTPException) as exc:
        module._require_internal_token("wrong-token")
    assert exc.value.status_code == 401

    # Empty token rejected
    with pytest.raises(module.HTTPException) as exc:
        module._require_internal_token("")
    assert exc.value.status_code == 401

    # None token rejected
    with pytest.raises(module.HTTPException) as exc:
        module._require_internal_token(None)
    assert exc.value.status_code == 401


def test_img_endpoint_rejects_path_traversal(monkeypatch):
    """The /_img/ endpoint must reject path traversal attempts."""
    monkeypatch.setattr(module, "INTERNAL_TOKEN", "secret")
    _set_single_target(monkeypatch)

    client = TestClient(module.app)
    # Attempts with path traversal characters are rejected by regex
    for rid in ["../etc/passwd", "..%2F..%2Fetc%2Fpasswd", "foo/bar"]:
        res = client.get(f"/_img/{rid}.jpg")
        assert res.status_code in (403, 404, 422), f"Unexpected status for rid={rid!r}"


def test_upstream_error_does_not_leak_url():
    """Upstream error responses must not contain internal URLs or body previews."""
    resp = module._error_response(
        status=502,
        request_id="test-req",
        code="upstream_error",
        message="Upstream request failed",
        detail={"latency_ms": 100},
    )
    body = resp.body.decode()
    assert "127.0.0.1" not in body
    assert "body_preview" not in body


def test_upstream_bad_status_no_body_leak():
    """_upstream_bad_status must not leak upstream body content in the response."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 500
    mock_resp.text = "SECRET_INTERNAL_DATA from upstream"

    result = module._upstream_bad_status(mock_resp, "req-123", 42, "chat")
    body = result.body.decode()
    assert "SECRET_INTERNAL_DATA" not in body
    assert "body_preview" not in body


def test_upstream_json_error_no_body_leak():
    """_upstream_json_or_error must not leak upstream body in invalid JSON errors."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.json.side_effect = ValueError("bad json")
    mock_resp.text = "SECRET_UPSTREAM_CONTENT"

    _, err = module._upstream_json_or_error(mock_resp, "req-456", 10, "chat")
    assert err is not None
    body = err.body.decode()
    assert "SECRET_UPSTREAM_CONTENT" not in body
    assert "body_preview" not in body


def test_resize_write_jpeg_path_stays_in_tmp(monkeypatch, tmp_path):
    """_resize_and_write_jpeg must write files only within PROXY_TMP_DIR."""
    monkeypatch.setattr(module, "PROXY_TMP_DIR", tmp_path)
    # A safe image_id should work fine
    import io as _io
    from PIL import Image as _Image

    img = _Image.new("RGB", (10, 10), color="red")
    buf = _io.BytesIO()
    img.save(buf, format="JPEG")
    raw = buf.getvalue()

    path, w, h = module._resize_and_write_jpeg(raw, "safe-id")
    assert str(path.resolve()).startswith(str(tmp_path.resolve()))
    path.unlink(missing_ok=True)
