import re
import sys
from pathlib import Path

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
