import re
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as module


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


def test_caption_requires_token(monkeypatch):
    monkeypatch.setattr(module, "INTERNAL_TOKEN", "secret")
    client = TestClient(module.app)
    res = client.post(
        "/internal/caption",
        files={"image": ("a.jpg", b"x", "image/jpeg")},
    )
    assert res.status_code == 401


def test_parse_vllm_targets_fallback():
    targets = module._parse_vllm_targets("", "http://vllm.local/v1/chat/completions", "vision-model")
    assert targets == {
        "default": {"url": "http://vllm.local/v1/chat/completions", "model": "vision-model"}
    }


def test_parse_vllm_targets_from_json():
    raw = (
        '{"small":{"url":"http://127.0.0.1:8100/v1/chat/completions","model":"small-model"},'
        '"large":{"url":"http://127.0.0.1:8200/v1/chat/completions","model":"large-model"}}'
    )
    targets = module._parse_vllm_targets(raw, "ignored", "ignored")
    assert sorted(targets.keys()) == ["large", "small"]
    assert targets["small"]["model"] == "small-model"


def test_parse_vllm_targets_invalid_json():
    with pytest.raises(RuntimeError):
        module._parse_vllm_targets("{bad", "u", "m")


def test_resolve_vllm_target_unknown(monkeypatch):
    monkeypatch.setattr(
        module,
        "VLLM_TARGETS",
        {"default": {"url": "http://127.0.0.1:8000/v1/chat/completions", "model": "demo"}},
    )
    monkeypatch.setattr(module, "DEFAULT_VLLM_TARGET", "default")
    with pytest.raises(module.HTTPException) as exc:
        module._resolve_vllm_target("unknown")
    assert exc.value.status_code == 400


def test_internal_models_requires_token(monkeypatch):
    monkeypatch.setattr(module, "INTERNAL_TOKEN", "secret")
    monkeypatch.setattr(
        module,
        "VLLM_TARGETS",
        {"default": {"url": "http://127.0.0.1:8000/v1/chat/completions", "model": "demo"}},
    )
    monkeypatch.setattr(module, "DEFAULT_VLLM_TARGET", "default")

    client = TestClient(module.app)
    res = client.get("/internal/models")
    assert res.status_code == 401


def test_internal_models_returns_targets(monkeypatch):
    monkeypatch.setattr(module, "INTERNAL_TOKEN", "secret")
    monkeypatch.setattr(
        module,
        "VLLM_TARGETS",
        {
            "default": {"url": "http://127.0.0.1:8000/v1/chat/completions", "model": "demo"},
            "fast": {"url": "http://127.0.0.1:8100/v1/chat/completions", "model": "demo-fast"},
        },
    )
    monkeypatch.setattr(module, "DEFAULT_VLLM_TARGET", "default")

    client = TestClient(module.app)
    res = client.get("/internal/models", headers={"x-internal-token": "secret"})
    assert res.status_code == 200
    body = res.json()
    assert body["default_target"] == "default"
    assert [item["alias"] for item in body["targets"]] == ["default", "fast"]


def test_caption_unknown_target(monkeypatch):
    monkeypatch.setattr(module, "INTERNAL_TOKEN", "secret")
    monkeypatch.setattr(
        module,
        "VLLM_TARGETS",
        {"default": {"url": "http://127.0.0.1:8000/v1/chat/completions", "model": "demo"}},
    )
    monkeypatch.setattr(module, "DEFAULT_VLLM_TARGET", "default")
    client = TestClient(module.app)
    res = client.post(
        "/internal/caption",
        headers={"x-internal-token": "secret", "x-vllm-target": "missing"},
        files={"image": ("a.jpg", b"x", "image/jpeg")},
    )
    assert res.status_code == 400
