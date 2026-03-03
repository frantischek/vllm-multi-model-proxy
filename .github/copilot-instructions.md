# Copilot Instructions for vLLM Multi-Model Proxy

## Project Overview

This is a lightweight **FastAPI** proxy (Python 3.11+) for routing requests to multiple vLLM model backends. It supports four task types: image captioning, chat, image generation, and text-to-speech (TTS). All endpoints live under `/internal/*` and are protected by a shared token (`INTERNAL_TOKEN`).

## Repository Layout

- `app.py` — The entire application in a single file (FastAPI app, config, routes, helpers).
- `tests/test_app.py` — Pytest test suite using `fastapi.testclient.TestClient`.
- `requirements.txt` — Python dependencies (fastapi, uvicorn, httpx, pillow, python-multipart, pytest).
- `.env.example` — All supported environment variables with documentation.
- `.github/workflows/ci.yml` — CI workflow: installs deps and runs `pytest -q` on Python 3.11.
- `CONTRIBUTING.md` — Contribution guidelines and PR checklist.
- `SECURITY.md` — Security policy and practices.

## Build & Test Commands

Always run these from the repository root:

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (always do this before building or testing)
pip install -r requirements.txt

# Run the test suite
pytest -q

# Run a specific test
pytest tests/test_app.py::test_health_reflects_missing_token -q

# Start the dev server (requires .env file)
cp .env.example .env
uvicorn app:app --env-file .env --host 0.0.0.0 --port 9000
```

There is no separate build step; the project is pure Python.

## Coding Conventions

- All application code is in `app.py`. Keep it as a single-file application.
- Use Python type hints (e.g., `Optional[str]`, `Dict[str, Any]`).
- Use Pydantic `BaseModel` for request body schemas.
- Environment variables drive all configuration — never hard-code secrets or URLs.
- Prefix internal helper functions with an underscore (e.g., `_get_request_id`, `_resolve_vllm_target`).
- Return structured JSON error responses with `request_id` on failures.
- Use `logging` (module-level `log` logger) for all log output; do not use `print`.
- Regex patterns are pre-compiled as module-level constants (e.g., `SAFE_ID_RE`, `TARGET_ALIAS_RE`).

## Testing Conventions

- Tests are in `tests/test_app.py` and use `pytest` with `monkeypatch` for patching module-level config.
- Use `fastapi.testclient.TestClient` to test endpoints.
- Use the `_set_single_target(monkeypatch)` helper to set up a standard single-target fixture.
- Test names follow `test_<feature>_<scenario>` (e.g., `test_caption_requires_token`).
- Add or update tests for any behavior changes.

## Security Rules

- Never commit secrets or tokens — `.env` is gitignored.
- Always validate and sanitize user-supplied IDs (request IDs, target aliases) using the existing regex patterns.
- Enforce upload size limits (`MAX_UPLOAD_BYTES`) and MIME type validation (`ALLOWED_MIME`).
- Keep `INTERNAL_TOKEN` checks on all `/internal/*` endpoints.

## CI Pipeline

The CI workflow (`.github/workflows/ci.yml`) runs on every push to `main` and on all pull requests:
1. Checks out the repo
2. Sets up Python 3.11
3. Installs dependencies via `pip install -r requirements.txt`
4. Runs `pytest -q`

Ensure all tests pass locally with `pytest -q` before submitting a PR.
