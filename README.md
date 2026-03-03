# GX10 Inference Proxy

A lightweight FastAPI proxy that accepts an uploaded image, resizes it safely,
forwards it to a vLLM vision model, and returns a single German caption sentence.

The service is designed for internal usage (`/internal/caption`) and can be hosted
publicly on GitHub for collaborative improvements.

## Features

- Token-protected internal endpoint
- Multi-target routing for multiple vLLM model backends
- Strict upload limits and MIME type validation
- Image resize pipeline for stable vLLM input
- Structured JSON error responses with `request_id`
- Temporary image cleanup after each request
- Basic CI and contribution templates

## Architecture

1. Client uploads an image to `POST /internal/caption`
2. Proxy validates token and input
3. Proxy resizes image and stores it as temporary JPEG
4. vLLM fetches image from `GET /_img/{rid}.jpg`
5. Proxy returns model caption and metadata
6. Temporary file is deleted

## Quickstart

### 1) Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure environment

```bash
cp .env.example .env
```

Set at least:

- `INTERNAL_TOKEN`
- `VLLM_URL` (single-target mode) or `VLLM_TARGETS_JSON` (multi-target mode)
- `PROXY_PUBLIC_BASE`

### 4) Run service

```bash
uvicorn app:app --env-file .env --host 0.0.0.0 --port 9000
```

## API

### `GET /health`

Health endpoint.

### `POST /internal/caption`

Headers:

- `x-internal-token: <INTERNAL_TOKEN>`
- `x-request-id: <optional custom id>`
- `x-vllm-target: <optional target alias>`

Multipart form:

- `image`: jpeg/png/webp
- `target`: optional alias (overrides `x-vllm-target`)

Example:

```bash
curl -sS -X POST "http://127.0.0.1:9000/internal/caption" \
  -H "x-internal-token: dev-token" \
  -H "x-vllm-target: default" \
  -F "image=@example.jpg"
```

### `GET /internal/models`

Returns configured model targets (protected by `x-internal-token`).

## Configuration

All variables are optional except `INTERNAL_TOKEN` in real deployments.
See [.env.example](.env.example).

Multi-target format example:

```json
{
  "default": {
    "url": "http://127.0.0.1:8000/v1/chat/completions",
    "model": "Qwen/Qwen2.5-VL-7B-Instruct"
  },
  "fast": {
    "url": "http://127.0.0.1:8001/v1/chat/completions",
    "model": "Qwen/Qwen2.5-VL-3B-Instruct"
  }
}
```

## Security Notes

- Keep `INTERNAL_TOKEN` secret.
- Run behind a private network or trusted gateway.
- Prefer localhost-only access between proxy and vLLM.
- See [SECURITY.md](SECURITY.md).

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License. See [LICENSE](LICENSE).
