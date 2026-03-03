# vLLM Multi-Model Proxy

A lightweight FastAPI proxy for multiple vLLM targets and multiple task types:
image captioning, chat, image generation, and text-to-speech (TTS).

The service is designed for internal usage (`/internal/*`) and can be hosted
publicly on GitHub for collaborative improvements.

## Features

- Token-protected internal endpoint
- Multi-target routing for multiple vLLM model backends
- Multi-task API: caption, chat, image generation, and TTS
- Strict upload limits and MIME type validation
- Image resize pipeline for stable vLLM input
- Structured JSON error responses with `request_id`
- Temporary image cleanup after each request
- Basic CI and contribution templates

## Architecture

1. Client calls a task endpoint under `/internal/*`
2. Proxy validates token, target alias, and payload
3. For captioning, proxy resizes image and serves it via `/_img/{rid}.jpg`
4. Proxy forwards to the configured upstream URL for the selected task
5. Proxy returns structured result and metadata

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
- `VLLM_CHAT_URL`/`VLLM_CHAT_MODEL` (single-target mode)
  or `VLLM_TARGETS_JSON` (multi-target mode)
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

### `POST /internal/chat`

Headers:

- `x-internal-token: <INTERNAL_TOKEN>`
- `x-vllm-target: <optional target alias>`

Body:

- `messages`: OpenAI-style chat messages
- `target`: optional alias (overrides header)

Example:

```bash
curl -sS -X POST "http://127.0.0.1:9000/internal/chat" \
  -H "content-type: application/json" \
  -H "x-internal-token: dev-token" \
  -d '{
    "messages":[{"role":"user","content":"Schreibe einen kurzen Gruß."}],
    "target":"default"
  }'
```

### `POST /internal/image-generate`

Body:

- `prompt`: required
- optional: `n`, `size`, `quality`, `response_format`, `target`

### `POST /internal/tts`

Body:

- `input`: required text
- optional: `voice`, `response_format`, `speed`, `target`

Response:

- Audio bytes if upstream returns audio content
- JSON if upstream returns JSON content

### `GET /internal/models`

Returns configured model targets (protected by `x-internal-token`).

## Configuration

All variables are optional except `INTERNAL_TOKEN` in real deployments.
See [.env.example](.env.example).

Multi-target format example:

```json
{
  "default": {
    "chat_url": "http://127.0.0.1:8000/v1/chat/completions",
    "chat_model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "caption_model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "image_url": "http://127.0.0.1:8000/v1/images/generations",
    "image_model": "Qwen/Qwen-Image",
    "tts_url": "http://127.0.0.1:8000/v1/audio/speech",
    "tts_model": "Qwen/Qwen-TTS"
  },
  "fast": {
    "chat_url": "http://127.0.0.1:8001/v1/chat/completions",
    "chat_model": "Qwen/Qwen2.5-VL-3B-Instruct"
  }
}
```

Legacy mode is still supported (`url` + `model`), mapped internally to
`chat_url` + `chat_model`.

## Security Notes

- Keep `INTERNAL_TOKEN` secret.
- Run behind a private network or trusted gateway.
- Prefer localhost-only access between proxy and vLLM.
- See [SECURITY.md](SECURITY.md).

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License. See [LICENSE](LICENSE).
