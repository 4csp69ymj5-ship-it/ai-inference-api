# Multi-Modal AI Inference API

> Fast, efficient, and optimized AI inference API for text, image, and audio — built for scale.

## Features
- **Streaming responses** — ChatGPT-style token streaming
- **Redis semantic caching** — repeat queries served in <5ms, zero AI compute
- **Multi-modal** — text, image, and audio in one unified API
- **4 async workers** — handles concurrent requests without blocking
- **Fully containerized** — Docker + Docker Compose, runs anywhere

## Architecture
## Quick Start

```bash
git clone https://github.com/4csp69ymj5-ship-it/ai-inference-api.git
cd ai-inference-api
cp .env.example .env  # Add your GROQ_API_KEY
docker compose up --build
```

API live at `http://localhost:8000`
Docs at `http://localhost:8000/docs`

## API Usage

### Text Inference
```bash
curl -X POST http://localhost:8000/v1/infer \
  -H "Content-Type: application/json" \
  -d '{"modality": "text", "text": "Explain quantum computing", "max_tokens": 100}'
```

### Streaming
```bash
curl -X POST http://localhost:8000/v1/infer \
  -H "Content-Type: application/json" \
  -d '{"modality": "text", "text": "Write a poem about AI", "stream": true}'
```

### Audio Transcription
```bash
curl -X POST http://localhost:8000/v1/infer \
  -H "Content-Type: application/json" \
  -d '{"modality": "audio", "audio_url": "https://example.com/audio.mp3"}'
```

## Performance
| Metric | Value |
|--------|-------|
| First response latency | ~440ms |
| Cache hit latency | <5ms |
| Concurrent workers | 4 |
| Cache TTL | 1 hour |

## Stack
- **FastAPI** — async Python API framework
- **Groq** — ultra-fast LLM inference (Llama 3.1)
- **Redis** — response caching layer
- **Docker** — containerized deployment
- **Whisper** — audio transcription

## Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API status |
| GET | `/v1/health` | Health check |
| POST | `/v1/infer` | Run inference |
| GET | `/docs` | Swagger UI |

## Built for Google Cloud
Designed to deploy on Google Cloud Run with Cloud Memorystore (Redis) for enterprise-scale inference.
