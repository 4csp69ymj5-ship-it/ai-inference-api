from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from app.cache import get_cached, set_cached
import httpx
import os
import json
import time

router = APIRouter()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

class InferRequest(BaseModel):
    modality: str
    text: Optional[str] = None
    image_url: Optional[str] = None
    audio_url: Optional[str] = None
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False

@router.get("/health")
async def health():
    return {"status": "healthy", "timestamp": time.time()}

async def stream_groq(text: str, max_tokens: int):
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream(
            "POST",
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": text}],
                "max_tokens": max_tokens,
                "stream": True
            }
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break
                    try:
                        chunk = json.loads(data)
                        token = chunk["choices"][0]["delta"].get("content", "")
                        if token:
                            yield f"data: {json.dumps({'token': token})}\n\n"
                    except Exception:
                        continue

@router.post("/infer")
async def infer(req: InferRequest):
    payload = req.model_dump()

    # Streaming path — no cache
    if req.stream and req.modality == "text" and req.text:
        return StreamingResponse(
            stream_groq(req.text, req.max_tokens),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )

    # Non-streaming path — use cache
    cached = get_cached(payload)
    if cached:
        cached["cache"] = "HIT"
        return cached

    start = time.time()

    try:
        if req.modality == "text":
            if not req.text:
                raise HTTPException(400, "text field required")
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    GROQ_URL,
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json={
                        "model": "llama-3.1-8b-instant",
                        "messages": [{"role": "user", "content": req.text}],
                        "max_tokens": req.max_tokens
                    }
                )
                data = response.json()
            result = {
                "result": data["choices"][0]["message"]["content"],
                "tokens_used": data["usage"]["total_tokens"],
                "model": "llama-3.1-8b-instant"
            }

        elif req.modality == "image":
            if not req.image_url or not req.text:
                raise HTTPException(400, "text and image_url required")
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    GROQ_URL,
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json={
                        "model": "llama-3.1-8b-instant",
                        "messages": [{"role": "user", "content": f"{req.text} [Image: {req.image_url}]"}],
                        "max_tokens": req.max_tokens
                    }
                )
                data = response.json()
            result = {
                "result": data["choices"][0]["message"]["content"],
                "tokens_used": data["usage"]["total_tokens"],
                "model": "llama-3.1-8b-instant"
            }

        elif req.modality == "audio":
            if not req.audio_url:
                raise HTTPException(400, "audio_url required")
            async with httpx.AsyncClient(timeout=60) as client:
                audio_response = await client.get(req.audio_url)
                files = {"file": ("audio.mp3", audio_response.content, "audio/mpeg")}
                data = {"model": "whisper-large-v3-turbo"}
                response = await client.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    files=files,
                    data=data
                )
                result_data = response.json()
            result = {
                "result": result_data.get("text", "Transcription failed"),
                "model": "whisper-large-v3-turbo"
            }

        elif req.modality == "multimodal":
            results = {}
            if req.text:
                async with httpx.AsyncClient(timeout=60) as client:
                    response = await client.post(
                        GROQ_URL,
                        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                        json={
                            "model": "llama-3.1-8b-instant",
                            "messages": [{"role": "user", "content": req.text}],
                            "max_tokens": req.max_tokens
                        }
                    )
                    data = response.json()
                results["text"] = data["choices"][0]["message"]["content"]
            result = {"result": results, "model": "multimodal"}

        else:
            raise HTTPException(400, f"Unknown modality: {req.modality}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

    response_data = {
        **result,
        "latency_ms": round((time.time() - start) * 1000, 2),
        "cache": "MISS"
    }

    set_cached(payload, response_data)
    return response_data
