import os
import httpx

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_AUDIO_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

async def run_text_inference(text: str, max_tokens: int = 1024) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": text}],
                "max_tokens": max_tokens
            }
        )
        data = response.json()
    return {
        "result": data["choices"][0]["message"]["content"],
        "tokens_used": data["usage"]["total_tokens"],
        "model": "llama-3.1-8b-instant"
    }

async def run_image_inference(text: str, image_url: str, max_tokens: int = 1024) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": f"{text} [Image URL: {image_url}]"}],
                "max_tokens": max_tokens
            }
        )
        data = response.json()
    return {
        "result": data["choices"][0]["message"]["content"],
        "tokens_used": data["usage"]["total_tokens"],
        "model": "llama-3.1-8b-instant"
    }

async def run_audio_inference(audio_url: str) -> dict:
    async with httpx.AsyncClient() as client:
        audio_response = await client.get(audio_url)
        audio_bytes = audio_response.content
        files = {"file": ("audio.mp3", audio_bytes, "audio/mpeg")}
        data = {"model": "whisper-large-v3-turbo"}
        response = await client.post(
            GROQ_AUDIO_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files=files,
            data=data
        )
        result = response.json()
    return {
        "result": result.get("text", "Transcription failed"),
        "model": "whisper-large-v3-turbo"
    }
