import os
import httpx

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_AUDIO_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

# llama-3.1-8b-instant is the fastest groq model i tested,
# 70b is smarter but adds ~200ms which kills the point of this whole thing
TEXT_MODEL = "llama-3.1-8b-instant"
AUDIO_MODEL = "whisper-large-v3-turbo"

async def run_text_inference(text: str, max_tokens: int = 1024) -> dict:
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": TEXT_MODEL,
                "messages": [{"role": "user", "content": text}],
                "max_tokens": max_tokens
            }
        )
        data = response.json()

    # groq returns usage stats which is handy for tracking costs later
    return {
        "result": data["choices"][0]["message"]["content"],
        "tokens_used": data["usage"]["total_tokens"],
        "model": TEXT_MODEL
    }

async def run_image_inference(text: str, image_url: str, max_tokens: int = 1024) -> dict:
    # groq doesnt support vision natively yet so we pass the url in the prompt
    # not ideal but works fine for most use cases
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": TEXT_MODEL,
                "messages": [{"role": "user", "content": f"{text} [Image: {image_url}]"}],
                "max_tokens": max_tokens
            }
        )
        data = response.json()
    return {
        "result": data["choices"][0]["message"]["content"],
        "tokens_used": data["usage"]["total_tokens"],
        "model": TEXT_MODEL
    }

async def run_audio_inference(audio_url: str) -> dict:
    # download the audio first then send to whisper
    # groq's whisper is way faster than openai's hosted version
    async with httpx.AsyncClient(timeout=60) as client:
        audio_response = await client.get(audio_url)
        audio_bytes = audio_response.content
        files = {"file": ("audio.mp3", audio_bytes, "audio/mpeg")}
        data = {"model": AUDIO_MODEL}
        response = await client.post(
            GROQ_AUDIO_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files=files,
            data=data
        )
        result_data = response.json()
    return {
        "result": result_data.get("text", "transcription failed"),
        "model": AUDIO_MODEL
    }
