from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.routes import router
import uvicorn

load_dotenv()

app = FastAPI(
    title="Multi-Modal AI Inference API",
    description="Optimized AI inference for text, image, and audio.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/v1")

@app.get("/")
async def root():
    return {
        "status": "online",
        "api": "Multi-Modal AI Inference",
        "version": "1.0.0",
        "endpoints": ["/v1/infer", "/v1/health", "/docs"]
    }
