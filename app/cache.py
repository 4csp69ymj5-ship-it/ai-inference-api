import redis
import hashlib
import json
import os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_TTL = int(os.getenv("CACHE_TTL", 3600))

try:
    client = redis.from_url(REDIS_URL, decode_responses=True)
except Exception:
    client = None

def make_key(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True)
    return "infer:" + hashlib.sha256(raw.encode()).hexdigest()

def get_cached(payload: dict):
    if not client:
        return None
    try:
        key = make_key(payload)
        result = client.get(key)
        return json.loads(result) if result else None
    except Exception:
        return None

def set_cached(payload: dict, response: dict):
    if not client:
        return
    try:
        key = make_key(payload)
        client.setex(key, CACHE_TTL, json.dumps(response))
    except Exception:
        pass
