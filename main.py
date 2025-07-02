from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
import redis.asyncio as redis
import aiohttp
from datetime import datetime
import os
import json
from dotenv import load_dotenv
import asyncio
from time import time
from collections import deque
from typing import List, Dict

load_dotenv()
app = FastAPI()

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
TOGETHER_KEY = os.getenv("TOGETHER_KEY")
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Redis connection
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = redis.from_url(REDIS_URL)
API_KEY_CACHE_KEY = "active_api_key"
RATE_LIMIT_KEY = "chat_completions_rate_limit"
MAX_REQUESTS = 600
RATE_LIMIT_WINDOW = 60  # 1 minute

# Concurrency controls
CONCURRENT_REQUESTS = 5
semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
request_queue = deque()

TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

# Pydantic models
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]


class ChatResponse(BaseModel):
    choices: list[dict]


async def check_rate_limit():
    """Check and update rate limit using Redis"""
    current_time = int(time())
    async with redis_client.pipeline() as pipe:
        pipe.zadd(RATE_LIMIT_KEY, {str(current_time): current_time})
        pipe.zremrangebyscore(RATE_LIMIT_KEY, 0, current_time - RATE_LIMIT_WINDOW)
        pipe.zcard(RATE_LIMIT_KEY)
        _, _, count = await pipe.execute()

    if count > MAX_REQUESTS:
        raise HTTPException(status_code=429, detail="Rate limit exceeded: 10 requests per minute")
    return count


async def get_active_api_key():
    """Get active API key from Redis or MongoDB"""
    cached_key = await redis_client.get(API_KEY_CACHE_KEY)
    if cached_key:
        return cached_key.decode()

    api_key_doc = collection.find_one(
        {"status": "active"},
        sort=[("created_at", 1)]
    )

    if not api_key_doc:
        return None

    api_key = api_key_doc["api_key"]
    await redis_client.setex(API_KEY_CACHE_KEY, 36000, api_key)
    return api_key


async def deactivate_and_clear_api_key(api_key: str):
    """Deactivate API key and clear Redis cache"""
    collection.update_one(
        {"api_key": api_key},
        {"$set": {"status": "deactive", "updated_at": datetime.utcnow()}}
    )
    await redis_client.delete(API_KEY_CACHE_KEY)


# async def process_single_request(request: ChatRequest, api_key: str, retry_count: int = 0) -> Dict:
#     """Process a single chat completion request with retry on 402/429"""
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": f"Bearer {api_key}"
#     }

#     async with aiohttp.ClientSession() as session:
#         async with session.post(
#             "https://api.deepinfra.com/v1/openai/chat/completions",
#             json=request.dict(),
#             headers=headers
#         ) as response:
#             if response.status in (402, 429):
#                 await deactivate_and_clear_api_key(api_key)
#                 if retry_count < 1:  # Allow one retry
#                     new_api_key = await get_active_api_key()
#                     if not new_api_key:
#                         raise HTTPException(status_code=503, detail="No active API keys available")
#                     return await process_single_request(request, new_api_key, retry_count + 1)
#                 raise HTTPException(status_code=429, detail="Rate limit or payment issue after retry")

#             if response.status != 200:
#                 raise HTTPException(status_code=response.status, detail=await response.text())

#             return await response.json()

async def process_single_request(request: ChatRequest, api_key: str, retry_count: int = 0) -> Dict:
    """Process a single chat completion request to Together.ai with retry on 402/429"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TOGETHER_KEY}"
    }

    # Convert request to dict and force model to DeepSeek
    payload = request.dict()

    async with aiohttp.ClientSession() as session:
        async with session.post(TOGETHER_API_URL, json=payload, headers=headers) as response:
            if response.status in (402, 429):
                await deactivate_and_clear_api_key(api_key)
                if retry_count < 1:
                    new_api_key = await get_active_api_key()
                    if not new_api_key:
                        raise HTTPException(status_code=503, detail="No active API keys available")
                    return await process_single_request(request, new_api_key, retry_count + 1)
                raise HTTPException(status_code=429, detail="Rate limit or payment issue after retry")

            if response.status != 200:
                raise HTTPException(status_code=response.status, detail=await response.text())

            return await response.json()


async def process_requests_concurrently(requests: List[ChatRequest]) -> List[Dict]:
    """Process multiple requests concurrently within rate and concurrency limits"""
    api_key = await get_active_api_key()
    if not api_key:
        raise HTTPException(status_code=503, detail="No active API keys available")

    # Check rate limit
    count = await check_rate_limit()
    available_slots = max(0, MAX_REQUESTS - count)
    if available_slots < len(requests):
        raise HTTPException(status_code=429, detail="Rate limit would be exceeded")

    # Process requests concurrently with semaphore
    async with semaphore:
        try:
            results = await asyncio.gather(
                *(process_single_request(req, api_key) for req in requests),
                return_exceptions=True
            )
            return results
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    """Handle single or queued requests"""
    global request_queue

    # Add request to queue
    request_queue.append(request)

    # Process up to CONCURRENT_REQUESTS at a time
    if len(request_queue) >= CONCURRENT_REQUESTS or len(request_queue) > 0:
        requests_to_process = []
        while request_queue and len(requests_to_process) < CONCURRENT_REQUESTS:
            requests_to_process.append(request_queue.popleft())

        results = await process_requests_concurrently(requests_to_process)

        # Return result for the current request
        for req, result in zip(requests_to_process, results):
            if req == request:
                if isinstance(result, Exception):
                    raise HTTPException(status_code=500, detail=str(result))
                return result

    # If request wasn't processed in batch, process it individually
    result = await process_requests_concurrently([request])
    return result[0]


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await redis_client.close()