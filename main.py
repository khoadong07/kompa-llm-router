from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import aiohttp
from typing import List, Dict, Optional
import asyncio
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# Fireworks AI client configuration
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
FIREWORKS_API_URL = "https://api.fireworks.ai/inference/v1"
client = AsyncOpenAI(
    base_url=FIREWORKS_API_URL,
    api_key=FIREWORKS_API_KEY
)

# Concurrency controls
CONCURRENT_REQUESTS = 5
semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

# Pydantic models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str]
    messages: List[Message]

class ChatResponse(BaseModel):
    choices: List[Dict]

async def process_single_request(request: ChatRequest, retry_count: int = 0) -> Dict:
    """Process a single chat completion request to Fireworks AI"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {FIREWORKS_API_KEY}"
    }

    # Ensure the model is set to Fireworks AI's model
    payload = request.dict()
    payload["model"] = "accounts/fireworks/models/llama4-scout-instruct-basic"

    async with aiohttp.ClientSession() as session:
        async with session.post(
            FIREWORKS_API_URL + "/chat/completions",
            json=payload,
            headers=headers
        ) as response:
            if response.status in (402, 429):
                if retry_count < 1:  # Allow one retry
                    return await process_single_request(request, retry_count + 1)
                raise HTTPException(status_code=429, detail="Rate limit or payment issue after retry")

            if response.status != 200:
                raise HTTPException(status_code=response.status, detail=await response.text())

            return await response.json()

async def process_requests_concurrently(requests: List[ChatRequest]) -> List[Dict]:
    """Process multiple requests concurrently within concurrency limits"""
    async with semaphore:
        try:
            results = await asyncio.gather(
                *(process_single_request(req) for req in requests),
                return_exceptions=True
            )
            return results
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    """Handle single chat completion request"""
    result = await process_requests_concurrently([request])
    if isinstance(result[0], Exception):
        raise HTTPException(status_code=500, detail=str(result[0]))
    return result[0]