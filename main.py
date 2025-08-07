from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncio
import os
from dotenv import load_dotenv

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# === FastAPI app ===
app = FastAPI()

# === Langfuse tracking setup ===
langfuse_handler = CallbackHandler()
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

# === Chat prompt template ===
prompt = ChatPromptTemplate.from_template("""
Bạn là trợ lý AI. Hãy trả lời yêu cầu người dùng một cách tự nhiên và chính xác.

Lịch sử hội thoại:
{messages}
""")

# === Format messages into string for prompt ===
def format_messages(messages: List[Dict]) -> str:
    return "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages])

# === Pydantic models ===
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = "gpt-4o-mini"
    messages: List[Message]

class ChatResponse(BaseModel):
    choices: List[Dict]

# === Concurrency controls ===
CONCURRENT_REQUESTS = 5
semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

# === Chain builder (per request) ===
def build_chain(model_name: str):
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.4,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    chain = (
        {
            "messages": lambda x: format_messages(x["messages"]),
        }
        | prompt
        | llm
    )
    return chain

# === Single request processing ===
async def process_single_request(request: ChatRequest, retry_count: int = 0) -> Dict:
    try:
        chain = build_chain(request.model or "gpt-4o-mini")
        input_data = {
            "messages": [msg.dict() for msg in request.messages]
        }

        response = await asyncio.to_thread(
            lambda: chain.invoke(input_data, config={"callbacks": [langfuse_handler]})
        )

        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": response.content
                    },
                    "finish_reason": "stop",
                    "index": 0
                }
            ]
        }
    except Exception as e:
        if retry_count < 1:
            return await process_single_request(request, retry_count + 1)
        raise HTTPException(status_code=500, detail=f"LangChain/OpenAI error: {str(e)}")

# === Process multiple requests ===
async def process_requests_concurrently(requests: List[ChatRequest]) -> List[Dict]:
    async with semaphore:
        results = await asyncio.gather(
            *(process_single_request(req) for req in requests),
            return_exceptions=True
        )
        return results

# === FastAPI endpoint ===
@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    result = await process_requests_concurrently([request])
    if isinstance(result[0], Exception):
        raise HTTPException(status_code=500, detail=str(result[0]))
    return result[0]
