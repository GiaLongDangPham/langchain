from fastapi import APIRouter
from app.models.chat import ChatRequest
from app.core.llm import chat_with_rag

router = APIRouter()

@router.post("/chat")
def chat(req: ChatRequest):
    return {"reply": chat_with_rag(req)}
