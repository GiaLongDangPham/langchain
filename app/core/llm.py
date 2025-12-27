from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from app.core.memory import chat_memory, MAX_TURNS
from app.models.chat import ChatRequest

llm = ChatOllama(model="llama3")
embeddings = OllamaEmbeddings(model="llama3")

vectorstore = FAISS.load_local(
    "vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3
    }
)

def chat_with_rag(req: ChatRequest):
    sessionId = req.sessionId
    question = req.message

    # Lấy history
    history = chat_memory[sessionId]
    history_text = "\n".join(history[-MAX_TURNS:])

    # RAG retrieval
    docs = retriever.invoke(rewrite_query(question))
    context = "\n".join(d.page_content for d in docs)

    # Prompt có MEMORY + RAG
    prompt = f"""
            Bạn là trợ lý sinh viên.
            CHỈ sử dụng thông tin được cung cấp.
            Nếu không đủ thông tin, hãy trả lời: "Tôi không biết".

            Lịch sử hội thoại:
            {history_text}

            Thông tin từ dữ liệu:
            {context}

            Câu hỏi hiện tại:
            {question}
            """

    response = llm.invoke(prompt)
    answer = response.content

    # Lưu lại memory
    history.append(f"User: {question}")
    history.append(f"AI: {answer}")

    return answer

def rewrite_query(question: str):
    return f"Thông tin chi tiết về {question}"