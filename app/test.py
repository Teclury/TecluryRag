from fastapi import FastAPI, HTTPException
import os
from pydantic import BaseModel
from datetime import datetime, timedelta
from dotenv import load_dotenv
import threading
import time

# LangChain Imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

# Memory Imports
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

app = FastAPI(
    title="Teclury RAG Chatbot with Auto Session Cleanup",
    description="RAG + Session Memory cleared if idle > 1 hour",
    version="4.1.0"
)

vectorstore: FAISS | None = None
rag_chain = None

# ----------------------------------------------------
# Chat Memory Store
# ----------------------------------------------------
# { session_id: {history: ChatMessageHistory, last_activity: datetime}}
store = {}


class ChatRequest(BaseModel):
    query: str
    session_id: str


# ----------------------------------------------------
# Memory Functions
# ----------------------------------------------------
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    now = datetime.utcnow()

    if session_id not in store:
        history = ChatMessageHistory()
        store[session_id] = {
            "history": history,
            "last_activity": now
        }
    else:
        store[session_id]["last_activity"] = now

    return store[session_id]["history"]


# ----------------------------------------------------
# Vectorstore
# ----------------------------------------------------
def build_vectorstore():
    file_path = "data/knowledge.txt"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Knowledge file not found at: {file_path}")

    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)


# ----------------------------------------------------
# RAG Chain with Memory
# ----------------------------------------------------
def setup_rag_chain(vector_store):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,
        convert_system_message_to_human=True
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a friendly and professional AI assistant for Teclury, an IT and AI solutions company and your name is **Nora**.

    YOUR INSTRUCTIONS:
    1. **Language Detection & Adaptation (Crucial):** - **General Rule:** Identify the language of the user's question (English, Tamil, Malayalam, Telugu, Kannada, Hindi, etc.) and **reply in that exact same language and script.**
       - **Tanglish Rule:** If the user types in **Tanglish** (Tamil words using English letters), your response must be converted into **proper Tamil script (தமிழ்)**.
       - **Example:** - User: "Neenga enna services tharinga?" (Tanglish) -> You: "நாங்கள் Web Development மற்றும் AI சேவைகளை வழங்குகிறோம்." (Tamil)
         - User: "Sughamaano?" (Malayalam) -> You: "അതെ, സുഖമാണ്! നിങ്ങൾക്ക് എന്ത് ഐടി സഹായമാണ് വേണ്ടത്?" (Malayalam)

    2. **Tone & Style:** - Keep responses short, warm, and professional.
       - Do not be robotic. Talk like a helpful team member.

    3. **Handling Specific Topics:**
       - **Greetings:** If they say "Hi", "Hello", "How are you", greet them warmly in their language and ask how you can help.
       - **Achievements:** If asked about success, reply like a founder: "We are heads-down working on innovative products. Our clients' success is our real biggest achievement."
       - **Pricing:** NEVER mention price or budget unless the user explicitly asks for it.
       - **Unknown Info:** If the answer is not in the 'Context' below, say (in the user's language): "I don't have those specific details right now. I recommend you talk to our expert team directly at +91 8526521533 or contact@teclury.in."

    4. **Goal:** - Always try to gently gather their requirements. End your answers with a helpful follow-up question like "What kind of project are you looking to build?"
    5.**Use memory if it is attached** -Use users memory to give them a custmized rsponse response as based on their chat history mre friendly and caring .

Context:
{context}
"""
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    print(prompt)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    conversational = RunnableWithMessageHistory(
        retrieval_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational


# ----------------------------------------------------
# CLEANUP JOB (idle > 1 hour)
# ----------------------------------------------------
def cleanup_idle_sessions():
    while True:
        now = datetime.utcnow()
        sessions = list(store.keys())

        for session_id in sessions:
            last_activity = store[session_id]["last_activity"]
            idle_minutes = (now - last_activity).total_seconds() / 60

            # delete if idle for > 60 minutes
            if idle_minutes >= 60:
                del store[session_id]
                print(f"🧹 Deleted session {session_id} due to 1+ hour inactivity")

        time.sleep(3600)  


# ----------------------------------------------------
# FastAPI Events
# ----------------------------------------------------
@app.on_event("startup")
def startup_event():
    global vectorstore, rag_chain

    vectorstore = build_vectorstore()
    rag_chain = setup_rag_chain(vectorstore)

    threading.Thread(target=cleanup_idle_sessions, daemon=True).start()

    print("🚀 RAG ready — inactive chat sessions auto-clean after 1 hour")


@app.get("/")
def health():
    return {"status": "OK", "ready": rag_chain is not None}


# ----------------------------------------------------
# Chat Endpoint
# ----------------------------------------------------
@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        response = rag_chain.invoke(
            {"input": request.query},
            config={"configurable": {"session_id": request.session_id}}
        )

        # update last activity
        store[request.session_id]["last_activity"] = datetime.utcnow()

        return {
            "status": "success",
            "answer": response["answer"],
            "session_id": request.session_id,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        print("Chat Error:", e)
        raise HTTPException(status_code=500, detail=str(e))
