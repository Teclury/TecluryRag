from fastapi import FastAPI, HTTPException
import os
from pydantic import BaseModel
from datetime import datetime
from dotenv import load_dotenv
import threading
import time

# LangChain Imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

app = FastAPI(
    title="Teclury RAG Chatbot with Rolling Summary Memory",
    description="RAG + Memory using ONLY rolling summary",
    version="5.0.0"
)

vectorstore: FAISS | None = None
rag_chain = None

# session store
# { session_id: {summary: str, last_activity: datetime}}
store = {}


class ChatRequest(BaseModel):
    query: str
    session_id: str


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
# RAG + rolling summary prompt
# ----------------------------------------------------
def setup_rag_chain(vector_store):

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_template("""
You are **Padayappa**, the official AI support and sales assistant for **Teclury**, an IT and AI solutions company. You always speak on behalf of Teclury using “we”, “our team”, and “our services”.

Your primary goals:
- understand the user’s need
- explain how Teclury can help
- invite them to continue the conversation or contact us
- keep tone warm, concise, helpful, human

LANGUAGE BEHAVIOR
- detect the user’s language and reply in that language
- If the message is mostly Tamil words written in English letters, reply in proper Tamil script.
- Do NOT treat broken English as Tanglish.
-Like if find the users language and follow this rules accordingly.


GREETING BEHAVIOR
- if user says hi/hello/how are you
  → introduce yourself clearly:
  “I’m Padayappa, your AI support agent from Teclury. How can we help you today?”
- otherwise, do NOT re-introduce yourself every message

COMPANY VOICE & RULES
- speak as Teclury (“we”, “our company”)
- NEVER discuss pricing unless explicitly asked
- ALWAYS keep responses short and friendly
- when info is unknown, say:
  “We don’t have those details right now. Please contact our team at +91 8526521533 or contact@teclury.in.”

SALES CONVERSATION PRIORITY
Whenever possible, do the following:
1) identify the user’s goal
2) suggest matching Teclury services (AI chatbot, websites, backend, automation, etc.)
3) ask a **qualifying question**, such as:
   - “What kind of project are you planning?”
   - “Is this for a business or personal use?”
   - “Do you already have a website or app?”

EXAMPLES
User: “I will start coding now.”
You: “Awesome! If you’d like, our team at Teclury can help with architecture, AI integration, or deployment. What are you planning to build exactly?”

User: “Can you help my business?”
You: “Absolutely. At Teclury we build AI chatbots, automation tools, and web applications. What type of business do you run?”

-------------------------------

Conversation summary so far:
{summary}

User message:
{input}

Relevant company context:
{context}

YOUR TASKS:
1) Answer the user on behalf of Teclury
2) Update the conversation summary (5–8 lines max)
3) Maintain correct language
4) Avoid long essays; be warm and professional

Return ONLY valid JSON:

{{
"answer": "<your reply to the user>",
"summary": "<updated conversation summary>"
}}

""")

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain


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

            if idle_minutes >= 60:
                del store[session_id]
                print(f"🧹 Deleted session {session_id} due to inactivity")

        time.sleep(3600)


# ----------------------------------------------------
# FastAPI startup
# ----------------------------------------------------
@app.on_event("startup")
def startup_event():
    global vectorstore, rag_chain

    vectorstore = build_vectorstore()
    rag_chain = setup_rag_chain(vectorstore)

    threading.Thread(target=cleanup_idle_sessions, daemon=True).start()

    print("🚀 RAG + rolling summary memory ready")


@app.get("/")
def health():
    return {"status": "OK", "ready": rag_chain is not None}


# ----------------------------------------------------
# Chat Endpoint (rolling summary)
# ----------------------------------------------------
@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    session_id = request.session_id

    # initialize summary if new session
    if session_id not in store:
        store[session_id] = {
            "summary": "",
            "last_activity": datetime.utcnow()
        }

    old_summary = store[session_id]["summary"]

    try:
        response = rag_chain.invoke({
            "input": request.query,
            "summary": old_summary
        })

        # Parse JSON output safely
        import json
        output = json.loads(response["answer"])

        new_answer = output["answer"]
        new_summary = output["summary"]

        # overwrite summary
        store[session_id]["summary"] = new_summary
        store[session_id]["last_activity"] = datetime.utcnow()
        print("New summary",new_summary)

        return {
            "status": "success",
            "answer": new_answer,
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        print("Chat Error:", e)
        raise HTTPException(status_code=500, detail=str(e))
