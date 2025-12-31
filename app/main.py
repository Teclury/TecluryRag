from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import hashlib
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
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

vectorstore: FAISS | None = None
rag_chain = None


# session summary store
store = {}

# simple per-session rate limiter
rate_limiters = {}
MAX_RPM = 5
MAX_RPD = 30


class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None


# ---------------- RATE LIMIT ---------------- #

def check_rate_limit(session_id: str):
    now = datetime.utcnow()

    if session_id not in rate_limiters:
        rate_limiters[session_id] = {
            "minute": [0, now],
            "day": [0, now]
        }

    minute_count, minute_time = rate_limiters[session_id]["minute"]
    day_count, day_time = rate_limiters[session_id]["day"]

    # reset minute bucket
    if (now - minute_time).total_seconds() >= 60:
        minute_count = 0
        minute_time = now

    # reset day bucket
    if (now - day_time).days >= 1:
        day_count = 0
        day_time = now

    # minute limit
    if minute_count >= MAX_RPM:
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "answer": "Sorry about that! You’ve hit the rate limit for now. Please try again in a minute.",
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    # daily limit
    if day_count >= MAX_RPD:
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "answer": "Daily limit reached. Please try again tomorrow or contact our team.",
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    # increment counters
    rate_limiters[session_id]["minute"] = [minute_count + 1, minute_time]
    rate_limiters[session_id]["day"] = [day_count + 1, day_time]

    return None


# ---------------- VECTOR STORE ---------------- #

def build_vectorstore():
    file_path = "data/knowledge.txt"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Knowledge file not found at: {file_path}")

    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()

    # smaller chunks = fewer tokens
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    return FAISS.from_documents(chunks, embeddings)


# ---------------- RAG SETUP ---------------- #

def setup_rag_chain(vector_store):

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.3,
        max_tokens=300  # cap output tokens
    )

    prompt = ChatPromptTemplate.from_template("""
You are Nora, a friendly and professional AI assistant for Teclury (IT & AI solutions company). Speak like a real human team member.

ROLE
- understand what the user needs
- reply briefly and clearly
- explain how Teclury can help
- end with ONE small follow-up question
- always speak as “we / our team / our services”

LANGUAGE RULES (STRICT)
**Language Detection & Adaptation (Crucial):** - **General Rule:** Identify the language of the user's question (English, Tamil, Malayalam, Telugu, Kannada, Hindi, etc.) and **reply in that exact same language and script.**
       - **Tanglish Rule:** If the user types in **Tanglish** (Tamil words using English letters), your response must be converted into **proper Tamil script (தமிழ்)**.
       - **Example:** - User: "Neenga enna services tharinga?" (Tanglish) -> You: "நாங்கள் Web Development மற்றும் AI சேவைகளை வழங்குகிறோம்." (Tamil)
         - User: "Sughamaano?" (Malayalam) -> You: "അതെ, സുഖമാണ്! നിങ്ങൾക്ക് എന്ത് ഐടി സഹായമാണ് വേണ്ടത്?" (Malayalam)


GREETING RULE
If user says only hello/hi/vanakkam/namaste/how are you:
→ Introduce yourself ONCE per conversation:
“Hi! I’m Nora, your AI assistant from Teclury. How can we help you today?”
If they ask how are you → say you’re doing great and ask them back
Do not introduce again if summary exists

SCOPE OF ANSWERS
Answer ONLY Teclury-related topics:
- software
- websites / apps
- AI / chatbots / automation
- Teclury company / team / services / products
If unrelated:
“I’m here to help with Teclury’s products and services. What are you looking to build or improve?”

ACHIEVEMENTS ANSWER
“We’re focused on building innovative products. Our clients’ success is our biggest achievement.”

PRICING RULE
- Do NOT mention price unless user directly asks

REPLY LENGTH RULE
- 3–5 short lines max
- no long paragraphs
- no repeated sentences
- no marketing stories
- use friendly emojis naturally (🤝🚀😊🎯✨)

CONFIRMATION RULE (IMPORTANT)
If user says: ok / we can start / proceed / start project / let’s go
→ ALWAYS reply briefly AND share contact:
“Awesome! We’re happy to work with you 🤝 Please contact us at +91 8526521533 or contact@teclury.in to finalize the next step.”

UNKNOWN INFORMATION RULE
If answer not in context:
“I don’t have exact details right now. Please contact our team at +91 8526521533 or contact@teclury.in.”

MEMORY
Use summary to avoid repeating introductions and personalize replies

ALWAYS END WITH ONE QUESTION
Examples:
- What are you planning to build?
- Is this for business or personal use?
- Do you already have a website or app?

INPUTS
summary = {summary}
input = {input}
context = {context}

OUTPUT (valid JSON only)
{{
"answer": "<reply>",
"summary": "<updated short summary>"
}}
"""
)
    
    
    # fewer docs = fewer tokens
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain


# ---------------- CLEANUP THREAD ---------------- #

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


# ---------------- STARTUP ---------------- #

@app.on_event("startup")
def startup_event():
    global vectorstore, rag_chain,retriever

    vectorstore = build_vectorstore()
    rag_chain= setup_rag_chain(vectorstore)

    threading.Thread(target=cleanup_idle_sessions, daemon=True).start()

    print("🚀 RAG + rolling summary memory ready")


@app.get("/")
def health():
    return {"status": "OK", "ready": rag_chain is not None}


# ---------------- CHAT ENDPOINT ---------------- #

@app.post("/chat")
def chat_endpoint(request: ChatRequest, http_request: Request):

    if rag_chain is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        # identify user via IP hash if no session
        client_ip = http_request.client.host or "unknown-ip"
        ip_session = hashlib.sha256(client_ip.encode()).hexdigest()[:16]

        if not request.session_id:
            session_id = ip_session
        elif request.session_id in store:
            session_id = request.session_id
        else:
            session_id = ip_session

        # rate limit check
        limit_response = check_rate_limit(session_id)
        if limit_response:
            return limit_response

        # initialize session
        if session_id not in store:
            store[session_id] = {
                "summary": "",
                "last_activity": datetime.utcnow()
            }

        old_summary = store[session_id]["summary"]
        
        # -------- model call with 429 handling -------- #
        try:
            response = rag_chain.invoke({
                "input": request.query,
                "summary": old_summary
            })
        except Exception as e:
            print(e)
            if "rate_limit" in str(e).lower() or "429" in str(e):
                return {
                    "status": "success",
                    "answer": "We’re receiving a high number of requests right now 😅 Please try again in a minute.",
                    "session_id": session_id,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            raise

        
        

        import json
        try:
            output = json.loads(response["answer"])
        except Exception:
            raise HTTPException(
                status_code=500,
                detail="Model returned invalid JSON. Try again."
            )

        new_answer = output["answer"]
        new_summary = output["summary"]
        

        # trim summary to avoid ballooning tokens
        store[session_id]["summary"] = new_summary[-2000:]
        store[session_id]["last_activity"] = datetime.utcnow()
       
        return {
            "status": "success",
            "answer": new_answer,
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        print("Chat Error:", e)
        raise HTTPException(status_code=500, detail=str(e))
