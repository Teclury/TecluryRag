import threading
import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import numpy as np
import hashlib
import os
import json

# -------------------- ENV + CLIENT -------------------- #

load_dotenv()
client = OpenAI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found")

# -------------------- FASTAPI APP -------------------- #

app = FastAPI(title="Teclury RAG Chatbot with Rolling AI Summary")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- FILE PATHS -------------------- #

DATA_FILE = "data/knowledge.txt"
VECTOR_FILE = "vectors.json"

# -------------------- MEMORY -------------------- #

vector_store = []   # list of {id,text,embedding}

# session_id -> {summary, last_activity}
session_summaries = {}

# -------------------- RATE LIMIT -------------------- #

rate_limiters = {}
MAX_RPM = 5
MAX_RPD = 30

def chunk_text(text, chunk_size=600, overlap=150):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

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

    rate_limiters[session_id]["minute"] = [minute_count + 1, minute_time]
    rate_limiters[session_id]["day"] = [day_count + 1, day_time]

    return None


def cleanup_idle_sessions():
    while True:
        now = datetime.utcnow()
        sessions = list(session_summaries.keys())

        for session_id in sessions:
            try:
                last_activity = session_summaries[session_id]["last_activity"]
            except KeyError:
                continue

            idle_minutes = (now - last_activity).total_seconds() / 60

            if idle_minutes >= 60:
                del session_summaries[session_id]
                print(f"🧹 Deleted session {session_id} due to inactivity")

        time.sleep(3600)

SYSTEM_PROMPT = """
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

def get_embedding(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def build_vectors_from_text():
    global vector_store

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found")

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    print("Chunks created:", len(chunks))

    vector_store = []

    for i, chunk in enumerate(chunks):
        emb = get_embedding(chunk)
        vector_store.append({
            "id": i,
            "text": chunk,
            "embedding": emb
        })

    with open(VECTOR_FILE, "w") as f:
        json.dump(vector_store, f)

    print("✅ vectors.json created")


def load_vectors():
    global vector_store
    with open(VECTOR_FILE) as f:
        vector_store = json.load(f)
    print("📂 vectors.json loaded")


@app.on_event("startup")
def startup():
    if os.path.exists(VECTOR_FILE):
        load_vectors()
    else:
        build_vectors_from_text()

    threading.Thread(target=cleanup_idle_sessions, daemon=True).start()

    print("🚀 Teclury RAG chatbot ready")


def cosine(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search_chunks(query: str, k: int = 3):
    query_vec = get_embedding(query)

    scored = []
    for item in vector_store:
        score = cosine(query_vec, item["embedding"])
        scored.append((score, item))

    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:k]


class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None


@app.post("/chat")
def chat(req: ChatRequest, http_request: Request):

    try:
        # session
        client_ip = http_request.client.host or "unknown"
        default_session = hashlib.sha256(client_ip.encode()).hexdigest()[:16]
        session_id = default_session

        # rate limit
        limit_response = check_rate_limit(session_id)
        if limit_response:
            return limit_response

        # init session
        if session_id not in session_summaries:
            session_summaries[session_id] = {
                "summary": "",
                "last_activity": datetime.utcnow()
            }

        old_summary = session_summaries[session_id]["summary"]

        # retrieve chunks
        results = search_chunks(req.query)
        context = "\n\n".join([doc["text"] for _, doc in results])

        # LLM call
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": json.dumps({
                        "summary": old_summary,
                        "context": context,
                        "input": req.query
                    })
                }
            ]
        )

        raw = completion.choices[0].message.content

        parsed = json.loads(raw)

        answer = parsed["answer"]
        new_summary = parsed["summary"]

        # store new summary
        session_summaries[session_id] = {
            "summary": new_summary,
            "last_activity": datetime.utcnow()
        }

        return {
            "status": "success",
            "answer": answer,
            "summary": new_summary,
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        print("Chat error:", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def health():
    return {
        "ok": True,
        "vector_count": len(vector_store),
        "session_count": len(session_summaries)
    }

