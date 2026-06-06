import os
import re
import json
import uuid
import requests
from datetime import datetime
from fastapi import FastAPI, Request, Query
from fastapi.responses import JSONResponse, Response
from groq import Groq
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

app = FastAPI(title="Phoenix AI Sovereign Core Engine", version="5.3.0")

# --- 1. SOVEREIGN MULTI-TENANT DATABASE STORAGE ---
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./phoenix.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Profile(Base):
    __tablename__ = "profiles"
    profile_id = Column(String(50), primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class PlatformIdentity(Base):
    __tablename__ = "platform_identities"
    platform = Column(String(20), primary_key=True)          
    platform_user_id = Column(String(100), primary_key=True) 
    profile_id = Column(String(50), ForeignKey("profiles.profile_id"), nullable=False)

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    profile_id = Column(String(50), ForeignKey("profiles.profile_id"), nullable=False, index=True)
    role = Column(String(20), nullable=False)                
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

class TrainingCorpus(Base):
    __tablename__ = "training_corpus"
    id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_type = Column(String(30), default="fine_tuning_jsonl")
    payload = Column(Text, nullable=False)                   
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# --- 2. CONFIGURATION & KEYS ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

WHATSAPP_TOKEN = os.environ.get("WHATSAPP_ACCESS_TOKEN")
WHATSAPP_PHONE_ID = os.environ.get("WHATSAPP_PHONE_NUMBER_ID")
WHATSAPP_VERIFY_TOKEN = os.environ.get("WHATSAPP_VERIFY_TOKEN", "phoenix_secret_verify_token")

# --- 3. INTELLIGENCE UTILITIES ---
def live_multi_source_search(query: str) -> str:
    sources_found = []
    clean_q = query.strip().strip('"').strip("'").strip()
    TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
    
    if TAVILY_API_KEY:
        try:
            url = "https://api.tavily.com/search"
            payload = {
                "api_key": TAVILY_API_KEY, 
                "query": clean_q, 
                "search_depth": "basic", 
                "topic": "general", 
                "max_results": 5
            }
            response = requests.post(url, json=payload, timeout=8)
            if response.status_code == 200:
                results = response.json().get("results", [])
                for idx, res in enumerate(results):
                    content = res.get("content", "")
                    sources_found.append(f"SOURCE [{idx+1}]: {content}\n")
                if sources_found:
                    return "\n".join(sources_found)
        except Exception:
            pass

    try:
        url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(clean_q)}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        res = requests.get(url, headers=headers, timeout=8)
        if res.status_code == 200:
            body = res.text
            snippets = re.findall(r'<a class="result__snippet"[^>]*>(.*?)</a>', body, re.DOTALL)
            for i in range(min(4, len(snippets))):
                clean_snippet = re.sub('<[^<]+?>', '', snippets[i]).strip()
                sources_found.append(f"SOURCE [{i+1}]: {clean_snippet}\n")
            if sources_found:
                return "\n".join(sources_found)
    except Exception:
        pass
    return ""

def needs_web_search(question: str) -> bool:
    keywords = ["current", "now", "today", "latest", "recent", "who is", "president", "price", "weather", "score", "vs", "crypto", "match", "rate", "news", "deputy", "vice president"]
    return any(k in question.lower() for k in keywords) or bool(re.search(r'\b(202[4-9]|20[3-9]\d)\b', question))

def send_whatsapp_reply(to_phone: str, text: str):
    if not WHATSAPP_TOKEN or not WHATSAPP_PHONE_ID:
        return False
    url = f"https://graph.facebook.com/v18.0/{WHATSAPP_PHONE_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to_phone,
        "type": "text",
        "text": {"preview_url": False, "body": text}
    }
    try:
        res = requests.post(url, json=payload, headers=headers, timeout=10)
        return res.status_code in [200, 201]
    except Exception:
        return False

# --- 4. CORE PIPELINE BUSINESS LOGIC (SHARED ENGINE) ---
def execute_phoenix_core_pipeline(question: str, platform: str, platform_user_id: str, lang: str = "English") -> str:
    db = SessionLocal()
    try:
        # STEP A: Identity Segmentation
        identity = db.query(PlatformIdentity).filter(
            PlatformIdentity.platform == platform, 
            PlatformIdentity.platform_user_id == str(platform_user_id)
        ).first()

        if identity:
            profile_id = identity.profile_id
        else:
            profile_id = str(uuid.uuid4())
            db.add(Profile(profile_id=profile_id))
            db.add(PlatformIdentity(platform=platform, platform_user_id=str(platform_user_id), profile_id=profile_id))
            db.commit()

        # STEP B: Load Segmented Sandbox Memory
        db_messages = db.query(ChatMessage).filter(ChatMessage.profile_id == profile_id).order_by(ChatMessage.timestamp.asc()).all()
        history_pipeline = [{"role": msg.role, "content": msg.content} for msg in db_messages[-10:]]

        # STEP C: Temporal System Dynamic Base Anchor
        current_datetime = datetime.utcnow()
        current_time_anchor = current_datetime.strftime("%B %d, %Y")
        current_year = current_datetime.strftime("%Y")
        
        system_prompt = (
            f"You are Phoenix AI, an elite, highly objective intelligence assistant built by Chidibless from Nigeria.\n"
            f"REAL-WORLD TIME CONTEXT: Today's date is precisely {current_time_anchor}.\n\n"
            f"CRITICAL ASSISTANT DIRECTIVES:\n"
            f"1. Rely completely on the provided live real-time index data to resolve facts for the current year. Stale historical search entries must be discarded if they conflict with modern context.\n"
            f"2. ABSOLUTE SECRET PROCESSES: Speak authoritatively and natively. Never use meta-language phrases like 'According to the source text', 'Based on the blocks provided', or 'There is no information in the data'. If data is missing or conflicting, state what you know cleanly or state that information is unavailable without referencing your background software pipelines.\n"
            f"3. Do not include bracket citations, footnote links, or text annotations anywhere in your spoken vocabulary.\n"
            f"4. Respond naturally and directly in: {lang}."
        )

        # STEP D: Dynamic Search Query Hardening (Injects current temporal frame cleanly)
        if needs_web_search(question):
            search_query = question
            if not re.search(r'\b\d{4}\b', search_query):
                search_query = f"{search_query} {current_year}"
                
            live_intel = live_multi_source_search(search_query)
            if live_intel:
                system_prompt += f"\n\n[LIVE SEARCH INDEX DATA FOR {current_time_anchor}]:\n{live_intel}"

        payload_messages = [{"role": "system", "content": system_prompt}] + history_pipeline + [{"role": "user", "content": question}]

        # STEP E: LLM Model Pipeline Execution
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=payload_messages,
            max_tokens=1000,
            timeout=25
        )
        answer = response.choices[0].message.content

        # STEP F: Post-Process Output Sanitization (Wipes raw bracket citation leaks)
        answer = re.sub(r'【[^】]*】', '', answer)
        answer = re.sub(r'\[\d+†[^\]]*\]', '', answer)
        answer = answer.strip()

        # STEP G: Record Segmented Memory Trace
        db.add(ChatMessage(profile_id=profile_id, role="user", content=question))
        db.add(ChatMessage(profile_id=profile_id, role="assistant", content=answer))
        
        # STEP H: Internal Flywheel Dataset Capture
        try:
            training_block = {
                "messages": [
                    {"role": "system", "content": "You are Phoenix AI, a highly objective sovereign assistant built by Chidibless."},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
            }
            db.add(TrainingCorpus(payload=json.dumps(training_block)))
        except Exception:
            pass

        db.commit()
        return answer
    except Exception as err:
        db.rollback()
        return f"Core Engine Intercept Failure: {str(err)}"
    finally:
        db.close()

# --- 5. ENDPOINT INTERFACES ---
@app.post("/chat")
async def HTTP_chat_endpoint(request: Request):
    body = await request.json()
    question = body.get("question", "")
    lang = body.get("language", "English")
    platform = body.get("platform", "api_direct")
    platform_user_id = str(body.get("user_id", body.get("chat_id", "global_user")))
    
    if not question:
        return JSONResponse({"error": "No prompt input received"}, status_code=400)
        
    reply = execute_phoenix_core_pipeline(question, platform, platform_user_id, lang)
    return JSONResponse({"answer": reply})

@app.get("/webhook/whatsapp")
async def whatsapp_webhook_verification(
    mode: str = Query(None, alias="hub.mode"),
    token: str = Query(None, alias="hub.verify_token"),
    challenge: str = Query(None, alias="hub.challenge")
):
    if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN:
        return Response(content=challenge, media_type="text/plain")
    return Response(content="Verification Mismatch", status_code=403)

@app.post("/webhook/whatsapp")
async def whatsapp_webhook_receiver(request: Request):
    try:
        payload = await request.json()
        if "object" in payload:
            if payload.get("entry") and payload["entry"][0].get("changes"):
                value = payload["entry"][0]["changes"][0]["value"]
                if "messages" in value:
                    message_data = value["messages"][0]
                    sender_phone_number = message_data["from"]
                    if message_data.get("type") == "text":
                        incoming_text = message_data["text"]["body"]
                        
                        ai_reply = execute_phoenix_core_pipeline(
                            question=incoming_text,
                            platform="whatsapp",
                            platform_user_id=sender_phone_number
                        )
                        send_whatsapp_reply(sender_phone_number, ai_reply)
        return Response(content="EVENT_RECEIVED", status_code=200)
    except Exception:
        return Response(content="HANDLED", status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
