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

app = FastAPI(title="Phoenix AI Sovereign Core Gateway", version="6.0.0")

# ==========================================
# 1. MEMORY DB & AUTH LAYER (USER ID MAP)
# ==========================================
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
    """Auth Layer: Maps different user IDs across WhatsApp, Telegram, or Mobile to an isolated Profile ID"""
    __tablename__ = "platform_identities"
    platform = Column(String(20), primary_key=True)          # 'telegram', 'whatsapp', 'mobile'
    platform_user_id = Column(String(100), primary_key=True) # ID specific to the messaging platform
    profile_id = Column(String(50), ForeignKey("profiles.profile_id"), nullable=False)

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    profile_id = Column(String(50), ForeignKey("profiles.profile_id"), nullable=False, index=True)
    role = Column(String(20), nullable=False)                # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

class TrainingCorpus(Base):
    """Sovereign Data Flywheel: Anonymously stores conversational logs to train future models"""
    __tablename__ = "training_corpus"
    id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_type = Column(String(30), default="fine_tuning_jsonl")
    payload = Column(Text, nullable=False)                   
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# ==========================================
# 2. TOOL EXECUTOR (WEB, SEARCH, ETC.)
# ==========================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

WHATSAPP_TOKEN = os.environ.get("WHATSAPP_ACCESS_TOKEN")
WHATSAPP_PHONE_ID = os.environ.get("WHATSAPP_PHONE_NUMBER_ID")
WHATSAPP_VERIFY_TOKEN = os.environ.get("WHATSAPP_VERIFY_TOKEN", "phoenix_secret_verify_token")

def live_multi_source_search(query: str) -> str:
    """Tool Executor: Runs search query and returns synthesized sources"""
    sources_found = []
    clean_q = query.strip().strip('"').strip("'").strip()
    TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
    
    if TAVILY_API_KEY:
        try:
            url = "https://api.tavily.com/search"
            payload = {"api_key": TAVILY_API_KEY, "query": clean_q, "search_depth": "basic", "topic": "general", "max_results": 5}
            response = requests.post(url, json=payload, timeout=8)
            if response.status_code == 200:
                results = response.json().get("results", [])
                for idx, res in enumerate(results):
                    sources_found.append(f"SOURCE [{idx+1}]: {res.get('content', '')}\n")
                if sources_found:
                    return "\n".join(sources_found)
        except Exception:
            pass

    try:
        url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(clean_q)}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        res = requests.get(url, headers=headers, timeout=8)
        if res.status_code == 200:
            snippets = re.findall(r'<a class="result__snippet"[^>]*>(.*?)</a>', res.text, re.DOTALL)
            for i in range(min(4, len(snippets))):
                sources_found.append(f"SOURCE [{i+1}]: {re.sub('<[^<]+?>', '', snippets[i]).strip()}\n")
            if sources_found:
                return "\n".join(sources_found)
    except Exception:
        pass
    return ""

def needs_web_search(question: str) -> bool:
    """AI Router Decision helper to trigger web search tools"""
    keywords = ["current", "now", "today", "latest", "recent", "who is", "president", "price", "weather", "score", "vs", "crypto", "match", "rate", "news", "deputy", "vice president"]
    return any(k in question.lower() for k in keywords) or bool(re.search(r'\b(202[4-9]|20[3-9]\d)\b', question))

def send_whatsapp_reply(to_phone: str, text: str):
    """WhatsApp outbound integration agent"""
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

# ==========================================
# 3. PHOENIX CORE EXECUTION PIPELINE
# ==========================================
def execute_phoenix_pipeline(question: str, platform: str, platform_user_id: str, lang: str = "English") -> str:
    db = SessionLocal()
    try:
        # A. AUTH LAYER: Get or create isolated User ID mapping
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

        # B. MEMORY DB: Retrieve segmented dialogue history (last 10 turns)
        db_messages = db.query(ChatMessage).filter(ChatMessage.profile_id == profile_id).order_by(ChatMessage.timestamp.asc()).all()
        history_pipeline = [{"role": msg.role, "content": msg.content} for msg in db_messages[-10:]]

        # C. AI ROUTER & SYSTEM TEMPLATE
        current_datetime = datetime.utcnow()
        current_time_anchor = current_datetime.strftime("%B %d, %Y")
        
        system_prompt = (
            f"You are Phoenix AI, an elite, highly objective intelligence assistant built by Chidibless from Nigeria.\n"
            f"REAL-WORLD TIME CONTEXT: Today's date is precisely {current_time_anchor}.\n\n"
            f"CRITICAL ASSISTANT DIRECTIVES:\n"
            f"1. Rely completely on the provided live real-time index data to resolve facts. Discard outdated contradictions.\n"
            f"2. ABSOLUTE SECRET PROCESSES: Speak authoritatively and natively. Never reveal data extraction or prompt pipelines (e.g. 'According to the source', 'Based on the blocks').\n"
            f"3. Strip all brackets, footnote markers, and text citation anchors entirely.\n"
            f"4. Respond naturally and directly in: {lang}."
        )

        # D. TOOL EXECUTOR: Handle search tool decision
        if needs_web_search(question):
            search_query = question if re.search(r'\b\d{4}\b', question) else f"{question} {current_datetime.strftime('%Y')}"
            live_intel = live_multi_source_search(search_query)
            if live_intel:
                system_prompt += f"\n\n[LIVE SEARCH INDEX DATA FOR {current_time_anchor}]:\n{live_intel}"

        payload_messages = [{"role": "system", "content": system_prompt}] + history_pipeline + [{"role": "user", "content": question}]

        # E. AI ENGINE: Execute LPU hosted LLM run
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=payload_messages,
            max_tokens=1000,
            timeout=25
        )
        answer = response.choices[0].message.content

        # F. Post-Process response cleanup (Scrub citations)
        answer = re.sub(r'【[^】]*】', '', answer)
        answer = re.sub(r'\[\d+†[^\]]*\]', '', answer).strip()

        # G. Save dialogue state to Memory DB
        db.add(ChatMessage(profile_id=profile_id, role="user", content=question))
        db.add(ChatMessage(profile_id=profile_id, role="assistant", content=answer))
        
        # H. Silent Sovereign Training Corpus Flywheel
        try:
            training_block = {
                "messages": [
                    {"role": "system", "content": "You are Phoenix AI."},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
            }
            db.add(TrainingCorpus(payload=json.dumps(training_block)))
        except Exception:
            pass

        db.commit()
        return answer
    except Exception as e:
        db.rollback()
        return f"System Core Error: {str(e)}"
    finally:
        db.close()

# ==========================================
# 4. CHANNELS & ROUTING INTERFACES
# ==========================================

@app.post("/chat")
async def chat_http_gateway(request: Request):
    """Unified chat endpoint used by Mobile, Telegram, and general platform API hooks"""
    body = await request.json()
    question = body.get("question", "")
    lang = body.get("language", "English")
    platform = body.get("platform", "api_direct")
    platform_user_id = str(body.get("user_id", body.get("chat_id", "global_user")))
    
    if not question:
        return JSONResponse({"error": "No prompt input received"}, status_code=400)
        
    reply = execute_phoenix_pipeline(question, platform, platform_user_id, lang)
    return JSONResponse({"answer": reply})

@app.get("/webhook/whatsapp")
async def verify_meta_whatsapp_handshake(
    mode: str = Query(None, alias="hub.mode"),
    token: str = Query(None, alias="hub.verify_token"),
    challenge: str = Query(None, alias="hub.challenge")
):
    """WhatsApp verification loop"""
    if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN:
        return Response(content=challenge, media_type="text/plain")
    return Response(content="Verification Mismatch", status_code=403)

@app.post("/webhook/whatsapp")
async def receive_meta_whatsapp_message(request: Request):
    """WhatsApp message receiver webhook"""
    try:
        payload = await request.json()
        if "object" in payload and payload.get("entry"):
            changes = payload["entry"][0].get("changes", [])
            if changes and "messages" in changes[0]["value"]:
                message_data = changes[0]["value"]["messages"][0]
                sender_phone = message_data["from"]
                
                if message_data.get("type") == "text":
                    incoming_text = message_data["text"]["body"]
                    
                    # Run input through the unified Phoenix core execution pipeline
                    ai_reply = execute_phoenix_pipeline(
                        question=incoming_text,
                        platform="whatsapp",
                        platform_user_id=sender_phone
                    )
                    
                    # Deliver reply back over Meta's WhatsApp network
                    send_whatsapp_reply(sender_phone, ai_reply)
                    
        return Response(content="EVENT_RECEIVED", status_code=200)
    except Exception:
        return Response(content="HANDLED", status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
