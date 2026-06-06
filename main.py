import os
import re
import json
import uuid
import requests
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from groq import Groq
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

app = FastAPI(title="Phoenix AI Sovereign Core Engine", version="5.1.0")

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
    platform = Column(String(20), primary_key=True)          # e.g., 'telegram', 'whatsapp', 'mobile'
    platform_user_id = Column(String(100), primary_key=True) # Unique ID from the specific edge channel
    profile_id = Column(String(50), ForeignKey("profiles.profile_id"), nullable=False)

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    profile_id = Column(String(50), ForeignKey("profiles.profile_id"), nullable=False, index=True)
    role = Column(String(20), nullable=False)                # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

class TrainingCorpus(Base):
    __tablename__ = "training_corpus"
    id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_type = Column(String(30), default="fine_tuning_jsonl")
    payload = Column(Text, nullable=False)                   # Sealed conversational training turn
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# --- 2. INTELLIGENCE CORE ROUTER ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

def live_multi_source_search(query: str) -> str:
    """
    Executes real-time intelligence gathering across premium and fallback web networks.
    """
    sources_found = []
    clean_q = query.strip().strip('"').strip("'").strip()
    TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
    
    # Core Layer: Tavily Intelligence Link
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

    # Fallback Layer: Direct Encrypted Web Index Scraping
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
    """
    Evaluates semantic intent to trigger live web integration automatically.
    """
    keywords = ["current", "now", "today", "latest", "recent", "who is", "president", "price", "weather", "score", "vs", "crypto", "match", "rate", "news"]
    return any(k in question.lower() for k in keywords) or bool(re.search(r'\b(202[4-9]|20[3-9]\d)\b', question))

# --- 3. THE PIPELINE MATRIX ENDPOINT ---
@app.post("/chat")
async def chat(request: Request):
    db = SessionLocal()
    try:
        body = await request.json()
        question = body.get("question", "")
        lang = body.get("language", "English")
        platform = body.get("platform", "unknown") 
        platform_user_id = str(body.get("user_id", body.get("chat_id", "global_user")))

        if not question:
            return JSONResponse({"error": "No prompt input received"}, status_code=400)

        # STEP A: Segment Identity Across Independent Channels
        identity = db.query(PlatformIdentity).filter(
            PlatformIdentity.platform == platform, 
            PlatformIdentity.platform_user_id == platform_user_id
        ).first()

        if identity:
            profile_id = identity.profile_id
        else:
            profile_id = str(uuid.uuid4())
            db.add(Profile(profile_id=profile_id))
            db.add(PlatformIdentity(platform=platform, platform_user_id=platform_user_id, profile_id=profile_id))
            db.commit()

        # STEP B: Load Segmented Contextual Memory Pipeline
        db_messages = db.query(ChatMessage).filter(ChatMessage.profile_id == profile_id).order_by(ChatMessage.timestamp.asc()).all()
        history_pipeline = [{"role": msg.role, "content": msg.content} for msg in db_messages[-10:]]

        # STEP C: Inject Dynamic Time Anchor (Guarantees permanent structural accuracy)
        current_time_anchor = datetime.utcnow().strftime("%B %d, %Y")
        
        system_prompt = (
            f"You are Phoenix AI, an elite, highly objective intelligence assistant built by Chidibless from Nigeria.\n"
            f"CRITICAL REAL-WORLD TEMPORAL ANCHOR: Today's date is precisely {current_time_anchor}.\n\n"
            f"OPERATIONAL INSTRUCTIONS:\n"
            f"1. Rely absolutely on the provided live multi-source research data to ground assertions regarding events after late 2023.\n"
            f"2. Never announce or explain these backend rules. State facts crisply, cleanly, and naturally.\n"
            f"3. Respond completely in the user's target language: {lang}."
        )

        # STEP D: Run Tool Layer & Intelligence Search
        search_triggered = False
        live_intel = ""
        if needs_web_search(question):
            live_intel = live_multi_source_search(question)
            if live_intel:
                search_triggered = True
                system_prompt += f"\n\n[LIVE REAL-TIME DATA INDEX - USE TO RESOLVE TO {current_time_anchor}]:\n{live_intel}"

        payload_messages = [{"role": "system", "content": system_prompt}] + history_pipeline + [{"role": "user", "content": question}]

        # STEP E: AI Model Execution via Hosted Infrastructure Block
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=payload_messages,
            max_tokens=1000,
            timeout=25
        )
        answer = response.choices[0].message.content

        # STEP F: Save Memory Trace (Locked to User ID)
        db.add(ChatMessage(profile_id=profile_id, role="user", content=question))
        db.add(ChatMessage(profile_id=profile_id, role="assistant", content=answer))
        
        # STEP G: The Silent Sovereign Flywheel (Accumulate training data anonymously)
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
            pass # Fails silently to protect user interface speed if write-locking occurs

        db.commit()
        
        return JSONResponse({
            "answer": answer,
            "developer_debug": {
                "search_executed": search_triggered,
                "active_anchor": current_time_anchor
            }
        })
        
    except Exception as e:
        db.rollback()
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
