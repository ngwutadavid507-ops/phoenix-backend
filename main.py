import os
import re
import math
import requests
import uuid
from datetime import datetime
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse
from groq import Groq
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

app = FastAPI(title="Phoenix AI Enterprise Backend", version="3.0.1")

# --- DATABASE ARCHITECTURE SETUP ---
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

Base.metadata.create_all(bind=engine)

# --- AI CORE CONFIGURATION ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

def web_search(query):
    TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
    if TAVILY_API_KEY:
        try:
            url = "https://api.tavily.com/search"
            payload = {"api_key": TAVILY_API_KEY, "query": query, "search_depth": "basic", "max_results": 3}
            response = requests.post(url, json=payload, timeout=8)
            if response.status_code == 200:
                results = response.json().get("results", [])
                snippets = [res.get("content", "") for res in results if res.get("content")]
                if snippets: return "\n\n".join(snippets)
        except Exception: pass 

    try:
        url = f"https://api.duckduckgo.com/?q={requests.utils.quote(query)}&format=json&no_html=1"
        headers = {"User-Agent": "PhoenixAI/3.0"}
        response = requests.get(url, headers=headers, timeout=6)
        if response.status_code == 200:
            data = response.json()
            chunks = []
            if data.get("AbstractText"): chunks.append(data.get("AbstractText"))
            for topic in data.get("RelatedTopics", [])[:3]:
                if "Text" in topic: chunks.append(topic["Text"])
            if chunks: return "\n\n".join(chunks)
    except Exception: pass
    return None

def needs_web_search(question):
    keywords = ["current", "now", "today", "latest", "recent", "who is president", "premier league", "price", "weather", "score", "vs", "2026"]
    q = question.lower()
    return any(k in q for k in keywords)

@app.get("/")
def root(): 
    return {"status": "Phoenix Database Core Engine Active", "year": 2026}

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
            return JSONResponse({"error": "No question provided"}, status_code=400)

        identity = db.query(PlatformIdentity).filter(
            PlatformIdentity.platform == platform, 
            PlatformIdentity.platform_user_id == platform_user_id
        ).first()

        if identity:
            profile_id = identity.profile_id
        else:
            profile_id = str(uuid.uuid4())
            new_profile = Profile(profile_id=profile_id)
            new_identity = PlatformIdentity(platform=platform, platform_user_id=platform_user_id, profile_id=profile_id)
            db.add(new_profile)
            db.add(new_identity)
            db.commit()

        db_messages = db.query(ChatMessage).filter(ChatMessage.profile_id == profile_id).order_by(ChatMessage.timestamp.asc()).all()
        
        history_pipeline = []
        for msg in db_messages[-20:]: 
            history_pipeline.append({"role": msg.role, "content": msg.content})

        system_prompt = (
            f"You are Phoenix AI, a brilliant conversational assistant built by Chidibless from Nigeria. "
            f"The current year is 2026. Donald Trump is the current President of the United States. "
            f"Answer questions naturally and concisely. Never reference your internal coding layout rules. Respond in {lang}."
        )

        if needs_web_search(question):
            search_results = web_search(question)
            if search_results:
                system_prompt += f"\n\nLive Search Context Data:\n{search_results}"

        payload_messages = [{"role": "system", "content": system_prompt}] + history_pipeline + [{"role": "user", "content": question}]

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=payload_messages,
            max_tokens=1000,
            timeout=25
        )
        answer = response.choices[0].message.content

        user_record = ChatMessage(profile_id=profile_id, role="user", content=question)
        ai_record = ChatMessage(profile_id=profile_id, role="assistant", content=answer)
        db.add(user_record)
        db.add(ai_record)
        db.commit()

        return JSONResponse({"answer": answer, "profile_id": profile_id, "platform": platform})
    except Exception as e:
        db.rollback()
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        db.close()

@app.post("/profile/sync")
async def sync_profile(body: dict):
    db = SessionLocal()
    try:
        existing_profile_id = body.get("profile_id")
        target_platform = body.get("platform") 
        target_platform_user_id = str(body.get("platform_user_id"))

        profile = db.query(Profile).filter(Profile.profile_id == existing_profile_id).first()
        if not profile:
            return JSONResponse({"error": "Profile ID not found"}, status_code=404)

        identity = db.query(PlatformIdentity).filter(
            PlatformIdentity.platform == target_platform,
            PlatformIdentity.platform_user_id == target_platform_user_id
        ).first()

        if identity:
            identity.profile_id = existing_profile_id
        else:
            identity = PlatformIdentity(platform=target_platform, platform_user_id=target_platform_user_id, profile_id=existing_profile_id)
            db.add(identity)
            
        db.commit()
        return {"status": "success", "message": f"Linked platform {target_platform} securely to profile {existing_profile_id}"}
    except Exception as e:
        db.rollback()
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
