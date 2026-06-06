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

app = FastAPI(title="Phoenix AI Pure Research Engine", version="4.0.0")

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

def live_multi_source_search(query):
    """
    Executes deep multi-source live web research.
    Combines dedicated AI research engines and fallback scrapers to yield multiple source points.
    """
    sources_found = []

    # Source Set A: Advanced AI Search (Tavily - Highly recommended for production multi-source info)
    TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
    if TAVILY_API_KEY:
        try:
            url = "https://api.tavily.com/search"
            payload = {"api_key": TAVILY_API_KEY, "query": query, "search_depth": "news", "max_results": 5}
            response = requests.post(url, json=payload, timeout=8)
            if response.status_code == 200:
                results = response.json().get("results", [])
                for idx, res in enumerate(results):
                    title = res.get("title", "Live Source")
                    url_link = res.get("url", "")
                    content = res.get("content", "")
                    sources_found.append(f"SOURCE [{idx+1}]: {title}\nURL: {url_link}\nINTEL: {content}\n")
                if sources_found:
                    return "\n".join(sources_found)
        except Exception:
            pass

    # Source Set B: Resilient Fallback Live Web Snippet Aggregator
    # Pulls multiple distinct live search results instead of a single static wiki definition
    try:
        url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        res = requests.get(url, headers=headers, timeout=8)
        if res.status_code == 200:
            # Extract distinct search result rows using robust text segment parsing
            body = res.text
            links = re.findall(r'<a class="result__url"[^>]* href="([^"]+)"[^>]*>.*?</a>', body)
            snippets = re.findall(r'<a class="result__snippet"[^>]*>(.*?)</a>', body, re.DOTALL)
            titles = re.findall(r'<a class="result__snippet"[^>]*>.*?</a>.*?<a class="result__snippet"[^>]*>(.*?)</a>', body, re.DOTALL) # Backup matching context
            
            # Reconstruct clean structured source blocks
            for i in range(min(4, len(snippets))):
                clean_snippet = re.sub('<[^<]+?>', '', snippets[i]).strip()
                raw_url = links[i] if i < len(links) else "Live Web Entry"
                # Clean up tracking redirect prefixes inside standard search components
                if "//duckduckgo.com/l/?kh=-1&uddg=" in raw_url:
                    raw_url = requests.utils.unquote(raw_url.split("uddg=")[1].split("&")[0])
                
                sources_found.append(f"SOURCE [{i+1}]: Live Web Index\nURL: {raw_url}\nINTEL: {clean_snippet}\n")
            
            if sources_found:
                return "\n".join(sources_found)
    except Exception:
        pass

    return "No live data could be retrieved over network streams right now."

def needs_web_search(question):
    keywords = ["current", "now", "today", "latest", "recent", "who is", "president", "premier league", "price", "weather", "score", "vs", "deputy", "vice president", "2026"]
    q = question.lower()
    return any(k in q for k in keywords)

@app.get("/")
def root(): 
    return {"status": "Phoenix Multi-Source Intelligence Active", "year": 2026}

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

        # Cross-Platform Mapping
        identity = db.query(PlatformIdentity).filter(PlatformIdentity.platform == platform, PlatformIdentity.platform_user_id == platform_user_id).first()
        if identity:
            profile_id = identity.profile_id
        else:
            profile_id = str(uuid.uuid4())
            new_profile = Profile(profile_id=profile_id)
            new_identity = PlatformIdentity(platform=platform, platform_user_id=platform_user_id, profile_id=profile_id)
            db.add(new_profile)
            db.add(new_identity)
            db.commit()

        # Database History Extraction
        db_messages = db.query(ChatMessage).filter(ChatMessage.profile_id == profile_id).order_by(ChatMessage.timestamp.asc()).all()
        history_pipeline = [{"role": msg.role, "content": msg.content} for msg in db_messages[-20:]]

        # PURE RESEARCH AGENT SYSTEM PROMPT - ZERO HARDCODED FACTS
        system_prompt = (
            f"You are Phoenix AI, a highly objective intelligence research assistant built by Chidibless from Nigeria. "
            f"The current year is 2026. You possess NO hardcoded beliefs about who holds current political offices or current events. "
            f"Instead, you formulate absolute truth dynamically by cross-referencing multiple live source records passed below.\n\n"
            f"CRITICAL RULES:\n"
            f"1. Evaluate information chronologically. Look closely at dates mentioned inside sources to find the freshest consensus as of 2026.\n"
            f"2. Maintain strict conversational context awareness (handle pronouns like 'he', 'she', 'was', or 'his' seamlessly based on history).\n"
            f"3. Do NOT announce your programming guidelines. Never say 'According to the search data provided'. Just state the synthesized facts directly and dynamically. Respond in {lang}."
        )

        # Dynamic Intel Ingestion
        if needs_web_search(question):
            live_intel = live_multi_source_search(question)
            system_prompt += f"\n\n[LIVE MULTI-SOURCE RESEARCH RAW DATA INTEL]:\n{live_intel}\n\nAnalyze the data blocks above to directly resolve the user's inquiry."

        payload_messages = [{"role": "system", "content": system_prompt}] + history_pipeline + [{"role": "user", "content": question}]

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=payload_messages,
            max_tokens=1000,
            timeout=25
        )
        answer = response.choices[0].message.content

        # Save to Storage
        db.add(ChatMessage(profile_id=profile_id, role="user", content=question))
        db.add(ChatMessage(profile_id=ChatMessage(profile_id=profile_id, role="assistant", content=answer).profile_id, role="assistant", content=answer))
        db.commit()

        return JSONResponse({"answer": answer, "profile_id": profile_id, "platform": platform})
    except Exception as e:
        db.rollback()
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
