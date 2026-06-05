import os
import re
import math
import requests
from collections import defaultdict
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Phoenix AI Unified Backend", version="2.0.1")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

def web_search(query):
    try:
        # Fallback multi-agent structure to scrape text streams securely
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        r = requests.get(url, headers=headers, timeout=10)
        
        snippets = re.findall(r'<a class="result__snippet"[^>]*>(.*?)</a>', r.text, re.DOTALL)
        clean_results = []
        for snip in snippets[:4]:
            clean_text = re.sub(r'<[^>]*>', '', snip).strip()
            clean_results.append(clean_text)
            
        return "\n\n".join(clean_results) if clean_results else None
    except Exception:
        return None

def needs_web_search(question):
    keywords = [
        "current", "now", "today", "latest", "recent", "right now",
        "2024", "2025", "2026", "who is president", "premier league",
        "prime minister", "price", "weather", "news", "score", "winner", 
        "election", "who is", "what happened", "vs"
    ]
    q = question.lower()
    return any(k in q for k in keywords)

def ask_groq(system_prompt, user_prompt, model="llama-3.3-70b-versatile"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=1000,
        timeout=25
    )
    return response.choices[0].message.content

# Custom TF-IDF Indexing Utilities
def tokenize(text): return re.findall(r'\b[a-z]{2,}\b', text.lower())
def chunk_text(text, chunk_size=120, overlap=20):
    words = text.split(); chunks = []; i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_size]))
        if i + chunk_size >= len(words): break
        i += chunk_size - overlap
    return chunks

def build_index(chunks):
    index = defaultdict(list); dfreq = defaultdict(int)
    for cid, chunk in enumerate(chunks):
        tks = tokenize(chunk)
        if not tks: continue
        freq = defaultdict(int)
        for t in tks: freq[t] += 1
        for t, count in freq.items():
            index[t].append((cid, count / len(tks)))
            dfreq[t] += 1
    return index, dfreq

def retrieve_chunks(query, chunks, index, doc_freq, top_k=4):
    tks = tokenize(query); scores = defaultdict(float)
    for t in tks:
        if t not in index: continue
        idf = math.log((len(chunks) + 1) / (doc_freq[t] + 1)) + 1.0
        for cid, tf in index[t]: scores[cid] += tf * idf
    if not scores: return chunks[:top_k]
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunks[cid] for cid, _ in ranked[:top_k]]

@app.get("/")
def root(): return {"status": "Phoenix Core Engine Active", "year": 2026}

@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.json()
        question = body.get("question", "")
        lang = body.get("language", "English")
        platform = body.get("platform", "unknown")

        if not question:
            return JSONResponse({"error": "No question provided"}, status_code=400)

        if needs_web_search(question):
            search_results = web_search(question)
            if search_results:
                answer = ask_groq(
                    f"You are Phoenix AI. Answer using ONLY the live search results below. "
                    f"Be direct, accurate, and completely factual. Current Year: 2026. Respond in {lang}.",
                    f"Live Search Data:\n{search_results}\n\nQuestion: {question}"
                )
            else:
                # Upgraded Fallback Guardrail System Prompt
                answer = ask_groq(
                    f"You are Phoenix AI, built by Chidibless from Nigeria. The current year is 2026. "
                    f"Donald Trump is the President of the United States (inaugurated Jan 20, 2025). "
                    f"Answer the user directly using this 2026 timeline context. Note that live web search failed, "
                    f"so briefly advise the user to confirm rapidly changing details live. Respond in {lang}.",
                    question
                )
        else:
            answer = ask_groq(
                f"You are Phoenix AI, built by Chidibless from Nigeria. The current year is 2026. "
                f"Answer any question helpfully, clearly, and concisely. Respond in {lang}.",
                question
            )

        return JSONResponse({"answer": answer, "platform": platform, "language": lang})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/voice")
async def process_voice(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        transcription = client.audio.transcriptions.create(
            file=(file.filename, file_bytes), model="whisper-large-v3", response_format="json"
        )
        return JSONResponse({"transcript": transcription.text})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/vision")
async def process_vision(request: Request):
    try:
        body = await request.json()
        image_url = body.get("image_url")
        prompt = body.get("prompt", "Analyze this image and describe it clearly.")
        lang = body.get("language", "English")

        if not image_url: return JSONResponse({"error": "No image URL provided"}, status_code=400)

        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[{"role": "user", "content": [{"type": "text", "text": f"{prompt} Respond in {lang}."}, {"type": "image_url", "image_url": {"url": image_url}}]}],
            max_tokens=1000
        )
        return JSONResponse({"answer": response.choices[0].message.content})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/analyse")
async def analyse_document(request: Request):
    try:
        body = await request.json()
        text = body.get("text", "")
        action = body.get("action", "summarise")
        question = body.get("question", "")
        lang = body.get("language", "English")

        if not text: return JSONResponse({"error": "No text provided"}, status_code=400)

        chunks = chunk_text(text)
        idx, dfreq = build_index(chunks)
        total_chunks = len(chunks)

        if action == "summarise":
            answer = ask_groq(f"You are Phoenix AI. Summarise this document clearly. Respond in {lang}.", f"Content:\n{text[:8000]}")
        elif action == "quiz":
            answer = ask_groq(f"You are Phoenix AI. Create a 5-question multiple choice test with choices A, B, C, D based on this context. Respond in {lang}.", f"Content:\n{text[:8000]}")
        elif action == "intelligence":
            answer = ask_groq(f"You are Phoenix AI. Provide an executive summary, type evaluation, key facts, and internal layout overview. Respond in {lang}.", f"Content:\n{text[:8000]}")
        elif action == "question" and question:
            context_data = build_rag_context(question, {"chunks": chunks, "index": idx, "doc_freq": dfreq})
            answer = ask_groq(f"You are Phoenix AI. Answer using ONLY these excerpts. Respond in {lang}.", f"Excerpts:\n{context_data}\n\nQuestion: {question}")
        else:
            answer = ask_groq(f"You are Phoenix AI. Review this information. Respond in {lang}.", f"Content:\n{text[:8000]}")

        return JSONResponse({"answer": answer, "chunks": total_chunks, "action": action, "language": lang})
    except Exception as e: return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/compare")
async def compare_documents(request: Request):
    try:
        body = await request.json()
        doc1 = body.get("document1", "")
        doc2 = body.get("document2", "")
        lang = body.get("language", "English")

        if not doc1 or not doc2: return JSONResponse({"error": "Both documents required"}, status_code=400)

        answer = ask_groq(f"You are Phoenix AI. Structurally compare both text sets. List similarities, sharp contrasts, and structural completeness. Respond in {lang}.", f"Doc1:\n{doc1[:6000]}\n\nDoc2:\n{doc2[:6000]}")
        return JSONResponse({"answer": answer, "language": lang})
    except Exception as e: return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
