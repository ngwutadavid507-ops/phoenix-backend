import os
import re
import math
import requests
from collections import defaultdict
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Phoenix AI Backend", version="1.0.0")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

def web_search(query):
    try:
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1"
        }
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        results = []
        if data.get("Abstract"):
            results.append(data["Abstract"])
        if data.get("Answer"):
            results.append(data["Answer"])
        for topic in data.get("RelatedTopics", [])[:3]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(topic["Text"])
        return "\n\n".join(results) if results else None
    except Exception:
        return None

def needs_web_search(question):
    keywords = [
        "current", "now", "today", "latest", "recent",
        "2024", "2025", "2026", "who is president",
        "prime minister", "price", "weather", "news",
        "score", "winner", "election", "who is"
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

def tokenize(text):
    return re.findall(r'\b[a-z]{2,}\b', text.lower())

def chunk_text(text, chunk_size=120, overlap=20):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
        i += chunk_size - overlap
    return chunks

def build_index(chunks):
    index = defaultdict(list)
    doc_freq = defaultdict(int)
    for cid, chunk in enumerate(chunks):
        tokens = tokenize(chunk)
        if not tokens:
            continue
        freq = defaultdict(int)
        for token in tokens:
            freq[token] += 1
        for token, count in freq.items():
            tf = count / len(tokens)
            index[token].append((cid, tf))
            doc_freq[token] += 1
    return index, doc_freq

def retrieve_chunks(query, chunks, index, doc_freq, top_k=4):
    query_tokens = tokenize(query)
    n = len(chunks)
    scores = defaultdict(float)
    for token in query_tokens:
        if token not in index:
            continue
        idf = math.log((n + 1) / (doc_freq[token] + 1)) + 1.0
        for cid, tf in index[token]:
            scores[cid] += tf * idf
    if not scores:
        return chunks[:top_k]
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunks[cid] for cid, _ in ranked[:top_k]]

def index_document(text):
    chunks = chunk_text(text)
    index, doc_freq = build_index(chunks)
    return {
        "chunks": chunks,
        "index": index,
        "doc_freq": doc_freq,
        "total_chunks": len(chunks)
    }

def build_rag_context(query, rag_store):
    top_chunks = retrieve_chunks(
        query,
        rag_store["chunks"],
        rag_store["index"],
        rag_store["doc_freq"],
        top_k=4
    )
    return "\n\n---\n\n".join(top_chunks)

@app.get("/")
def root():
    return {"status": "Phoenix AI Backend is running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "alive"}

@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.json()
        question = body.get("question", "")
        lang = body.get("language", "English")
        platform = body.get("platform", "unknown")

        if not question:
            return JSONResponse(
                {"error": "No question provided"},
                status_code=400
            )

        if needs_web_search(question):
            search_results = web_search(question)
            if search_results:
                answer = ask_groq(
                    f"You are Phoenix AI. Answer using ONLY the search "
                    f"results below. Be direct and confident. "
                    f"The year is 2026. Respond in {lang}.",
                    f"Search results:\n{search_results}\n\n"
                    f"Question: {question}\n\n"
                    f"Answer directly from search results only."
                )
            else:
                answer = ask_groq(
                    f"You are Phoenix AI. Answer helpfully. "
                    f"For current events tell user to verify online. "
                    f"Respond in {lang}.",
                    question
                )
        else:
            answer = ask_groq(
                f"You are Phoenix AI, built by Chidibless from Nigeria. "
                f"Answer any question helpfully and accurately. "
                f"Be clear and concise. Respond in {lang}.",
                question
            )

        return JSONResponse({
            "answer": answer,
            "platform": platform,
            "language": lang
        })

    except Exception as e:
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )

@app.post("/analyse")
async def analyse_document(request: Request):
    try:
        body = await request.json()
        text = body.get("text", "")
        action = body.get("action", "summarise")
        question = body.get("question", "")
        lang = body.get("language", "English")

        if not text:
            return JSONResponse(
                {"error": "No text provided"},
                status_code=400
            )

        rag_store = index_document(text)

        if action == "summarise":
            answer = ask_groq(
                f"You are Phoenix AI. Summarise this document. "
                f"Respond in {lang}.",
                f"Document:\n{text[:8000]}\n\nProvide a detailed summary."
            )

        elif action == "quiz":
            answer = ask_groq(
                f"You are Phoenix AI. Generate quiz questions. "
                f"Respond in {lang}.",
                f"Document:\n{text[:8000]}\n\n"
                f"Generate 5 multiple choice questions with 4 options "
                f"each (A, B, C, D). Include correct answer."
            )

        elif action == "intelligence":
            answer = ask_groq(
                f"You are Phoenix AI. Analyse this document. "
                f"Respond in {lang}.",
                f"Document:\n{text[:8000]}\n\n"
                f"Provide intelligence report:\n"
                f"1. Document Type\n"
                f"2. Main Topic\n"
                f"3. Key Sections\n"
                f"4. Important Facts\n"
                f"5. Suggested Questions"
            )

        elif action == "question" and question:
            context = build_rag_context(question, rag_store)
            answer = ask_groq(
                f"You are Phoenix AI. Answer from document excerpts only. "
                f"Respond in {lang}.",
                f"Document excerpts:\n{context}\n\n"
                f"Question: {question}"
            )

        else:
            answer = ask_groq(
                f"You are Phoenix AI. Help with this document. "
                f"Respond in {lang}.",
                f"Document:\n{text[:8000]}"
            )

        return JSONResponse({
            "answer": answer,
            "chunks": rag_store["total_chunks"],
            "action": action,
            "language": lang
        })

    except Exception as e:
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )

@app.post("/compare")
async def compare_documents(request: Request):
    try:
        body = await request.json()
        doc1 = body.get("document1", "")
        doc2 = body.get("document2", "")
        lang = body.get("language", "English")

        if not doc1 or not doc2:
            return JSONResponse(
                {"error": "Both documents required"},
                status_code=400
            )

        answer = ask_groq(
            f"You are Phoenix AI. Compare two documents. "
            f"Respond in {lang}.",
            f"Document 1:\n{doc1[:6000]}\n\n"
            f"Document 2:\n{doc2[:6000]}\n\n"
            f"Compare:\n"
            f"1. Key similarities\n"
            f"2. Key differences\n"
            f"3. Which is more comprehensive and why"
        )

        return JSONResponse({
            "answer": answer,
            "language": lang
        })

    except Exception as e:
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
