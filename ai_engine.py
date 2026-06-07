import os
import asyncio
from groq import Groq
from tavily import TavilyClient
from serpapi import GoogleSearch

# Core Integration API Clients
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

MODEL_NAME = "llama-3.3-70b-versatile"

def fetch_tavily_results(query: str) -> str:
    """Queries Tavily API for semantic, LLM-optimized web documentation chunks."""
    if not tavily_client:
        return "⚠️ Tavily Engine: Client unconfigured (Missing Key).\n"
    try:
        response = tavily_client.search(query=query, max_results=3, search_depth="advanced")
        results = response.get("results", [])
        
        context_str = "--- TAVILY LIVE INSIGHTS ---\n"
        for res in results:
            context_str += f"• {res.get('title')} ({res.get('url')}):\n  {res.get('content')}\n\n"
        return context_str
    except Exception as e:
        return f"⚠️ Tavily Lookup Failed: {e}\n"

def fetch_serpapi_results(query: str) -> str:
    """Queries SerpApi for direct raw organic Google Search results coverage."""
    if not SERPAPI_API_KEY:
        return "⚠️ SerpApi Engine: Client unconfigured (Missing Key).\n"
    try:
        search = GoogleSearch({
            "q": query,
            "api_key": SERPAPI_API_KEY,
            "num": 3
        })
        dictionary_results = search.get_dict()
        organic_results = dictionary_results.get("organic_results", [])
        
        context_str = "--- SERPAPI GOOGLE SEARCH EXTRACTIONS ---\n"
        # Check for instant direct answers if available on Google
        if "answer_box" in dictionary_results:
            box = dictionary_results["answer_box"]
            if "answer" in box:
                context_str += f"⚡ Direct Answer Box: {box.get('answer')}\n\n"
            elif "snippet" in box:
                context_str += f"⚡ Direct Answer Box Snippet: {box.get('snippet')}\n\n"

        for res in organic_results:
            context_str += f"• {res.get('title')} ({res.get('link')}):\n  {res.get('snippet')}\n\n"
        return context_str
    except Exception as e:
        return f"⚠️ SerpApi Lookup Failed: {e}\n"

async def aggregate_dual_search(query: str) -> str:
    """Runs Tavily and SerpApi queries concurrently via asyncio threads to preserve speed."""
    loop = asyncio.get_event_loop()
    
    # Offload blocking synchronous network IO requests to worker threads
    tavily_task = loop.run_in_executor(None, fetch_tavily_results, query)
    serpapi_task = loop.run_in_executor(None, fetch_serpapi_results, query)
    
    tavily_res, serpapi_res = await asyncio.gather(tavily_task, serpapi_task)
    
    combined_context = (
        "=== COOPERATIVE REAL-TIME GROUND TRUTH SEARCH MATRIX ===\n"
        f"{tavily_res}"
        f"{serpapi_res}"
        "========================================================\n"
    )
    return combined_context

def router_needs_search(history_messages: list) -> bool:
    """
    Intelligent Router Engine: Evaluates the latest user prompt 
    to decide if it requires external real-time data lookup.
    """
    if not history_messages:
        return False
        
    last_user_msg = history_messages[-1]["content"].lower()
    
    search_triggers = [
        "president", "governor", "deputy", "minister", "prime minister",
        "who is", "current", "latest", "news", "today", "price", "rate", 
        "exchange", "vs", "versus", "winner", "match", "score", "weather",
        "now", "2025", "2026", "dollar", "naira", "jamb"
    ]
    
    return any(trigger in last_user_msg for trigger in search_triggers)

async def process_text_or_vision(user_id: str, messages_list: list, image_bytes: bytes = None):
    """
    Executes core routing pattern combining Multi-turn Memory, 
    Dual-Search Cross-Verification RAG, and Vision pipelines.
    """
    try:
        if not GROQ_API_KEY:
            return "❌ Engine Configuration Error: GROQ_API_KEY environment variable is missing on Render."

        # A. MULTIMODAL VISION TRACK
        if image_bytes:
            vision_model = "llama-3.2-11b-vision-preview"
            import base64
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            last_prompt = messages_list[-1]["content"] if messages_list else "Analyze this image payload."
            
            response = groq_client.chat.completions.create(
                model=vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": last_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1024
            )
            return response.choices[0].message.content

        # B. TEXT PROCESSING WITH ROUTER + DUAL SEARCH AGGREGATION
        else:
            if router_needs_search(messages_list):
                last_prompt = messages_list[-1]["content"]
                print(f"📡 Dual Router Fired: Gathering live search context for: '{last_prompt}'")
                
                # Fetch fresh aggregated multi-source ground truth
                live_web_context = await aggregate_dual_search(last_prompt)
                
                # Append context directly into the current execution thread safely
                augmented_messages = messages_list[:-1] + [
                    {
                        "role": "user", 
                        "content": f"{live_web_context}\nUsing the highly verified search context provided above, answer this request accurately. If information contradicts your internal pre-training cutoff data, give absolute priority to the verified live search context: {last_prompt}"
                    }
                ]
                
                response = groq_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=augmented_messages
                )
            else:
                response = groq_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages_list
                )
                
            return response.choices[0].message.content

    except Exception as e:
        print(f"Error processing context engine pipeline: {e}")
        return f"⚠️ Phoenix AI Engine Error: Unable to complete request. ({e})"

async def generate_image_url(prompt: str) -> str:
    import urllib.parse
    encoded_prompt = urllib.parse.quote(prompt)
    return f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true"

async def transcribe_voice_bytes(audio_bytes: bytes, filename: str) -> str:
    try:
        with open(filename, "wb") as f:
            f.write(audio_bytes)
        with open(filename, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                file=(filename, audio_file.read()),
                model="whisper-large-v3",
                response_format="text"
            )
        if os.path.exists(filename):
            os.remove(filename)
        return transcription
    except Exception as e:
        print(f"Voice note transcription failure: {e}")
        return "[Audio transcription asset unreadable]"
