import os
import asyncio
from groq import Groq
from tavily import TavilyClient
from serpapi import GoogleSearch

# Core Integration API Clients (Read gracefully on startup)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

MODEL_NAME = "llama-3.3-70b-versatile"

def fetch_tavily_results(query: str) -> str:
    """Queries Tavily API for semantic, LLM-optimized web documentation chunks."""
    if not tavily_client:
        return ""
    try:
        response = tavily_client.search(query=query, max_results=3, search_depth="advanced")
        results = response.get("results", [])
        
        context_str = "--- TAVILY LIVE DATA ---\n"
        for res in results:
            context_str += f"Context Chunk: {res.get('content')}\n\n"
        return context_str
    except Exception as e:
        print(f"Tavily background error: {e}")
        return ""

def fetch_serpapi_results(query: str) -> str:
    """Queries SerpApi safely inside an execution block only when requested."""
    if not SERPAPI_API_KEY:
        return ""
    try:
        # ✅ Instantiated dynamically inside the function scope to prevent boot blocking
        search = GoogleSearch({
            "q": query,
            "api_key": SERPAPI_API_KEY,
            "num": 3
        })
        dictionary_results = search.get_dict()
        organic_results = dictionary_results.get("organic_results", [])
        
        context_str = "--- SERPAPI GOOGLE SEARCH DATA ---\n"
        if "answer_box" in dictionary_results:
            box = dictionary_results["answer_box"]
            context_str += f"Direct Answer: {box.get('answer') or box.get('snippet')}\n\n"

        for res in organic_results:
            context_str += f"Snippet: {res.get('snippet')}\n\n"
        return context_str
    except Exception as e:
        print(f"SerpApi background error: {e}")
        return ""

async def aggregate_dual_search(query: str) -> str:
    """Runs Tavily and SerpApi queries concurrently via asyncio threads to preserve speed."""
    loop = asyncio.get_event_loop()
    
    tavily_task = loop.run_in_executor(None, fetch_tavily_results, query)
    serpapi_task = loop.run_in_executor(None, fetch_serpapi_results, query)
    
    tavily_res, serpapi_res = await asyncio.gather(tavily_task, serpapi_task)
    
    combined_context = (
        f"{tavily_res}\n"
        f"{serpapi_res}"
    )
    return combined_context

def router_needs_search(history_messages: list) -> bool:
    """Intelligent Router Engine: Evaluates the latest user prompt for real-time triggers."""
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
    """Executes core routing pattern with high-fidelity, polished markdown outputs."""
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
                            {"type": "text", "text": f"{last_prompt}\nProvide a beautifully formatted, clean analysis of this image. Use bullet points and bold headers where appropriate."},
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

        # B. TEXT PROCESSING WITH BEAUTIFIED STRUCTURAL FRAMING
        else:
            presentation_instructions = (
                "\n\n[PRESENTATION INSTRUCTIONS]: Format your output beautifully for a premium mobile chat application (Telegram/WhatsApp). "
                "1. Use bold headings (**Heading**) to split distinct thoughts.\n"
                "2. Use emojis naturally at the start of sections to make it visually engaging but keep it professional.\n"
                "3. Use clean, spaced bullet points for lists.\n"
                "4. Absolutely never print raw URLs, source markers, code variables, or snippets from the background search data text.\n"
                "5. Keep the language direct, elegant, and crisp. No fluff or conversational filler like 'Sure, let me search that for you'."
            )

            if router_needs_search(messages_list):
                last_prompt = messages_list[-1]["content"]
                
                live_web_context = await aggregate_dual_search(last_prompt)
                
                augmented_messages = messages_list[:-1] + [
                    {
                        "role": "user", 
                        "content": (
                            f"=== BACKGROUND SEARCH CONTEXT ===\n{live_web_context}\n\n"
                            f"User Query: {last_prompt}\n"
                            f"Using the verified search context above, synthesize a complete answer. {presentation_instructions}"
                        )
                    }
                ]
                
                response = groq_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=augmented_messages
                )
            else:
                modified_messages = messages_list[:-1] + [
                    {
                        "role": "user",
                        "content": f"{messages_list[-1]['content']}{presentation_instructions}"
                    }
                ]
                response = groq_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=modified_messages
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
