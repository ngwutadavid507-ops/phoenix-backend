import os
import asyncio
from groq import Groq
from tavily import TavilyClient
from serpapi import GoogleSearch

# Core Integration API Clients
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
    return f"{tavily_res}\n{serpapi_res}"

def router_needs_search(history_messages: list) -> bool:
    """Intelligent Router Engine: Evaluates the latest user prompt for real-time triggers."""
    if not history_messages:
        return False
        
    last_user_msg = history_messages[-1]["content"].lower()
    search_triggers = [
        "president", "governor", "deputy", "minister", "prime minister",
        "who is", "current", "latest", "news", "today", "price", "rate", 
        "exchange", "vs", "versus", "winner", "match", "score", "weather",
        "now", "2025", "2026", "dollar", "naira", "jamb", "his", "her", "they", "age", "old"
    ]
    return any(trigger in last_user_msg for trigger in search_triggers)

async def rewrite_query_with_context(messages_list: list) -> str:
    """
    🧠 QUERY REWRITER LAYER:
    Analyzes conversation history to resolve pronouns ('he', 'his deputy') 
    into an explicit standalone Google search string.
    """
    if len(messages_list) <= 2:
        return messages_list[-1]["content"]
        
    try:
        # Create a lightweight context summary for a fast model pass
        history_summary = ""
        for msg in messages_list[-4:-1]:  # Look back up to 3 turns max
            history_summary += f"{msg['role'].upper()}: {msg['content']}\n"
            
        last_prompt = messages_list[-1]["content"]
        
        rewrite_prompt = (
            f"Given the following chat history conversation:\n{history_summary}\n"
            f"And the new follow-up question: '{last_prompt}'\n\n"
            f"Rewrite a single standalone search engine query that resolves pronouns (like 'he', 'him', 'his deputy', 'them') into the actual subjects mentioned earlier in the conversation. "
            f"Output ONLY the final plain search query string. Do not add any conversational text or explanation."
        )
        
        # Use a super fast completion to generate the optimized search string
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",  # Small, blindingly fast model for internal utilities
            messages=[{"role": "user", "content": rewrite_prompt}],
            temperature=0.0,
            max_tokens=60
        )
        optimized_query = response.choices[0].message.content.strip().strip('"')
        print(f"🔄 Query Rewriter transformed '{last_prompt}' -> '{optimized_query}'")
        return optimized_query
    except Exception as e:
        print(f"Query rewriter exception: {e}")
        return messages_list[-1]["content"]

async def process_text_or_vision(user_id: str, messages_list: list, image_bytes: bytes = None):
    """Executes core routing pattern, maintaining total chat context while infusing RAG data."""
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

        # B. TEXT PROCESSING WITH FULL MEMORY RETENTION + QUERY REWRITING
        else:
            presentation_instructions = (
                "\n\n[CRITICAL SYSTEM INSTRUCTIONS]: You are Phoenix AI, an interactive, smart chat companion—NOT a generic search engine wrapper. "
                "1. Maintain a natural, organic conversational flow. Follow the multi-turn memory thread explicitly.\n"
                "2. Synthesize answers dynamically using background context. Never copy-paste raw definitions, links, or chunks.\n"
                "3. Format beautifully using bold titles and spaced mobile-optimized bullet points.\n"
                "4. Keep your response laser-focused on exactly what was asked. Avoid irrelevant text dumps."
            )

            if router_needs_search(messages_list):
                # Run the contextual query rewriter to build a real standalone search string
                optimized_search_string = await rewrite_query_with_context(messages_list)
                
                # Fetch fresh aggregated multi-source ground truth using the optimized query
                live_web_context = await aggregate_dual_search(optimized_search_string)
                
                # Inject the verified search context as background data stream
                augmented_messages = [
                    {
                        "role": "system",
                        "content": f"LIVE SEARCH BACKGROUND MATRIX DATA:\n{live_web_context}\nUse this live context to verify real-time claims seamlessly based on the conversation history thread. Current year is 2026."
                    }
                ] + messages_list[:-1] + [
                    {
                        "role": "user",
                        "content": f"{messages_list[-1]['content']}{presentation_instructions}"
                    }
                ]
                
                response = groq_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=augmented_messages
                )
            else:
                # Regular conversation tracking matching memory stream
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
