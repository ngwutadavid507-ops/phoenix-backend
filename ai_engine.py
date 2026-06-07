# ai_engine.py
import os
import json
import asyncio
from groq import Groq
from tavily import TavilyClient
from serpapi import GoogleSearch

# Pull feature modules dynamically from local package pathways
from modules.rag_processor import extract_document_context
from modules.personalization import load_user_profile, save_user_profile
from modules.crypto_ticker import fetch_crypto_asset_metrics

# Infrastructure Key Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

MODEL_NAME = "llama-3.3-70b-versatile"

def fetch_tavily_results(query: str) -> str:
    if not tavily_client: return ""
    try:
        response = tavily_client.search(query=query, max_results=3, search_depth="advanced")
        results = response.get("results", [])
        context_str = "--- TAVILY LIVE DATA ---\n"
        for res in results:
            context_str += f"Context: {res.get('content')}\n\n"
        return context_str
    except: return ""

def fetch_serpapi_results(query: str) -> str:
    if not SERPAPI_API_KEY: return ""
    try:
        search = GoogleSearch({"q": query, "api_key": SERPAPI_API_KEY, "num": 3})
        dictionary_results = search.get_dict()
        organic_results = dictionary_results.get("organic_results", [])
        context_str = "--- SERPAPI GOOGLE SEARCH DATA ---\n"
        if "answer_box" in dictionary_results:
            box = dictionary_results["answer_box"]
            context_str += f"Direct Answer: {box.get('answer') or box.get('snippet')}\n\n"
        for res in organic_results:
            context_str += f"Snippet: {res.get('snippet')}\n\n"
        return context_str
    except: return ""

async def aggregate_dual_search(query: str) -> str:
    loop = asyncio.get_event_loop()
    tavily_task = loop.run_in_executor(None, fetch_tavily_results, query)
    serpapi_task = loop.run_in_executor(None, fetch_serpapi_results, query)
    tavily_res, serpapi_res = await asyncio.gather(tavily_task, serpapi_task)
    return f"{tavily_res}\n{serpapi_res}"

# Expansive Tools Routing Schema Dictionary
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "aggregate_dual_search",
            "description": "Call this tool for live events, current presidents, real-time facts, conversion rates, or data requiring up-to-date info for the current year (2026).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The optimized search query resolving any shorthand pronoun references."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_user_preference",
            "description": "Call this tool to save specific things about the user permanently (e.g., their name, home city, favorite tech frameworks).",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "The preference element identifier, e.g., 'user_name', 'coding_language'."},
                    "value": {"type": "string", "description": "The specific information value profile variable to map."}
                },
                "required": ["key", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_crypto_metrics",
            "description": "Call this tool when the user requests current pricing or market metric positions for crypto or DeFi assets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "token_id": {"type": "string", "description": "The asset identifier string (e.g., 'bitcoin', 'ethereum', 'solana', 'hyperliquid')."}
                },
                "required": ["token_id"]
            }
        }
    }
]

async def process_text_or_vision(user_id: str, messages_list: list, image_bytes: bytes = None, document_path: str = None):
    try:
        if not GROQ_API_KEY:
            return "❌ Engine Configuration Error: GROQ_API_KEY is missing."

        if image_bytes:
            vision_model = "llama-3.2-11b-vision-preview"
            import base64
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            last_prompt = messages_list[-1]["content"] if messages_list else "Analyze this payload."
            response = groq_client.chat.completions.create(
                model=vision_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{last_prompt}\nProvide a clean, beautifully formatted mobile-ready layout."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }],
                max_tokens=1024
            )
            return response.choices[0].message.content

        else:
            user_profile = load_user_profile(user_id)
            profile_context_snippet = f"\n--- PERSISTENT USER PERSONALIZATION DATA ---\n{json.dumps(user_profile)}" if user_profile else ""
            doc_context_snippet = extract_document_context(document_path) if document_path else ""

            presentation_instructions = (
                f"\n\n[CRITICAL SYSTEM INSTRUCTIONS]: You are Phoenix AI, a smart chat companion. "
                f"The current year is 2026. Maintain multi-turn tracking thread consistency. "
                f"Format beautifully using bold titles and structured bullet points optimized for screens. "
                f"Never output debugging JSON strings or tool trace payloads directly."
                f"{profile_context_snippet}"
                f"{doc_context_snippet}"
            )

            execution_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages_list]
            execution_messages[-1]["content"] += presentation_instructions

            first_response = groq_client.chat.completions.create(
                model=MODEL_NAME,
                messages=execution_messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto"
            )
            
            response_message = first_response.choices[0].message
            tool_calls = response_message.tool_calls

            if tool_calls:
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments or tool_call.function.argv)
                    tool_content = ""
                    
                    if function_name == "aggregate_dual_search":
                        search_query = tool_args.get("query")
                        tool_content = await aggregate_dual_search(search_query)
                        
                    elif function_name == "update_user_preference":
                        key = tool_args.get("key")
                        val = tool_args.get("value")
                        user_profile[key] = val
                        save_user_profile(user_id, user_profile)
                        tool_content = f"[System Profile Status Update: Target variable state key '{key}' set to '{val}' successfully.]"
                        
                    elif function_name == "lookup_crypto_metrics":
                        token_id = tool_args.get("token_id")
                        tool_content = await fetch_crypto_asset_metrics(token_id)

                    execution_messages.append(response_message)
                    execution_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": tool_content
                    })
                    
                    final_response = groq_client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=execution_messages
                    )
                    return final_response.choices[0].message.content

            return response_message.content

    except Exception as e:
        print(f"Error handling functional orchestration logic execution pipeline: {e}")
        return f"⚠️ Phoenix AI Engine Error: Unable to resolve process context. ({e})"

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
        if os.path.exists(filename): os.remove(filename)
        return transcription
    except Exception as e:
        print(f"Transcription execution pipeline error: {e}")
        return "[Audio asset unreadable]"
