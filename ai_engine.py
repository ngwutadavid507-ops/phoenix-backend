# ai_engine.py
import os
import json
import re
import asyncio
import urllib.parse
import httpx
from groq import Groq

# Pull feature modules dynamically from local pathways
from modules.rag_processor import extract_document_context
from modules.personalization import load_user_profile, save_user_profile
from modules.crypto_ticker import fetch_crypto_asset_metrics

# Infrastructure Key Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
MODEL_NAME = "llama-3.3-70b-versatile"

# Fully asynchronous Tavily lookup engine
async def fetch_tavily_results_async(query: str) -> str:
    if not TAVILY_API_KEY: return ""
    url = "https://api.tavily.com/search"
    payload = {"api_key": TAVILY_API_KEY, "query": query, "search_depth": "advanced", "max_results": 3}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=6.0)
            if response.status_code == 200:
                results = response.json().get("results", [])
                context_str = "--- TAVILY LIVE DATA ---\n"
                for res in results:
                    context_str += f"Context: {res.get('content')}\n\n"
                return context_str
    except:
        pass
    return ""

# Fully asynchronous SerpApi Google lookup engine
async def fetch_serpapi_results_async(query: str) -> str:
    if not SERPAPI_API_KEY: return ""
    encoded_query = urllib.parse.quote(query)
    url = f"https://serpapi.com/search.json?q={encoded_query}&api_key={SERPAPI_API_KEY}&num=3"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=6.0)
            if response.status_code == 200:
                data = response.json()
                organic_results = data.get("organic_results", [])
                context_str = "--- SERPAPI GOOGLE SEARCH DATA ---\n"
                if "answer_box" in data:
                    box = data["answer_box"]
                    context_str += f"Direct Answer: {box.get('answer') or box.get('snippet')}\n\n"
                for res in organic_results:
                    context_str += f"Snippet: {res.get('snippet')}\n\n"
                return context_str
    except:
        pass
    return ""

# Aggregates both pipelines running completely in parallel
async def aggregate_dual_search(query: str) -> str:
    tavily_task = fetch_tavily_results_async(query)
    serpapi_task = fetch_serpapi_results_async(query)
    tavily_res, serpapi_res = await asyncio.gather(tavily_task, serpapi_task)
    return f"{tavily_res}\n{serpapi_res}"

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "aggregate_dual_search",
            "description": "Call this tool for live events, general news updates, real-time facts, fiat conversion rates, or data requiring up-to-date info.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The optimized search query resolving any shorthand references."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_user_preference",
            "description": "Call this tool to save specific metadata attributes about the user permanently.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "The preference key identifier."},
                    "value": {"type": "string", "description": "The information string value to save."}
                },
                "required": ["key", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_crypto_metrics",
            "description": "Call this tool when the user requests current pricing or live market metrics for crypto assets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "token_id": {"type": "string", "description": "The token symbol string (e.g., 'BTC', 'sol')."}
                },
                "required": ["token_id"]
            }
        }
    }
]

def parse_fallback_text_tool_calls(text: str):
    found_calls = []
    pattern = r"<function=(\w+)\s*(\{.*?\})\s*>"
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        func_name = match[0]
        try:
            func_args = json.loads(match[1])
            found_calls.append({"name": func_name, "args": func_args})
        except: continue
    return found_calls

async def process_text_or_vision(user_id: str, messages_list: list, image_bytes: bytes = None, document_path: str = None):
    try:
        if not GROQ_API_KEY:
            return "❌ Engine Configuration Error: GROQ_API_KEY is missing."

        try:
            user_profile = load_user_profile(user_id)
            if not isinstance(user_profile, dict): user_profile = {}
        except Exception: user_profile = {}

        if image_bytes:
            vision_model = "llama-3.2-11b-vision-instant"
            import base64
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            # Extract plain text safely for vision prompts to prevent array conversion bugs
            last_prompt = "Analyze this image payload."
            if messages_list and isinstance(messages_list[-1].get("content"), str):
                last_prompt = messages_list[-1]["content"]

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: groq_client.chat.completions.create(
                    model=vision_model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{last_prompt}\nProvide a clean, direct layout response."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }],
                    max_tokens=1024
                )
            )
            return response.choices[0].message.content

        else:
            # Safe extraction logic for text messaging streams
            execution_messages = []
            for msg in messages_list:
                if isinstance(msg.get("content"), str):
                    execution_messages.append({"role": msg["role"], "content": msg["content"]})
                else:
                    execution_messages.append({"role": msg["role"], "content": str(msg["content"])})

            last_user_message = execution_messages[-1]["content"].strip().lower() if execution_messages else ""
            is_casual_greeting = last_user_message in ["hi", "hello", "hey", "yo", "sup", "good morning", "good afternoon", "good evening"]

            profile_context_snippet = f"\n--- PERSISTENT USER PERSONALIZATION DATA ---\n{json.dumps(user_profile)}" if user_profile else ""
            doc_context_snippet = extract_document_context(document_path) if document_path else ""

            if is_casual_greeting:
                presentation_instructions = (
                    f"\n\n[SYSTEM NOTE]: The user is just saying hello. Respond with a very short, friendly, single-sentence greeting welcoming them back to Phoenix AI."
                )
            else:
                presentation_instructions = (
                    f"\n\n[CRITICAL SYSTEM INSTRUCTIONS]: You are Phoenix AI, a smart mobile chat companion. "
                    f"The current year is 2026. Maintain conversational consistency. "
                    f"Format answers beautifully using clean layouts. "
                    f"NEVER output raw pseudo-XML tags like '<function=...>' directly into user chat views."
                    f"{profile_context_snippet}"
                    f"{doc_context_snippet}"
                )

            execution_messages[-1]["content"] += presentation_instructions

            loop = asyncio.get_event_loop()
            first_response = await loop.run_in_executor(
                None,
                lambda: groq_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=execution_messages,
                    tools=TOOLS_SCHEMA,
                    tool_choice="auto",
                    extra_body={"disable_tool_validation": True}
                )
            )
            
            response_message = first_response.choices[0].message
            tool_calls = response_message.tool_calls
            raw_text = response_message.content or ""

            parsed_fallback_calls = parse_fallback_text_tool_calls(raw_text) if not tool_calls else []

            if tool_calls or parsed_fallback_calls:
                execution_messages.append(response_message)
                
                if tool_calls:
                    for tool_call in tool_calls:
                        function_name = tool_call.function.name
                        try: tool_args = json.loads(tool_call.function.arguments)
                        except: continue
                        
                        tool_content = ""
                        if function_name == "aggregate_dual_search":
                            tool_content = await aggregate_dual_search(tool_args.get("query", ""))
                        elif function_name == "update_user_preference":
                            key, val = tool_args.get("key"), tool_args.get("value")
                            user_profile[key] = val
                            save_user_profile(user_id, user_profile)
                            tool_content = f"[Preference '{key}' saved.]"
                        elif function_name == "lookup_crypto_metrics":
                            tool_content = await fetch_crypto_asset_metrics(tool_args.get("token_id", ""))

                        execution_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": tool_content
                        })

                else:
                    for idx, fallback_call in enumerate(parsed_fallback_calls):
                        function_name = fallback_call["name"]
                        tool_args = fallback_call["args"]
                        
                        tool_content = ""
                        if function_name == "aggregate_dual_search":
                            tool_content = await aggregate_dual_search(tool_args.get("query", ""))
                        elif function_name == "update_user_preference":
                            key, val = tool_args.get("key"), tool_args.get("value")
                            user_profile[key] = val
                            save_user_profile(user_id, user_profile)
                            tool_content = f"[Preference '{key}' saved.]"
                        elif function_name == "lookup_crypto_metrics":
                            tool_content = await fetch_crypto_asset_metrics(tool_args.get("token_id", ""))

                        execution_messages.append({
                            "role": "tool",
                            "tool_call_id": f"fallback_id_{idx}",
                            "name": function_name,
                            "content": tool_content
                        })
                
                final_response = await loop.run_in_executor(
                    None,
                    lambda: groq_client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=execution_messages
                    )
                )
                return final_response.choices[0].message.content

            return raw_text

    except Exception as e:
        print(f"Error handling engine runtime workflow execution loop: {e}")
        return f"⚠️ Phoenix AI Engine Error: Unable to resolve process context. ({e})"

async def generate_image_url(prompt: str) -> str:
    import urllib.parse
    encoded_prompt = urllib.parse.quote(prompt)
    return f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true"

async def transcribe_voice_bytes(audio_bytes: bytes, filename: str) -> str:
    try:
        # Run local file writing in the executor thread pool to stop server freezes
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: open(filename, "wb").write(audio_bytes))
        
        def run_whisper():
            with open(filename, "rb") as audio_file:
                return groq_client.audio.transcriptions.create(
                    file=(filename, audio_file.read()),
                    model="whisper-large-v3",
                    response_format="text"
                )
        
        transcription = await loop.run_in_executor(None, run_whisper)
        if os.path.exists(filename): 
            await loop.run_in_executor(None, os.remove, filename)
        return transcription
    except Exception as e:
        print(f"Transcription execution pipeline error: {e}")
        return "[Audio asset unreadable]"
