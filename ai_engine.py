import os
import json
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
    """Queries Tavily API for semantic web documentation chunks."""
    if not tavily_client:
        return ""
    try:
        response = tavily_client.search(query=query, max_results=3, search_depth="advanced")
        results = response.get("results", [])
        context_str = "--- TAVILY LIVE DATA ---\n"
        for res in results:
            context_str += f"Context: {res.get('content')}\n\n"
        return context_str
    except Exception as e:
        print(f"Tavily background error: {e}")
        return ""

def fetch_serpapi_results(query: str) -> str:
    """Queries SerpApi safely for live organic Google results."""
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
    """Runs Tavily and SerpApi concurrently to maximize execution speed."""
    print(f"📡 High-Speed Parallel Search triggering for query: '{query}'")
    loop = asyncio.get_event_loop()
    
    tavily_task = loop.run_in_executor(None, fetch_tavily_results, query)
    serpapi_task = loop.run_in_executor(None, fetch_serpapi_results, query)
    
    tavily_res, serpapi_res = await asyncio.gather(tavily_task, serpapi_task)
    return f"{tavily_res}\n{serpapi_res}"

# Define the unified tool structural schema for Groq's router
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "aggregate_dual_search",
            "description": "Call this tool when the user asks about live events, current presidents, real-time facts, conversion rates, or any information requiring up-to-date data for the current year (2026).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The optimized standalone search query string, resolving any pronoun contextual references (e.g., transforming 'how old is he' into 'Emmanuel Macron age')."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

async def process_text_or_vision(user_id: str, messages_list: list, image_bytes: bytes = None):
    """Executes high-speed native tool routing while strictly maintaining chat history threads."""
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
                            {"type": "text", "text": f"{last_prompt}\nProvide a beautifully formatted, clean analysis of this image."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                max_tokens=1024
            )
            return response.choices[0].message.content

        # B. TEXT PROCESSING WITH NATIVE SPEED ROUTING TOOL CALLS
        else:
            presentation_instructions = (
                "\n\n[CRITICAL SYSTEM INSTRUCTIONS]: You are Phoenix AI, an interactive, smart chat companion. "
                "The current year is 2026. Follow the multi-turn memory thread explicitly. "
                "Format beautifully using bold titles and clean spaced mobile-optimized bullet points. "
                "Never dump raw variables, code markers, or background JSON strings into the final chat output."
            )

            # Build contextual execution payload matching current instructions
            execution_messages = []
            for msg in messages_list:
                execution_messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Inject structural constraints onto the latest user interaction block
            execution_messages[-1]["content"] += presentation_instructions

            # First Pass: Ask Groq if it needs to execute a search tool function
            first_response = groq_client.chat.completions.create(
                model=MODEL_NAME,
                messages=execution_messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto"
            )
            
            response_message = first_response.choices[0].message
            tool_calls = response_message.tool_calls

            # If Llama decides it needs real-time search context data:
            if tool_calls:
                for tool_call in tool_calls:
                    if tool_call.function.name == "aggregate_dual_search":
                        # Safely parse the query generated natively by Llama's context understanding
                        tool_args = json.loads(tool_call.function.argv or tool_call.function.arguments)
                        search_query = tool_args.get("query")
                        
                        # Run the parallel dual lookup
                        live_web_context = await aggregate_dual_search(search_query)
                        
                        # Append tool invocation trace to the execution array
                        execution_messages.append(response_message)
                        execution_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": "aggregate_dual_search",
                            "content": live_web_context
                        })
                        
                        # Second Pass: Generate the final polished answer using the collected data
                        final_response = groq_client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=execution_messages
                        )
                        return final_response.choices[0].message.content
            
            # If no tools were required, return the initial message generation directly (Blazing Fast!)
            return response_message.content

    except Exception as e:
        print(f"Error processing native tool calling pipeline: {e}")
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
