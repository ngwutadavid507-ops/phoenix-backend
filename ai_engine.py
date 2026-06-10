import os
import sys
import asyncio
import httpx
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from groq import Groq
from telegram import Bot
from tavily import TavilyClient

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load and validate core environment tokens
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")

if not GROQ_API_KEY:
    logger.error("CRITICAL: GROQ_API_KEY environmental variable is missing!")

# Initialize third-party client drivers securely
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None
tg_bot = Bot(token=TELEGRAM_BOT_TOKEN) if TELEGRAM_BOT_TOKEN else None

class MobileAppResponse(BaseModel):
    ui_intent: str = Field(description="The intent of the message (e.g., 'chat', 'search_result', 'image_generation', 'error')")
    display_title: Optional[str] = Field(None, description="A clean, bold title for the UI card header")
    main_text: str = Field(description="The core conversational answer, supporting clean Markdown formatting")
    suggested_quick_replies: List[str] = Field(default=[], description="Contextual follow-up buttons for the app UI")
    citations: List[str] = Field(default=[], description="Clean URLs discovered during live web search analysis")

def get_structured_system_prompt() -> str:
    return """You are the core intelligence engine of Phoenix AI.
When handling native mobile app traffic requests, you MUST respond strictly with a valid JSON object matching this schema:
{
    "ui_intent": "chat" | "search_result" | "image_generation",
    "display_title": "Optional Title Header",
    "main_text": "Your markdown-supported response string here",
    "suggested_quick_replies": ["Follow up question 1", "Follow up question 2"],
    "citations": ["https://sourceurl.com"]
}
Do not include markdown code block formatting backticks (```json) in your final response payload string."""

async def send_whatsapp_api(payload: dict) -> dict:
    if not WHATSAPP_TOKEN or not WHATSAPP_PHONE_NUMBER_ID:
        logger.warning("WhatsApp credentials missing. Skipping API dispatch.")
        return {}
    url = f"[https://graph.facebook.com/v17.0/](https://graph.facebook.com/v17.0/){WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=10.0)
            return response.json()
        except Exception as e:
            logger.error(f"Failed to post payload to WhatsApp cloud endpoints: {e}")
            return {}

async def execute_web_search(query: str) -> str:
    if not tavily_client:
        return "[Search engine inactive: Missing API Key]"
    try:
        loop = asyncio.get_running_loop()
        # Offload synchronous Tavily SDK request execution to background worker thread context
        response = await loop.run_in_executor(
            None, 
            lambda: tavily_client.search(query=query, search_depth="advanced", max_results=4)
        )
        results = response.get("results", [])
        if not results:
            return "No relevant real-time search context was uncovered for this query."
        
        compiled_context = []
        for item in results:
            compiled_context.append(f"Source: {item.get('url')}\nContent: {item.get('content')}\n")
        return "\n---\n".join(compiled_context)
    except Exception as e:
        logger.error(f"Search engine task execution failure: {e}")
        return f"[Search error encountered: {e}]"

async def run_llama_inference(system_prompt: str, user_prompt: str, model_name: str = "llama-3.3-70b-versatile") -> str:
    if not groq_client:
        return "AI Core engine offline. Complete configuration profile to converse."
    try:
        loop = asyncio.get_running_loop()
        completion = await loop.run_in_executor(
            None,
            lambda: groq_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=2048
            )
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Groq reasoning model failure: {e}")
        return f"An operational error occurred while generating text answers: {e}"

async def process_telegram_with_status(chat_id: int, user_query: str):
    if not tg_bot:
        return
    try:
        await tg_bot.send_chat_action(chat_id=chat_id, action="typing")
        status_msg = await tg_bot.send_message(
            chat_id=chat_id, 
            text="🧠 *Phoenix AI is thinking...*", 
            parse_mode="Markdown"
        )
        
        await tg_bot.edit_message_text(
            chat_id=chat_id, message_id=status_msg.message_id,
            text="🔍 *Searching the web for live context...*", parse_mode="Markdown"
        )
        
        # Concurrent evaluation of context search and reasoning initialization
        search_context = await execute_web_search(user_query)
        
        await tg_bot.edit_message_text(
            chat_id=chat_id, message_id=status_msg.message_id,
            text="⚙️ *Analyzing sources and processing reasoning...*", parse_mode="Markdown"
        )
        
        system_rules = "You are Phoenix AI, an elite assistant. Combine the following web search context to fulfill requests cleanly:\n" + search_context
        final_answer = await run_llama_inference(system_rules, user_query)
        
        await tg_bot.edit_message_text(
            chat_id=chat_id, message_id=status_msg.message_id,
            text=final_answer, parse_mode="Markdown"
        )
    except Exception as e:
        logger.error(f"Telegram status tracking workflow broken: {e}")

async def process_whatsapp_with_status(user_phone: str, user_query: str):
    typing_payload = {
        "messaging_product": "whatsapp", "recipient_type": "individual",
        "to": user_phone, "type": "typing_indicator", "typing_indicator": {"type": "text"}
    }
    await send_whatsapp_api(typing_payload)
    
    ack_payload = {
        "messaging_product": "whatsapp", "to": user_phone, "type": "text",
        "text": {"body": "🔍 Phoenix AI is researching and analyzing your request now. Hold on a moment..."}
    }
    await send_whatsapp_api(ack_payload)
    
    try:
        search_context = await execute_web_search(user_query)
        system_rules = "You are Phoenix AI. Synthesize web findings to build answers with references:\n" + search_context
        final_answer = await run_llama_inference(system_rules, user_query)
        
        final_payload = {
            "messaging_product": "whatsapp", "to": user_phone, "type": "text",
            "text": {"body": final_answer}
        }
        await send_whatsapp_api(final_payload)
    except Exception as e:
        logger.error(f"WhatsApp execution thread exception: {e}")

async def transcribe_voice_bytes(filename: str, clean_filename: str) -> str:
    if not groq_client:
        return "[Transcription Engine Offline]"
    try:
        loop = asyncio.get_running_loop()
        
        def run_whisper():
            # Insulated binary file descriptor targeting Whisper engine directly
            with open(clean_filename, "rb") as audio_file:
                return groq_client.audio.transcriptions.create(
                    file=(os.path.basename(clean_filename), audio_file.read(), "audio/ogg"),
                    model="whisper-large-v3",
                    language="en"
                ).text

        transcription = await loop.run_in_executor(None, run_whisper)
        return transcription if transcription else "[Empty Audio]"
    except Exception as e:
        logger.error(f"Transcription execution pipeline error: {e}")
        return f"[Audio Parsing Error: {e}]"
    finally:
        # Combined single finally block safely executing cleanup tasks sequentially
        try:
            if os.path.exists(clean_filename):
                os.remove(clean_filename)
        except Exception as cleanup_err:
            logger.error(f"Failed deleting temporary file clean_filename: {cleanup_err}")
            
        try:
            if os.path.exists(filename):
                os.remove(filename)
        except Exception as cleanup_err:
            logger.error(f"Failed deleting temporary raw filename payload: {cleanup_err}")
