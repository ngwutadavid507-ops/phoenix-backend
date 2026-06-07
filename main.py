import os
import httpx
from fastapi import FastAPI, Request, Response
from ai_engine import process_text_or_vision, generate_image_url, transcribe_voice_bytes

app = FastAPI()

# Credentials Configuration
WHATSAPP_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# ==========================================
# 🧠 IN-MEMORY CONVERSATION STORAGE
# ==========================================
conversation_history = {}

def get_user_context(session_id: str) -> list:
    """Retrieves or builds a clean history thread with system framing instructions."""
    if session_id not in conversation_history:
        conversation_history[session_id] = [
            {
                "role": "system", 
                "content": "You are Phoenix AI, an ultra-responsive, privacy-centric assistant. The current year is 2026. You have conversational memory—always refer to previous messages in the context if the user asks follow-up questions."
            }
        ]
    return conversation_history[session_id]

def trim_context(session_id: str, max_turns: int = 12):
    """Prevents memory arrays from bloating the API payload context window."""
    history = conversation_history[session_id]
    if len(history) > max_turns:
        # Keep the system prompt, drop the oldest user/assistant pair
        system_prompt = history[0]
        conversation_history[session_id] = [system_prompt] + history[-(max_turns-1):]

# ==========================================
# 🎛️ SHARED PLATFORM UTILITIES
# ==========================================
async def send_telegram_message(chat_id: int, text: str):
    url = f"{TELEGRAM_API_URL}/sendMessage"
    async with httpx.AsyncClient() as client:
        await client.post(url, json={"chat_id": chat_id, "text": text})

async def send_telegram_photo(chat_id: int, photo_url: str):
    url = f"{TELEGRAM_API_URL}/sendPhoto"
    async with httpx.AsyncClient() as client:
        await client.post(url, json={"chat_id": chat_id, "photo": photo_url})

async def download_telegram_media(file_id: str) -> bytes:
    async with httpx.AsyncClient() as client:
        path_url = f"{TELEGRAM_API_URL}/getFile?file_id={file_id}"
        res = await client.get(path_url)
        if res.status_code == 200:
            file_path = res.json().get("result", {}).get("file_path")
            download_url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}"
            file_res = await client.get(download_url)
            return file_res.content
    return b""

async def download_meta_media(media_id: str) -> bytes:
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    async with httpx.AsyncClient() as client:
        url = f"https://graph.facebook.com/v25.0/{media_id}"
        res = await client.get(url, headers=headers)
        if res.status_code == 200:
            file_url = res.json().get("url")
            file_res = await client.get(file_url, headers=headers)
            return file_res.content
    return b""

async def send_whatsapp_message(to: str, text: str):
    url = f"https://graph.facebook.com/v25.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text}
    }
    async with httpx.AsyncClient() as client:
        await client.post(url, json=payload, headers=headers)

async def send_whatsapp_media(to: str, media_url: str, media_type: str = "image"):
    url = f"https://graph.facebook.com/v25.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": media_type,
        media_type: {"link": media_url}
    }
    async with httpx.AsyncClient() as client:
        await client.post(url, json=payload, headers=headers)


# ==========================================
# 🚀 1. TELEGRAM WEBHOOK ROUTE
# ==========================================
@app.post("/webhook/telegram")
async def telegram_webhook(request: Request):
    payload = await request.json()
    try:
        if "message" not in payload:
            return Response(content="OK", status_code=200)
            
        message = payload["message"]
        chat_id = message["chat"]["id"]
        session_key = f"tg_{chat_id}"
        
        # Pull conversational memory list
        history = get_user_context(session_key)
        
        # A. PARSE TELEGRAM TEXT
        if "text" in message:
            user_text = message["text"]
            
            if user_text.startswith("/start"):
                conversation_history[session_key] = [history[0]] # Reset conversation context
                await send_telegram_message(chat_id, "Phoenix AI memory refreshed. How can I assist you securely today?")
                return Response(content="OK", status_code=200)

            if user_text.lower().startswith("draw:") or user_text.lower().startswith("generate image:"):
                prompt = user_text.split(":", 1)[1].strip()
                await send_telegram_message(chat_id, "🎨 Sketching your canvas concept, one moment...")
                img_url = await generate_image_url(prompt)
                if img_url:
                    await send_telegram_photo(chat_id, img_url)
                else:
                    await send_telegram_message(chat_id, "❌ Failed to generate image canvas.")
            else:
                # Store user message input inside history
                history.append({"role": "user", "content": user_text})
                
                reply = await process_text_or_vision(session_key, history)
                
                # Store assistant output inside history
                history.append({"role": "assistant", "content": reply})
                trim_context(session_key)
                
                await send_telegram_message(chat_id, reply)
                
        # B. PARSE TELEGRAM PHOTOS / SCREENSHOTS
        elif "photo" in message:
            photo_asset = message["photo"][-1]
            file_id = photo_asset["file_id"]
            caption = message.get("caption", "Analyze this image payload.")
            
            await send_telegram_message(chat_id, "🔍 Reviewing screenshot assets...")
            img_bytes = await download_telegram_media(file_id)
            
            history.append({"role": "user", "content": caption})
            reply = await process_text_or_vision(session_key, history, img_bytes)
            history.append({"role": "assistant", "content": reply})
            trim_context(session_key)
            
            await send_telegram_message(chat_id, reply)

        # C. PARSE TELEGRAM VOICE NOTES
        elif "voice" in message:
            voice_asset = message["voice"]
            file_id = voice_asset["file_id"]
            
            await send_telegram_message(chat_id, "🎙️ Processing voice note transcription...")
            audio_bytes = await download_telegram_media(file_id)
            transcription = await transcribe_voice_bytes(audio_bytes, "voice.oga")
            
            transcript_msg = f"[Voice Note Transcribed: {transcription}]"
            history.append({"role": "user", "content": transcript_msg})
            reply = await process_text_or_vision(session_key, history)
            history.append({"role": "assistant", "content": reply})
            trim_context(session_key)
            
            await send_telegram_message(chat_id, reply)

    except Exception as e:
        print(f"Telegram processing error: {e}")
        
    return Response(content="OK", status_code=200)


# ==========================================
# 🍏 2. WHATSAPP WEBHOOK ROUTE
# ==========================================
@app.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request):
    payload = await request.json()
    try:
        value = payload["entry"][0]["changes"][0]["value"]
        if "messages" not in value:
            return Response(content="OK", status_code=200)

        message = value["messages"][0]
        sender = message["from"]
        msg_type = message["type"]
        session_key = f"wa_{sender}"
        
        # Pull conversational memory list
        history = get_user_context(session_key)
        
        if msg_type == "text":
            user_text = message["text"]["body"]
            if user_text.lower().startswith("draw:") or user_text.lower().startswith("generate image:"):
                prompt = user_text.split(":", 1)[1].strip()
                await send_whatsapp_message(sender, "🎨 Sketching your canvas concept, one moment...")
                img_url = await generate_image_url(prompt)
                if img_url:
                    await send_whatsapp_media(sender, img_url, "image")
                else:
                    await send_whatsapp_message(sender, "❌ Failed to generate image canvas asset.")
            else:
                history.append({"role": "user", "content": user_text})
                reply = await process_text_or_vision(session_key, history)
                history.append({"role": "assistant", "content": reply})
                trim_context(session_key)
                
                await send_whatsapp_message(sender, reply)

        elif msg_type == "image":
            img_id = message["image"]["id"]
            caption = message["image"].get("caption", "Analyze this image payload.")
            
            await send_whatsapp_message(sender, "🔍 Reviewing image payload assets...")
            img_bytes = await download_meta_media(img_id)
            
            history.append({"role": "user", "content": caption})
            reply = await process_text_or_vision(session_key, history, img_bytes)
            history.append({"role": "assistant", "content": reply})
            trim_context(session_key)
            
            await send_whatsapp_message(sender, reply)

        elif msg_type == "voice":
            voice_id = message["voice"]["id"]
            await send_whatsapp_message(sender, "🎙️ Processing audio transcription streaming...")
            audio_bytes = await download_meta_media(voice_id)
            transcription = await transcribe_voice_bytes(audio_bytes, "voice.ogg")
            
            transcript_msg = f"[Voice Note Transcribed: {transcription}]"
            history.append({"role": "user", "content": transcript_msg})
            reply = await process_text_or_vision(session_key, history)
            history.append({"role": "assistant", "content": reply})
            trim_context(session_key)
            
            await send_whatsapp_message(sender, reply)

    except Exception as e:
        print(f"WhatsApp processing error: {e}")
        
    return Response(content="OK", status_code=200)

@app.get("/webhook/whatsapp")
async def whatsapp_verification(request: Request):
    params = request.query_params
    verify_token = os.getenv("WHATSAPP_VERIFY_TOKEN", "phoenix_secret_verify_token")
    if params.get("hub.mode") == "subscribe" and params.get("hub.verify_token") == verify_token:
        return Response(content=params.get("hub.challenge"), status_code=200)
    return Response(content="Verification mismatch", status_code=403)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
