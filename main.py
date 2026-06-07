from fastapi import FastAPI, Request
from ai_engine import process_text_or_vision, generate_image_url, transcribe_voice_bytes
import httpx
import os

app = FastAPI()

# Read configuration from environment variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

@app.get("/")
async def root():
    return {"status": "Phoenix AI Backend is up and running"}

# ✅ THIS ENFORCES THE EXACT ROUTE TELEGRAM IS CALLING IN YOUR LOGS
@app.post("/webhook/telegram")
async def telegram_webhook(request: Request):
    try:
        payload = await request.json()
        
        # Guard clause for message events
        if "message" not in payload:
            return {"status": "ignored"}
            
        message = payload["message"]
        chat_id = message["chat"]["id"]
        text = message.get("text", "")
        
        # Handle '/start' command or empty setups
        if text.startswith("/start"):
            reply = "Phoenix AI is live and integrated. How can I assist you securely today?"
        else:
            # Process text using our updated Groq configuration
            reply = await process_text_or_vision(user_id=str(chat_id), prompt=text)
            
        # Send back to Telegram API
        telegram_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        async with httpx.AsyncClient() as client:
            await client.post(telegram_url, json={
                "chat_id": chat_id,
                "text": reply
            })
            
        return {"status": "success"}
        
    except Exception as e:
        print(f"Error handling Telegram webhook: {e}")
        return {"status": "error", "detail": str(e)}

# Keep your WhatsApp webhook intact if applicable
@app.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request):
    # Your WhatsApp handling logic goes here
    return {"status": "success"}
