import os
import httpx
from fastapi import FastAPI, Request, Response
from ai_engine import process_text_or_vision, generate_image_url, transcribe_voice_bytes

app = FastAPI()

WHATSAPP_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")

async def download_meta_media(media_id: str) -> bytes:
    """Retrieves raw media streaming binaries directly from Meta's asset nodes."""
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
    """Sends structured text logs and markdown responses back to the client."""
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
    """Pushes high-fidelity visual attachments or renders directly to the chat thread."""
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
        
        # 1. PARSE TEXT STRING & GENERATION PROMPTS
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
                reply = await process_text_or_vision(sender, prompt=user_text)
                await send_whatsapp_message(sender, reply)

        # 2. PARSE CODE SCREENSHOTS AND ARCHITECTURES
        elif msg_type == "image":
            img_id = message["image"]["id"]
            caption = message["image"].get("caption", "")
            await send_whatsapp_message(sender, "🔍 Reviewing image payload assets...")
            img_bytes = await download_meta_media(img_id)
            reply = await process_text_or_vision(sender, prompt=caption, image_bytes=img_bytes)
            await send_whatsapp_message(sender, reply)

        # 3. PARSE MULTILINGUAL VOICE NOTES
        elif msg_type == "voice":
            voice_id = message["voice"]["id"]
            await send_whatsapp_message(sender, "🎙️ Processing audio transcription streaming...")
            audio_bytes = await download_meta_media(voice_id)
            transcription = await transcribe_voice_bytes(audio_bytes, "voice.ogg")
            reply = await process_text_or_vision(sender, prompt=f"[Voice Transcribed: {transcription}]")
            await send_whatsapp_message(sender, reply)

    except Exception as e:
        print(f"Error processing context engine pipeline: {e}")
        
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
