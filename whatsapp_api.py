import os
from fastapi import APIRouter, Request, Query, Response
from dotenv import load_dotenv

load_dotenv()
router = APIRouter()

WHATSAPP_TOKEN = os.environ.get("WHATSAPP_ACCESS_TOKEN")
WHATSAPP_PHONE_ID = os.environ.get("WHATSAPP_PHONE_NUMBER_ID")
WHATSAPP_VERIFY_TOKEN = os.environ.get("WHATSAPP_VERIFY_TOKEN", "phoenix_secret_verify_token")

@router.get("/webhook/whatsapp")
async def verification_handshake(
    mode: str = Query(None, alias="hub.mode"),
    token: str = Query(None, alias="hub.verify_token"),
    challenge: str = Query(None, alias="hub.challenge")
):
    if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN:
        return Response(content=challenge, media_type="text/plain")
    return Response(content="Mismatch", status_code=403)

@router.post("/webhook/whatsapp")
async def receive_whatsapp_event(request: Request):
    from main import execute_phoenix_pipeline
    try:
        payload = await request.json()
        if "object" in payload and payload.get("entry"):
            changes = payload["entry"][0].get("changes", [])
            if changes and "messages" in changes[0]["value"]:
                msg = changes[0]["value"]["messages"][0]
                if msg.get("type") == "text":
                    sender = msg["from"]
                    incoming_text = msg["text"]["body"]
                    
                    ai_reply = execute_phoenix_pipeline(
                        question=incoming_text,
                        platform="whatsapp",
                        platform_user_id=sender
                    )
                    
                    import requests
                    url = f"https://graph.facebook.com/v18.0/{WHATSAPP_PHONE_ID}/messages"
                    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
                    payload = {
                        "messaging_product": "whatsapp",
                        "recipient_type": "individual",
                        "to": sender,
                        "type": "text",
                        "text": {"preview_url": False, "body": ai_reply}
                    }
                    requests.post(url, json=payload, headers=headers, timeout=10)
                    
        return Response(content="EVENT_RECEIVED", status_code=200)
    except Exception:
        return Response(content="HANDLED", status_code=200)
