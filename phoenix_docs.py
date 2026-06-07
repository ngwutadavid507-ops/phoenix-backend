import os
import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

PHOENIX_API_URL = os.environ.get("PHOENIX_API_URL", "http://localhost:8000/chat")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Phoenix Docs Core Active. Connected directly to local system runtime lifecycle.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_message = update.message.text
    
    placeholder = await update.message.reply_text("Phoenix is thinking...")

    payload = {
        "question": user_message,
        "platform": "telegram",
        "user_id": str(user_id)
    }

    try:
        response = requests.post(PHOENIX_API_URL, json=payload, timeout=30)
        ai_reply = response.json().get("answer", "Empty core network token.") if response.status_code == 200 else f"Core Gateway Error ({response.status_code})"
    except Exception as e:
        ai_reply = f"Pipeline Connection Fault: {str(e)}"

    await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=placeholder.message_id, text=ai_reply)

def initialize_telegram_application(token: str) -> Application:
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    return app
