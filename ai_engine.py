import os
import httpx
from openai import AsyncOpenAI

# Point to Groq's open-source gateway infrastructure
client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

async def process_text_or_vision(user_id: str, prompt: str, image_bytes: bytes = None) -> str:
    """Processes textual tech queries using a current, live production model on Groq."""
    system_instruction = (
        "You are Phoenix AI, a strict assistant for tech architecture and project execution. "
        "Keep your output clean, actionable, and formatted beautifully for mobile messaging screens."
    )
    
    if image_bytes and not prompt:
        prompt = "[User sent a screenshot/image asset for review]"
    elif image_bytes and prompt:
        prompt = f"{prompt} [Attached Image Asset]"

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": prompt}
    ]

    # Swapped decommissioned string for the active live flagship model ID
    response = await client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    return response.choices[0].message.content

async def generate_image_url(prompt: str) -> str:
    """Generates a free image via Pollinations AI without needing an API key."""
    try:
        encoded_prompt = httpx.utils.quote(prompt)
        target_url = f"https://image.pollinations.ai/p/{encoded_prompt}?width=1024&height=1024&seed=42"
        return target_url
    except Exception as e:
        print(f"Free Image Generation Error: {e}")
        return None

async def transcribe_voice_bytes(audio_bytes: bytes, file_name: str = "voice.ogg") -> str:
    """Sends raw speech binaries to Groq's Whisper architecture for free transcription."""
    try:
        response = await client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=(file_name, audio_bytes),
            response_format="text"
        )
        return response
    except Exception as e:
        print(f"Compatible Whisper Processing Error: {e}")
        return "System error: Unable to parse audio payload."
