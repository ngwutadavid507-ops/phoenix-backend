import os
import httpx
from openai import AsyncOpenAI

# We use the standard OpenAI client but route it directly to Groq's free endpoint
client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

async def process_text_or_vision(user_id: str, prompt: str, image_bytes: bytes = None) -> str:
    """Processes textual tech queries using a free Llama 3 model via Groq endpoint."""
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

    # Calling Llama 3 on Groq using the compatible OpenAI client format
    response = await client.chat.completions.create(
        model="llama3-70b-8192",
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
