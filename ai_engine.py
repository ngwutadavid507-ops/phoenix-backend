import os
import base64
from openai import AsyncOpenAI

# Shared asynchronous client initialization
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def process_text_or_vision(user_id: str, prompt: str, image_bytes: bytes = None) -> str:
    """Processes textual tech queries or reads incoming system screenshots using Vision."""
    system_instruction = (
        "You are Phoenix AI, a strict assistant for tech architecture and project execution. "
        "Keep your output clean, actionable, and formatted beautifully for mobile messaging screens."
    )
    
    messages = [{"role": "system", "content": system_instruction}]
    
    if image_bytes:
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt or "Analyze this bug or screenshot asset step-by-step."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        })
    else:
        messages.append({"role": "user", "content": prompt})

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return response.choices[0].message.content

async def generate_image_url(prompt: str) -> str:
    """Generates an image via DALL-E 3 and returns the cloud hosting URL."""
    try:
        response = await client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        return response.data[0].url
    except Exception as e:
        print(f"DALL-E Asset Generation Error: {e}")
        return None

async def transcribe_voice_bytes(audio_bytes: bytes, file_name: str = "voice.ogg") -> str:
    """Sends raw speech binaries directly to Whisper for multilingual translation/transcription."""
    try:
        response = await client.audio.transcriptions.create(
            model="whisper-1",
            file=(file_name, audio_bytes),
            response_format="text"
        )
        return response
    except Exception as e:
        print(f"Whisper Processing Error: {e}")
        return "System error: Unable to parse audio payload."
