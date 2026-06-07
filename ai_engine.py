import os
from groq import Groq

# Initialize Groq client using global environment credentials
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# ✅ UPDATED TO AN ACTIVE PRODUCTION MODEL TIER
MODEL_NAME = "llama-3.3-70b-versatile"

async def process_text_or_vision(user_id: str, prompt: str, image_bytes: bytes = None):
    """
    Processes multi-platform requests through the active Groq inference platform.
    If image_bytes is provided, handles multimodal vision analysis context.
    """
    try:
        if not GROQ_API_KEY:
            return "❌ Engine Configuration Error: GROQ_API_KEY environment variable is missing on Render."

        # A. MULTIMODAL VISION TRACK
        if image_bytes:
            # Note: Groq uses specific vision models like llama-3.2-11b-vision-preview for images
            vision_model = "llama-3.2-11b-vision-preview"
            import base64
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            response = client.chat.completions.create(
                model=vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt if prompt else "Analyze this image payload."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1024
            )
            return response.choices[0].message.content

        # B. TEXT PROCESSING TRACK
        else:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are Phoenix AI, a privacy-centric cross-platform system assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content

    except Exception as e:
        print(f"Error processing context engine pipeline: {e}")
        return f"⚠️ Phoenix AI Engine Error: Unable to complete message processing. ({e})"

async def generate_image_url(prompt: str) -> str:
    """Fallback utility for routing canvas design requests to open image endpoints."""
    import urllib.parse
    encoded_prompt = urllib.parse.quote(prompt)
    return f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true"

async def transcribe_voice_bytes(audio_bytes: bytes, filename: str) -> str:
    """Routes binary audio payloads to Groq's Whisper architecture for processing."""
    try:
        # Save bytes temporarily to pass to the client library file handler
        with open(filename, "wb") as f:
            f.write(audio_bytes)
            
        with open(filename, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
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
