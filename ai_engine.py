import os
from groq import Groq

# Initialize Groq client using global environment credentials
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Active, supported production model tier
MODEL_NAME = "llama-3.3-70b-versatile"

async def process_text_or_vision(user_id: str, messages_list: list, image_bytes: bytes = None):
    """
    Processes requests through Groq.
    - If image_bytes is provided, processes it using a vision-capable model.
    - Otherwise, processes the entire conversational history thread.
    """
    try:
        if not GROQ_API_KEY:
            return "❌ Engine Configuration Error: GROQ_API_KEY environment variable is missing on Render."

        # A. MULTIMODAL VISION TRACK
        if image_bytes:
            vision_model = "llama-3.2-11b-vision-preview"
            import base64
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            # For vision, we pull the last text content or fallback
            last_prompt = messages_list[-1]["content"] if messages_list else "Analyze this image payload."
            
            response = client.chat.completions.create(
                model=vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": last_prompt},
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

        # B. TEXT PROCESSING TRACK (With full conversation history context)
        else:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages_list
            )
            return response.choices[0].message.content

    except Exception as e:
        print(f"Error processing context engine pipeline: {e}")
        return f"⚠️ Phoenix AI Engine Error: Unable to complete message processing. ({e})"

async def generate_image_url(prompt: str) -> str:
    import urllib.parse
    encoded_prompt = urllib.parse.quote(prompt)
    return f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true"

async def transcribe_voice_bytes(audio_bytes: bytes, filename: str) -> str:
    try:
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
