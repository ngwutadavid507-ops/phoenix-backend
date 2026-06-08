# modules/personalization.py
import os
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Safely initialize client only if environmental variables are present
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if (SUPABASE_URL and SUPABASE_KEY) else None

def load_user_profile(user_id: str) -> dict:
    """Fetches user preference profiles instantly from the Supabase cloud layer."""
    if not supabase:
        return {}
    try:
        response = supabase.table("user_profiles").select("profile_data").eq("user_id", str(user_id)).execute()
        if response.data:
            return response.data[0].get("profile_data", {})
    except Exception as e:
        print(f"Database read synchronization anomaly: {e}")
    return {}

def save_user_profile(user_id: str, profile_data: dict):
    """Persists user data securely via atomic cloud upsert layers using native primary key matching."""
    if not supabase:
        return
    try:
        payload = {"user_id": str(user_id), "profile_data": profile_data}
        # SDK uses primary key constraint automatically without extra arguments
        supabase.table("user_profiles").upsert(payload).execute()
    except Exception as e:
        print(f"Database write execution breakdown: {e}")
