# modules/personalization.py
import os
import json

PROFILE_DIR = "user_profiles"
os.makedirs(PROFILE_DIR, exist_ok=True)

def get_profile_path(user_id: str) -> str:
    safe_id = "".join([c for c in user_id if c.isalpha() or c.isdigit() or c in ("-", "_")]).strip()
    return os.path.join(PROFILE_DIR, f"{safe_id}_profile.json")

def load_user_profile(user_id: str) -> dict:
    path = get_profile_path(user_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_user_profile(user_id: str, profile_data: dict):
    path = get_profile_path(user_id)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Failed to persist user schema allocation properties: {e}")
