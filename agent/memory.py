import json
import os
from typing import Dict, Any

MEMORY_DIR = os.path.join(os.path.dirname(__file__), "..", ".agent_memory")
os.makedirs(MEMORY_DIR, exist_ok=True)

def save_session(session_id: str, data: Dict[str, Any]):
    filepath = os.path.join(MEMORY_DIR, f"{session_id}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_session(session_id: str) -> Dict[str, Any]:
    filepath = os.path.join(MEMORY_DIR, f"{session_id}.json")
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def list_sessions() -> list[str]:
    return [f.replace(".json", "") for f in os.listdir(MEMORY_DIR) if f.endswith(".json")]
