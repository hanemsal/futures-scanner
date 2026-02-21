import json
import os
import time

DEFAULT_PATH = "state.json"

def load_state(path: str = DEFAULT_PATH) -> dict:
    if not os.path.exists(path):
        return {"last_sent": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_state(state: dict, path: str = DEFAULT_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def can_send(symbol: str, cooldown_sec: int, state: dict) -> bool:
    last = state.get("last_sent", {}).get(symbol)
    if not last:
        return True
    return (time.time() - last) >= cooldown_sec

def mark_sent(symbol: str, state: dict) -> None:
    state.setdefault("last_sent", {})[symbol] = time.time()
