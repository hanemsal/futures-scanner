import os
import time
import requests
from typing import Optional, Dict, Any, List

HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "10"))

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

API_BASE = f"https://api.telegram.org/bot{BOT_TOKEN}"

def _tg(method: str, params: Dict[str, Any]) -> Any:
    url = f"{API_BASE}/{method}"
    r = requests.post(url, data=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()

def send_message(text: str) -> None:
    if not BOT_TOKEN or not CHAT_ID:
        return
    _tg("sendMessage", {
        "chat_id": CHAT_ID,
        "text": text,
        "disable_web_page_preview": True
    })

def get_updates(offset: Optional[int], timeout_sec: int = 30) -> Any:
    params = {"timeout": timeout_sec}
    if offset is not None:
        params["offset"] = offset
    return _tg("getUpdates", params)

def is_allowed_chat(update: Dict[str, Any]) -> bool:
    # Sadece tek chat id’den komut dinle
    try:
        msg = update.get("message") or update.get("edited_message")
        if not msg:
            return False
        cid = str(msg["chat"]["id"])
        return cid == str(CHAT_ID)
    except Exception:
        return False

def extract_text(update: Dict[str, Any]) -> Optional[str]:
    msg = update.get("message") or update.get("edited_message")
    if not msg:
        return None
    return msg.get("text")
