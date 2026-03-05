import os
import requests
from typing import Optional, Dict, Any

HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "60"))

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# 1 ise sadece CHAT_ID'den gelen mesajları işler.
STRICT_CHAT = os.getenv("STRICT_CHAT", "1") == "1"

API_BASE = f"https://api.telegram.org/bot{BOT_TOKEN}"


def _tg(method: str, params: Dict[str, Any], req_timeout: Optional[float] = None) -> Any:
    url = f"{API_BASE}/{method}"
    timeout = req_timeout if req_timeout is not None else HTTP_TIMEOUT
    r = requests.post(url, data=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def send_message(text: str) -> None:
    if not BOT_TOKEN or not CHAT_ID:
        return
    _tg(
        "sendMessage",
        {"chat_id": CHAT_ID, "text": text, "disable_web_page_preview": True},
        req_timeout=20,
    )


def get_updates(offset: Optional[int], timeout_sec: int = 25) -> Any:
    params: Dict[str, Any] = {"timeout": timeout_sec}
    if offset is not None:
        params["offset"] = offset
    # long polling için request timeout'u yükseltiyoruz
    return _tg("getUpdates", params, req_timeout=max(70, timeout_sec + 45))


def is_allowed_chat(update: Dict[str, Any]) -> bool:
    if not STRICT_CHAT:
        return True
    if not CHAT_ID:
        return True
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
