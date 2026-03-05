import os
import requests
from typing import Optional, Dict, Any

HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30"))

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
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
        {
            "chat_id": CHAT_ID,
            "text": text,
            "disable_web_page_preview": True,
        },
        req_timeout=60,
    )


def get_updates(offset: Optional[int], timeout_sec: int = 30) -> Any:
    params: Dict[str, Any] = {"timeout": timeout_sec}
    if offset is not None:
        params["offset"] = offset
    # long polling 30s ise request timeout'u biraz yüksek tut
    return _tg("getUpdates", params, req_timeout=timeout_sec + 20)
