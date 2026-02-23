import os
import requests
from typing import Optional

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
TG_CHAT_ID = os.getenv("TG_CHAT_ID")

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"


def send_telegram(text: str, parse_mode: str = "HTML", disable_preview: bool = True) -> Optional[dict]:
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        raise RuntimeError("TG_BOT_TOKEN / TG_CHAT_ID missing")

    url = TELEGRAM_API.format(token=TG_BOT_TOKEN, method="sendMessage")
    payload = {
        "chat_id": TG_CHAT_ID,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": disable_preview,
    }

    r = requests.post(url, json=payload, timeout=20)
    if not r.ok:
        raise RuntimeError(f"Telegram error: {r.status_code} {r.text}")
    return r.json()
