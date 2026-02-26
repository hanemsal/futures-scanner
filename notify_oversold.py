import os
import requests

def send_telegram_oversold(text: str):
    token = os.getenv("OVERSOLD_BOT_TOKEN")
    chat_id = os.getenv("OVERSOLD_CHAT_ID")

    if not token or not chat_id:
        raise RuntimeError("OVERSOLD_BOT_TOKEN / OVERSOLD_CHAT_ID eksik")

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    requests.post(url, json=payload, timeout=15)
