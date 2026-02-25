import os
import requests

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "").strip()
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "").strip()

def send_telegram(text: str) -> None:
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        print("[WARN] Telegram env missing (TG_BOT_TOKEN / TG_CHAT_ID). Message:\n", text)
        return

    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TG_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",              # ✅ <b> çalışır
        "disable_web_page_preview": True
    }

    try:
        r = requests.post(url, json=payload, timeout=20)
        if r.status_code != 200:
            print("[WARN] Telegram send failed:", r.status_code, r.text)
    except Exception as e:
        print("[WARN] Telegram exception:", e)
