import os
import requests

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "").strip()
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "").strip()


def send_telegram(text: str) -> None:
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        print("⚠️ Telegram credentials missing (TG_BOT_TOKEN / TG_CHAT_ID).")
        return

    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TG_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    }

    r = requests.post(url, data=payload, timeout=20)

    # Debug için response görünsün
    if r.status_code != 200:
        print("❌ Telegram error:", r.status_code, r.text)
        r.raise_for_status()
    else:
        # İstersen DEBUG env ile açarız, şimdilik sade
        print("✅ Telegram sent")
