import os
import requests


def send_telegram_oversold(text: str) -> None:
    """
    OVERSOLD_BOT_TOKEN ve OVERSOLD_CHAT_ID env'leri ile Telegram'a mesaj yollar.
    Kanal ile uğraşma: şimdilik direkt kendi chat_id (senin id) ile gönder.
    """
    token = os.getenv("OVERSOLD_BOT_TOKEN", "").strip()
    chat_id = os.getenv("OVERSOLD_CHAT_ID", "").strip()

    if not token or not chat_id:
        print("[WARN] Telegram env eksik: OVERSOLD_BOT_TOKEN / OVERSOLD_CHAT_ID")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }

    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code != 200:
            print(f"[WARN] Telegram sendMessage failed: {r.status_code} {r.text}")
    except Exception as e:
        print(f"[WARN] Telegram exception: {e}")
