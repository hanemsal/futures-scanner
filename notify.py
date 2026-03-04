# notify.py
import requests


def send_telegram(bot_token: str, chat_id: str, text: str) -> None:
    """
    Telegram mesaj gönderir.
    bot_token ve chat_id boşsa Exception fırlatır (app.py bunu yakalayıp loglayacak).
    """
    if not bot_token or not chat_id:
        raise ValueError("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID eksik.")

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    r = requests.post(url, data=payload, timeout=15)
    r.raise_for_status()
