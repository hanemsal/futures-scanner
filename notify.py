import requests


def send_telegram(bot_token: str, chat_id: str, text: str) -> bool:
    """
    Sends a Telegram message (Markdown).
    """
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, json=payload, timeout=12)
        r.raise_for_status()
        data = r.json()
        return bool(data.get("ok"))
    except Exception:
        return False
