import requests

def send_telegram(bot_token: str, chat_id: str, text: str, timeout: int = 20) -> None:
    if not bot_token or not chat_id or not text:
        return
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        requests.post(url, json=payload, timeout=timeout)
    except Exception:
        pass
