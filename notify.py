import requests


def send_telegram(token: str, chat_id: str, text: str, dry_run: bool = False, timeout: int = 12) -> None:
    """
    Telegram message sender.
    - dry_run=True -> sadece konsola yazar, telegrama göndermez.
    """
    if dry_run:
        print("[DRY_RUN] Telegram message:\n" + text)
        return

    if not token or not chat_id:
        raise ValueError("TG_BOT_TOKEN / TG_CHAT_ID missing")

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
