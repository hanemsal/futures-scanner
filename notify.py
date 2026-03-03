import requests


def send_telegram(bot_token: str, chat_id: str, text: str, dry_run: bool = False):
    if dry_run:
        print("[DRY_RUN] Telegram message:\n" + text, flush=True)
        return

    if not bot_token or not chat_id:
        raise RuntimeError("TG_BOT_TOKEN / TG_CHAT_ID missing")

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    r = requests.post(url, json=payload, timeout=15)
    r.raise_for_status()
