# notify.py
import os
import time
import requests


def send_telegram(text: str, bot_token: str, chat_id: str, dry_run: bool = False, **kwargs) -> bool:
    """
    Telegram'a mesaj gönderir.
    - dry_run=True ise mesajı göndermek yerine loglar (test için).
    - **kwargs: app.py ileride ekstra keyword gönderirse hata vermesin diye.
    """
    if not bot_token or not chat_id:
        print("[WARN] TG_BOT_TOKEN veya TG_CHAT_ID boş. Mesaj gönderilmedi.")
        return False

    if dry_run:
        print("[DRY_RUN] Telegram message (not sent):")
        print(text)
        return True

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }

    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code != 200:
            print(f"[ERROR] Telegram send failed: {r.status_code} {r.text}")
            return False
        return True
    except Exception as e:
        print(f"[ERROR] Telegram exception: {e}")
        return False


def send_channel(text: str, bot_token: str, chat_id: str, dry_run: bool = False, **kwargs) -> bool:
    """
    Bazı eski kodlarda send_channel kullanılıyor olabilir.
    Aynı işi yapsın diye alias bıraktım.
    """
    return send_telegram(text, bot_token, chat_id, dry_run=dry_run)
