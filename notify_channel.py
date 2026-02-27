import os
from notify import send_telegram

# Kanal gönderimi varsayılan kapalı (eski sistem etkilenmesin)
USE_TG_CHANNEL = int(os.getenv("USE_TG_CHANNEL", "0"))

def send_channel(text: str, timeout: int = 20) -> None:
    """
    Kanal mesajı: TELEGRAM_* env ile kanala gönderir.
    USE_TG_CHANNEL=1 değilse hiçbir şey yapmaz.
    """
    if USE_TG_CHANNEL != 1:
        return

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TG_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")  # -100... kanal id

    if not bot_token or not chat_id:
        return

    send_telegram(bot_token=bot_token, chat_id=chat_id, text=text, timeout=timeout)
