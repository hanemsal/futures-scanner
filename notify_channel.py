import os
from notify import send_telegram

def send_channel(text: str) -> None:
    """
    Kanal mesajı: Eski notify.py'ye dokunmadan,
    sadece ENV'den TELEGRAM_* okuyup send_telegram'ı kullanır.
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")  # -100... kanal id

    # Eğer TELEGRAM_BOT_TOKEN boş bırakıldıysa, TG_BOT_TOKEN'a düş (opsiyonel)
    if not bot_token:
        bot_token = os.getenv("TG_BOT_TOKEN")

    send_telegram(bot_token=bot_token, chat_id=chat_id, text=text)
