# notify.py
import requests


TELEGRAM_LIMIT = 3900  # 4096'a yaklaşmayalım


def send_telegram(bot_token: str, chat_id: str, text: str) -> None:
    """
    Telegram mesaj gönderir.
    bot_token / chat_id boşsa Exception fırlatır (app.py yakalayıp loglar).
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


def send_telegram_chunked(bot_token: str, chat_id: str, text: str) -> None:
    """
    Telegram limitine takılmamak için mesajı parçalara böler.
    """
    if len(text) <= TELEGRAM_LIMIT:
        send_telegram(bot_token, chat_id, text)
        return

    lines = text.split("\n")
    buf = ""
    for line in lines:
        if len(buf) + len(line) + 1 > TELEGRAM_LIMIT:
            if buf.strip():
                send_telegram(bot_token, chat_id, buf.rstrip())
            buf = line + "\n"
        else:
            buf += line + "\n"

    if buf.strip():
        send_telegram(bot_token, chat_id, buf.rstrip())


def send_dip_list(bot_token: str, chat_id: str, symbols_sorted: list[str], total: int, top_n: int = 40) -> None:
    """
    Dip coin listesini Telegram'a basar.
    Top N + sonda Toplam: X
    """
    top = symbols_sorted[:max(0, int(top_n))]
    header = f"🧲 DIP LIST (Top {len(top)} / Total {total})"
    body_lines = [f"{i+1:02d}) {s}" for i, s in enumerate(top)]
    footer = f"✅ Toplam Dip: {total}"
    msg = "\n".join([header, ""] + body_lines + ["", footer])
    send_telegram_chunked(bot_token, chat_id, msg)
