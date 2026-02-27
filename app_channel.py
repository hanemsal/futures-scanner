import os
import time
import json
import requests
from datetime import datetime, timezone

from notify_channel import send_channel

BINANCE_FAPI = os.getenv("BINANCE_FAPI", "https://fapi.binance.com").rstrip("/")

# app.py ile aynı dosyayı okuyacağız
STORAGE_PATH = os.getenv("STORAGE_PATH", "/tmp/futures_scanner_storage.json")

# Worker tarama sıklığı
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "60"))

# Duplicate kalıcı cache (worker restart olunca tekrar basmasın)
SENT_PATH = os.getenv("SENT_PATH", "/tmp/channel_sent_cache.json")

# Sinyal format parametreleri
SWING_LEN = int(os.getenv("SWING_LEN", "10"))          # SL için swing low bakış
KLINE_LIMIT = int(os.getenv("CHANNEL_KLINE_LIMIT", "200"))

# Telegram rate limit koruması
SEND_SLEEP_SEC = float(os.getenv("SEND_SLEEP_SEC", "0.7"))


def load_sent() -> set:
    try:
        with open(SENT_PATH, "r", encoding="utf-8") as f:
            arr = json.load(f)
        return set(arr) if isinstance(arr, list) else set()
    except Exception:
        return set()


def save_sent(s: set) -> None:
    try:
        with open(SENT_PATH, "w", encoding="utf-8") as f:
            json.dump(sorted(list(s)), f)
    except Exception:
        pass


sent_cache = load_sent()


def get_klines(symbol: str, interval: str, limit: int = 200):
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def format_price(x: float) -> str:
    if x >= 100:
        return f"{x:.2f}"
    if x >= 1:
        return f"{x:.4f}"
    return f"{x:.6f}"


def build_signal(symbol: str, tf: str) -> str:
    kl = get_klines(symbol, tf, KLINE_LIMIT)

    closes = [float(x[4]) for x in kl]
    lows = [float(x[3]) for x in kl]

    i = len(closes) - 2  # last CLOSED candle
    if i < 2:
        raise ValueError("not enough candles")

    entry = closes[i]

    # Swing low (son SWING_LEN mum)
    start = max(0, i - SWING_LEN + 1)
    swing_low = min(lows[start:i + 1])
    sl = swing_low

    risk = entry - sl
    if risk <= 0:
        # risk negatif/0 ise yine de mesaj basmayalım
        raise ValueError("invalid risk (entry<=sl)")

    tp1 = entry + risk
    tp2 = entry + risk * 2
    tp3 = entry + risk * 3

    ts = datetime.fromtimestamp(int(kl[i][6]) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

    msg = f"""🚀 <b>LONG BREAKOUT</b>

<b>Symbol:</b> {symbol}
<b>TF:</b> {tf}
<b>Time:</b> {ts} UTC

<b>ENTRY:</b> {format_price(entry)}
<b>SL:</b> {format_price(sl)}

<b>TP1:</b> {format_price(tp1)}
<b>TP2:</b> {format_price(tp2)}
<b>TP3:</b> {format_price(tp3)}

#scanner"""

    return msg


def main():
    send_channel("✅ Premium channel worker started")

    last_error = ""
    while True:
        try:
            if os.path.exists(STORAGE_PATH):
                with open(STORAGE_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # data dict ise: {"BTCUSDT:1h:LONG": 123...}
                keys = list(data.keys()) if isinstance(data, dict) else list(data)

                for key in keys:
                    if key in sent_cache:
                        continue

                    parts = key.split(":")
                    if len(parts) != 3:
                        # beklenmeyen format
                        sent_cache.add(key)
                        save_sent(sent_cache)
                        continue

                    symbol, tf, side = parts

                    # Şimdilik sadece LONG mesajı (istersen side'a göre SHORT da ekleriz)
                    if side.upper() != "LONG":
                        sent_cache.add(key)
                        save_sent(sent_cache)
                        continue

                    msg = build_signal(symbol, tf)
                    send_channel(msg)

                    sent_cache.add(key)
                    save_sent(sent_cache)

                    # Telegram rate-limit
                    time.sleep(SEND_SLEEP_SEC)

            last_error = ""

        except Exception as e:
            # Aynı hatayı sürekli spamlemeyelim
            err = str(e)
            if err != last_error:
                send_channel(f"⚠️ Premium worker error: {err}")
                last_error = err

        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main()
