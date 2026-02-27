import os
import time
import json
import requests
from datetime import datetime, timezone

from notify_channel import send_channel

BINANCE_FAPI = "https://fapi.binance.com"
STORAGE_PATH = os.getenv("STORAGE_PATH", "/tmp/futures_scanner_storage.json")
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "60"))
TF = os.getenv("TF", "1h")

sent_cache = set()


def get_klines(symbol, interval="1h", limit=200):
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def format_price(x):
    if x >= 100:
        return f"{x:.2f}"
    if x >= 1:
        return f"{x:.4f}"
    return f"{x:.6f}"


def build_signal(symbol):
    kl = get_klines(symbol, TF, 200)

    closes = [float(x[4]) for x in kl]
    highs = [float(x[2]) for x in kl]
    lows = [float(x[3]) for x in kl]

    i = len(closes) - 2

    entry = closes[i]

    swing_low = min(lows[i-10:i+1])
    sl = swing_low

    risk = entry - sl

    tp1 = entry + risk
    tp2 = entry + risk * 2
    tp3 = entry + risk * 3

    ts = datetime.fromtimestamp(
        int(kl[i][6]) / 1000,
        tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M")

    msg = f"""🚀 <b>LONG BREAKOUT</b>

<b>Symbol:</b> {symbol}
<b>TF:</b> {TF}
<b>Time:</b> {ts}

<b>ENTRY:</b> {format_price(entry)}
<b>SL:</b> {format_price(sl)}

<b>TP1:</b> {format_price(tp1)}
<b>TP2:</b> {format_price(tp2)}
<b>TP3:</b> {format_price(tp3)}

#scanner"""

    return msg


def main():
    send_channel("✅ Premium channel worker started")

    while True:
        try:

            if os.path.exists(STORAGE_PATH):

                with open(STORAGE_PATH, "r") as f:
                    data = json.load(f)

                for key in data:

                    if key in sent_cache:
                        continue

                    symbol, tf, side = key.split(":")

                    msg = build_signal(symbol)

                    send_channel(msg)

                    sent_cache.add(key)

        except Exception as e:
            send_channel(f"⚠️ Worker error: {e}")

        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main()
