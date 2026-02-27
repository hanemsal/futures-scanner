import os
import time
from typing import List, Dict, Optional

import requests

from notify import send_telegram
from storage import Storage


# =========================
# ENV
# =========================

BINANCE_FAPI = os.getenv("BINANCE_FAPI", "https://fapi.binance.com").rstrip("/")

DEBUG = int(os.getenv("DEBUG", "1"))

INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "300"))
HEARTBEAT_SEC = int(os.getenv("HEARTBEAT_SEC", "900"))

KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "260"))

TOP_N = int(os.getenv("TOP_N", "80"))
MIN_QUOTE_VOLUME = float(os.getenv("MIN_QUOTE_VOLUME", "3000000"))

EMA_FAST = int(os.getenv("EMA_FAST", "3"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "44"))

TF = os.getenv("TF", "5m")
HTF = os.getenv("HTF", "1h")

USE_STORAGE = int(os.getenv("USE_STORAGE", "1"))
STORAGE_PATH = os.getenv("STORAGE_PATH", "/var/data/futures_state.json")
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", "21600"))


# =========================
# HTTP
# =========================

def get_json(url: str, params=None, retries=3, timeout=15):
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(i+1)
    raise last


def get_klines(symbol: str, tf: str, limit: int):
    return get_json(
        f"{BINANCE_FAPI}/fapi/v1/klines",
        params={"symbol": symbol, "interval": tf, "limit": limit},
    )


# =========================
# PARSE
# =========================

def parse_close(klines) -> List[float]:
    return [float(x[4]) for x in klines]


# =========================
# EMA
# =========================

def ema(data: List[float], length: int) -> List[float]:

    k = 2 / (length + 1)

    out = [data[0]]

    for i in range(1, len(data)):
        out.append(data[i]*k + out[-1]*(1-k))

    return out


# =========================
# SYMBOL LIST
# =========================

def get_symbols() -> List[str]:

    data = get_json(f"{BINANCE_FAPI}/fapi/v1/ticker/24hr")

    rows = []

    for x in data:

        symbol = x["symbol"]

        if not symbol.endswith("USDT"):
            continue

        vol = float(x["quoteVolume"])

        if vol >= MIN_QUOTE_VOLUME:

            rows.append((symbol, vol))

    rows.sort(key=lambda x: x[1], reverse=True)

    return [x[0] for x in rows[:TOP_N]]


# =========================
# PROFESSIONAL SIGNAL ENGINE
# =========================

def check_signal(symbol: str) -> Optional[Dict]:

    need = EMA_SLOW + 10

    # =====================
    # 5m SIGNAL TF
    # =====================

    kl = get_klines(symbol, TF, KLINE_LIMIT)

    kl = kl[:-1]  # last candle ignore

    close = parse_close(kl)

    if len(close) < need:
        return None

    ema_fast = ema(close, EMA_FAST)
    ema_slow = ema(close, EMA_SLOW)


    # =====================
    # 2 candle confirmation
    # =====================

    if not (
        ema_fast[-3] <= ema_slow[-3]
        and ema_fast[-2] > ema_slow[-2]
        and ema_fast[-1] > ema_slow[-1]
    ):
        return None


    # =====================
    # DISTANCE FILTER
    # =====================

    distance = (ema_fast[-1] - ema_slow[-1]) / ema_slow[-1]

    if distance < 0.0015:
        return None


    # =====================
    # HTF FILTER (1H)
    # =====================

    hkl = get_klines(symbol, HTF, KLINE_LIMIT)

    hkl = hkl[:-1]

    hclose = parse_close(hkl)

    if len(hclose) < need:
        return None

    h_fast = ema(hclose, EMA_FAST)
    h_slow = ema(hclose, EMA_SLOW)


    # Trend direction filter
    if h_fast[-1] <= h_slow[-1]:
        return None


    # =====================
    # SLOPE FILTER ⭐⭐⭐⭐⭐
    # =====================

    if h_slow[-1] <= h_slow[-4]:
        return None


    return {
        "symbol": symbol,
        "price": close[-1],
        "tf": TF,
        "htf": HTF
    }


# =========================
# MESSAGE
# =========================

def build_msg(sig):

    return (
        "🚀 LONG SIGNAL\n\n"
        f"Symbol: {sig['symbol']}\n"
        f"TF: {sig['tf']} | HTF: {sig['htf']}\n"
        f"Price: {sig['price']}\n\n"
        "#scanner"
    )


# =========================
# MAIN LOOP
# =========================

def main():

    storage = Storage(
        STORAGE_PATH,
        enabled=(USE_STORAGE == 1),
        cooldown_sec=COOLDOWN_SEC
    )

    print("BOT STARTED")
    print("Professional Institutional Version Active")


    last_hb = time.time()


    while True:

        sent = 0

        try:

            symbols = get_symbols()

            for sym in symbols:

                sig = check_signal(sym)

                if not sig:
                    continue

                key = f"{sym}_LONG"

                if storage.should_send(key):

                    msg = build_msg(sig)

                    send_telegram(msg)

                    storage.mark_sent(key)

                    print("SIGNAL SENT:", sym)

                    sent += 1

                    time.sleep(0.5)


            if DEBUG:
                print("Cycle done. Sent:", sent)


            if time.time() - last_hb > HEARTBEAT_SEC:

                print("HEARTBEAT OK")

                last_hb = time.time()


        except Exception as e:

            print("ERROR:", e)


        time.sleep(INTERVAL_SEC)



# =========================

if __name__ == "__main__":
    main()
