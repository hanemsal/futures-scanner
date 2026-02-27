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

# Signal TF ve HTF Filter
TF = os.getenv("TF", "5m").strip()            # sinyal timeframe (default 5m)
HTF = os.getenv("HTF", "1h").strip()          # trend timeframe (default 1h)
USE_HTF_FILTER = int(os.getenv("USE_HTF_FILTER", "1"))  # 1: aktif, 0: kapalı

USE_STORAGE = int(os.getenv("USE_STORAGE", "1"))
STORAGE_PATH = os.getenv("STORAGE_PATH", "/tmp/futures_state.json")
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", "3600"))


# =========================
# HTTP (retry'li)
# =========================
def get_json(url: str, params=None, retries: int = 3, timeout: int = 15):
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(1.0 * (i + 1))
    raise last_err


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
# INDICATORS
# =========================
def ema(data: List[float], length: int) -> List[float]:
    if not data:
        return []
    if length <= 1:
        return data[:]

    k = 2 / (length + 1)
    out = [data[0]]
    for i in range(1, len(data)):
        out.append(data[i] * k + out[-1] * (1 - k))
    return out


def crossed_up(prev_a: float, prev_b: float, now_a: float, now_b: float) -> bool:
    return prev_a <= prev_b and now_a > now_b


# =========================
# SYMBOL LIST
# =========================
def get_symbols() -> List[str]:
    data = get_json(f"{BINANCE_FAPI}/fapi/v1/ticker/24hr")
    rows = []

    for x in data:
        s = x.get("symbol", "")
        if not s.endswith("USDT"):
            continue

        try:
            vol = float(x.get("quoteVolume", 0))
        except Exception:
            continue

        if vol >= MIN_QUOTE_VOLUME:
            rows.append((s, vol))

    rows.sort(key=lambda t: t[1], reverse=True)
    return [s for s, _ in rows[:TOP_N]]


# =========================
# SIGNAL (Seçenek B: TF sinyal + HTF trend filtresi)
# =========================
def check_signal(symbol: str) -> Optional[Dict]:
    need = max(EMA_FAST, EMA_SLOW) + 5

    # ---- 1) Signal TF (default: 5m)
    kl = get_klines(symbol, TF, KLINE_LIMIT)
    if len(kl) < 10:
        return None

    # Son mum kapanmamış olabilir -> çıkar
    kl = kl[:-1]

    close = parse_close(kl)
    if len(close) < need:
        return None

    ema_fast = ema(close, EMA_FAST)
    ema_slow = ema(close, EMA_SLOW)

    if len(ema_fast) < 3 or len(ema_slow) < 3:
        return None

    cross = crossed_up(
        ema_fast[-2], ema_slow[-2],
        ema_fast[-1], ema_slow[-1]
    )
    if not cross:
        return None

    # ---- 2) HTF Filter (default: 1h) -> trend onayı
    if USE_HTF_FILTER == 1:
        hkl = get_klines(symbol, HTF, KLINE_LIMIT)
        if len(hkl) < 10:
            return None

        hkl = hkl[:-1]
        hclose = parse_close(hkl)
        if len(hclose) < need:
            return None

        h_fast = ema(hclose, EMA_FAST)
        h_slow = ema(hclose, EMA_SLOW)

        # LONG için: HTF'de fast > slow olmalı
        if not (h_fast[-1] > h_slow[-1]):
            return None

    return {"symbol": symbol, "price": close[-1], "tf": TF, "htf": HTF}


# =========================
# MESSAGE
# =========================
def build_msg(sig: Dict) -> str:
    return (
        "🚀 LONG SIGNAL\n\n"
        f"Symbol: {sig['symbol']}\n"
        f"TF: {sig.get('tf', '')} | HTF: {sig.get('htf', '')}\n"
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
    print("STORAGE_PATH =", STORAGE_PATH)
    print("TF =", TF, "| HTF =", HTF, "| USE_HTF_FILTER =", USE_HTF_FILTER)

    last_hb = time.time()

    while True:
        sent = 0
        try:
            symbols = get_symbols()
            if DEBUG:
                print(f"Symbols loaded: {len(symbols)}")

            for sym in symbols:
                try:
                    sig = check_signal(sym)
                    if not sig:
                        continue

                    key = f"{sym}_LONG_{TF}_HTF{HTF}"

                    if storage.should_send(key):
                        msg = build_msg(sig)
                        send_telegram(msg)
                        storage.mark_sent(key)
                        print("SIGNAL SENT:", sym)
                        sent += 1
                        time.sleep(0.5)

                except Exception as e:
                    if DEBUG:
                        print(f"SYM ERROR {sym}: {e}")

            if DEBUG:
                print("Cycle done. Sent:", sent)

            if time.time() - last_hb > HEARTBEAT_SEC:
                print("HEARTBEAT OK")
                last_hb = time.time()

        except Exception as e:
            print("ERROR (main loop):", e)

        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main()
