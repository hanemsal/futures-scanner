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

INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "180"))
HEARTBEAT_SEC = int(os.getenv("HEARTBEAT_SEC", "900"))

KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "260"))

TOP_N = int(os.getenv("TOP_N", "80"))
MIN_QUOTE_VOLUME = float(os.getenv("MIN_QUOTE_VOLUME", "3000000"))

EMA_FAST = int(os.getenv("EMA_FAST", "3"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "44"))

# Timeframes
TF_ENTRY = os.getenv("TF_ENTRY", "5m").strip()
TF_TREND = os.getenv("TF_TREND", "1h").strip()

# "Recent cross" window on HTF (bars): 6 => son 6 saat içinde 1h cross olduysa
HTF_CROSS_LOOKBACK = int(os.getenv("HTF_CROSS_LOOKBACK", "6"))

# BTC filter
USE_BTC_FILTER = int(os.getenv("USE_BTC_FILTER", "1"))
BTC_SYMBOL = os.getenv("BTC_SYMBOL", "BTCUSDT").strip()  # Binance futures symbol
BTC_TF = os.getenv("BTC_TF", "1h").strip()

# Risk params (price move %)
LEVERAGE = float(os.getenv("LEVERAGE", "5"))
TP_PCT = float(os.getenv("TP_PCT", "5"))   # price move %
SL_PCT = float(os.getenv("SL_PCT", "2"))   # price move %

# Storage/cooldown
USE_STORAGE = int(os.getenv("USE_STORAGE", "1"))
STORAGE_PATH = os.getenv("STORAGE_PATH", "/var/data/futures_state.json")
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", "21600"))


# =========================
# HTTP
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
            time.sleep(1 + i)
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


def crossed_down(prev_a: float, prev_b: float, now_a: float, now_b: float) -> bool:
    return prev_a >= prev_b and now_a < now_b


def slope_up(series: List[float], bars: int = 2) -> bool:
    if len(series) < bars + 1:
        return False
    return series[-1] > series[-1 - bars]


def slope_down(series: List[float], bars: int = 2) -> bool:
    if len(series) < bars + 1:
        return False
    return series[-1] < series[-1 - bars]


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
# BTC FILTER
# =========================
def btc_trend() -> Optional[str]:
    """
    Returns:
      "BULL" if BTC EMA_FAST > EMA_SLOW and slope up
      "BEAR" if BTC EMA_FAST < EMA_SLOW and slope down
      None otherwise (flat/unclear)
    """
    if USE_BTC_FILTER != 1:
        return "ANY"

    kl = get_klines(BTC_SYMBOL, BTC_TF, KLINE_LIMIT)
    if len(kl) < 30:
        return None

    kl = kl[:-1]
    close = parse_close(kl)

    need = max(EMA_FAST, EMA_SLOW) + 10
    if len(close) < need:
        return None

    ef = ema(close, EMA_FAST)
    es = ema(close, EMA_SLOW)

    if ef[-1] > es[-1] and slope_up(ef, bars=2):
        return "BULL"

    if ef[-1] < es[-1] and slope_down(ef, bars=2):
        return "BEAR"

    return None


# =========================
# HTF RECENT CROSS FILTER
# =========================
def htf_recent_cross(symbol: str, side: str) -> bool:
    """
    side: "LONG" or "SHORT"
    Kural:
      - HTF'de yön doğrulaması (fast>slow veya fast<slow) şart
      - Son HTF_CROSS_LOOKBACK bar içinde cross gerçekleşmiş olmalı
      - Ayrıca fast'in slope'u doğru yöne olmalı (trend güç filtresi)
    """
    kl = get_klines(symbol, TF_TREND, KLINE_LIMIT)
    if len(kl) < 50:
        return False

    kl = kl[:-1]
    close = parse_close(kl)

    need = max(EMA_FAST, EMA_SLOW) + HTF_CROSS_LOOKBACK + 10
    if len(close) < need:
        return False

    ef = ema(close, EMA_FAST)
    es = ema(close, EMA_SLOW)

    # current direction + slope (strength)
    if side == "LONG":
        if not (ef[-1] > es[-1] and slope_up(ef, bars=2)):
            return False
    else:
        if not (ef[-1] < es[-1] and slope_down(ef, bars=2)):
            return False

    # recent cross detection in last N bars
    n = max(1, HTF_CROSS_LOOKBACK)
    start = len(ef) - (n + 2)  # ensure we have i-1, i
    if start < 1:
        start = 1

    if side == "LONG":
        for i in range(start, len(ef)):
            if crossed_up(ef[i - 1], es[i - 1], ef[i], es[i]):
                return True
        return False

    # SHORT
    for i in range(start, len(ef)):
        if crossed_down(ef[i - 1], es[i - 1], ef[i], es[i]):
            return True
    return False


# =========================
# ENTRY TRIGGER (5m cross)
# =========================
def entry_cross(symbol: str, side: str) -> Optional[float]:
    kl = get_klines(symbol, TF_ENTRY, KLINE_LIMIT)
    if len(kl) < 30:
        return None

    kl = kl[:-1]
    close = parse_close(kl)

    need = max(EMA_FAST, EMA_SLOW) + 10
    if len(close) < need:
        return None

    ef = ema(close, EMA_FAST)
    es = ema(close, EMA_SLOW)

    if side == "LONG":
        if crossed_up(ef[-2], es[-2], ef[-1], es[-1]):
            return close[-1]
        return None

    # SHORT
    if crossed_down(ef[-2], es[-2], ef[-1], es[-1]):
        return close[-1]
    return None


# =========================
# TP/SL
# =========================
def calc_tp_sl(entry: float, side: str) -> Dict[str, float]:
    tp_mul = TP_PCT / 100.0
    sl_mul = SL_PCT / 100.0
    if side == "LONG":
        tp = entry * (1.0 + tp_mul)
        sl = entry * (1.0 - sl_mul)
    else:
        tp = entry * (1.0 - tp_mul)
        sl = entry * (1.0 + sl_mul)
    return {"tp": tp, "sl": sl}


def fmt_price(x: float) -> str:
    if x >= 1000:
        return f"{x:.2f}"
    if x >= 1:
        return f"{x:.4f}"
    if x >= 0.01:
        return f"{x:.6f}"
    return f"{x:.8f}"


# =========================
# MESSAGE
# =========================
def build_msg(symbol: str, side: str, entry: float) -> str:
    levels = calc_tp_sl(entry, side)
    tp = levels["tp"]
    sl = levels["sl"]

    emoji = "🚀" if side == "LONG" else "🔻"
    pnl_tp = TP_PCT * LEVERAGE
    pnl_sl = SL_PCT * LEVERAGE

    return (
        f"{emoji} {side} SIGNAL (Institutional)\n\n"
        f"Symbol: {symbol}\n"
        f"TF: {TF_ENTRY} | HTF: {TF_TREND} | BTC: {BTC_TF}\n\n"
        f"Entry: {fmt_price(entry)}\n"
        f"TP  (+{TP_PCT:.2f}% | ~+{pnl_tp:.1f}% PnL @x{LEVERAGE:g}): {fmt_price(tp)}\n"
        f"SL  (-{SL_PCT:.2f}% | ~-{pnl_sl:.1f}% PnL @x{LEVERAGE:g}): {fmt_price(sl)}\n\n"
        "#scanner"
    )


# =========================
# MAIN
# =========================
def main():
    storage = Storage(
        STORAGE_PATH,
        enabled=(USE_STORAGE == 1),
        cooldown_sec=COOLDOWN_SEC
    )

    print("BOT STARTED (Institutional)")
    print("STORAGE_PATH =", STORAGE_PATH)
    print("TF_ENTRY =", TF_ENTRY, "| TF_TREND =", TF_TREND)
    print("BTC_FILTER =", USE_BTC_FILTER, "| BTC_SYMBOL =", BTC_SYMBOL, "| BTC_TF =", BTC_TF)
    print("HTF_CROSS_LOOKBACK =", HTF_CROSS_LOOKBACK)
    print("TP/SL =", TP_PCT, "/", SL_PCT, "| LEVERAGE =", LEVERAGE)

    last_hb = time.time()

    while True:
        sent = 0
        try:
            btc_state = btc_trend()  # "BULL" / "BEAR" / None / "ANY"

            if DEBUG:
                print("BTC_STATE:", btc_state)

            symbols = get_symbols()
            if DEBUG:
                print("Symbols loaded:", len(symbols))

            for sym in symbols:
                try:
                    # LONG path
                    if btc_state in ("BULL", "ANY"):
                        if htf_recent_cross(sym, "LONG"):
                            entry = entry_cross(sym, "LONG")
                            if entry is not None:
                                key = f"{sym}_LONG"
                                if storage.should_send(key):
                                    send_telegram(build_msg(sym, "LONG", entry))
                                    storage.mark_sent(key)
                                    sent += 1
                                    time.sleep(0.25)

                    # SHORT path
                    if btc_state in ("BEAR", "ANY"):
                        if htf_recent_cross(sym, "SHORT"):
                            entry = entry_cross(sym, "SHORT")
                            if entry is not None:
                                key = f"{sym}_SHORT"
                                if storage.should_send(key):
                                    send_telegram(build_msg(sym, "SHORT", entry))
                                    storage.mark_sent(key)
                                    sent += 1
                                    time.sleep(0.25)

                except Exception as e:
                    if DEBUG:
                        print("SYM ERROR:", sym, e)

            if DEBUG:
                print("Cycle done. Sent:", sent)

            if time.time() - last_hb > HEARTBEAT_SEC:
                print("HEARTBEAT OK")
                last_hb = time.time()

        except Exception as e:
            print("MAIN ERROR:", e)

        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main()
