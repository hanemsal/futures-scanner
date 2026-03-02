import os
import time
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests

from notify import send_telegram
from storage import Storage


# =========================
# HELPERS
# =========================
def env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)).strip())
    except Exception:
        return default


def env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)).strip())
    except Exception:
        return default


def env_bool01(key: str, default: int = 0) -> bool:
    v = str(os.getenv(key, str(default))).strip()
    return v in ("1", "true", "True", "yes", "YES")


def now_utc() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def bump(d: Dict[str, int], key: str) -> None:
    d[key] = d.get(key, 0) + 1


# =========================
# CONFIG (ENV)
# =========================
BINANCE_FAPI = os.getenv("BINANCE_FAPI", "https://fapi.binance.com").rstrip("/")

# Timeframes
TF_ENTRY = os.getenv("TF_ENTRY", os.getenv("TF", "1h"))  # entry timeframe
HTF = os.getenv("HTF", os.getenv("TF_TREND", "4h"))      # higher timeframe trend
BTC_TF = os.getenv("BTC_TF", "1h")
BTC_SYMBOL = os.getenv("BTC_SYMBOL", "BTCUSDT")

INTERVAL_SEC = env_int("INTERVAL_SEC", 600)
KLINE_LIMIT = env_int("KLINE_LIMIT", 260)

# EMA
EMA_FAST = env_int("EMA_FAST", 3)
EMA_SLOW = env_int("EMA_SLOW", 44)
EMA_TREND = env_int("EMA_TREND", 123)  # "ema123" çizgin

# RSI
RSI_LEN = env_int("RSI_LEN", 21)
RSI_MIN = env_float("RSI_MIN", 42.0)

# MFI
USE_MFI_FILTER = env_bool01("USE_MFI_FILTER", 1)
MFI_LEN = env_int("MFI_LEN", 14)
MFI_LONG_MIN = env_float("MFI_LONG_MIN", 40.0)
MFI_LONG_MAX = env_float("MFI_LONG_MAX", 85.0)
MFI_SLOPE_ENABLE = env_bool01("MFI_SLOPE_ENABLE", 1)
MFI_SLOPE_BARS = env_int("MFI_SLOPE_BARS", 1)

# Volume filter
USE_VOL_FILTER = env_bool01("USE_VOL_FILTER", 1)
VOL_LEN = env_int("VOL_LEN", 20)
VOL_MULT = env_float("VOL_MULT", 1.1)
VOL_USE_QUOTE = env_bool01("VOL_USE_QUOTE", 1)  # (Binance kline quote volume)

# Universe / liquidity
TOP_N = env_int("TOP_N", 200)
MIN_QUOTE_VOLUME = env_float("MIN_QUOTE_VOLUME", 3_000_000.0)  # 24h quoteVolume USDT

# Filters toggles
USE_HTF_FILTER = env_bool01("USE_HTF_FILTER", 1)
USE_BTC_FILTER = env_bool01("USE_BTC_FILTER", 1)
LONG_ONLY = env_bool01("LONG_ONLY", 1)

# Signal behavior
HTF_CROSS_LOOKBACK = env_int("HTF_CROSS_LOOKBACK", 6)  # cross son kaç bar içinde olmalı
COOLDOWN_SEC = env_int("COOLDOWN_SEC", 21600)          # 6 saat
HEARTBEAT_SEC = env_int("HEARTBEAT_SEC", 900)          # 15 dk

# TP/SL suggestion (manual trade)
TP_PCT = env_float("TP_PCT", 8.0)  # sen %8-15 swing diyorsun -> default 8
SL_PCT = env_float("SL_PCT", 2.0)

# Storage
USE_STORAGE = env_bool01("USE_STORAGE", 1)
STORAGE_PATH = os.getenv("STORAGE_PATH", "/var/data/futures_state.json")

# Debug
DEBUG = env_bool01("DEBUG", 1)
DEBUG_REJECTS = env_bool01("DEBUG_REJECTS", 0)
DRY_RUN = env_bool01("DRY_RUN", 0)
TEST_ONCE = env_bool01("TEST_ONCE", 0)


# =========================
# BINANCE API
# =========================
def http_get(url: str, params: dict, timeout: int = 20):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def get_klines(symbol: str, interval: str, limit: int) -> List[list]:
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    return http_get(url, params)


def get_24h_tickers() -> List[dict]:
    url = f"{BINANCE_FAPI}/fapi/v1/ticker/24hr"
    return http_get(url, {})


def get_price(symbol: str) -> float:
    url = f"{BINANCE_FAPI}/fapi/v1/ticker/price"
    data = http_get(url, {"symbol": symbol})
    return float(data["price"])


# =========================
# INDICATORS
# =========================
def ema(values: List[float], length: int) -> List[float]:
    if not values or length <= 1:
        return values[:]
    k = 2 / (length + 1)
    out = []
    e = values[0]
    for v in values:
        e = v * k + e * (1 - k)
        out.append(e)
    return out


def rsi(values: List[float], length: int) -> List[float]:
    if len(values) < length + 2:
        return [50.0] * len(values)
    gains = [0.0]
    losses = [0.0]
    for i in range(1, len(values)):
        ch = values[i] - values[i - 1]
        gains.append(max(ch, 0.0))
        losses.append(max(-ch, 0.0))

    # Wilder smoothing
    avg_gain = sum(gains[1:length + 1]) / length
    avg_loss = sum(losses[1:length + 1]) / length

    out = [50.0] * (length)  # warmup
    for i in range(length, len(values)):
        avg_gain = (avg_gain * (length - 1) + gains[i]) / length
        avg_loss = (avg_loss * (length - 1) + losses[i]) / length
        if avg_loss == 0:
            out.append(100.0)
        else:
            rs = avg_gain / avg_loss
            out.append(100.0 - (100.0 / (1.0 + rs)))
    # pad if needed
    while len(out) < len(values):
        out.insert(0, 50.0)
    return out[-len(values):]


def mfi(highs: List[float], lows: List[float], closes: List[float], volumes: List[float], length: int) -> List[float]:
    n = len(closes)
    if n < length + 2:
        return [50.0] * n

    tp = [(highs[i] + lows[i] + closes[i]) / 3.0 for i in range(n)]
    rmf = [tp[i] * volumes[i] for i in range(n)]

    pos = [0.0] * n
    neg = [0.0] * n
    for i in range(1, n):
        if tp[i] > tp[i - 1]:
            pos[i] = rmf[i]
        elif tp[i] < tp[i - 1]:
            neg[i] = rmf[i]

    out = [50.0] * n
    for i in range(length, n):
        pos_sum = sum(pos[i - length + 1:i + 1])
        neg_sum = sum(neg[i - length + 1:i + 1])
        if neg_sum == 0:
            out[i] = 100.0
        else:
            mr = pos_sum / neg_sum
            out[i] = 100.0 - (100.0 / (1.0 + mr))
    return out


def sma(values: List[float], length: int) -> List[float]:
    if len(values) < length:
        return [sum(values) / max(1, len(values))] * len(values)
    out = []
    s = 0.0
    for i, v in enumerate(values):
        s += v
        if i >= length:
            s -= values[i - length]
        if i >= length - 1:
            out.append(s / length)
        else:
            out.append(s / (i + 1))
    return out


def crossed_up(a_prev: float, a_now: float, b_prev: float, b_now: float) -> bool:
    return a_prev <= b_prev and a_now > b_now


def cross_within_lookback(fast: List[float], slow: List[float], lookback: int) -> bool:
    # Son lookback bar içinde herhangi bir bullish cross var mı?
    n = len(fast)
    lb = min(lookback, n - 2)
    for i in range(n - lb, n):
        if i <= 0:
            continue
        if crossed_up(fast[i - 1], fast[i], slow[i - 1], slow[i]):
            return True
    return False


# =========================
# FILTERS
# =========================
def btc_ok(reject: Dict[str, int]) -> bool:
    if not USE_BTC_FILTER:
        return True
    try:
        k = get_klines(BTC_SYMBOL, BTC_TF, KLINE_LIMIT)
        closes = [float(x[4]) for x in k]
        highs = [float(x[2]) for x in k]
        lows = [float(x[3]) for x in k]
        vols = [float(x[5]) for x in k]

        ema_f = ema(closes, EMA_FAST)
        ema_s = ema(closes, EMA_SLOW)
        ema_t = ema(closes, EMA_TREND)

        # BTC trend: fast>=slow ve close>=ema123 olsun
        if ema_f[-1] < ema_s[-1]:
            bump(reject, "BTC_TREND_DOWN")
            return False
        if closes[-1] < ema_t[-1]:
            bump(reject, "BTC_BELOW_EMA123")
            return False

        return True
    except Exception:
        bump(reject, "BTC_ERR")
        return False


def htf_ok(symbol: str, reject: Dict[str, int]) -> bool:
    if not USE_HTF_FILTER:
        return True
    try:
        k = get_klines(symbol, HTF, KLINE_LIMIT)
        closes = [float(x[4]) for x in k]
        ema_f = ema(closes, EMA_FAST)
        ema_s = ema(closes, EMA_SLOW)
        ema_t = ema(closes, EMA_TREND)

        if ema_f[-1] < ema_s[-1]:
            bump(reject, "HTF_EMA_DOWN")
            return False
        if closes[-1] < ema_t[-1]:
            bump(reject, "HTF_BELOW_EMA123")
            return False
        return True
    except Exception:
        bump(reject, "HTF_ERR")
        return False


def entry_signal(symbol: str, reject: Dict[str, int]) -> bool:
    try:
        k = get_klines(symbol, TF_ENTRY, KLINE_LIMIT)
        closes = [float(x[4]) for x in k]
        highs = [float(x[2]) for x in k]
        lows = [float(x[3]) for x in k]
        vols = [float(x[5]) for x in k]
        quote_vols = [float(x[7]) for x in k]  # quote asset volume

        ema_f = ema(closes, EMA_FAST)
        ema_s = ema(closes, EMA_SLOW)

        # EMA cross son X barda
        if not cross_within_lookback(ema_f, ema_s, HTF_CROSS_LOOKBACK):
            bump(reject, "NO_CROSS")
            return False

        # RSI
        r = rsi(closes, RSI_LEN)
        if r[-1] < RSI_MIN:
            bump(reject, "RSI_LOW")
            return False

        # MFI
        if USE_MFI_FILTER:
            m = mfi(highs, lows, closes, vols, MFI_LEN)
            if not (MFI_LONG_MIN <= m[-1] <= MFI_LONG_MAX):
                bump(reject, "MFI_OUT")
                return False
            if MFI_SLOPE_ENABLE:
                bars = max(1, MFI_SLOPE_BARS)
                if len(m) > bars and (m[-1] - m[-1 - bars]) <= 0:
                    bump(reject, "MFI_NOT_RISING")
                    return False

        # Volume spike
        if USE_VOL_FILTER:
            series = quote_vols if VOL_USE_QUOTE else vols
            v_sma = sma(series, VOL_LEN)
            if series[-1] < VOL_MULT * v_sma[-1]:
                bump(reject, "VOL_WEAK")
                return False

        return True
    except Exception:
        bump(reject, "ENTRY_ERR")
        return False


# =========================
# UNIVERSE (TOP N)
# =========================
def pick_symbols() -> List[Tuple[str, float]]:
    tickers = get_24h_tickers()
    out = []
    for t in tickers:
        s = t.get("symbol", "")
        if not s.endswith("USDT"):
            continue
        # perpetual USDT futures list zaten fapi/24hr döküyor
        try:
            qv = float(t.get("quoteVolume", 0.0))
        except Exception:
            continue
        if qv < MIN_QUOTE_VOLUME:
            continue
        out.append((s, qv))
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:TOP_N]


# =========================
# MAIN LOOP
# =========================
def cfg_banner() -> None:
    print("[BOOT] Futures scanner started")
    print(f"[CFG] TF_ENTRY={TF_ENTRY} HTF={HTF} EMA={EMA_FAST}/{EMA_SLOW} EMA_TREND={EMA_TREND} RSI_LEN={RSI_LEN} RSI_MIN={RSI_MIN}")
    print(f"[CFG] MFI={int(USE_MFI_FILTER)} VOL={int(USE_VOL_FILTER)} BTC_FILTER={int(USE_BTC_FILTER)} HTF_FILTER={int(USE_HTF_FILTER)}")
    print(f"[CFG] TOP_N={TOP_N} MIN_QUOTE_VOLUME={MIN_QUOTE_VOLUME} COOLDOWN_SEC={COOLDOWN_SEC} LONG_ONLY={int(LONG_ONLY)}")
    print(f"[CFG] DRY_RUN={int(DRY_RUN)} STORAGE={int(USE_STORAGE)} PATH={STORAGE_PATH}")


def main():
    cfg_banner()

    storage = Storage(STORAGE_PATH) if USE_STORAGE else None
    last_heartbeat = 0

    while True:
        t0 = time.time()
        reject_counts: Dict[str, int] = {}

        # Heartbeat
        if HEARTBEAT_SEC > 0 and now_utc() - last_heartbeat >= HEARTBEAT_SEC:
            msg = f"✅ worker alive | TF={TF_ENTRY} HTF={HTF} TOP_N={TOP_N} BTC_FILTER={int(USE_BTC_FILTER)}"
            if not DRY_RUN:
                send_telegram(msg)
            print("[HB]", msg)
            last_heartbeat = now_utc()

        # BTC filter once per loop
        if not btc_ok(reject_counts):
            if DEBUG_REJECTS:
                print("[REJECT] BTC filter failed:", reject_counts)
            # BTC down ise tüm turu boş geç
            if TEST_ONCE:
                break
            time.sleep(INTERVAL_SEC)
            continue

        symbols = pick_symbols()
        if DEBUG:
            print(f"[INFO] universe size={len(symbols)} top1={symbols[0][0] if symbols else '-'}")

        for sym, qv in symbols:
            # cooldown
            if storage:
                last_ts = storage.get_last_signal_ts(sym)
                if last_ts and (now_utc() - last_ts) < COOLDOWN_SEC:
                    bump(reject_counts, "COOLDOWN")
                    continue

            # HTF trend
            if not htf_ok(sym, reject_counts):
                continue

            # Entry signal
            if not entry_signal(sym, reject_counts):
                continue

            # signal!
            try:
                px = get_price(sym)
            except Exception:
                px = float("nan")

            text = (
                "🚀 LONG SIGNAL\n"
                f"Symbol: {sym}\n"
                f"TF: {TF_ENTRY} | HTF: {HTF}\n"
                f"Price: {px}\n"
                f"TP: {TP_PCT:.2f}% | SL: {SL_PCT:.2f}%\n"
                f"Liquidity(24h qV): {qv:,.0f}\n"
            )

            print("[SIGNAL]", sym, "price=", px)

            if not DRY_RUN:
                send_telegram(text)

            if storage:
                storage.set_last_signal_ts(sym, now_utc())

        # Reject summary
        if DEBUG_REJECTS:
            top = sorted(reject_counts.items(), key=lambda x: x[1], reverse=True)[:12]
            print("[REJECTS]", top)

        if TEST_ONCE:
            break

        # sleep with loop timing
        elapsed = time.time() - t0
        nap = max(1, INTERVAL_SEC - int(elapsed))
        time.sleep(nap)


if __name__ == "__main__":
    main()
