import os
import time
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests

from notify import send_telegram
from storage import Storage


# =========================
# ENV / AYARLAR
# =========================
BINANCE_FAPI = os.getenv("BINANCE_FAPI", "https://fapi.binance.com").rstrip("/")

# Timeframes
TF_ENTRY = os.getenv("TF_ENTRY", "1h")          # entry TF (TV'de 1h)
HTF = os.getenv("HTF", "4h")                    # trend TF (opsiyonel)
BTC_TF = os.getenv("BTC_TF", "1h")              # BTC filter TF
KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "260"))
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "600"))  # 10 dk tarama

# Universe / likidite
TOP_N = int(os.getenv("TOP_N", "200"))
MIN_QUOTE_VOLUME = float(os.getenv("MIN_QUOTE_VOLUME", "3000000"))  # 24h quoteVolume (USDT)
ONLY_USDT_PERP = int(os.getenv("ONLY_USDT_PERP", "1"))  # 1 = sadece USDT perpetual

# EMA cross
EMA_FAST = int(os.getenv("EMA_FAST", "3"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "44"))

# RSI
RSI_LEN = int(os.getenv("RSI_LEN", "21"))
RSI_MIN = float(os.getenv("RSI_MIN", "42"))

# MFI
USE_MFI_FILTER = int(os.getenv("USE_MFI_FILTER", "1"))
MFI_LEN = int(os.getenv("MFI_LEN", "14"))
MFI_LONG_MIN = float(os.getenv("MFI_LONG_MIN", "40"))
MFI_LONG_MAX = float(os.getenv("MFI_LONG_MAX", "85"))

# Volume spike filter (son bar vs ortalama)
USE_VOL_FILTER = int(os.getenv("USE_VOL_FILTER", "1"))
VOL_LEN = int(os.getenv("VOL_LEN", "20"))
VOL_MULT = float(os.getenv("VOL_MULT", "1.1"))
VOL_USE_QUOTE = int(os.getenv("VOL_USE_QUOTE", "1"))  # 1=vol*close, 0=raw volume

# Filters
USE_BTC_FILTER = int(os.getenv("USE_BTC_FILTER", "1"))
BTC_SYMBOL = os.getenv("BTC_SYMBOL", "BTCUSDT")

USE_HTF_FILTER = int(os.getenv("USE_HTF_FILTER", "1"))
HTF_STRICT_CROSS = int(os.getenv("HTF_STRICT_CROSS", "0"))  # 0=trend (EMAfast>=EMAslow), 1=HTF'de cross şart

# Signal behavior
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", "21600"))  # 6 saat
USE_STORAGE = int(os.getenv("USE_STORAGE", "1"))
STORAGE_PATH = os.getenv("STORAGE_PATH", "/var/data/futures_state.json")

# Telegram
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "").strip()
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "").strip()

# Debug
DEBUG = int(os.getenv("DEBUG", "0"))
DEBUG_REJECTS = int(os.getenv("DEBUG_REJECTS", "0"))
TEST_ONCE = int(os.getenv("TEST_ONCE", "0"))

# Long-only
LONG_ONLY = int(os.getenv("LONG_ONLY", "1"))

# Take profit suggestions (mesaj için)
TP1_PCT = float(os.getenv("TP1_PCT", "8"))
TP2_PCT = float(os.getenv("TP2_PCT", "12"))
TP3_PCT = float(os.getenv("TP3_PCT", "15"))
SL_PCT_SUGGEST = float(os.getenv("SL_PCT_SUGGEST", "4"))

# HTTP
TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "12"))
SESSION = requests.Session()


# =========================
# Binance API
# =========================
def _get_json(path: str, params: Optional[dict] = None) -> dict:
    url = f"{BINANCE_FAPI}{path}"
    r = SESSION.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_exchange_info() -> dict:
    return _get_json("/fapi/v1/exchangeInfo")


def get_24h_tickers() -> List[dict]:
    # /fapi/v1/ticker/24hr returns list
    return _get_json("/fapi/v1/ticker/24hr")


def get_klines(symbol: str, interval: str, limit: int) -> List[List]:
    return _get_json("/fapi/v1/klines", params={"symbol": symbol, "interval": interval, "limit": limit})


# =========================
# Indicator helpers
# =========================
def ema(series: List[float], length: int) -> List[float]:
    if length <= 1:
        return series[:]
    k = 2.0 / (length + 1.0)
    out = []
    prev = series[0]
    out.append(prev)
    for x in series[1:]:
        prev = (x * k) + (prev * (1 - k))
        out.append(prev)
    return out


def rsi_wilder(closes: List[float], length: int) -> List[float]:
    if len(closes) < length + 2:
        return [50.0] * len(closes)
    gains = [0.0]
    losses = [0.0]
    for i in range(1, len(closes)):
        ch = closes[i] - closes[i - 1]
        gains.append(max(ch, 0.0))
        losses.append(max(-ch, 0.0))

    avg_gain = sum(gains[1:length + 1]) / length
    avg_loss = sum(losses[1:length + 1]) / length

    out = [50.0] * len(closes)
    # First RSI value at index length
    rs = (avg_gain / avg_loss) if avg_loss > 0 else float("inf")
    out[length] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(length + 1, len(closes)):
        avg_gain = ((avg_gain * (length - 1)) + gains[i]) / length
        avg_loss = ((avg_loss * (length - 1)) + losses[i]) / length
        rs = (avg_gain / avg_loss) if avg_loss > 0 else float("inf")
        out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out


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


def vol_spike_ok(closes: List[float], volumes: List[float], length: int, mult: float, use_quote: int) -> bool:
    if len(closes) < length + 2:
        return False
    vals = []
    for i in range(-length - 1, -1):
        v = volumes[i]
        if use_quote:
            v = v * closes[i]
        vals.append(v)
    avg = sum(vals) / len(vals) if vals else 0.0

    last_v = volumes[-2]
    if use_quote:
        last_v = last_v * closes[-2]

    if avg <= 0:
        return False
    return last_v >= (avg * mult)


def is_cross_up(fast: List[float], slow: List[float]) -> bool:
    # closed candle logic: use [-3] -> [-2]
    if len(fast) < 3 or len(slow) < 3:
        return False
    return (fast[-3] <= slow[-3]) and (fast[-2] > slow[-2])


def is_cross_down(fast: List[float], slow: List[float]) -> bool:
    if len(fast) < 3 or len(slow) < 3:
        return False
    return (fast[-3] >= slow[-3]) and (fast[-2] < slow[-2])


def fmt_time(ts_ms: int) -> str:
    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M UTC")


# =========================
# Symbol universe
# =========================
def build_tradeable_symbols() -> List[str]:
    info = get_exchange_info()
    symbols = []
    for s in info.get("symbols", []):
        if s.get("status") != "TRADING":
            continue
        if ONLY_USDT_PERP:
            if s.get("quoteAsset") != "USDT":
                continue
            if s.get("contractType") != "PERPETUAL":
                continue
        symbols.append(s["symbol"])
    return symbols


def top_by_quote_volume(all_symbols: List[str]) -> List[Tuple[str, float]]:
    tickers = get_24h_tickers()
    m: Dict[str, float] = {}
    for t in tickers:
        sym = t.get("symbol")
        if sym in all_symbols:
            try:
                qv = float(t.get("quoteVolume", "0"))
            except Exception:
                qv = 0.0
            m[sym] = qv

    ranked = sorted(m.items(), key=lambda x: x[1], reverse=True)
    # filter by MIN_QUOTE_VOLUME first
    ranked = [(s, qv) for (s, qv) in ranked if qv >= MIN_QUOTE_VOLUME]
    return ranked[:TOP_N]


# =========================
# Strategy evaluation
# =========================
def load_ohlcv(symbol: str, tf: str, limit: int) -> Optional[Dict[str, List[float]]]:
    try:
        kl = get_klines(symbol, tf, limit)
    except Exception as e:
        if DEBUG:
            print(f"[ERR] klines {symbol} {tf}: {e}")
        return None

    # Binance kline: [ openTime, open, high, low, close, volume, closeTime, quoteAssetVolume, ... ]
    o = []
    h = []
    l = []
    c = []
    v = []
    ot = []
    for row in kl:
        ot.append(int(row[0]))
        o.append(float(row[1]))
        h.append(float(row[2]))
        l.append(float(row[3]))
        c.append(float(row[4]))
        v.append(float(row[5]))
    return {"open_time": ot, "open": o, "high": h, "low": l, "close": c, "volume": v}


def trend_ok_htf(symbol: str, reject: Dict[str, int]) -> bool:
    if not USE_HTF_FILTER:
        return True
    data = load_ohlcv(symbol, HTF, KLINE_LIMIT)
    if not data:
        reject["HTF_NO_DATA"] += 1
        return False

    closes = data["close"]
    ef = ema(closes, EMA_FAST)
    es = ema(closes, EMA_SLOW)

    if HTF_STRICT_CROSS:
        ok = is_cross_up(ef, es)
        if not ok:
            reject["HTF_NO_CROSS"] += 1
        return ok
    else:
        ok = ef[-2] >= es[-2]
        if not ok:
            reject["HTF_TREND_DOWN"] += 1
        return ok


def btc_ok(reject: Dict[str, int]) -> bool:
    if not USE_BTC_FILTER:
        return True
    data = load_ohlcv(BTC_SYMBOL, BTC_TF, KLINE_LIMIT)
    if not data:
        reject["BTC_NO_DATA"] += 1
        return False
    closes = data["close"]
    ef = ema(closes, EMA_FAST)
    es = ema(closes, EMA_SLOW)
    ok = ef[-2] >= es[-2]
    if not ok:
        reject["BTC_TREND_DOWN"] += 1
    return ok


def entry_signal(symbol: str) -> Tuple[Optional[dict], Dict[str, int]]:
    reject = {
    "NO_DATA": 0,
    "NO_CROSS": 0,
    "RSI_LOW": 0,
    "MFI_OUT": 0,
    "VOL_LOW": 0,
    "HTF_FAIL": 0,
    "BTC_FAIL": 0,
    "SHORT_DISABLED": 0,

    # missing keys fix
    "BTC_TREND_DOWN": 0,
    "BTC_NO_DATA": 0,
    "HTF_NO_DATA": 0,
    "HTF_NO_CROSS": 0,
    "HTF_TREND_DOWN": 0,
}

    data = load_ohlcv(symbol, TF_ENTRY, KLINE_LIMIT)
    if not data:
        reject["NO_DATA"] += 1
        return None, reject

    closes = data["close"]
    highs = data["high"]
    lows = data["low"]
    vols = data["volume"]
    ot = data["open_time"]

    ef = ema(closes, EMA_FAST)
    es = ema(closes, EMA_SLOW)

    # Direction decision based on cross
    direction = None
    if is_cross_up(ef, es):
        direction = "LONG"
    elif is_cross_down(ef, es):
        direction = "SHORT"

    if direction is None:
        reject["NO_CROSS"] += 1
        return None, reject

    if LONG_ONLY and direction == "SHORT":
        reject["SHORT_DISABLED"] += 1
        return None, reject

    # BTC filter
    if not btc_ok(reject):
        reject["BTC_FAIL"] += 1
        return None, reject

    # HTF trend filter
    if not trend_ok_htf(symbol, reject):
        reject["HTF_FAIL"] += 1
        return None, reject

    # RSI
    rsi = rsi_wilder(closes, RSI_LEN)
    if rsi[-2] < RSI_MIN:
        reject["RSI_LOW"] += 1
        return None, reject

    # MFI
    if USE_MFI_FILTER:
        m = mfi(highs, lows, closes, vols, MFI_LEN)
        mfi_last = m[-2]
        if not (MFI_LONG_MIN <= mfi_last <= MFI_LONG_MAX):
            reject["MFI_OUT"] += 1
            return None, reject
    else:
        mfi_last = None

    # Volume spike
    if USE_VOL_FILTER:
        if not vol_spike_ok(closes, vols, VOL_LEN, VOL_MULT, VOL_USE_QUOTE):
            reject["VOL_LOW"] += 1
            return None, reject

    # Build signal payload (use closed candle close price)
    price = closes[-2]
    candle_time = ot[-2]
    payload = {
        "symbol": symbol,
        "tf": TF_ENTRY,
        "direction": direction,
        "price": price,
        "time": candle_time,
        "ema_fast": ef[-2],
        "ema_slow": es[-2],
        "rsi": rsi[-2],
        "mfi": mfi_last,
    }
    return payload, reject


# =========================
# Main loop
# =========================
def make_message(sig: dict) -> str:
    sym = sig["symbol"]
    tf = sig["tf"]
    direction = sig["direction"]
    price = sig["price"]
    t = fmt_time(sig["time"])

    rsi_v = sig.get("rsi")
    mfi_v = sig.get("mfi")

    lines = []
    lines.append("🚀 *LONG SIGNAL*")
    lines.append(f"*Symbol:* `{sym}`")
    lines.append(f"*TF:* `{tf}`")
    lines.append(f"*Time:* `{t}`")
    lines.append(f"*Price (close):* `{price:.8f}`")
    lines.append(f"*EMA{EMA_FAST} / EMA{EMA_SLOW}:* `{sig['ema_fast']:.8f}` / `{sig['ema_slow']:.8f}`")
    lines.append(f"*RSI({RSI_LEN}):* `{rsi_v:.2f}`")
    if mfi_v is not None:
        lines.append(f"*MFI({MFI_LEN}):* `{mfi_v:.2f}`")

    # Swing plan suggestion
    lines.append("")
    lines.append("📌 *Manual Swing Plan (suggestion)*")
    lines.append(f"• TP1: `+{TP1_PCT:.1f}%`  | TP2: `+{TP2_PCT:.1f}%` | TP3: `+{TP3_PCT:.1f}%`")
    lines.append(f"• SL (suggest): `-{SL_PCT_SUGGEST:.1f}%`  (coin volatilitesine göre ayarla)")
    lines.append("")
    lines.append("#scanner")

    return "\n".join(lines)


def main():
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        print("[FATAL] TG_BOT_TOKEN / TG_CHAT_ID eksik.")
        return

    storage = Storage(STORAGE_PATH) if USE_STORAGE else None

    print("[BOOT] Futures scanner started")
    print(f"[CFG] TF_ENTRY={TF_ENTRY} HTF={HTF} EMA={EMA_FAST}/{EMA_SLOW} RSI_LEN={RSI_LEN} RSI_MIN={RSI_MIN}")
    print(f"[CFG] MFI={USE_MFI_FILTER} VOL={USE_VOL_FILTER} BTC_FILTER={USE_BTC_FILTER} HTF_FILTER={USE_HTF_FILTER}")
    print(f"[CFG] TOP_N={TOP_N} MIN_QUOTE_VOLUME={MIN_QUOTE_VOLUME} COOLDOWN_SEC={COOLDOWN_SEC} LONG_ONLY={LONG_ONLY}")

    all_syms = build_tradeable_symbols()

    while True:
        started = time.time()

        reject_counts: Dict[str, int] = {}
        def bump_rejects(local: Dict[str, int]):
            for k, v in local.items():
                reject_counts[k] = reject_counts.get(k, 0) + v

        # global filter first
        if not btc_ok(reject_counts):
            if DEBUG:
                print("[INFO] BTC filter failed, skipping this cycle.")
            time.sleep(INTERVAL_SEC)
            if TEST_ONCE:
                break
            continue

        ranked = top_by_quote_volume(all_syms)
        symbols = [s for (s, qv) in ranked]

        if DEBUG:
            print(f"[LOOP] scanning {len(symbols)} symbols (top by quote volume)")

        sent = 0
        for sym in symbols:
            # cooldown check
            if storage and storage.is_cooldown(sym, COOLDOWN_SEC):
                continue

            sig, rej = entry_signal(sym)
            bump_rejects(rej)

            if not sig:
                continue

            msg = make_message(sig)
            ok = send_telegram(TG_BOT_TOKEN, TG_CHAT_ID, msg)
            if ok:
                sent += 1
                if storage:
                    storage.mark_sent(sym)

                if DEBUG:
                    print(f"[SENT] {sym} @ {sig['price']:.8f}")

        if DEBUG or DEBUG_REJECTS:
            dur = time.time() - started
            print(f"[STATS] sent={sent} cycle_sec={dur:.2f}")
            if DEBUG_REJECTS:
                # Print top reject reasons
                top = sorted(reject_counts.items(), key=lambda x: x[1], reverse=True)
                print("[REJECTS]")
                for k, v in top[:15]:
                    print(f"  - {k}: {v}")

        if TEST_ONCE:
            break

        # sleep
        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main()
