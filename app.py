import os
import time
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import requests

from notify import send_telegram
from storage import Storage

# =========================
# ENV / CONFIG
# =========================
BINANCE_FAPI = os.getenv("BINANCE_FAPI", "https://fapi.binance.com").rstrip("/")

# Timeframes
TF_ENTRY = os.getenv("TF_ENTRY", os.getenv("TF", "1h"))  # entry tf
HTF = os.getenv("HTF", os.getenv("TF_TREND", "4h"))      # higher tf trend

INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "600"))
KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "260"))

# Signal logic
EMA_FAST = int(os.getenv("EMA_FAST", "3"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "44"))
LOOKBACK = int(os.getenv("HTF_CROSS_LOOKBACK", os.getenv("LOOKBACK", "6")))

RSI_LEN = int(os.getenv("RSI_LEN", "21"))
RSI_MIN = float(os.getenv("RSI_MIN", "42"))

# Stoch RSI
STOCH_RSI_LEN = int(os.getenv("STOCH_RSI_LEN", "14"))
STOCH_K = int(os.getenv("STOCH_K", "5"))
STOCH_D = int(os.getenv("STOCH_D", "5"))

# WaveTrend (LazyBear style)
WT_CH_LEN = int(os.getenv("WT_CH_LEN", "9"))     # Channel Length
WT_AVG_LEN = int(os.getenv("WT_AVG_LEN", "12"))  # Average Length
WT_OB1 = float(os.getenv("WT_OB1", "60"))
WT_OB2 = float(os.getenv("WT_OB2", "53"))
WT_OS1 = float(os.getenv("WT_OS1", "-60"))
WT_OS2 = float(os.getenv("WT_OS2", "-53"))

# Optional Filters (0/1)
USE_WT = int(os.getenv("USE_WT", os.getenv("USE_WT_FILTER", "1"))) == 1
USE_STOCH_RSI = int(os.getenv("USE_STOCH_RSI", "1")) == 1
USE_HTF_FILTER = int(os.getenv("USE_HTF_FILTER", "0")) == 1
USE_BTC_FILTER = int(os.getenv("USE_BTC_FILTER", "0")) == 1
USE_VOL_FILTER = int(os.getenv("USE_VOL_FILTER", "0")) == 1
USE_MFI_FILTER = int(os.getenv("USE_MFI_FILTER", "0")) == 1  # mevcut ama kapalı default

# Volume filter
VOL_LEN = int(os.getenv("VOL_LEN", "20"))
VOL_MULT = float(os.getenv("VOL_MULT", "1.1"))
MIN_QUOTE_VOLUME = float(os.getenv("MIN_QUOTE_VOLUME", "3000000"))

# Universe selection
TOP_N = int(os.getenv("TOP_N", "200"))
VOL_USE_QUOTE = int(os.getenv("VOL_USE_QUOTE", "1")) == 1

# Risk / ops
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", "21600"))  # 6h default
LONG_ONLY = int(os.getenv("LONG_ONLY", "1")) == 1
DRY_RUN = int(os.getenv("DRY_RUN", "0")) == 1
TEST_ONCE = int(os.getenv("TEST_ONCE", "0")) == 1
DEBUG = int(os.getenv("DEBUG", "1")) == 1
DEBUG_REJECTS = int(os.getenv("DEBUG_REJECTS", "0")) == 1
HEARTBEAT_SEC = int(os.getenv("HEARTBEAT_SEC", "900"))

# Storage / telegram
STORAGE_PATH = os.getenv("STORAGE_PATH", "/var/data/futures_state.json")
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "")

# BTC filter params
BTC_SYMBOL = os.getenv("BTC_SYMBOL", "BTCUSDT")
BTC_TF = os.getenv("BTC_TF", HTF)  # trend tf for btc filter
EMA_TREND = int(os.getenv("EMA_TREND", "123"))

# =========================
# Data structures
# =========================
@dataclass
class Series:
    ts: List[int]
    opens: List[float]
    highs: List[float]
    lows: List[float]
    closes: List[float]
    volumes: List[float]

# =========================
# Binance helpers
# =========================
def _get(url: str, params: Optional[dict] = None, timeout: int = 15):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def get_exchange_info() -> dict:
    return _get(f"{BINANCE_FAPI}/fapi/v1/exchangeInfo")

def get_24h_tickers() -> list:
    return _get(f"{BINANCE_FAPI}/fapi/v1/ticker/24hr")

def get_klines(symbol: str, interval: str, limit: int) -> Series:
    data = _get(f"{BINANCE_FAPI}/fapi/v1/klines", params={"symbol": symbol, "interval": interval, "limit": limit})
    ts, o, h, l, c, v = [], [], [], [], [], []
    for k in data:
        ts.append(int(k[0]))
        o.append(float(k[1]))
        h.append(float(k[2]))
        l.append(float(k[3]))
        c.append(float(k[4]))
        v.append(float(k[5]))
    return Series(ts=ts, opens=o, highs=h, lows=l, closes=c, volumes=v)

def get_usdt_perp_symbols_top_by_volume(top_n: int, min_quote_vol: float) -> List[str]:
    info = get_exchange_info()
    allowed = set()
    for s in info.get("symbols", []):
        if s.get("contractType") != "PERPETUAL":
            continue
        if s.get("quoteAsset") != "USDT":
            continue
        if s.get("status") != "TRADING":
            continue
        sym = s.get("symbol")
        if sym:
            allowed.add(sym)

    tickers = get_24h_tickers()
    rows = []
    for t in tickers:
        sym = t.get("symbol")
        if sym not in allowed:
            continue
        try:
            quote_vol = float(t.get("quoteVolume", 0.0))
        except Exception:
            quote_vol = 0.0
        if quote_vol < min_quote_vol:
            continue
        rows.append((sym, quote_vol))

    rows.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in rows[:top_n]]

# =========================
# Indicators
# =========================
def _ema(values: List[float], length: int) -> List[float]:
    if length <= 1:
        return values[:]
    k = 2 / (length + 1.0)
    out = []
    ema = values[0]
    out.append(ema)
    for x in values[1:]:
        ema = x * k + ema * (1 - k)
        out.append(ema)
    return out

def _sma(values: List[float], length: int) -> List[float]:
    out = []
    s = 0.0
    q = []
    for x in values:
        q.append(x)
        s += x
        if len(q) > length:
            s -= q.pop(0)
        if len(q) < length:
            out.append(sum(q) / len(q))
        else:
            out.append(s / length)
    return out

def _rsi(closes: List[float], length: int) -> List[float]:
    if len(closes) < 2:
        return [50.0] * len(closes)
    gains = [0.0]
    losses = [0.0]
    for i in range(1, len(closes)):
        ch = closes[i] - closes[i - 1]
        gains.append(max(ch, 0.0))
        losses.append(max(-ch, 0.0))
    avg_g = _ema(gains, length)
    avg_l = _ema(losses, length)
    out = []
    for g, l in zip(avg_g, avg_l):
        if l == 0:
            out.append(100.0)
        else:
            rs = g / l
            out.append(100.0 - (100.0 / (1.0 + rs)))
    return out

def _stoch_rsi(closes: List[float], rsi_len: int = None, stoch_len: int = None, k_len: int = None, d_len: int = None) -> Tuple[List[float], List[float]]:
    # defaults from ENV
    rsi_len = rsi_len or RSI_LEN
    stoch_len = stoch_len or STOCH_RSI_LEN
    k_len = k_len or STOCH_K
    d_len = d_len or STOCH_D

    rsi = _rsi(closes, rsi_len)
    k_raw = []
    for i in range(len(rsi)):
        start = max(0, i - stoch_len + 1)
        window = rsi[start:i + 1]
        lo = min(window)
        hi = max(window)
        if hi - lo == 0:
            k_raw.append(0.0)
        else:
            k_raw.append(100.0 * (rsi[i] - lo) / (hi - lo))
    k = _sma(k_raw, k_len)
    d = _sma(k, d_len)
    return k, d

def _wavetrend_lb(highs: List[float], lows: List[float], closes: List[float]) -> Tuple[List[float], List[float]]:
    # LazyBear-ish WaveTrend using typical price + EMA smoothing
    # Inputs: WT_CH_LEN (n1), WT_AVG_LEN (n2)
    ap = [(h + l + c) / 3.0 for h, l, c in zip(highs, lows, closes)]
    esa = _ema(ap, WT_CH_LEN)
    d = [abs(a - e) for a, e in zip(ap, esa)]
    de = _ema(d, WT_CH_LEN)
    ci = []
    for a, e, denom in zip(ap, esa, de):
        if denom == 0:
            ci.append(0.0)
        else:
            ci.append((a - e) / (0.015 * denom))
    tci = _ema(ci, WT_AVG_LEN)
    wt1 = tci
    wt2 = _sma(wt1, 4)  # signal line (common setting)
    return wt1, wt2

def _cross_up(a: List[float], b: List[float], lookback: int = 1) -> bool:
    # any cross up within last lookback bars
    n = len(a)
    if n < 2:
        return False
    start = max(1, n - lookback)
    for i in range(start, n):
        if a[i - 1] <= b[i - 1] and a[i] > b[i]:
            return True
    return False

def _cross_down(a: List[float], b: List[float], lookback: int = 1) -> bool:
    n = len(a)
    if n < 2:
        return False
    start = max(1, n - lookback)
    for i in range(start, n):
        if a[i - 1] >= b[i - 1] and a[i] < b[i]:
            return True
    return False

# =========================
# Filters / Conditions
# =========================
def ema_cross_ok(series: Series, reject: Dict[str, int]) -> bool:
    ema_f = _ema(series.closes, EMA_FAST)
    ema_s = _ema(series.closes, EMA_SLOW)

    # cross within last LOOKBACK bars (on TF_ENTRY)
    crossed = False
    for i in range(max(1, len(ema_f) - LOOKBACK), len(ema_f)):
        if ema_f[i - 1] <= ema_s[i - 1] and ema_f[i] > ema_s[i]:
            crossed = True
            break

    if not crossed:
        reject["EMA_CROSS"] = reject.get("EMA_CROSS", 0) + 1
        return False

    return True

def rsi_ok(series: Series, reject: Dict[str, int]) -> bool:
    rsi = _rsi(series.closes, RSI_LEN)
    if rsi[-1] < RSI_MIN:
        reject["RSI_MIN"] = reject.get("RSI_MIN", 0) + 1
        return False
    return True

def stoch_ok(series: Series, reject: Dict[str, int]) -> bool:
    if not USE_STOCH_RSI:
        return True
    k, d = _stoch_rsi(series.closes, RSI_LEN, STOCH_RSI_LEN, STOCH_K, STOCH_D)
    if not (k[-1] > d[-1]):
        reject["STOCH_KD"] = reject.get("STOCH_KD", 0) + 1
        return False
    return True

def vol_spike_ok(series: Series, reject: Dict[str, int]) -> bool:
    if not USE_VOL_FILTER:
        return True
    if len(series.volumes) < VOL_LEN + 2:
        return True

    v = series.volumes
    avg = sum(v[-(VOL_LEN + 1):-1]) / VOL_LEN
    cur = v[-1]
    if avg <= 0:
        return True
    if cur < avg * VOL_MULT:
        reject["VOL_SPIKE"] = reject.get("VOL_SPIKE", 0) + 1
        return False
    return True

def htf_trend_ok(symbol: str, reject: Dict[str, int]) -> bool:
    if not USE_HTF_FILTER:
        return True
    try:
        s = get_klines(symbol, HTF, min(KLINE_LIMIT, 200))
    except Exception:
        reject["HTF_FETCH"] = reject.get("HTF_FETCH", 0) + 1
        return False

    ema_f = _ema(s.closes, EMA_FAST)
    ema_s = _ema(s.closes, EMA_SLOW)
    # trend ok if fast > slow currently
    if not (ema_f[-1] > ema_s[-1]):
        reject["HTF_TREND"] = reject.get("HTF_TREND", 0) + 1
        return False
    return True

def btc_ok(reject: Dict[str, int]) -> bool:
    if not USE_BTC_FILTER:
        return True
    try:
        s = get_klines(BTC_SYMBOL, BTC_TF, min(KLINE_LIMIT, 200))
    except Exception:
        reject["BTC_FETCH"] = reject.get("BTC_FETCH", 0) + 1
        return False

    ema_t = _ema(s.closes, EMA_TREND)
    # BTC trend ok if price above EMA_TREND
    if not (s.closes[-1] >= ema_t[-1]):
        reject["BTC_TREND_DOWN"] = reject.get("BTC_TREND_DOWN", 0) + 1
        return False
    return True

def wt_confirm_ok(series: Series, reject: Dict[str, int]) -> bool:
    """
    HYBRID WT:
    A) Dip reversal (priority):
       - WT1 dipped <= WT_OS2 within last 5 bars
       - WT1 crosses UP WT2 recently
       - WT1 rising
       - Stoch K > D (confirm)
    B) Strong continuation (secondary):
       - WT1 > 0 and WT1 < WT_OB2
       - WT1 > WT2 and rising
       - RSI > 55
       - Stoch K > D
    """
    if not USE_WT:
        return True

    wt1, wt2 = _wavetrend_lb(series.highs, series.lows, series.closes)
    rsi = _rsi(series.closes, RSI_LEN)
    k, d = _stoch_rsi(series.closes, RSI_LEN, STOCH_RSI_LEN, STOCH_K, STOCH_D)

    rising = wt1[-1] > wt1[-2]
    cross_up_recent = _cross_up(wt1, wt2, lookback=2)

    # A) Dip reversal
    dip_zone = min(wt1[-5:]) <= WT_OS2
    dip_reversal = dip_zone and cross_up_recent and rising and (k[-1] > d[-1])

    # B) Continuation (but not overbought)
    continuation = (
        wt1[-1] > 0
        and wt1[-1] < WT_OB2
        and wt1[-1] > wt2[-1]
        and rising
        and rsi[-1] > 55
        and (k[-1] > d[-1])
    )

    if dip_reversal or continuation:
        return True

    reject["WT_CONFIRM"] = reject.get("WT_CONFIRM", 0) + 1
    return False

# =========================
# Message formatting
# =========================
def build_signal_message(symbol: str, series: Series) -> str:
    price = series.closes[-1]
    ema_f = _ema(series.closes, EMA_FAST)[-1]
    ema_s = _ema(series.closes, EMA_SLOW)[-1]
    rsi = _rsi(series.closes, RSI_LEN)[-1]
    k, d = _stoch_rsi(series.closes, RSI_LEN, STOCH_RSI_LEN, STOCH_K, STOCH_D)
    wt1, wt2 = _wavetrend_lb(series.highs, series.lows, series.closes)

    msg = []
    msg.append("🚀 LONG SIGNAL")
    msg.append(f"Symbol: {symbol}")
    msg.append(f"TF: {TF_ENTRY} | HTF: {HTF}")
    msg.append(f"Price: {price:.6g}")
    msg.append("")
    msg.append(f"EMA{EMA_FAST}: {ema_f:.6g} | EMA{EMA_SLOW}: {ema_s:.6g}")
    msg.append(f"RSI({RSI_LEN}): {rsi:.2f}")
    msg.append(f"StochRSI K/D (K={STOCH_K},D={STOCH_D}): {k[-1]:.2f}/{d[-1]:.2f}")
    msg.append(f"WT_LB (ch={WT_CH_LEN},avg={WT_AVG_LEN}) WT1/WT2: {wt1[-1]:.2f}/{wt2[-1]:.2f}")
    msg.append("")
    msg.append("Exit plan (manual):")
    msg.append("- TP1: +8.0% (suggestion)")
    msg.append("- SL:  -2.0% (suggestion)")
    msg.append(f"- WT exit: if WT1 crosses DOWN WT2 while WT1>{WT_OB2:.0f} consider close/trim")
    msg.append(f"- WT warning: if WT1>{WT_OB1:.0f} and turns down -> tighten stop")
    return "\n".join(msg)

def log_cfg():
    print(f"[BOOT] Futures scanner started")
    print(f"[CFG] TF_ENTRY={TF_ENTRY} HTF={HTF} EMA={EMA_FAST}/{EMA_SLOW} LOOKBACK={LOOKBACK}")
    print(f"[CFG] RSI_LEN={RSI_LEN} RSI_MIN={RSI_MIN} MFI={1 if USE_MFI_FILTER else 0} VOL={1 if USE_VOL_FILTER else 0} WT={1 if USE_WT else 0} STOCH_RSI={1 if USE_STOCH_RSI else 0}")
    print(f"[CFG] TOP_N={TOP_N} MIN_QUOTE_VOLUME={MIN_QUOTE_VOLUME} COOLDOWN_SEC={COOLDOWN_SEC} LONG_ONLY={1 if LONG_ONLY else 0} DRY_RUN={1 if DRY_RUN else 0}")

def heartbeat(storage: Storage):
    if HEARTBEAT_SEC <= 0:
        return
    key = "_last_heartbeat"
    now = int(time.time())
    last = storage.get_int(key, 0)
    if now - last >= HEARTBEAT_SEC:
        storage.set_int(key, now)
        try:
            send_telegram(f"✅ worker alive | TF={TF_ENTRY} HTF={HTF} TOP_N={TOP_N} BTC_FILTER={1 if USE_BTC_FILTER else 0}")
        except Exception:
            pass

# =========================
# Main loop
# =========================
def should_send(symbol: str, storage: Storage) -> bool:
    now = int(time.time())
    last = storage.get_int(f"last_sent:{symbol}", 0)
    return (now - last) >= COOLDOWN_SEC

def mark_sent(symbol: str, storage: Storage):
    storage.set_int(f"last_sent:{symbol}", int(time.time()))

def main():
    log_cfg()

    storage = Storage(STORAGE_PATH, enabled=True)

    if TG_BOT_TOKEN and TG_CHAT_ID:
        # set notify module env usage
        pass

    # loop
    while True:
        start_ts = time.time()
        reject_counts: Dict[str, int] = {}

        # optional btc filter
        if not btc_ok(reject_counts):
            if DEBUG_REJECTS:
                print("[REJECT] BTC filter failed:", reject_counts)
            time.sleep(INTERVAL_SEC)
            continue

        symbols = []
        try:
            symbols = get_usdt_perp_symbols_top_by_volume(TOP_N, MIN_QUOTE_VOLUME)
        except Exception as e:
            print("[ERR] Failed to fetch symbols:", str(e))
            time.sleep(INTERVAL_SEC)
            continue

        sent = 0
        scanned = 0

        for sym in symbols:
            scanned += 1

            if not should_send(sym, storage):
                continue

            try:
                s = get_klines(sym, TF_ENTRY, KLINE_LIMIT)
            except Exception:
                reject_counts["KLINES_FETCH"] = reject_counts.get("KLINES_FETCH", 0) + 1
                continue

            # core conditions
            ok = True
            if not ema_cross_ok(s, reject_counts):
                ok = False
            elif not rsi_ok(s, reject_counts):
                ok = False
            elif not stoch_ok(s, reject_counts):
                ok = False
            elif not wt_confirm_ok(s, reject_counts):
                ok = False
            elif not vol_spike_ok(s, reject_counts):
                ok = False
            elif not htf_trend_ok(sym, reject_counts):
                ok = False

            if not ok:
                continue

            # send
            msg = build_signal_message(sym, s)
            if DRY_RUN:
                print("[DRY_RUN] would send:\n", msg)
                mark_sent(sym, storage)
                sent += 1
                continue

            try:
                send_telegram(msg)
                mark_sent(sym, storage)
                sent += 1
            except Exception as e:
                print("[ERR] telegram send failed:", str(e))

        heartbeat(storage)

        if DEBUG:
            dt = time.time() - start_ts
            top_rejects = sorted(reject_counts.items(), key=lambda x: x[1], reverse=True)[:8]
            print(f"[LOOP] scanned={scanned} sent={sent} dt={dt:.1f}s rejects={dict(top_rejects)}")

        if TEST_ONCE:
            break

        time.sleep(INTERVAL_SEC)

if __name__ == "__main__":
    main()
