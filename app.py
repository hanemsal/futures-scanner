import os
import time
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import requests

from notify import send_telegram
from storage import Storage

BINANCE_FAPI = os.getenv("BINANCE_FAPI", "https://fapi.binance.com").rstrip("/")

# =========================
# ENV (defaults)
# =========================
TF_ENTRY = os.getenv("TF_ENTRY", os.getenv("TF", "1h"))          # entry TF (TradingView: 1h)
HTF = os.getenv("HTF", os.getenv("TF_TREND", "4h"))              # trend TF (TradingView: 4h)
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "600"))
KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "260"))

# EMA
EMA_FAST = int(os.getenv("EMA_FAST", "3"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "44"))
EMA_TREND = int(os.getenv("EMA_TREND", "123"))
HTF_CROSS_LOOKBACK = int(os.getenv("HTF_CROSS_LOOKBACK", "6"))   # last N bars on entry TF

# RSI
RSI_LEN = int(os.getenv("RSI_LEN", "21"))
RSI_MIN = float(os.getenv("RSI_MIN", "42"))

# MFI (optional)
USE_MFI_FILTER = int(os.getenv("USE_MFI_FILTER", "1"))
MFI_LEN = int(os.getenv("MFI_LEN", "14"))
MFI_LONG_MIN = float(os.getenv("MFI_LONG_MIN", "40"))
MFI_LONG_MAX = float(os.getenv("MFI_LONG_MAX", "85"))
MFI_SLOPE_ENABLE = int(os.getenv("MFI_SLOPE_ENABLE", "1"))
MFI_SLOPE_BARS = int(os.getenv("MFI_SLOPE_BARS", "1"))

# Volume filter (optional)
USE_VOL_FILTER = int(os.getenv("USE_VOL_FILTER", "1"))
VOL_LEN = int(os.getenv("VOL_LEN", "20"))
VOL_MULT = float(os.getenv("VOL_MULT", "1.1"))
VOL_USE_QUOTE = int(os.getenv("VOL_USE_QUOTE", "1"))  # if 1 use quoteVolume, else base volume

# Quote volume pre-filter (USDT)
MIN_QUOTE_VOLUME = float(os.getenv("MIN_QUOTE_VOLUME", "3000000"))

# HTF/BTC filters
USE_HTF_FILTER = int(os.getenv("USE_HTF_FILTER", "1"))
USE_BTC_FILTER = int(os.getenv("USE_BTC_FILTER", "1"))
BTC_SYMBOL = os.getenv("BTC_SYMBOL", "BTCUSDT")
BTC_TF = os.getenv("BTC_TF", HTF)

# Stoch RSI (TradingView: K=5 D=5)
USE_STOCH_RSI = int(os.getenv("USE_STOCH_RSI", "1"))
STOCH_RSI_LEN = int(os.getenv("STOCH_RSI_LEN", "14"))  # TV default 14 unless you change
STOCH_K = int(os.getenv("STOCH_K", "5"))
STOCH_D = int(os.getenv("STOCH_D", "5"))
STOCH_OS = float(os.getenv("STOCH_OS", "20"))
STOCH_OB = float(os.getenv("STOCH_OB", "80"))

# WaveTrend (LazyBear WT_LB) TradingView params: channel=9 avg=12, levels 60/53, -60/-53
USE_WT = int(os.getenv("USE_WT", "1"))
WT_CH = int(os.getenv("WT_CH", "9"))
WT_AVG = int(os.getenv("WT_AVG", "12"))
WT_OB1 = float(os.getenv("WT_OB1", "60"))
WT_OB2 = float(os.getenv("WT_OB2", "53"))
WT_OS1 = float(os.getenv("WT_OS1", "-60"))
WT_OS2 = float(os.getenv("WT_OS2", "-53"))

# Cooldown / storage
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", "21600"))  # 6h default
USE_STORAGE = int(os.getenv("USE_STORAGE", "1"))
STORAGE_PATH = os.getenv("STORAGE_PATH", "/var/data/futures_state.json")

# Bot operation
TOP_N = int(os.getenv("TOP_N", "200"))
LONG_ONLY = int(os.getenv("LONG_ONLY", "1"))
DEBUG = int(os.getenv("DEBUG", "1"))
DEBUG_REJECTS = int(os.getenv("DEBUG_REJECTS", "0"))
DRY_RUN = int(os.getenv("DRY_RUN", "0"))
TEST_ONCE = int(os.getenv("TEST_ONCE", "0"))

HEARTBEAT_SEC = int(os.getenv("HEARTBEAT_SEC", "900"))

# Risk/Info shown in message (manual trading)
TP_PCT = float(os.getenv("TP_PCT", "5"))  # default suggestion
SL_PCT = float(os.getenv("SL_PCT", "2"))  # default suggestion


# =========================
# Helpers
# =========================
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _log(msg: str) -> None:
    if DEBUG:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"{ts} {msg}", flush=True)


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _ema(values: List[float], length: int) -> List[float]:
    if length <= 1:
        return values[:]
    out: List[float] = []
    k = 2 / (length + 1)
    ema_prev = values[0]
    out.append(ema_prev)
    for v in values[1:]:
        ema_prev = (v * k) + (ema_prev * (1 - k))
        out.append(ema_prev)
    return out


def _sma(values: List[float], length: int) -> List[float]:
    out = []
    s = 0.0
    q = []
    for v in values:
        q.append(v)
        s += v
        if len(q) > length:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def _rsi(closes: List[float], length: int) -> List[float]:
    # Wilder RSI
    if length <= 1:
        return [50.0] * len(closes)
    gains = [0.0]
    losses = [0.0]
    for i in range(1, len(closes)):
        chg = closes[i] - closes[i - 1]
        gains.append(max(chg, 0.0))
        losses.append(max(-chg, 0.0))

    avg_gain = _sma(gains[:length], length)[-1]
    avg_loss = _sma(losses[:length], length)[-1]

    rsis = [50.0] * len(closes)
    # seed
    for i in range(length, len(closes)):
        if i == length:
            avg_gain = sum(gains[1:length + 1]) / length
            avg_loss = sum(losses[1:length + 1]) / length
        else:
            avg_gain = (avg_gain * (length - 1) + gains[i]) / length
            avg_loss = (avg_loss * (length - 1) + losses[i]) / length

        if avg_loss == 0:
            rsis[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsis[i] = 100 - (100 / (1 + rs))
    return rsis


def _stoch(values: List[float], length: int) -> List[float]:
    out = []
    for i in range(len(values)):
        start = max(0, i - length + 1)
        window = values[start:i + 1]
        lo = min(window)
        hi = max(window)
        if hi - lo == 0:
            out.append(0.0)
        else:
            out.append(100.0 * (values[i] - lo) / (hi - lo))
    return out


def _stoch_rsi(closes: List[float]) -> Tuple[List[float], List[float]]:
    r = _rsi(closes, RSI_LEN)
    st = _stoch(r, STOCH_RSI_LEN)
    k = _sma(st, STOCH_K)
    d = _sma(k, STOCH_D)
    return k, d


def _typical_price(highs: List[float], lows: List[float], closes: List[float]) -> List[float]:
    return [(h + l + c) / 3.0 for h, l, c in zip(highs, lows, closes)]


def _mfi(highs: List[float], lows: List[float], closes: List[float], vols: List[float], length: int) -> List[float]:
    tp = _typical_price(highs, lows, closes)
    raw_money_flow = [tp[i] * vols[i] for i in range(len(tp))]

    pos = [0.0] * len(tp)
    neg = [0.0] * len(tp)
    for i in range(1, len(tp)):
        if tp[i] > tp[i - 1]:
            pos[i] = raw_money_flow[i]
        elif tp[i] < tp[i - 1]:
            neg[i] = raw_money_flow[i]

    out = [50.0] * len(tp)
    for i in range(len(tp)):
        start = max(0, i - length + 1)
        pos_sum = sum(pos[start:i + 1])
        neg_sum = sum(neg[start:i + 1])
        if neg_sum == 0:
            out[i] = 100.0
        else:
            mr = pos_sum / neg_sum
            out[i] = 100.0 - (100.0 / (1.0 + mr))
    return out


def _wavetrend_lb(highs: List[float], lows: List[float], closes: List[float]) -> Tuple[List[float], List[float]]:
    """
    LazyBear WaveTrend Oscillator (WT)
    params: WT_CH (channel length), WT_AVG (average length)
    Produces wt1, wt2 (signal)
    """
    ap = _typical_price(highs, lows, closes)
    esa = _ema(ap, WT_CH)
    abs_diff = [abs(ap[i] - esa[i]) for i in range(len(ap))]
    d = _ema(abs_diff, WT_CH)
    ci = []
    for i in range(len(ap)):
        denom = 0.015 * (d[i] if d[i] != 0 else 1e-9)
        ci.append((ap[i] - esa[i]) / denom)
    wt1 = _ema(ci, WT_AVG)
    wt2 = _sma(wt1, 4)  # LazyBear uses SMA 4 for signal
    return wt1, wt2


def _cross_up(a: List[float], b: List[float], lookback: int = 1) -> bool:
    # crossed above within last lookback bars (inclusive)
    n = len(a)
    lb = min(lookback, n - 1)
    for i in range(n - lb, n):
        if i <= 0:
            continue
        if a[i - 1] <= b[i - 1] and a[i] > b[i]:
            return True
    return False


def _cross_down(a: List[float], b: List[float], lookback: int = 1) -> bool:
    n = len(a)
    lb = min(lookback, n - 1)
    for i in range(n - lb, n):
        if i <= 0:
            continue
        if a[i - 1] >= b[i - 1] and a[i] < b[i]:
            return True
    return False


# =========================
# Binance API
# =========================
def get_exchange_info() -> dict:
    url = f"{BINANCE_FAPI}/fapi/v1/exchangeInfo"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()


def get_24h_tickers() -> List[dict]:
    url = f"{BINANCE_FAPI}/fapi/v1/ticker/24hr"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()


def get_klines(symbol: str, interval: str, limit: int) -> List[list]:
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


@dataclass
class Series:
    opens: List[float]
    highs: List[float]
    lows: List[float]
    closes: List[float]
    vols: List[float]
    quote_vols: List[float]

    @classmethod
    def from_klines(cls, klines: List[list]) -> "Series":
        o, h, l, c, v, qv = [], [], [], [], [], []
        for k in klines:
            o.append(_safe_float(k[1]))
            h.append(_safe_float(k[2]))
            l.append(_safe_float(k[3]))
            c.append(_safe_float(k[4]))
            v.append(_safe_float(k[5]))
            qv.append(_safe_float(k[7]))
        return cls(o, h, l, c, v, qv)


# =========================
# Filters
# =========================
def init_reject_counts() -> Dict[str, int]:
    return {
        "BTC_TREND_DOWN": 0,
        "HTF_TREND_DOWN": 0,
        "MIN_QUOTE_VOLUME": 0,
        "EMA_CROSS": 0,
        "RSI_MIN": 0,
        "MFI_RANGE": 0,
        "MFI_SLOPE": 0,
        "VOL_SPIKE": 0,
        "STOCH_RSI": 0,
        "WT_CONFIRM": 0,
    }


def btc_ok(reject_counts: Dict[str, int]) -> bool:
    if not USE_BTC_FILTER:
        return True
    try:
        kl = get_klines(BTC_SYMBOL, BTC_TF, KLINE_LIMIT)
        s = Series.from_klines(kl)
        ema_tr = _ema(s.closes, EMA_TREND)
        # BTC trend condition: close >= EMA123 (simple)
        if s.closes[-1] < ema_tr[-1]:
            reject_counts["BTC_TREND_DOWN"] += 1
            return False
        return True
    except Exception as e:
        _log(f"[WARN] BTC filter error: {e}. Allowing signals.")
        return True


def htf_ok(symbol: str, reject_counts: Dict[str, int]) -> bool:
    if not USE_HTF_FILTER:
        return True
    kl = get_klines(symbol, HTF, KLINE_LIMIT)
    s = Series.from_klines(kl)
    ema_tr = _ema(s.closes, EMA_TREND)
    if s.closes[-1] < ema_tr[-1]:
        reject_counts["HTF_TREND_DOWN"] += 1
        return False
    return True


def vol_spike_ok(series: Series, reject_counts: Dict[str, int]) -> bool:
    if not USE_VOL_FILTER:
        return True
    vols = series.quote_vols if VOL_USE_QUOTE else series.vols
    if len(vols) < VOL_LEN + 1:
        return True
    base = sum(vols[-(VOL_LEN + 1):-1]) / VOL_LEN
    if base <= 0:
        return True
    if vols[-1] < base * VOL_MULT:
        reject_counts["VOL_SPIKE"] += 1
        return False
    return True


def mfi_ok(series: Series, reject_counts: Dict[str, int]) -> bool:
    if not USE_MFI_FILTER:
        return True
    mfi = _mfi(series.highs, series.lows, series.closes, series.vols, MFI_LEN)
    cur = mfi[-1]
    if not (MFI_LONG_MIN <= cur <= MFI_LONG_MAX):
        reject_counts["MFI_RANGE"] += 1
        return False
    if MFI_SLOPE_ENABLE:
        bars = max(1, MFI_SLOPE_BARS)
        if len(mfi) > bars and mfi[-1] < mfi[-1 - bars]:
            reject_counts["MFI_SLOPE"] += 1
            return False
    return True


def stoch_rsi_ok(series: Series, reject_counts: Dict[str, int]) -> bool:
    if not USE_STOCH_RSI:
        return True
    k, d = _stoch_rsi(series.closes)
    # For long: K > D and preferably K not extremely overbought
    if not (k[-1] > d[-1]):
        reject_counts["STOCH_RSI"] += 1
        return False
    return True


def wt_confirm_ok(series: Series, reject_counts: Dict[str, int]) -> bool:
    if not USE_WT:
        return True
    wt1, wt2 = _wavetrend_lb(series.highs, series.lows, series.closes)

    # Long confirm idea (balanced):
    # - Prefer wt1 rising and wt1 > wt2 OR coming out of oversold (wt1 crossed up wt2 below OS2)
    rising = wt1[-1] > wt1[-2]
    cross_up_recent = _cross_up(wt1, wt2, lookback=2)
    coming_from_os = (min(wt1[-5:]) <= WT_OS2) and cross_up_recent
    if not ((rising and wt1[-1] > wt2[-1]) or coming_from_os):
        reject_counts["WT_CONFIRM"] += 1
        return False
    return True


# =========================
# Signal logic
# =========================
def long_signal(symbol: str, series: Series, reject_counts: Dict[str, int]) -> bool:
    # Pre filters
    if series.quote_vols[-1] < MIN_QUOTE_VOLUME:
        reject_counts["MIN_QUOTE_VOLUME"] += 1
        return False

    ema_fast = _ema(series.closes, EMA_FAST)
    ema_slow = _ema(series.closes, EMA_SLOW)

    # Cross condition: ema_fast crosses above ema_slow in last lookback bars
    if not _cross_up(ema_fast, ema_slow, lookback=HTF_CROSS_LOOKBACK):
        reject_counts["EMA_CROSS"] += 1
        return False

    rsi = _rsi(series.closes, RSI_LEN)
    if rsi[-1] < RSI_MIN:
        reject_counts["RSI_MIN"] += 1
        return False

    if not mfi_ok(series, reject_counts):
        return False

    if not vol_spike_ok(series, reject_counts):
        return False

    if not stoch_rsi_ok(series, reject_counts):
        return False

    if not wt_confirm_ok(series, reject_counts):
        return False

    return True


def build_message(symbol: str, series: Series) -> str:
    price = series.closes[-1]

    # Indicators snapshot
    ema_fast = _ema(series.closes, EMA_FAST)[-1]
    ema_slow = _ema(series.closes, EMA_SLOW)[-1]

    rsi = _rsi(series.closes, RSI_LEN)[-1]
    mfi = _mfi(series.highs, series.lows, series.closes, series.vols, MFI_LEN)[-1] if USE_MFI_FILTER else float("nan")

    k, d = _stoch_rsi(series.closes)
    st_k, st_d = k[-1], d[-1]

    wt1, wt2 = _wavetrend_lb(series.highs, series.lows, series.closes)
    wt1v, wt2v = wt1[-1], wt2[-1]

    # Exit plan (manual)
    # Suggested exit triggers based on WT:
    # - Partial profit if WT enters OB2+ zone and rolls over
    # - Full exit if WT1 crosses below WT2 while above OB2
    exit_plan = (
        f"Exit plan (manual):\n"
        f"- TP1: +{TP_PCT:.1f}% (suggestion)\n"
        f"- SL: -{SL_PCT:.1f}% (suggestion)\n"
        f"- WT exit: if WT1 crosses DOWN WT2 while WT1>{WT_OB2:.0f} consider close/trim\n"
        f"- WT warning: if WT1>{WT_OB1:.0f} and turns down -> tighten stop"
    )

    msg = (
        f"🚀 LONG SIGNAL\n"
        f"Symbol: {symbol}\n"
        f"TF: {TF_ENTRY} | HTF: {HTF}\n"
        f"Price: {price:.8g}\n\n"
        f"EMA{EMA_FAST}: {ema_fast:.8g} | EMA{EMA_SLOW}: {ema_slow:.8g}\n"
        f"RSI({RSI_LEN}): {rsi:.2f}\n"
        f"StochRSI K/D (K={STOCH_K},D={STOCH_D}): {st_k:.2f}/{st_d:.2f}\n"
        f"WT_LB (ch={WT_CH},avg={WT_AVG}) WT1/WT2: {wt1v:.2f}/{wt2v:.2f}\n"
    )
    if USE_MFI_FILTER:
        msg += f"MFI({MFI_LEN}): {mfi:.2f} (range {MFI_LONG_MIN:.0f}-{MFI_LONG_MAX:.0f})\n"

    msg += "\n" + exit_plan
    return msg


# =========================
# Universe selection
# =========================
def tradable_usdt_perps() -> List[str]:
    info = get_exchange_info()
    symbols = []
    for s in info.get("symbols", []):
        if s.get("contractType") != "PERPETUAL":
            continue
        if s.get("quoteAsset") != "USDT":
            continue
        if s.get("status") != "TRADING":
            continue
        symbols.append(s["symbol"])
    return symbols


def top_by_quote_volume(symbols: List[str], top_n: int) -> List[str]:
    tickers = get_24h_tickers()
    vol_map = {}
    for t in tickers:
        sym = t.get("symbol")
        if sym in symbols:
            vol_map[sym] = _safe_float(t.get("quoteVolume", 0))
    ranked = sorted(vol_map.items(), key=lambda x: x[1], reverse=True)
    return [s for s, _ in ranked[:top_n]]


# =========================
# Main loop
# =========================
def main():
    _log("[BOOT] Futures scanner started")
    _log(f"[CFG] TF_ENTRY={TF_ENTRY} HTF={HTF} EMA={EMA_FAST}/{EMA_SLOW} EMA_TREND={EMA_TREND} LOOKBACK={HTF_CROSS_LOOKBACK}")
    _log(f"[CFG] RSI_LEN={RSI_LEN} RSI_MIN={RSI_MIN} MFI={USE_MFI_FILTER} VOL={USE_VOL_FILTER} WT={USE_WT} STOCH_RSI={USE_STOCH_RSI}")
    _log(f"[CFG] TOP_N={TOP_N} MIN_QUOTE_VOLUME={MIN_QUOTE_VOLUME} COOLDOWN_SEC={COOLDOWN_SEC} LONG_ONLY={LONG_ONLY} DRY_RUN={DRY_RUN}")

    storage = Storage(STORAGE_PATH) if USE_STORAGE else None
    last_heartbeat = 0.0

    all_symbols = tradable_usdt_perps()
    scan_symbols = top_by_quote_volume(all_symbols, TOP_N)

    while True:
        t0 = time.time()

        reject_counts = init_reject_counts()

        if time.time() - last_heartbeat > HEARTBEAT_SEC:
            send_telegram(f"✅ worker alive | TF={TF_ENTRY} HTF={HTF} TOP_N={TOP_N} BTC_FILTER={USE_BTC_FILTER}")
            last_heartbeat = time.time()

        if not btc_ok(reject_counts):
            if DEBUG_REJECTS:
                _log(f"[REJECT] BTC filter -> {reject_counts}")
            if TEST_ONCE:
                return
            time.sleep(INTERVAL_SEC)
            continue

        for sym in scan_symbols:
            # Cooldown
            if storage and storage.is_on_cooldown(sym, COOLDOWN_SEC):
                continue

            try:
                if USE_HTF_FILTER and not htf_ok(sym, reject_counts):
                    continue

                kl = get_klines(sym, TF_ENTRY, KLINE_LIMIT)
                s = Series.from_klines(kl)

                if LONG_ONLY:
                    ok = long_signal(sym, s, reject_counts)
                    if ok:
                        msg = build_message(sym, s)
                        if not DRY_RUN:
                            send_telegram(msg)
                        _log(f"[SIGNAL] {sym} @ {s.closes[-1]}")
                        if storage:
                            storage.mark_sent(sym)
                else:
                    # for later (shorts)
                    pass

            except Exception as e:
                _log(f"[WARN] {sym} error: {e}")

        if DEBUG_REJECTS:
            _log(f"[REJECT_COUNTS] {reject_counts}")

        if TEST_ONCE:
            return

        # Sleep
        elapsed = time.time() - t0
        sleep_for = max(1.0, INTERVAL_SEC - elapsed)
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
