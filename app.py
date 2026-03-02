import os
import time
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import requests

from notify import send_telegram
from storage import Storage

BINANCE_FAPI = os.getenv("BINANCE_FAPI", "https://fapi.binance.com").rstrip("/")
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "12"))

# ===== ENV =====
TF_ENTRY = os.getenv("TF_ENTRY", os.getenv("TF", "1h"))  # compat
KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "260"))
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "600"))
HEARTBEAT_SEC = int(os.getenv("HEARTBEAT_SEC", "900"))
TEST_ONCE = int(os.getenv("TEST_ONCE", "0"))

EMA_FAST = int(os.getenv("EMA_FAST", "3"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "44"))
LOOKBACK = int(os.getenv("LOOKBACK", os.getenv("HTF_CROSS_LOOKBACK", "6")))

RSI_LEN = int(os.getenv("RSI_LEN", "21"))
RSI_MIN = float(os.getenv("RSI_MIN", "42"))

TOP_N = int(os.getenv("TOP_N", "200"))
MIN_QUOTE_VOLUME = float(os.getenv("MIN_QUOTE_VOLUME", "3000000"))
ONLY_USDT_PERP = int(os.getenv("ONLY_USDT_PERP", "1"))

COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", "21600"))
DEBUG = int(os.getenv("DEBUG", "1"))
DEBUG_REJECTS = int(os.getenv("DEBUG_REJECTS", "0"))
DRY_RUN = int(os.getenv("DRY_RUN", "0"))

USE_STORAGE = int(os.getenv("USE_STORAGE", "1"))
STORAGE_PATH = os.getenv("STORAGE_PATH", "/var/data/futures_state.json")

# Optional filters toggles (off => TV setup'a daha yakın)
USE_BTC_FILTER = int(os.getenv("USE_BTC_FILTER", "0"))
USE_HTF_FILTER = int(os.getenv("USE_HTF_FILTER", "0"))
USE_MFI_FILTER = int(os.getenv("USE_MFI_FILTER", "0"))
USE_VOL_FILTER = int(os.getenv("USE_VOL_FILTER", "0"))

# Stoch RSI
USE_STOCH_RSI = int(os.getenv("USE_STOCH_RSI", "1"))
STOCH_RSI_LEN = int(os.getenv("STOCH_RSI_LEN", "14"))
STOCH_K = int(os.getenv("STOCH_K", "5"))
STOCH_D = int(os.getenv("STOCH_D", "5"))

# WaveTrend
USE_WT = int(os.getenv("USE_WT", "1"))
WT_CH_LEN = int(os.getenv("WT_CH_LEN", "9"))
WT_AVG_LEN = int(os.getenv("WT_AVG_LEN", "12"))
WT_OB1 = float(os.getenv("WT_OB1", "60"))
WT_OB2 = float(os.getenv("WT_OB2", "53"))
WT_OS1 = float(os.getenv("WT_OS1", "-60"))
WT_OS2 = float(os.getenv("WT_OS2", "-53"))

USE_WT_DIP = int(os.getenv("USE_WT_DIP", "1"))
USE_WT_CONTINUATION = int(os.getenv("USE_WT_CONTINUATION", "0"))

LONG_ONLY = int(os.getenv("LONG_ONLY", "1"))

# Manual plan suggestions (message only)
TP_PCT = float(os.getenv("TP_PCT", "8"))
SL_PCT = float(os.getenv("SL_PCT", "2"))

# --- Optional trend filters params (keep env but default off) ---
HTF = os.getenv("HTF", "4h")
EMA_TREND = int(os.getenv("EMA_TREND", "123"))
HTF_STRICT_CROSS = int(os.getenv("HTF_STRICT_CROSS", "0"))

BTC_SYMBOL = os.getenv("BTC_SYMBOL", "BTCUSDT")
BTC_TF = os.getenv("BTC_TF", "4h")

# MFI params (if enabled)
MFI_LEN = int(os.getenv("MFI_LEN", "14"))
MFI_LONG_MIN = float(os.getenv("MFI_LONG_MIN", "40"))
MFI_LONG_MAX = float(os.getenv("MFI_LONG_MAX", "85"))
MFI_SLOPE_ENABLE = int(os.getenv("MFI_SLOPE_ENABLE", "1"))
MFI_SLOPE_BARS = int(os.getenv("MFI_SLOPE_BARS", "1"))

# Volume spike params (if enabled)
VOL_LEN = int(os.getenv("VOL_LEN", "20"))
VOL_MULT = float(os.getenv("VOL_MULT", "1.1"))
VOL_USE_QUOTE = int(os.getenv("VOL_USE_QUOTE", "1"))


@dataclass
class Kline:
    o: float
    h: float
    l: float
    c: float
    v: float
    t: int  # open time ms


def _get(url: str, params: Optional[dict] = None) -> dict:
    r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_klines(symbol: str, interval: str, limit: int) -> List[Kline]:
    data = _get(f"{BINANCE_FAPI}/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})
    out: List[Kline] = []
    for row in data:
        out.append(Kline(
            o=float(row[1]), h=float(row[2]), l=float(row[3]), c=float(row[4]), v=float(row[5]), t=int(row[0])
        ))
    return out


def get_top_symbols() -> List[Tuple[str, float]]:
    tickers = _get(f"{BINANCE_FAPI}/fapi/v1/ticker/24hr")
    pairs: List[Tuple[str, float]] = []
    for t in tickers:
        sym = t.get("symbol", "")
        if ONLY_USDT_PERP and not sym.endswith("USDT"):
            continue
        # Binance futures 24h quoteVolume is a string
        qv = float(t.get("quoteVolume", 0) or 0)
        if qv < MIN_QUOTE_VOLUME:
            continue
        pairs.append((sym, qv))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:TOP_N]


def ema(values: List[float], length: int) -> List[float]:
    if length <= 1:
        return values[:]
    k = 2 / (length + 1.0)
    out = []
    prev = values[0]
    out.append(prev)
    for v in values[1:]:
        prev = prev + k * (v - prev)
        out.append(prev)
    return out


def rsi(values: List[float], length: int) -> List[float]:
    # Wilder RSI
    if len(values) < length + 1:
        return [50.0] * len(values)

    gains = [0.0]
    losses = [0.0]
    for i in range(1, len(values)):
        ch = values[i] - values[i - 1]
        gains.append(max(ch, 0.0))
        losses.append(max(-ch, 0.0))

    avg_gain = sum(gains[1:length + 1]) / length
    avg_loss = sum(losses[1:length + 1]) / length
    out = [50.0] * len(values)

    rs = (avg_gain / avg_loss) if avg_loss > 0 else float("inf")
    out[length] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(length + 1, len(values)):
        avg_gain = (avg_gain * (length - 1) + gains[i]) / length
        avg_loss = (avg_loss * (length - 1) + losses[i]) / length
        rs = (avg_gain / avg_loss) if avg_loss > 0 else float("inf")
        out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out


def stoch_rsi(values: List[float], rsi_len: int, stoch_len: int, k: int, d: int) -> Tuple[List[float], List[float]]:
    r = rsi(values, rsi_len)
    srsi = [50.0] * len(values)
    for i in range(len(values)):
        start = max(0, i - stoch_len + 1)
        window = r[start:i + 1]
        lo = min(window)
        hi = max(window)
        if hi - lo == 0:
            srsi[i] = 0.0
        else:
            srsi[i] = 100.0 * (r[i] - lo) / (hi - lo)

    k_line = sma(srsi, k)
    d_line = sma(k_line, d)
    return k_line, d_line


def sma(values: List[float], length: int) -> List[float]:
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


def wavetrend(klines: List[Kline], ch_len: int, avg_len: int) -> Tuple[List[float], List[float]]:
    # LazyBear WT core: uses HLC3
    hlc3 = [(k.h + k.l + k.c) / 3.0 for k in klines]
    esa = ema(hlc3, ch_len)
    de = [abs(hlc3[i] - esa[i]) for i in range(len(hlc3))]
    d = ema(de, ch_len)
    ci = []
    for i in range(len(hlc3)):
        denom = (0.015 * d[i]) if d[i] != 0 else 1e-9
        ci.append((hlc3[i] - esa[i]) / denom)
    wt1 = ema(ci, avg_len)
    wt2 = sma(wt1, 4)  # LazyBear often uses 4 as signal smoothing
    return wt1, wt2


def mfi(klines: List[Kline], length: int) -> List[float]:
    # Money Flow Index
    tp = [(k.h + k.l + k.c) / 3.0 for k in klines]
    mf = [tp[i] * klines[i].v for i in range(len(klines))]
    out = [50.0] * len(klines)
    for i in range(length, len(klines)):
        pos = 0.0
        neg = 0.0
        for j in range(i - length + 1, i + 1):
            if tp[j] > tp[j - 1]:
                pos += mf[j]
            elif tp[j] < tp[j - 1]:
                neg += mf[j]
        mr = (pos / neg) if neg > 0 else float("inf")
        out[i] = 100.0 - (100.0 / (1.0 + mr))
    return out


def crossed_up(a_prev: float, a_now: float, b_prev: float, b_now: float) -> bool:
    return a_prev <= b_prev and a_now > b_now


def ema_cross_recent(closes: List[float], fast: int, slow: int, lookback: int) -> bool:
    ef = ema(closes, fast)
    es = ema(closes, slow)
    # within last lookback bars (exclude current forming? burada kline kapanışları olduğu için OK)
    start = max(1, len(closes) - lookback - 1)
    for i in range(start, len(closes)):
        if crossed_up(ef[i - 1], ef[i], es[i - 1], es[i]):
            return True
    return False


def wt_dip_ok(wt1: List[float], wt2: List[float]) -> bool:
    # Dip reversal: oversold zone + bullish cross
    if len(wt1) < 3:
        return False
    w1p, w1 = wt1[-2], wt1[-1]
    w2p, w2 = wt2[-2], wt2[-1]
    did_cross = crossed_up(w1p, w1, w2p, w2)

    # must be oversold-ish near the dip
    was_os = min(w1p, w1) <= WT_OS2  # -53 default
    deep_os = min(w1p, w1) <= WT_OS1  # -60 default
    return did_cross and (was_os or deep_os)


def wt_cont_ok(wt1: List[float], wt2: List[float]) -> bool:
    # Continuation (disabled by default): bullish cross when WT above -20 or above 0 zone
    if len(wt1) < 3:
        return False
    w1p, w1 = wt1[-2], wt1[-1]
    w2p, w2 = wt2[-2], wt2[-1]
    did_cross = crossed_up(w1p, w1, w2p, w2)
    return did_cross and (w1 > -20)


def vol_spike_ok(klines: List[Kline]) -> bool:
    if len(klines) < VOL_LEN + 1:
        return False
    vols = [k.v for k in klines]
    base = sum(vols[-VOL_LEN-1:-1]) / VOL_LEN
    last = vols[-1]
    return last >= base * VOL_MULT


def btc_ok() -> bool:
    # optional: BTC above EMA_TREND on BTC_TF
    if not USE_BTC_FILTER:
        return True
    ks = get_klines(BTC_SYMBOL, BTC_TF, KLINE_LIMIT)
    closes = [k.c for k in ks]
    e = ema(closes, EMA_TREND)
    return closes[-1] >= e[-1]


def htf_ok(symbol: str) -> bool:
    if not USE_HTF_FILTER:
        return True
    ks = get_klines(symbol, HTF, KLINE_LIMIT)
    closes = [k.c for k in ks]
    # simple trend filter: close above EMA_TREND
    e = ema(closes, EMA_TREND)
    if closes[-1] < e[-1]:
        return False
    # optionally require recent fast/slow cross on HTF too
    if HTF_STRICT_CROSS:
        return ema_cross_recent(closes, EMA_FAST, EMA_SLOW, LOOKBACK)
    return True


def mfi_ok(symbol_klines: List[Kline]) -> Tuple[bool, float]:
    if not USE_MFI_FILTER:
        return True, float("nan")
    m = mfi(symbol_klines, MFI_LEN)
    val = m[-1]
    if not (MFI_LONG_MIN <= val <= MFI_LONG_MAX):
        return False, val
    if MFI_SLOPE_ENABLE and MFI_SLOPE_BARS >= 1 and len(m) > MFI_SLOPE_BARS:
        if m[-1] < m[-1 - MFI_SLOPE_BARS]:
            return False, val
    return True, val


def fmt(x: float, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{x:.{nd}f}"


def build_message(symbol: str, price: float,
                  ema_f: float, ema_s: float,
                  rsi_v: float,
                  k: float, d: float,
                  wt1: float, wt2: float) -> str:
    lines = []
    lines.append("🚀 LONG SIGNAL")
    lines.append(f"Symbol: {symbol}")
    lines.append(f"TF: {TF_ENTRY} | HTF: {HTF}")
    lines.append(f"Price: {fmt(price, 6)}")
    lines.append("")
    lines.append(f"EMA{EMA_FAST}: {fmt(ema_f, 6)} | EMA{EMA_SLOW}: {fmt(ema_s, 6)}")
    lines.append(f"RSI({RSI_LEN}): {fmt(rsi_v, 2)}")
    if USE_STOCH_RSI:
        lines.append(f"StochRSI K/D (K={STOCH_K},D={STOCH_D}): {fmt(k,2)}/{fmt(d,2)}")
    if USE_WT:
        lines.append(f"WT_LB (ch={WT_CH_LEN},avg={WT_AVG_LEN}) WT1/WT2: {fmt(wt1,2)}/{fmt(wt2,2)}")

    lines.append("")
    lines.append("Exit plan (manual):")
    lines.append(f"- TP1: +{TP_PCT:.1f}% (suggestion)")
    lines.append(f"- SL: -{SL_PCT:.1f}% (suggestion)")
    lines.append(f"- WT exit: if WT1 crosses DOWN WT2 while WT1>{WT_OB2:.0f} consider close/trim")
    lines.append(f"- WT warning: if WT1>{WT_OB1:.0f} and turns down -> tighten stop")
    return "\n".join(lines)


def main() -> None:
    st = Storage(STORAGE_PATH) if USE_STORAGE else None
    last_heartbeat = 0.0

    print(f"[BOOT] Futures scanner started")
    print(f"[CFG] TF_ENTRY={TF_ENTRY} HTF={HTF} EMA={EMA_FAST}/{EMA_SLOW} LOOKBACK={LOOKBACK}")
    print(f"[CFG] RSI_LEN={RSI_LEN} RSI_MIN={RSI_MIN} MFI={USE_MFI_FILTER} VOL={USE_VOL_FILTER} BTC={USE_BTC_FILTER} HTF_FILTER={USE_HTF_FILTER}")
    print(f"[CFG] WT={USE_WT} DIP={USE_WT_DIP} CONT={USE_WT_CONTINUATION} STOCH_RSI={USE_STOCH_RSI}")
    print(f"[CFG] TOP_N={TOP_N} MIN_QUOTE_VOLUME={MIN_QUOTE_VOLUME} COOLDOWN_SEC={COOLDOWN_SEC} LONG_ONLY={LONG_ONLY} DRY_RUN={DRY_RUN}")
    print(f"[CFG] STORAGE={USE_STORAGE} PATH={STORAGE_PATH}")

    send_telegram(f"✅ worker alive | TF={TF_ENTRY} HTF={HTF} TOP_N={TOP_N} BTC_FILTER={USE_BTC_FILTER}")

    while True:
        now = time.time()
        if now - last_heartbeat >= HEARTBEAT_SEC:
            last_heartbeat = now
            if DEBUG:
                print(f"[HB] alive | TF={TF_ENTRY} TOP_N={TOP_N} cooldown={COOLDOWN_SEC}s")

        try:
            if not btc_ok():
                if DEBUG_REJECTS:
                    print("[REJECT] BTC filter says NO")
                time.sleep(INTERVAL_SEC)
                if TEST_ONCE:
                    return
                continue

            symbols = get_top_symbols()
            if DEBUG:
                print(f"[SCAN] symbols={len(symbols)} (top by quoteVolume)")

            for sym, qv in symbols:
                cooldown_key = f"{sym}:{TF_ENTRY}:LONG"
                if st and not st.can_send(cooldown_key, COOLDOWN_SEC):
                    continue

                ks = get_klines(sym, TF_ENTRY, KLINE_LIMIT)
                closes = [k.c for k in ks]
                if len(closes) < max(EMA_SLOW, RSI_LEN) + 5:
                    continue

                if not htf_ok(sym):
                    if DEBUG_REJECTS:
                        print(f"[REJECT] {sym} HTF trend filter")
                    continue

                # Core: EMA cross recent + RSI >= min
                if not ema_cross_recent(closes, EMA_FAST, EMA_SLOW, LOOKBACK):
                    if DEBUG_REJECTS:
                        print(f"[REJECT] {sym} EMA cross not found")
                    continue

                r = rsi(closes, RSI_LEN)[-1]
                if r < RSI_MIN:
                    if DEBUG_REJECTS:
                        print(f"[REJECT] {sym} RSI {r:.2f} < {RSI_MIN}")
                    continue

                # Optional: volume spike
                if USE_VOL_FILTER and not vol_spike_ok(ks):
                    if DEBUG_REJECTS:
                        print(f"[REJECT] {sym} volume spike")
                    continue

                # Optional: MFI
                ok_mfi, mfi_v = mfi_ok(ks)
                if not ok_mfi:
                    if DEBUG_REJECTS:
                        print(f"[REJECT] {sym} MFI {mfi_v:.2f}")
                    continue

                # Stoch RSI (info / optional gating)
                k_val = d_val = float("nan")
                if USE_STOCH_RSI:
                    k_line, d_line = stoch_rsi(closes, RSI_LEN, STOCH_RSI_LEN, STOCH_K, STOCH_D)
                    k_val, d_val = k_line[-1], d_line[-1]

                # WT
                wt1_val = wt2_val = float("nan")
                if USE_WT:
                    wt1, wt2 = wavetrend(ks, WT_CH_LEN, WT_AVG_LEN)
                    wt1_val, wt2_val = wt1[-1], wt2[-1]

                    dip_ok = (USE_WT_DIP and wt_dip_ok(wt1, wt2))
                    cont_ok = (USE_WT_CONTINUATION and wt_cont_ok(wt1, wt2))
                    if USE_WT_DIP or USE_WT_CONTINUATION:
                        if not (dip_ok or cont_ok):
                            if DEBUG_REJECTS:
                                print(f"[REJECT] {sym} WT gating (dip/cont)")
                            continue

                price = closes[-1]
                msg = build_message(sym, price, ema(closes, EMA_FAST)[-1], ema(closes, EMA_SLOW)[-1], r, k_val, d_val, wt1_val, wt2_val)

                if DRY_RUN:
                    print("[DRY_RUN] would send:\n" + msg)
                else:
                    send_telegram(msg)
                    if DEBUG:
                        print(f"[SEND] {sym} @ {price} | qv={qv:.0f}")

                if st:
                    st.mark_sent(cooldown_key)

            if TEST_ONCE:
                return

        except Exception as e:
            print(f"[ERR] scan loop error: {e}")

        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main()
