# app.py
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests

from notify import send_telegram
from storage import Storage

# =========================
# ENV / SETTINGS
# =========================
BINANCE_FAPI = os.getenv("BINANCE_FAPI", "https://fapi.binance.com").rstrip("/")
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "12"))

TF_ENTRY = os.getenv("TF_ENTRY", os.getenv("TF", "1h"))
KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "260"))
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "600"))
HEARTBEAT_SEC = int(os.getenv("HEARTBEAT_SEC", "900"))

TOP_N = int(os.getenv("TOP_N", "200"))
MIN_QUOTE_VOLUME = float(os.getenv("MIN_QUOTE_VOLUME", "3000000"))
ONLY_USDT_PERP = int(os.getenv("ONLY_USDT_PERP", "1")) == 1

EMA_FAST = int(os.getenv("EMA_FAST", "3"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "44"))
LOOKBACK = int(os.getenv("LOOKBACK", "6"))

RSI_LEN = int(os.getenv("RSI_LEN", "21"))
RSI_MIN = float(os.getenv("RSI_MIN", "42"))

USE_STOCH_RSI = int(os.getenv("USE_STOCH_RSI", "1")) == 1
STOCH_RSI_LEN = int(os.getenv("STOCH_RSI_LEN", "14"))
STOCH_K = int(os.getenv("STOCH_K", "5"))
STOCH_D = int(os.getenv("STOCH_D", "5"))

USE_WT = int(os.getenv("USE_WT", "1")) == 1
USE_WT_DIP = int(os.getenv("USE_WT_DIP", "1")) == 1
USE_WT_CONTINUATION = int(os.getenv("USE_WT_CONTINUATION", "0")) == 1
WT_CH = int(os.getenv("WT_CH", "12"))
WT_AVG = int(os.getenv("WT_AVG", "12"))
WT_OS1 = float(os.getenv("WT_OS1", "-60"))
WT_OS2 = float(os.getenv("WT_OS2", "-53"))

# filtreler (şimdilik kapalı)
USE_BTC_FILTER = int(os.getenv("USE_BTC_FILTER", "0")) == 1
USE_HTF_FILTER = int(os.getenv("USE_HTF_FILTER", "0")) == 1
USE_MFI_FILTER = int(os.getenv("USE_MFI_FILTER", "0")) == 1
USE_VOL_FILTER = int(os.getenv("USE_VOL_FILTER", "0")) == 1

TP_PCT = float(os.getenv("TP_PCT", "8"))
SL_PCT = float(os.getenv("SL_PCT", "2"))

COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", "21600"))
LONG_ONLY = int(os.getenv("LONG_ONLY", "1")) == 1

DEBUG = int(os.getenv("DEBUG", "1")) == 1
DEBUG_REJECTS = int(os.getenv("DEBUG_REJECTS", "0")) == 1
DRY_RUN = int(os.getenv("DRY_RUN", "0")) == 1

# TradingView ile eşleşme: varsayılan kapanmış mum
USE_LAST_CANDLE = int(os.getenv("USE_LAST_CANDLE", "0")) == 1

USE_STORAGE = int(os.getenv("USE_STORAGE", "1")) == 1
STORAGE_PATH = os.getenv("STORAGE_PATH", "/var/data/futures_state.json")

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "")

session = requests.Session()


def http_get(path: str, params: Optional[dict] = None):
    url = f"{BINANCE_FAPI}{path}"
    r = session.get(url, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


# =========================
# Indicators
# =========================
def ema(values: List[float], length: int) -> List[float]:
    if length <= 1:
        return values[:]
    out = [values[0]]
    k = 2 / (length + 1)
    for v in values[1:]:
        out.append(out[-1] + k * (v - out[-1]))
    return out


def sma(values: List[float], length: int) -> List[float]:
    if length <= 1:
        return values[:]
    out = []
    s = 0.0
    for i, v in enumerate(values):
        s += v
        if i >= length:
            s -= values[i - length]
        if i < length - 1:
            out.append(values[i])
        else:
            out.append(s / length)
    return out


def rsi(values: List[float], length: int) -> List[float]:
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

    out = [50.0] * length
    rs = (avg_gain / avg_loss) if avg_loss != 0 else float("inf")
    out.append(100 - (100 / (1 + rs)))

    for i in range(length + 1, len(values)):
        avg_gain = (avg_gain * (length - 1) + gains[i]) / length
        avg_loss = (avg_loss * (length - 1) + losses[i]) / length
        rs = (avg_gain / avg_loss) if avg_loss != 0 else float("inf")
        out.append(100 - (100 / (1 + rs)))

    while len(out) < len(values):
        out.insert(0, 50.0)
    return out[:len(values)]


def stoch_rsi(rsi_series: List[float], length: int, k: int, d: int) -> Tuple[List[float], List[float]]:
    if len(rsi_series) < length + 1:
        return ([50.0] * len(rsi_series), [50.0] * len(rsi_series))

    raw = []
    for i in range(len(rsi_series)):
        start = max(0, i - length + 1)
        window = rsi_series[start:i + 1]
        lo, hi = min(window), max(window)
        if hi - lo == 0:
            raw.append(0.0)
        else:
            raw.append((rsi_series[i] - lo) / (hi - lo) * 100.0)

    k_line = sma(raw, k)
    d_line = sma(k_line, d)
    return k_line, d_line


def crossed_up(a_prev: float, a: float, b_prev: float, b: float) -> bool:
    return a_prev <= b_prev and a > b


def crossed_down(a_prev: float, a: float, b_prev: float, b: float) -> bool:
    return a_prev >= b_prev and a < b


def ema_cross_hold(closes: List[float], fast: int, slow: int, lookback: int) -> Tuple[bool, int]:
    ef = ema(closes, fast)
    es = ema(closes, slow)
    start = max(1, len(closes) - lookback - 1)

    last_cross = -1
    for i in range(start, len(closes)):
        if crossed_up(ef[i - 1], ef[i], es[i - 1], es[i]):
            last_cross = i

    if last_cross == -1:
        return False, -1

    for j in range(last_cross, len(closes)):
        if ef[j] < es[j]:
            return False, last_cross

    return True, last_cross


def wavetrend_lb(high: List[float], low: List[float], close: List[float], ch_len: int, avg_len: int):
    ap = [(h + l + c) / 3.0 for h, l, c in zip(high, low, close)]
    esa = ema(ap, ch_len)
    abs_diff = [abs(a - e) for a, e in zip(ap, esa)]
    d = ema(abs_diff, ch_len)
    ci = []
    for a, e, di in zip(ap, esa, d):
        if di == 0:
            ci.append(0.0)
        else:
            ci.append((a - e) / (0.015 * di))
    wt1 = ema(ci, avg_len)
    wt2 = sma(wt1, 4)
    return wt1, wt2


def wt_dip_ok(wt1: List[float], wt2: List[float]) -> bool:
    if len(wt1) < 3:
        return False
    w1p, w1 = wt1[-2], wt1[-1]
    w2p, w2 = wt2[-2], wt2[-1]
    did_cross = crossed_up(w1p, w1, w2p, w2)
    was_os = min(w1p, w1) <= WT_OS2
    deep_os = min(w1p, w1) <= WT_OS1
    return did_cross and (was_os or deep_os)


def wt_cont_ok(wt1: List[float], wt2: List[float]) -> bool:
    if len(wt1) < 3:
        return False
    w1p, w1 = wt1[-2], wt1[-1]
    w2p, w2 = wt2[-2], wt2[-1]
    did_cross = crossed_up(w1p, w1, w2p, w2)
    return did_cross and w1 > -20  # continuation band


# =========================
# Data fetch
# =========================
def get_exchange_info():
    return http_get("/fapi/v1/exchangeInfo")


def get_24h_tickers():
    return http_get("/fapi/v1/ticker/24hr")


def get_klines(symbol: str, interval: str, limit: int):
    return http_get("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})


def pick_symbols() -> List[Tuple[str, float]]:
    info = get_exchange_info()
    symbols = set()
    for s in info.get("symbols", []):
        if s.get("contractType") != "PERPETUAL":
            continue
        if s.get("quoteAsset") != "USDT":
            continue
        if s.get("status") != "TRADING":
            continue
        symbols.add(s.get("symbol"))

    tickers = get_24h_tickers()
    rows = []
    for t in tickers:
        sym = t.get("symbol")
        if sym not in symbols:
            continue
        if ONLY_USDT_PERP and not sym.endswith("USDT"):
            continue
        try:
            qv = float(t.get("quoteVolume", "0") or "0")
        except Exception:
            qv = 0.0
        if qv < MIN_QUOTE_VOLUME:
            continue
        rows.append((sym, qv))

    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:TOP_N]


def to_ohlc(klines) -> Tuple[List[float], List[float], List[float], List[float]]:
    o, h, l, c = [], [], [], []
    for k in klines:
        o.append(float(k[1]))
        h.append(float(k[2]))
        l.append(float(k[3]))
        c.append(float(k[4]))
    return o, h, l, c


def kline_close_time_ms(kline) -> int:
    # kline[6] = close time (ms)
    return int(kline[6])


def fmt_ts(ms: int) -> str:
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M UTC")


# =========================
# Signal logic
# =========================
def long_signal(symbol: str, reject_counts: Dict[str, int]) -> Optional[str]:
    try:
        kl = get_klines(symbol, TF_ENTRY, KLINE_LIMIT)
        if len(kl) < max(EMA_SLOW, RSI_LEN, STOCH_RSI_LEN) + 10:
            reject_counts["KLINES_SHORT"] = reject_counts.get("KLINES_SHORT", 0) + 1
            return None

        if not USE_LAST_CANDLE:
            kl_use = kl[:-1]  # forming candle'ı at
        else:
            kl_use = kl

        # signal candle = son kullandığımız kapanmış mum
        signal_kline = kl_use[-1]
        signal_close_time = fmt_ts(kline_close_time_ms(signal_kline))

        o, h, l, c = to_ohlc(kl_use)

        # EMA cross (hold)
        ok_cross, _cross_i = ema_cross_hold(c, EMA_FAST, EMA_SLOW, LOOKBACK)
        if not ok_cross:
            reject_counts["EMA_CROSS_HOLD"] = reject_counts.get("EMA_CROSS_HOLD", 0) + 1
            return None

        ef_series = ema(c, EMA_FAST)
        es_series = ema(c, EMA_SLOW)
        ef = ef_series[-1]
        es = es_series[-1]

        # RSI
        r = rsi(c, RSI_LEN)
        r_now = r[-1]
        if r_now < RSI_MIN:
            reject_counts["RSI_MIN"] = reject_counts.get("RSI_MIN", 0) + 1
            return None

        # Stoch RSI (turn up + K>D)
        if USE_STOCH_RSI:
            k_line, d_line = stoch_rsi(r, STOCH_RSI_LEN, STOCH_K, STOCH_D)
            k_now, d_now = k_line[-1], d_line[-1]
            k_prev = k_line[-2]
            if not (k_now > k_prev and k_now > d_now):
                reject_counts["STOCH_TURN"] = reject_counts.get("STOCH_TURN", 0) + 1
                return None
        else:
            k_now, d_now = float("nan"), float("nan")

        # WT (hibrit)
        wt_mode = None
        if USE_WT:
            wt1, wt2 = wavetrend_lb(h, l, c, WT_CH, WT_AVG)
            w1, w2 = wt1[-1], wt2[-1]

            dip_ok = USE_WT_DIP and wt_dip_ok(wt1, wt2)
            cont_ok = USE_WT_CONTINUATION and wt_cont_ok(wt1, wt2)

            if dip_ok:
                wt_mode = "WT_DIP"
            elif cont_ok:
                wt_mode = "WT_CONT"
            else:
                reject_counts["WT"] = reject_counts.get("WT", 0) + 1
                return None
        else:
            w1, w2 = float("nan"), float("nan")

        entry_price = c[-1]  # kapanışta entry (TV ile aynı)
        last_price = c[-1]   # mesajda entry=close gösteriyoruz

        head = "🚀 LONG SIGNAL"
        if wt_mode == "WT_DIP":
            head += " (DIP REVERSAL)"
        elif wt_mode == "WT_CONT":
            head += " (CONTINUATION)"

        msg = (
            f"{head}\n"
            f"Symbol: {symbol}\n"
            f"TF: {TF_ENTRY}\n"
            f"Signal candle close: {signal_close_time}\n"
            f"Entry/Close: {entry_price:.6f}\n\n"
            f"EMA{EMA_FAST}: {ef:.6f} | EMA{EMA_SLOW}: {es:.6f}\n"
            f"RSI({RSI_LEN}): {r_now:.2f}\n"
        )

        if USE_STOCH_RSI:
            msg += f"StochRSI K/D (K={STOCH_K},D={STOCH_D}): {k_now:.2f}/{d_now:.2f}\n"
        if USE_WT:
            msg += f"WT_LB (ch={WT_CH},avg={WT_AVG}) WT1/WT2: {w1:.2f}/{w2:.2f}\n"

        msg += (
            f"\nExit plan (manual):\n"
            f"- TP1: +{TP_PCT:.1f}% (suggestion)\n"
            f"- SL:  -{SL_PCT:.1f}% (suggestion)\n"
            f"- WT exit: if WT1 crosses DOWN WT2 while WT1>{abs(WT_OS2):.0f} consider close/trim\n"
            f"- WT warning: if WT1>{abs(WT_OS2)+7:.0f} and turns down -> tighten stop\n"
        )
        return msg

    except Exception as e:
        reject_counts["EXC"] = reject_counts.get("EXC", 0) + 1
        if DEBUG:
            print(f"[ERR] {symbol} {e}")
        return None


def main():
    storage = Storage(STORAGE_PATH) if USE_STORAGE else None

    print("[BOOT] Futures scanner started")
    print(
        f"[CFG] TF_ENTRY={TF_ENTRY} EMA={EMA_FAST}/{EMA_SLOW} LOOKBACK={LOOKBACK} "
        f"RSI_LEN={RSI_LEN} RSI_MIN={RSI_MIN} WT={int(USE_WT)} DIP={int(USE_WT_DIP)} CONT={int(USE_WT_CONTINUATION)} "
        f"STOCH_RSI={int(USE_STOCH_RSI)}"
    )
    print(
        f"[CFG] TOP_N={TOP_N} MIN_QUOTE_VOLUME={MIN_QUOTE_VOLUME} "
        f"COOLDOWN_SEC={COOLDOWN_SEC} DRY_RUN={int(DRY_RUN)} USE_LAST_CANDLE={int(USE_LAST_CANDLE)}"
    )
    if USE_STORAGE:
        print(f"[CFG] STORAGE_PATH={STORAGE_PATH}")

    last_hb = 0.0

    while True:
        now = time.time()
        if now - last_hb >= HEARTBEAT_SEC:
            hb = f"✅ worker alive | TF={TF_ENTRY} TOP_N={TOP_N} DIP={int(USE_WT_DIP)} CONT={int(USE_WT_CONTINUATION)}"
            if DRY_RUN:
                hb += " DRY_RUN=1"
            send_telegram(hb, TG_BOT_TOKEN, TG_CHAT_ID, dry_run=False)
            last_hb = now

        symbols = pick_symbols()
        reject_counts: Dict[str, int] = {}

        for sym, _qv in symbols:
            if storage and storage.is_cooldown(sym, COOLDOWN_SEC):
                continue

            msg = long_signal(sym, reject_counts)
            if not msg:
                continue

            if not DRY_RUN:
                send_telegram(msg, TG_BOT_TOKEN, TG_CHAT_ID, dry_run=False)
                if storage:
                    storage.set_last(sym, int(time.time()))
            else:
                print(msg)

            time.sleep(0.2)

        if DEBUG_REJECTS and reject_counts:
            top = sorted(reject_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            print("[REJECTS]", top)

        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main()
