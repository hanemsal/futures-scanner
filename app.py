import os
import sys
import time
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict
import math
import requests

from notify import send_telegram
from storage import Storage

# stdout line-buffering (Render log'a aninda dussun)
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

def log(msg: str):
    print(msg, flush=True)

# =========================
# ENV
# =========================
BINANCE_FAPI = os.getenv("BINANCE_FAPI", "https://fapi.binance.com").rstrip("/")

TF_ENTRY = os.getenv("TF_ENTRY", os.getenv("TF", "5m"))
HTF = os.getenv("HTF", os.getenv("TF_TREND", "1h"))

INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "180"))
KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "260"))

TOP_N = int(os.getenv("TOP_N", "400"))
MIN_QUOTE_VOLUME_24H = float(os.getenv("MIN_QUOTE_VOLUME", "3000000"))

EMA_FAST = int(os.getenv("EMA_FAST", "3"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "44"))

USE_HTF_FILTER = int(os.getenv("USE_HTF_FILTER", "1"))
# Yeni: HTF sertliği
HTF_STRICT_CROSS = int(os.getenv("HTF_STRICT_CROSS", "0"))  # 1 yaparsan eski gibi cross arar
HTF_CROSS_LOOKBACK = int(os.getenv("HTF_CROSS_LOOKBACK", "6"))

USE_BTC_FILTER = int(os.getenv("USE_BTC_FILTER", "1"))
BTC_SYMBOL = os.getenv("BTC_SYMBOL", "BTCUSDT")
BTC_TF = os.getenv("BTC_TF", HTF)

USE_VOL_FILTER = int(os.getenv("USE_VOL_FILTER", "1"))
VOL_LEN = int(os.getenv("VOL_LEN", "20"))
VOL_MULT = float(os.getenv("VOL_MULT", "1.1"))
# Yeni: VOL filtre USDT bazlı mı?
VOL_USE_QUOTE = int(os.getenv("VOL_USE_QUOTE", "1"))  # 1 => volume*close (önerilen)

USE_RSI_FILTER = int(os.getenv("USE_RSI_FILTER", "1"))
RSI_LEN = int(os.getenv("RSI_LEN", "21"))          # öneri: 14-21
RSI_EMA_LEN = int(os.getenv("RSI_EMA_LEN", "14"))  # daha kısa daha stabil
RSI_MIN = float(os.getenv("RSI_MIN", "42"))        # öneri: 42

USE_MACD_NEAR0 = int(os.getenv("USE_MACD_NEAR0", "0"))  # başlangıçta 0 öneririm
MACD_FAST = int(os.getenv("MACD_FAST", "12"))
MACD_SLOW = int(os.getenv("MACD_SLOW", "26"))
MACD_SIGNAL = int(os.getenv("MACD_SIGNAL", "9"))
MACD_NEAR0_PCT = float(os.getenv("MACD_NEAR0_PCT", "0.15"))

USE_MFI_FILTER = int(os.getenv("USE_MFI_FILTER", "1"))
MFI_LEN = int(os.getenv("MFI_LEN", "14"))
# Öneri aralıklar:
MFI_LONG_MIN = float(os.getenv("MFI_LONG_MIN", "40"))
MFI_LONG_MAX = float(os.getenv("MFI_LONG_MAX", "85"))
MFI_SHORT_MIN = float(os.getenv("MFI_SHORT_MIN", "15"))
MFI_SHORT_MAX = float(os.getenv("MFI_SHORT_MAX", "65"))

LEVERAGE = int(os.getenv("LEVERAGE", "5"))
TP_PCT = float(os.getenv("TP_PCT", "5"))
SL_PCT = float(os.getenv("SL_PCT", "2"))

COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", "21600"))
USE_STORAGE = int(os.getenv("USE_STORAGE", "1"))
STORAGE_PATH = os.getenv("STORAGE_PATH", "/var/data/futures_state.json")

DEBUG = int(os.getenv("DEBUG", "1"))
DEBUG_REJECTS = int(os.getenv("DEBUG_REJECTS", "1"))
HEARTBEAT_SEC = int(os.getenv("HEARTBEAT_SEC", "900"))

DRY_RUN = int(os.getenv("DRY_RUN", "0"))
TEST_ONCE = int(os.getenv("TEST_ONCE", "0"))
DEBUG_EVERY_N = int(os.getenv("DEBUG_EVERY_N", "0"))

storage = Storage(STORAGE_PATH) if USE_STORAGE else None

# =========================
# HTTP
# =========================
def _get(url: str, params: dict, timeout: int = 12):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def get_symbols_top_by_volume(top_n: int, min_quote_volume_24h: float) -> List[str]:
    url = f"{BINANCE_FAPI}/fapi/v1/ticker/24hr"
    data = _get(url, params={}, timeout=20)
    rows = []
    for d in data:
        sym = d.get("symbol", "")
        if not sym.endswith("USDT"):
            continue
        qv = float(d.get("quoteVolume", 0.0) or 0.0)
        if qv < min_quote_volume_24h:
            continue
        rows.append((sym, qv))
    rows.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in rows[:top_n]]

def get_klines(symbol: str, interval: str, limit: int) -> List[list]:
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    return _get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=15)

# =========================
# Indicators
# =========================
def ema(series: List[float], length: int) -> List[float]:
    if not series:
        return []
    k = 2 / (length + 1)
    out = [series[0]]
    for x in series[1:]:
        out.append(out[-1] + k * (x - out[-1]))
    return out

def rsi_wilder(series: List[float], length: int) -> List[float]:
    if len(series) < length + 1:
        return [50.0] * len(series)

    gains, losses = [], []
    for i in range(1, len(series)):
        ch = series[i] - series[i-1]
        gains.append(max(ch, 0.0))
        losses.append(max(-ch, 0.0))

    avg_gain = sum(gains[:length]) / length
    avg_loss = sum(losses[:length]) / length

    rsis = [50.0] * length
    rs = (avg_gain / avg_loss) if avg_loss != 0 else float("inf")
    rsis.append(100 - (100 / (1 + rs)))

    for i in range(length, len(gains)):
        avg_gain = (avg_gain * (length - 1) + gains[i]) / length
        avg_loss = (avg_loss * (length - 1) + losses[i]) / length
        rs = (avg_gain / avg_loss) if avg_loss != 0 else float("inf")
        rsis.append(100 - (100 / (1 + rs)))

    if len(rsis) < len(series):
        rsis = ([rsis[0]] * (len(series) - len(rsis))) + rsis
    return rsis[-len(series):]

def macd_hist(series: List[float], fast: int, slow: int, signal: int) -> List[float]:
    if len(series) < max(fast, slow, signal) + 2:
        return [0.0] * len(series)
    ef = ema(series, fast)
    es = ema(series, slow)
    macd_line = [a - b for a, b in zip(ef, es)]
    sig = ema(macd_line, signal)
    return [m - s for m, s in zip(macd_line, sig)]

def mfi(high: List[float], low: List[float], close: List[float], volume: List[float], length: int) -> List[float]:
    n = min(len(high), len(low), len(close), len(volume))
    if n == 0:
        return []
    high, low, close, volume = high[-n:], low[-n:], close[-n:], volume[-n:]
    tp = [(h + l + c) / 3.0 for h, l, c in zip(high, low, close)]
    raw = [tp[i] * volume[i] for i in range(n)]

    pos = [0.0] * n
    neg = [0.0] * n
    for i in range(1, n):
        if tp[i] > tp[i-1]:
            pos[i] = raw[i]
        elif tp[i] < tp[i-1]:
            neg[i] = raw[i]

    out = [50.0] * n
    for i in range(n):
        if i < length:
            continue
        p = sum(pos[i-length+1:i+1])
        m = sum(neg[i-length+1:i+1])
        if m == 0 and p == 0:
            out[i] = 50.0
        elif m == 0:
            out[i] = 100.0
        else:
            mr = p / m
            out[i] = 100.0 - (100.0 / (1.0 + mr))
    return out

# =========================
# Kline parse
# =========================
def parse_hlcva(kl: List[list]) -> Tuple[List[float], List[float], List[float], List[float]]:
    high = [float(k[2]) for k in kl]
    low  = [float(k[3]) for k in kl]
    close = [float(k[4]) for k in kl]
    vol = [float(k[5]) for k in kl]
    return high, low, close, vol

def parse_close(kl: List[list]) -> List[float]:
    return [float(k[4]) for k in kl]

# =========================
# Filters
# =========================
def btc_ok() -> bool:
    if not USE_BTC_FILTER:
        return True
    try:
        k = get_klines(BTC_SYMBOL, BTC_TF, max(EMA_SLOW, 120))
        closes = parse_close(k)
        ef = ema(closes, EMA_FAST)[-1]
        es = ema(closes, EMA_SLOW)[-1]
        return ef >= es
    except Exception as e:
        if DEBUG:
            log(f"BTC filter error: {repr(e)}")
        return True  # hata olursa kilitleme

def vol_ok(vol: List[float], close: List[float]) -> Tuple[bool, float, float]:
    """
    VOL_USE_QUOTE=1 => USDT bazlı volume: vol*close
    """
    if not USE_VOL_FILTER:
        return True, float("nan"), float("nan")
    if len(vol) < VOL_LEN + 1 or len(close) < VOL_LEN + 1:
        return True, float("nan"), float("nan")

    if VOL_USE_QUOTE:
        last = vol[-1] * close[-1]
        ma = sum(v * c for v, c in zip(vol[-VOL_LEN:], close[-VOL_LEN:])) / VOL_LEN
    else:
        last = vol[-1]
        ma = sum(vol[-VOL_LEN:]) / VOL_LEN

    return (last >= ma * VOL_MULT), last, ma

def rsi_ok(closes: List[float], direction: str) -> Tuple[bool, float, float]:
    if not USE_RSI_FILTER:
        return True, float("nan"), float("nan")
    r = rsi_wilder(closes, RSI_LEN)
    r_last = r[-1]
    r_ema = ema(r, RSI_EMA_LEN)[-1]
    if direction == "LONG":
        return (r_last >= RSI_MIN), r_last, r_ema
    else:
        return (r_last <= (100 - RSI_MIN)), r_last, r_ema

def macd_ok(closes: List[float]) -> Tuple[bool, float]:
    if not USE_MACD_NEAR0:
        return True, float("nan")
    hist = macd_hist(closes, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    h = hist[-1]
    price = closes[-1]
    if price == 0:
        return True, h
    ok = abs(h) / price <= (MACD_NEAR0_PCT / 100.0)
    return ok, h

def mfi_ok(high: List[float], low: List[float], close: List[float], vol: List[float], direction: str) -> Tuple[bool, float]:
    if not USE_MFI_FILTER:
        return True, float("nan")
    m = mfi(high, low, close, vol, MFI_LEN)
    m_last = m[-1] if m else float("nan")
    if direction == "LONG":
        ok = (m_last >= MFI_LONG_MIN) and (m_last <= MFI_LONG_MAX)
    else:
        ok = (m_last >= MFI_SHORT_MIN) and (m_last <= MFI_SHORT_MAX)
    return ok, m_last

def htf_confirm(symbol: str, direction: str) -> bool:
    if not USE_HTF_FILTER:
        return True
    try:
        k = get_klines(symbol, HTF, max(EMA_SLOW + HTF_CROSS_LOOKBACK + 10, 160))
        closes = parse_close(k)
        ef = ema(closes, EMA_FAST)
        es = ema(closes, EMA_SLOW)

        # STRICT mod: cross arar
        if HTF_STRICT_CROSS:
            for i in range(1, HTF_CROSS_LOOKBACK + 1):
                a1, b1 = ef[-i-1], es[-i-1]
                a2, b2 = ef[-i], es[-i]
                if direction == "LONG" and (a1 <= b1 and a2 > b2):
                    return True
                if direction == "SHORT" and (a1 >= b1 and a2 < b2):
                    return True
            return False

        # SOFT mod: sadece trend hizası yeter
        if direction == "LONG":
            return ef[-1] >= es[-1]
        return ef[-1] <= es[-1]

    except Exception as e:
        if DEBUG:
            log(f"HTF confirm error {symbol}: {repr(e)}")
        return True  # hata olursa kilitleme

# =========================
# State
# =========================
def now_ts() -> int:
    return int(time.time())

def can_send(symbol: str, direction: str) -> bool:
    if not USE_STORAGE or storage is None:
        return True
    key = f"{symbol}:{direction}"
    last = storage.get(key)
    if last is None:
        return True
    return (now_ts() - int(last)) >= COOLDOWN_SEC

def mark_sent(symbol: str, direction: str):
    if USE_STORAGE and storage is not None:
        storage.set(f"{symbol}:{direction}", now_ts())

# =========================
# Signal
# =========================
def entry_cross(symbol: str, reject_counts: Dict[str, int]) -> Optional[Tuple[str, float, float]]:
    kl = get_klines(symbol, TF_ENTRY, max(KLINE_LIMIT, EMA_SLOW + 80))
    high, low, close, vol = parse_hlcva(kl)

    if len(close) < EMA_SLOW + 5:
        reject_counts["DATA_SHORT"] += 1
        return None

    ef = ema(close, EMA_FAST)
    es = ema(close, EMA_SLOW)

    # last completed bar cross
    a1, b1 = ef[-3], es[-3]
    a2, b2 = ef[-2], es[-2]

    if a1 <= b1 and a2 > b2:
        direction = "LONG"
    elif a1 >= b1 and a2 < b2:
        direction = "SHORT"
    else:
        reject_counts["NO_CROSS"] += 1
        return None

    reasons = []

    if not htf_confirm(symbol, direction):
        reasons.append("HTF_FAIL")

    if direction == "LONG" and USE_BTC_FILTER and not btc_ok():
        reasons.append("BTC_FAIL")

    okv, v_last, v_ma = vol_ok(vol, close)
    if not okv:
        reasons.append("VOL_FAIL")

    okr, r_last, r_ema = rsi_ok(close, direction)
    if not okr:
        reasons.append("RSI_FAIL")

    okm, h = macd_ok(close)
    if not okm:
        reasons.append("MACD0_FAIL")

    okf, mfi_value = mfi_ok(high, low, close, vol, direction)
    if not okf:
        reasons.append("MFI_FAIL")

    if reasons:
        for rr in reasons:
            reject_counts[rr] += 1
        if DEBUG and DEBUG_REJECTS:
            # detaylı log
            vol_detail = f"v={v_last:.4g}, ma={v_ma:.4g}, mult={VOL_MULT}, quote={VOL_USE_QUOTE}"
            rsi_detail = f"r={r_last:.2f}, ema={r_ema:.2f}, min={RSI_MIN}, len={RSI_LEN}"
            mfi_detail = f"mfi={mfi_value:.2f}, L={MFI_LONG_MIN}-{MFI_LONG_MAX}, S={MFI_SHORT_MIN}-{MFI_SHORT_MAX}"
            log(f"REJECT {symbol} {direction}: {', '.join(reasons)} | {vol_detail} | {rsi_detail} | {mfi_detail}")
        return None

    price = close[-1]
    return direction, price, mfi_value

def build_message(symbol: str, direction: str, price: float, mfi_value: float) -> str:
    return (
        f"🚀 {direction} SIGNAL\n\n"
        f"Symbol: {symbol}\n"
        f"TF: {TF_ENTRY} | HTF: {HTF}\n"
        f"Price: {price:.6g}\n"
        f"MFI({MFI_LEN}): {mfi_value:.2f}\n\n"
        f"Leverage: x{LEVERAGE}\n"
        f"TP: {TP_PCT:.2f}% | SL: {SL_PCT:.2f}%\n\n"
        f"#scanner"
    )

def heartbeat(last_hb: int) -> int:
    if HEARTBEAT_SEC <= 0:
        return last_hb
    t = now_ts()
    if (t - last_hb) >= HEARTBEAT_SEC:
        log(f"[{datetime.now(timezone.utc).isoformat()}] heartbeat OK")
        return t
    return last_hb

def main():
    log("✅ futures-scanner started")
    log(f"BOOT | BINANCE_FAPI={BINANCE_FAPI}")
    log(f"BOOT | TF_ENTRY={TF_ENTRY} HTF={HTF} EMA={EMA_FAST}/{EMA_SLOW} TOP_N={TOP_N} MIN_QV_24H={MIN_QUOTE_VOLUME_24H}")
    log(f"BOOT | Filters: BTC={USE_BTC_FILTER} HTF={USE_HTF_FILTER}(strict={HTF_STRICT_CROSS}) VOL={USE_VOL_FILTER}(quote={VOL_USE_QUOTE}) RSI={USE_RSI_FILTER} MACD0={USE_MACD_NEAR0} MFI={USE_MFI_FILTER}")
    log(f"BOOT | Params: RSI_LEN={RSI_LEN} RSI_MIN={RSI_MIN} MFI_L={MFI_LONG_MIN}-{MFI_LONG_MAX} MFI_S={MFI_SHORT_MIN}-{MFI_SHORT_MAX} VOL_MULT={VOL_MULT}")
    log(f"BOOT | Debug: DEBUG={DEBUG} DEBUG_REJECTS={DEBUG_REJECTS} DRY_RUN={DRY_RUN} TEST_ONCE={TEST_ONCE}")

    last_hb = now_ts()

    while True:
        reject_counts = {
            "DATA_SHORT": 0,
            "NO_CROSS": 0,
            "HTF_FAIL": 0,
            "BTC_FAIL": 0,
            "VOL_FAIL": 0,
            "RSI_FAIL": 0,
            "MACD0_FAIL": 0,
            "MFI_FAIL": 0,
        }

        try:
            last_hb = heartbeat(last_hb)

            symbols = get_symbols_top_by_volume(TOP_N, MIN_QUOTE_VOLUME_24H)
            log(f"Scanning {len(symbols)} symbols...")

            scanned = 0
            hits = 0

            for sym in symbols:
                scanned += 1
                if DEBUG and DEBUG_EVERY_N > 0 and (scanned % DEBUG_EVERY_N == 0):
                    log(f"PROGRESS | scanned={scanned}/{len(symbols)} hits={hits}")

                try:
                    res = entry_cross(sym, reject_counts)
                except Exception as e:
                    if DEBUG:
                        log(f"Symbol error {sym}: {repr(e)}")
                    continue

                if not res:
                    continue

                direction, price, mfi_value = res
                hits += 1

                if not can_send(sym, direction):
                    if DEBUG and DEBUG_REJECTS:
                        log(f"COOLDOWN {sym} {direction} (skip)")
                    continue

                msg = build_message(sym, direction, price, mfi_value)

                if DRY_RUN:
                    log(f"DRY_RUN WOULD_SEND: {sym} {direction} price={price:.6g} mfi={mfi_value:.2f}")
                else:
                    send_telegram(msg)
                    mark_sent(sym, direction)
                    log(f"SENT: {sym} {direction} price={price:.6g} mfi={mfi_value:.2f}")

            log(f"SCAN_DONE | scanned={scanned} hits={hits}")
            # ÖZET: hangi filtre kaç kez reddetti?
            log("REJECT_SUMMARY | " + " | ".join([f"{k}={v}" for k, v in reject_counts.items()]))

        except Exception as e:
            log(f"Loop error: {repr(e)}")

        if TEST_ONCE:
            log("TEST_ONCE=1 -> exiting after one scan cycle.")
            return

        time.sleep(INTERVAL_SEC)

if __name__ == "__main__":
    main()
