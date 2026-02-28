import os
import time
import math
from datetime import datetime, timezone
from typing import List, Tuple, Optional

import requests

from notify import send_telegram
from storage import Storage

# ============================================================
# Futures Scanner (EMA cross + filters) + MFI filter + REJECT DEBUG
# + NET DEBUG ENTEGRASYONU (heartbeat / dry-run / test-once)
# ============================================================

BINANCE_FAPI = os.getenv("BINANCE_FAPI", "https://fapi.binance.com").rstrip("/")

# --- Timeframes / polling
TF_ENTRY = os.getenv("TF_ENTRY", os.getenv("TF", "5m"))
TF_TREND = os.getenv("TF_TREND", os.getenv("HTF", "1h"))
HTF = os.getenv("HTF", TF_TREND)
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "180"))
KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "260"))

# --- Universe / liquidity
TOP_N = int(os.getenv("TOP_N", "400"))
MIN_QUOTE_VOLUME = float(os.getenv("MIN_QUOTE_VOLUME", "3000000"))

# --- Signal logic (EMA cross)
EMA_FAST = int(os.getenv("EMA_FAST", "3"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "44"))
HTF_CROSS_LOOKBACK = int(os.getenv("HTF_CROSS_LOOKBACK", "6"))

# --- BTC filter (optional)
USE_BTC_FILTER = int(os.getenv("USE_BTC_FILTER", "1"))
BTC_SYMBOL = os.getenv("BTC_SYMBOL", "BTCUSDT")
BTC_TF = os.getenv("BTC_TF", TF_TREND)

# --- Volume filter (optional)
USE_VOL_FILTER = int(os.getenv("USE_VOL_FILTER", "1"))
VOL_LEN = int(os.getenv("VOL_LEN", "20"))
VOL_MULT = float(os.getenv("VOL_MULT", "1.1"))

# --- MACD near-zero filter (optional)
USE_MACD_NEAR0 = int(os.getenv("USE_MACD_NEAR0", "1"))
MACD_FAST = int(os.getenv("MACD_FAST", "12"))
MACD_SLOW = int(os.getenv("MACD_SLOW", "26"))
MACD_SIGNAL = int(os.getenv("MACD_SIGNAL", "9"))
MACD_NEAR0_PCT = float(os.getenv("MACD_NEAR0_PCT", "0.15"))  # histogram near 0 as % of price

# --- RSI filter (optional)
USE_RSI_FILTER = int(os.getenv("USE_RSI_FILTER", "1"))
RSI_LEN = int(os.getenv("RSI_LEN", "123"))
RSI_EMA_LEN = int(os.getenv("RSI_EMA_LEN", "47"))
RSI_MIN = float(os.getenv("RSI_MIN", "50"))

# --- MFI filter
USE_MFI_FILTER = int(os.getenv("USE_MFI_FILTER", "1"))
MFI_LEN = int(os.getenv("MFI_LEN", "14"))
MFI_LONG_MIN = float(os.getenv("MFI_LONG_MIN", "50"))
MFI_LONG_MAX = float(os.getenv("MFI_LONG_MAX", "80"))
MFI_SHORT_MIN = float(os.getenv("MFI_SHORT_MIN", "20"))
MFI_SHORT_MAX = float(os.getenv("MFI_SHORT_MAX", "50"))

# --- Risk params (message only; you enter on Binance)
LEVERAGE = int(os.getenv("LEVERAGE", "5"))
TP_PCT = float(os.getenv("TP_PCT", "5"))
SL_PCT = float(os.getenv("SL_PCT", "2"))
TP_R1 = float(os.getenv("TP_R1", "1"))
TP_R2 = float(os.getenv("TP_R2", "2"))
TP_R3 = float(os.getenv("TP_R3", "3"))

# --- Cooldown / state
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", "21600"))
USE_STORAGE = int(os.getenv("USE_STORAGE", "1"))
STORAGE_PATH = os.getenv("STORAGE_PATH", "/var/data/futures_state.json")

# --- Debug / health
DEBUG = int(os.getenv("DEBUG", "1"))
DEBUG_REJECTS = int(os.getenv("DEBUG_REJECTS", "1"))  # 1=print reject reasons
HEARTBEAT_SEC = int(os.getenv("HEARTBEAT_SEC", "900"))

# --- NEW: test switches (çalışıyor mu teyidi)
DRY_RUN = int(os.getenv("DRY_RUN", "0"))          # 1 => telegrama gondermez, log basar
TEST_ONCE = int(os.getenv("TEST_ONCE", "0"))      # 1 => tek tur scan yapar ve cikar
DEBUG_EVERY_N = int(os.getenv("DEBUG_EVERY_N", "0"))  # 50 gibi => her N symbol'de progress log

storage = Storage(STORAGE_PATH) if USE_STORAGE else None


# ============================================================
# Helpers: Binance
# ============================================================

def _get(url: str, params: dict, timeout: int = 12):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def get_symbols_top_by_volume(top_n: int, min_quote_volume: float) -> List[str]:
    url = f"{BINANCE_FAPI}/fapi/v1/ticker/24hr"
    data = _get(url, params={}, timeout=20)

    rows = []
    for d in data:
        sym = d.get("symbol", "")
        if not sym.endswith("USDT"):
            continue
        qv = float(d.get("quoteVolume", 0.0) or 0.0)
        if qv < min_quote_volume:
            continue
        rows.append((sym, qv))

    rows.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in rows[:top_n]]

def get_klines(symbol: str, interval: str, limit: int) -> List[list]:
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    return _get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=15)


# ============================================================
# Indicators
# ============================================================

def ema(series: List[float], length: int) -> List[float]:
    if length <= 0:
        raise ValueError("EMA length must be > 0")
    if not series:
        return []
    k = 2 / (length + 1)
    out = [series[0]]
    for x in series[1:]:
        out.append(out[-1] + k * (x - out[-1]))
    return out

def rsi_wilder(series: List[float], length: int) -> List[float]:
    if length <= 0:
        raise ValueError("RSI length must be > 0")
    if len(series) < length + 1:
        return [50.0] * len(series)

    gains, losses = [], []
    for i in range(1, len(series)):
        ch = series[i] - series[i-1]
        gains.append(max(ch, 0.0))
        losses.append(max(-ch, 0.0))

    avg_gain = sum(gains[:length]) / length
    avg_loss = sum(losses[:length]) / length

    rsis = [50.0] * (length)
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
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = [a - b for a, b in zip(ema_fast, ema_slow)]
    signal_line = ema(macd_line, signal)
    return [m - s for m, s in zip(macd_line, signal_line)]

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


# ============================================================
# Parsing klines
# ============================================================

def parse_hlcva(klines: List[list]) -> Tuple[List[float], List[float], List[float], List[float]]:
    high = [float(k[2]) for k in klines]
    low  = [float(k[3]) for k in klines]
    close = [float(k[4]) for k in klines]
    vol = [float(k[5]) for k in klines]
    return high, low, close, vol

def parse_close(klines: List[list]) -> List[float]:
    return [float(k[4]) for k in klines]


# ============================================================
# Filters / conditions
# ============================================================

def btc_ok() -> bool:
    if not USE_BTC_FILTER:
        return True
    try:
        k = get_klines(BTC_SYMBOL, BTC_TF, max(EMA_SLOW, 80))
        closes = parse_close(k)
        e_fast = ema(closes, EMA_FAST)[-1]
        e_slow = ema(closes, EMA_SLOW)[-1]
        return e_fast >= e_slow
    except Exception as e:
        if DEBUG:
            print("BTC filter error:", e)
        return True  # hata olursa filtreyi fail yapmayalım

def vol_ok(volume: List[float]) -> bool:
    if not USE_VOL_FILTER:
        return True
    if len(volume) < VOL_LEN + 1:
        return True
    v = volume[-1]
    ma = sum(volume[-VOL_LEN:]) / VOL_LEN
    return v >= ma * VOL_MULT

def rsi_ok(closes: List[float], direction: str) -> Tuple[bool, float, float]:
    if not USE_RSI_FILTER:
        return True, float("nan"), float("nan")
    r = rsi_wilder(closes, RSI_LEN)
    r_last = r[-1]
    r_ema = ema(r, RSI_EMA_LEN)[-1]
    if direction == "LONG":
        return (r_last >= RSI_MIN) and (r_last >= r_ema), r_last, r_ema
    else:
        return (r_last <= (100 - RSI_MIN)) and (r_last <= r_ema), r_last, r_ema

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
    if not int(os.getenv("USE_HTF_FILTER", "1")):
        return True
    try:
        k = get_klines(symbol, HTF, max(EMA_SLOW + HTF_CROSS_LOOKBACK + 5, 120))
        closes = parse_close(k)
        ef = ema(closes, EMA_FAST)
        es = ema(closes, EMA_SLOW)

        for i in range(1, HTF_CROSS_LOOKBACK + 1):
            a1, b1 = ef[-i-1], es[-i-1]
            a2, b2 = ef[-i], es[-i]
            if direction == "LONG" and (a1 <= b1 and a2 > b2):
                return True
            if direction == "SHORT" and (a1 >= b1 and a2 < b2):
                return True

        if direction == "LONG":
            return ef[-1] >= es[-1]
        return ef[-1] <= es[-1]
    except Exception as e:
        if DEBUG:
            print("HTF confirm error:", symbol, e)
        return True  # hata olursa filtreyi fail yapmayalım


# ============================================================
# Message building
# ============================================================

def calc_tp_sl(price: float, direction: str) -> Tuple[float, float, float, float]:
    if direction == "LONG":
        sl = price * (1 - SL_PCT / 100.0)
    else:
        sl = price * (1 + SL_PCT / 100.0)

    tp1 = price * (1 + (TP_R1 / 100.0)) if direction == "LONG" else price * (1 - (TP_R1 / 100.0))
    tp2 = price * (1 + (TP_R2 / 100.0)) if direction == "LONG" else price * (1 - (TP_R2 / 100.0))
    tp3 = price * (1 + (TP_R3 / 100.0)) if direction == "LONG" else price * (1 - (TP_R3 / 100.0))
    return sl, tp1, tp2, tp3

def build_message(symbol: str, direction: str, price: float, mfi_value: float = float("nan")) -> str:
    sl, tp1, tp2, tp3 = calc_tp_sl(price, direction)
    mfi_part = f"\nMFI({MFI_LEN}): {mfi_value:.2f}" if (USE_MFI_FILTER and not math.isnan(mfi_value)) else ""
    return (
        f"🚀 {direction} SIGNAL\n\n"
        f"Symbol: {symbol}\n"
        f"TF: {TF_ENTRY} | HTF: {HTF}\n"
        f"Price: {price:.6g}{mfi_part}\n\n"
        f"Leverage: x{LEVERAGE}\n"
        f"TP: {TP_PCT:.2f}%  (TP1 {tp1:.6g} | TP2 {tp2:.6g} | TP3 {tp3:.6g})\n"
        f"SL: {SL_PCT:.2f}%  ({sl:.6g})\n\n"
        f"#scanner"
    )


# ============================================================
# Debug reject reasons
# ============================================================

def reject_log(symbol: str, direction: str, reasons: List[str]):
    if DEBUG and DEBUG_REJECTS and reasons:
        print(f"REJECT {symbol} {direction}: " + " | ".join(reasons))

def evaluate_filters(symbol: str, direction: str, high: List[float], low: List[float], close: List[float], vol: List[float]) -> Tuple[bool, float]:
    """
    Returns (ok, mfi_value). Prints reject reasons when DEBUG_REJECTS=1.
    """
    reasons = []

    # HTF
    if not htf_confirm(symbol, direction):
        reasons.append("HTF_CONFIRM_FAIL")

    # BTC (sadece LONG tarafinda)
    if direction == "LONG" and USE_BTC_FILTER and not btc_ok():
        reasons.append("BTC_FILTER_FAIL")

    # VOL
    if not vol_ok(vol):
        v = vol[-1] if vol else float("nan")
        reasons.append(f"VOL_FAIL(v={v:.4g}, mult={VOL_MULT})")

    # RSI
    ok_rsi, r_last, r_ema = rsi_ok(close, direction)
    if not ok_rsi:
        reasons.append(f"RSI_FAIL(r={r_last:.2f}, ema={r_ema:.2f}, min={RSI_MIN})")

    # MACD
    ok_macd, h = macd_ok(close)
    if not ok_macd:
        reasons.append(f"MACD_NEAR0_FAIL(hist={h:.6g}, near0%={MACD_NEAR0_PCT})")

    # MFI
    ok_mfi, mfi_value = mfi_ok(high, low, close, vol, direction)
    if not ok_mfi:
        if direction == "LONG":
            reasons.append(f"MFI_FAIL(mfi={mfi_value:.2f}, range={MFI_LONG_MIN}-{MFI_LONG_MAX})")
        else:
            reasons.append(f"MFI_FAIL(mfi={mfi_value:.2f}, range={MFI_SHORT_MIN}-{MFI_SHORT_MAX})")

    if reasons:
        reject_log(symbol, direction, reasons)
        return False, mfi_value

    return True, mfi_value


# ============================================================
# Core: detect entry EMA cross on TF_ENTRY
# ============================================================

def entry_cross(symbol: str) -> Optional[Tuple[str, float, float]]:
    kl = get_klines(symbol, TF_ENTRY, max(KLINE_LIMIT, EMA_SLOW + 50))
    high, low, close, vol = parse_hlcva(kl)

    if len(close) < EMA_SLOW + 5:
        return None

    ef = ema(close, EMA_FAST)
    es = ema(close, EMA_SLOW)

    # last completed bar (bar kapanisi)
    a1, b1 = ef[-3], es[-3]
    a2, b2 = ef[-2], es[-2]

    direction = None
    if a1 <= b1 and a2 > b2:
        direction = "LONG"
    elif a1 >= b1 and a2 < b2:
        direction = "SHORT"
    else:
        return None

    ok, mfi_value = evaluate_filters(symbol, direction, high, low, close, vol)
    if not ok:
        return None

    price = close[-1]
    return direction, price, mfi_value


# ============================================================
# Runner helpers
# ============================================================

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

def heartbeat(last_hb: int) -> int:
    if HEARTBEAT_SEC <= 0:
        return last_hb
    t = now_ts()
    if (t - last_hb) >= HEARTBEAT_SEC:
        print(f"[{datetime.now(timezone.utc).isoformat()}] heartbeat OK (TOP_N={TOP_N}, TF={TF_ENTRY}/{HTF}, interval={INTERVAL_SEC}s)")
        return t
    return last_hb


# ============================================================
# Main
# ============================================================

def main():
    print("✅ futures-scanner started")
    print(f"BOOT | BINANCE_FAPI={BINANCE_FAPI}")
    print(f"BOOT | TF_ENTRY={TF_ENTRY} HTF={HTF} EMA={EMA_FAST}/{EMA_SLOW} TOP_N={TOP_N} MIN_QV={MIN_QUOTE_VOLUME}")
    print(f"BOOT | Filters: BTC={USE_BTC_FILTER} HTF={int(os.getenv('USE_HTF_FILTER','1'))} VOL={USE_VOL_FILTER} RSI={USE_RSI_FILTER} MACD0={USE_MACD_NEAR0} MFI={USE_MFI_FILTER}")
    print(f"BOOT | Debug: DEBUG={DEBUG} DEBUG_REJECTS={DEBUG_REJECTS} DRY_RUN={DRY_RUN} TEST_ONCE={TEST_ONCE} DEBUG_EVERY_N={DEBUG_EVERY_N}")
    print(f"BOOT | Storage: USE_STORAGE={USE_STORAGE} PATH={STORAGE_PATH} COOLDOWN_SEC={COOLDOWN_SEC}")

    last_hb = now_ts()

    while True:
        try:
            last_hb = heartbeat(last_hb)

            symbols = get_symbols_top_by_volume(TOP_N, MIN_QUOTE_VOLUME)
            if DEBUG:
                print(f"Scanning {len(symbols)} symbols...")

            scanned = 0
            hits = 0

            for sym in symbols:
                scanned += 1

                if DEBUG and DEBUG_EVERY_N > 0 and (scanned % DEBUG_EVERY_N == 0):
                    print(f"PROGRESS | scanned={scanned}/{len(symbols)} hits={hits}")

                try:
                    res = entry_cross(sym)
                    if not res:
                        continue

                    direction, price, mfi_value = res
                    hits += 1

                    if not can_send(sym, direction):
                        if DEBUG and DEBUG_REJECTS:
                            print(f"COOLDOWN {sym} {direction} (skip)")
                        continue

                    msg = build_message(sym, direction, price, mfi_value=mfi_value)

                    if DRY_RUN:
                        print("DRY_RUN WOULD_SEND:", sym, direction, "price", price, "mfi", mfi_value)
                    else:
                        send_telegram(msg)
                        mark_sent(sym, direction)
                        if DEBUG:
                            print("SENT:", sym, direction, "price", price, "mfi", mfi_value)

                except Exception as e:
                    if DEBUG:
                        print("Symbol error:", sym, e)
                    continue

            if DEBUG:
                print(f"SCAN_DONE | scanned={scanned} hits={hits}")

        except Exception as e:
            print("Loop error:", e)

        if TEST_ONCE:
            print("TEST_ONCE=1 -> exiting after one scan cycle.")
            return

        time.sleep(INTERVAL_SEC)

if __name__ == "__main__":
    main()
