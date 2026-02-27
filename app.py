import os
import time
import math
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional

import requests

from notify import send_telegram
from storage import Storage

# =========================
# ENV / AYARLAR
# =========================
BINANCE_FAPI = os.getenv("BINANCE_FAPI", "https://fapi.binance.com").rstrip("/")

DEBUG = int(os.getenv("DEBUG", "0"))
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "60"))
HEARTBEAT_SEC = int(os.getenv("HEARTBEAT_SEC", "600"))

KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "260"))
TOP_N = int(os.getenv("TOP_N", "80"))
MIN_QUOTE_VOLUME = float(os.getenv("MIN_QUOTE_VOLUME", "3000000"))

# EMA
EMA_FAST = int(os.getenv("EMA_FAST", "3"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "44"))

# RSI (30m) – RSI123 + EMA47
RSI_LEN = int(os.getenv("RSI_LEN", "123"))
RSI_EMA_LEN = int(os.getenv("RSI_EMA_LEN", "47"))
RSI_MIN = float(os.getenv("RSI_MIN", "50"))
REQUIRE_RSI_CROSS = int(os.getenv("REQUIRE_RSI_CROSS", "0"))
RSI_CROSS_LOOKBACK = int(os.getenv("RSI_CROSS_LOOKBACK", "3"))

# MACD (5m)
USE_MACD_NEAR0 = int(os.getenv("USE_MACD_NEAR0", "0"))  # 1 => |macd| <= threshold
MACD_NEAR0 = float(os.getenv("MACD_NEAR0", "0.00002"))
MACD_POSITIVE = int(os.getenv("MACD_POSITIVE", "1"))    # 1 => macd>0 (near0 kapalıysa)

# BTC filtresi (opsiyonel)
USE_BTC_FILTER = int(os.getenv("USE_BTC_FILTER", "0"))
BTC_SYMBOL = os.getenv("BTC_SYMBOL", "BTCUSDT")
BTC_RSI_MIN = float(os.getenv("BTC_RSI_MIN", "0"))

# Storage / cooldown
USE_STORAGE = int(os.getenv("USE_STORAGE", "1"))
STORAGE_PATH = os.getenv("STORAGE_PATH", "/tmp/futures_scanner_storage.json")
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", "3600"))

# Risk / TP-Stop
ATR_LEN = int(os.getenv("ATR_LEN", "14"))
ATR_MULT = float(os.getenv("ATR_MULT", "1.0"))  # SwingLow - ATR*mult
SWING_LEN = int(os.getenv("SWING_LEN", "20"))   # SwingLow lookback (1H)
TP_R1 = float(os.getenv("TP_R1", "1.0"))
TP_R2 = float(os.getenv("TP_R2", "2.0"))
TP_R3 = float(os.getenv("TP_R3", "3.0"))

# Volume filter (opsiyonel)
USE_VOL_FILTER = int(os.getenv("USE_VOL_FILTER", "0"))
VOL_LEN = int(os.getenv("VOL_LEN", "20"))
VOL_MULT = float(os.getenv("VOL_MULT", "1.1"))

# timeframes (sabit)
TF_TREND_1H = "1h"
TF_TREND_30M = "30m"
TF_TRIGGER_5M = "5m"

# =========================
# HTTP
# =========================
def _get_json(url: str, params: Optional[dict] = None, timeout: int = 20):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def get_klines(symbol: str, interval: str, limit: int) -> List[List]:
    return _get_json(
        f"{BINANCE_FAPI}/fapi/v1/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
    )

def parse_ohlcv(kl: List[List]) -> Tuple[List[int], List[float], List[float], List[float], List[float], List[float]]:
    # returns close_time_ms, open, high, low, close, volume
    ct = [int(k[6]) for k in kl]
    o = [float(k[1]) for k in kl]
    h = [float(k[2]) for k in kl]
    l = [float(k[3]) for k in kl]
    c = [float(k[4]) for k in kl]
    v = [float(k[5]) for k in kl]
    return ct, o, h, l, c, v

def get_top_symbols_by_quote_volume(top_n: int, min_quote_vol: float) -> List[str]:
    data = _get_json(f"{BINANCE_FAPI}/fapi/v1/ticker/24hr")
    rows = []
    for x in data:
        sym = x.get("symbol", "")
        if not sym.endswith("USDT"):
            continue
        try:
            qv = float(x.get("quoteVolume", 0.0))
        except:
            continue
        if qv >= min_quote_vol:
            rows.append((sym, qv))
    rows.sort(key=lambda t: t[1], reverse=True)
    return [s for s, _ in rows[:top_n]]

# =========================
# INDICATORS
# =========================
def ema(series: List[float], length: int) -> List[float]:
    if not series or length <= 0:
        return []
    k = 2 / (length + 1)
    out = [series[0]]
    for i in range(1, len(series)):
        out.append(series[i] * k + out[-1] * (1 - k))
    return out

def crossed_up(prev_a: float, prev_b: float, now_a: float, now_b: float) -> bool:
    return (prev_a <= prev_b) and (now_a > now_b)

def rsi_series(closes: List[float], length: int) -> List[float]:
    if len(closes) < length + 1:
        return []
    rsis = [float("nan")] * len(closes)
    for i in range(length, len(closes)):
        gains = 0.0
        losses = 0.0
        for j in range(i - length + 1, i + 1):
            diff = closes[j] - closes[j - 1]
            if diff >= 0:
                gains += diff
            else:
                losses += -diff
        if losses == 0:
            rsis[i] = 100.0
        else:
            rs = gains / losses
            rsis[i] = 100 - (100 / (1 + rs))
    return rsis

def find_recent_cross_up(a: List[float], b: List[float], lookback: int) -> bool:
    if len(a) < lookback + 2 or len(b) < lookback + 2:
        return False
    for k in range(1, lookback + 1):
        if crossed_up(a[-(k+1)], b[-(k+1)], a[-k], b[-k]):
            return True
    return False

def macd_last(closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
    if len(closes) < slow + signal + 5:
        return (float("nan"), float("nan"), float("nan"))
    ef = ema(closes, fast)
    es = ema(closes, slow)
    n = min(len(ef), len(es))
    macd_line = [ef[-n + i] - es[-n + i] for i in range(n)]
    sig = ema(macd_line, signal)
    if not sig:
        return (float("nan"), float("nan"), float("nan"))
    hist = macd_line[-1] - sig[-1]
    return (macd_line[-1], sig[-1], hist)

def atr_wilder(highs: List[float], lows: List[float], closes: List[float], length: int) -> float:
    """
    Wilder ATR (RMA). Return last ATR.
    """
    if len(closes) < length + 2:
        return float("nan")

    trs: List[float] = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)

    # First ATR = SMA of first 'length' TRs
    first = sum(trs[:length]) / length
    atr = first
    for tr in trs[length:]:
        atr = (atr * (length - 1) + tr) / length
    return atr

def sma_last(series: List[float], length: int) -> float:
    if len(series) < length:
        return float("nan")
    return sum(series[-length:]) / length

# =========================
# FILTERS
# =========================
def btc_filter_ok() -> bool:
    if USE_BTC_FILTER != 1:
        return True
    try:
        kl = get_klines(BTC_SYMBOL, TF_TREND_1H, KLINE_LIMIT)
        _, _, _, _, closes, _ = parse_ohlcv(kl)
        rsi_btc = rsi_series(closes, RSI_LEN)
        rsi_btc_clean = [x for x in rsi_btc if not math.isnan(x)]
        if not rsi_btc_clean:
            return False
        return rsi_btc_clean[-1] >= BTC_RSI_MIN
    except Exception:
        return False

def check_signal(symbol: str) -> Optional[Dict]:
    # ---- 1H data (trend + ATR + swing + vol)
    kl1h = get_klines(symbol, TF_TREND_1H, KLINE_LIMIT)
    ct1h, o1h, h1h, l1h, c1h, v1h = parse_ohlcv(kl1h)

    e1h_fast = ema(c1h, EMA_FAST)
    e1h_slow = ema(c1h, EMA_SLOW)
    if not e1h_fast or not e1h_slow:
        return None
    trend_1h = e1h_fast[-1] > e1h_slow[-1]

    # ATR + SwingLow (1H)
    atr = atr_wilder(h1h, l1h, c1h, ATR_LEN)
    if math.isnan(atr):
        return None
    if len(l1h) < SWING_LEN + 2:
        return None
    swing_low = min(l1h[-SWING_LEN:])

    # Volume ratio (1H)
    vol_ratio = None
    vol_sma = sma_last(v1h, VOL_LEN)
    if not math.isnan(vol_sma) and vol_sma > 0:
        vol_ratio = v1h[-1] / vol_sma

    if USE_VOL_FILTER == 1:
        if vol_ratio is None:
            return None
        if vol_ratio < VOL_MULT:
            return None

    closed_1h_time = datetime.fromtimestamp(ct1h[-1] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ---- 30m data (trend + RSI123)
    kl30 = get_klines(symbol, TF_TREND_30M, KLINE_LIMIT)
    _, _, _, _, c30, _ = parse_ohlcv(kl30)

    e30_fast = ema(c30, EMA_FAST)
    e30_slow = ema(c30, EMA_SLOW)
    if not e30_fast or not e30_slow:
        return None
    trend_30m = e30_fast[-1] > e30_slow[-1]

    rsi30 = rsi_series(c30, RSI_LEN)
    if not rsi30 or math.isnan(rsi30[-1]):
        return None
    rsi30_clean = [x for x in rsi30 if not math.isnan(x)]
    if len(rsi30_clean) < RSI_EMA_LEN + 5:
        return None
    rsi30_ema = ema(rsi30_clean, RSI_EMA_LEN)

    rsi_last = rsi30_clean[-1]
    rsi_ema_last = rsi30_ema[-1]
    rsi_level_ok = rsi_last >= RSI_MIN
    rsi_above_ema = rsi_last > rsi_ema_last

    rsi_cross_ok = True
    if REQUIRE_RSI_CROSS == 1:
        n = min(len(rsi30_clean), len(rsi30_ema))
        a = rsi30_clean[-n:]
        b = rsi30_ema[-n:]
        rsi_cross_ok = find_recent_cross_up(a, b, RSI_CROSS_LOOKBACK)

    # ---- 5m trigger (EMA cross) + MACD
    kl5 = get_klines(symbol, TF_TRIGGER_5M, KLINE_LIMIT)
    ct5, _, _, _, c5, _ = parse_ohlcv(kl5)

    e5_fast = ema(c5, EMA_FAST)
    e5_slow = ema(c5, EMA_SLOW)
    if not e5_fast or not e5_slow or len(e5_fast) < 3 or len(e5_slow) < 3:
        return None
    cross_5m_up = crossed_up(e5_fast[-2], e5_slow[-2], e5_fast[-1], e5_slow[-1])

    macd_line, macd_sig, macd_hist = macd_last(c5)
    macd_ok = True
    if USE_MACD_NEAR0 == 1:
        macd_ok = (not math.isnan(macd_line)) and (abs(macd_line) <= MACD_NEAR0)
    else:
        if MACD_POSITIVE == 1:
            macd_ok = (not math.isnan(macd_line)) and (macd_line > 0)

    # ---- final OK
    ok = (
        trend_1h and
        trend_30m and
        rsi_level_ok and
        rsi_above_ema and
        rsi_cross_ok and
        cross_5m_up and
        macd_ok
    )
    if not ok:
        return None

    # ---- Trade levels
    entry = c5[-1]  # 5m last close
    sl = swing_low - (atr * ATR_MULT)
    risk = entry - sl
    if risk <= 0:
        return None

    tp1 = entry + risk * TP_R1
    tp2 = entry + risk * TP_R2
    tp3 = entry + risk * TP_R3

    # MACD hist info (senin ekrandaki gibi)
    macd_hist_str = "nan" if math.isnan(macd_hist) else f"{macd_hist:+.8f}"

    return {
        "symbol": symbol,
        "tf": TF_TREND_1H,
        "closed_time": closed_1h_time,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "rsi": rsi_last,
        "rsi_ema": rsi_ema_last,
        "macd_hist": macd_hist,
        "macd_hist_str": macd_hist_str,
        "macd_near0_ok": (abs(macd_line) <= MACD_NEAR0) if (USE_MACD_NEAR0 == 1 and not math.isnan(macd_line)) else None,
        "vol_ratio": vol_ratio,
        "vol_sma": vol_sma,
    }

# =========================
# TELEGRAM FORMAT
# =========================
def fmt(x: float) -> str:
    # küçük fiyatlar için daha çok digit
    if x == 0 or math.isnan(x) or math.isinf(x):
        return "nan"
    if abs(x) >= 1:
        return f"{x:.4f}"
    return f"{x:.6f}"

def build_signal_message(sig: Dict) -> str:
    # MACD near0 işareti
    near0_mark = ""
    if sig.get("macd_near0_ok") is True:
        near0_mark = " ✅"
    elif sig.get("macd_near0_ok") is False:
        near0_mark = " ❌"

    # Vol metni
    vol_line = ""
    if sig.get("vol_ratio") is not None and not math.isnan(sig.get("vol_sma", float("nan"))):
        vol_line = f"Vol: {sig['vol_ratio']:.2f}x (vs SMA{VOL_LEN}, mult={VOL_MULT})"

    # Plan
    plan = [
        "Plan:",
        "- TP1 görünce SL = ENTRY (break-even)",
        "- Kapanış EMA44 altına inerse çık (trail)",
    ]

    msg = []
    msg.append("🚀 LONG BREAKOUT (EMA3↑EMA44)")
    msg.append(f"SYMBOL: {sig['symbol']}")
    msg.append(f"TF: {sig['tf']}")
    msg.append(f"Time (closed candle): {sig['closed_time']}")
    msg.append("")
    msg.append(f"ENTRY: {fmt(sig['entry'])}")
    msg.append(f"SL:   {fmt(sig['sl'])}  (SwingLow-ATR)")
    msg.append(f"TP1:  {fmt(sig['tp1'])}  (1R)")
    msg.append(f"TP2:  {fmt(sig['tp2'])}  (2R)")
    msg.append(f"TP3:  {fmt(sig['tp3'])}  (3R)")
    msg.append("")
    msg.append(f"RSI({RSI_LEN}): {sig['rsi']:.2f} (↑)")
    msg.append(f"MACD hist: {sig['macd_hist_str']} (↑){near0_mark}")
    if vol_line:
        msg.append(vol_line)
    msg.append("")
    msg.extend(plan)
    msg.append("")
    msg.append("#scanner")
    return "\n".join(msg)

# =========================
# MAIN
# =========================
def main():
    # Storage / cooldown wrapper (enabled param yoksa böyle çöz)
storage = Storage(STORAGE_PATH, cooldown_sec=COOLDOWN_SEC)

def can_send(key: str) -> bool:
    if USE_STORAGE == 1:
        return storage.can_send(key)
    return True

def mark_sent(key: str):
    if USE_STORAGE == 1:
        storage.mark_sent(key)

    print("✅ futures-scanner started (BOT mode)")
    print(f"TOP_N={TOP_N} MIN_QUOTE_VOLUME={MIN_QUOTE_VOLUME} INTERVAL_SEC={INTERVAL_SEC}")
    print(f"EMA_FAST={EMA_FAST} EMA_SLOW={EMA_SLOW} | RSI_LEN={RSI_LEN} RSI_EMA_LEN={RSI_EMA_LEN} RSI_MIN={RSI_MIN}")
    print(f"ATR_LEN={ATR_LEN} ATR_MULT={ATR_MULT} SWING_LEN={SWING_LEN} | TP: {TP_R1}/{TP_R2}/{TP_R3}R")
    print(f"VOL_FILTER={USE_VOL_FILTER} VOL_LEN={VOL_LEN} VOL_MULT={VOL_MULT}")
    print(f"USE_STORAGE={USE_STORAGE} COOLDOWN_SEC={COOLDOWN_SEC}")

    last_hb = time.time()

    while True:
        try:
            if not btc_filter_ok():
                if DEBUG:
                    print("BTC filter: FAIL (skip cycle)")
                time.sleep(INTERVAL_SEC)
                continue

            symbols = get_top_symbols_by_quote_volume(TOP_N, MIN_QUOTE_VOLUME)

            sent_count = 0
            scanned = 0
            for sym in symbols:
                scanned += 1
                try:
                    sig = check_signal(sym)
                    if not sig:
                        continue

                    key = f"{sym}:LONG_EMA_MTF_RSI123_ATR"
                   if can_send(key):
                        text = build_signal_message(sig)
                        if DEBUG:
                            print(text)
                        send_telegram(text)
                         mark_sent(key)
                        sent_count += 1
                        time.sleep(1)  # telegram rate limit yumuşatma
                except Exception:
                    continue

            if DEBUG:
                print(f"candidates={len(symbols)} scanned={scanned} signals={sent_count}")

           if HEARTBEAT_SEC > 0 and (time.time() - last_hb) >= HEARTBEAT_SEC:
    hb = f"💓 HB | TF={TF_TREND_1H} | TOP_N={TOP_N} | MIN_QV={MIN_QUOTE_VOLUME} | BTC filter={'ON' if USE_BTC_FILTER==1 else 'OFF'}"
    print(hb)  # DEBUG'e bağlı olmasın
    last_hb = time.time()

        except Exception as e:
            print(f"⚠️ main loop error: {e}")

        time.sleep(INTERVAL_SEC)

if __name__ == "__main__":
    main()
