# app.py
import os
import time
import math
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests

from notify import send_telegram
from storage import Storage

# =========================
# ENV / AYARLAR
# =========================
BINANCE_FAPI = os.getenv("BINANCE_FAPI", "https://fapi.binance.com").rstrip("/")

TF = os.getenv("TF", "30m")  # Sinyal timeframe
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "600"))
KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "200"))

# Trend Indicator A (v2.3) yaklaşımı:
# Heikin Ashi + EMA(len=9) -> renk: HA close EMA üstü bullish, altı bearish
USE_HEIKIN_ASHI = int(os.getenv("USE_HEIKIN_ASHI", "1")) == 1
TREND_MA_TYPE = os.getenv("TREND_MA_TYPE", "EMA").upper()
TREND_MA_LEN = int(os.getenv("TREND_MA_LEN", "9"))
TREND_CONFIRM_CLOSE = int(os.getenv("TREND_CONFIRM_CLOSE", "1")) == 1  # candle close ile onay

# RSI
RSI_LEN = int(os.getenv("RSI_LEN", "14"))
RSI_MIN = float(os.getenv("RSI_MIN", "42"))
RSI_CROSS_CONFIRM = int(os.getenv("RSI_CROSS_CONFIRM", "1")) == 1
RSI_MA_LEN = int(os.getenv("RSI_MA_LEN", "14"))  # mor (RSI) sarıyı (RSI MA) kessin

# MACD
MACD_FAST = int(os.getenv("MACD_FAST", "12"))
MACD_SLOW = int(os.getenv("MACD_SLOW", "26"))
MACD_SIGNAL = int(os.getenv("MACD_SIGNAL", "9"))
MACD_CROSS_CONFIRM = int(os.getenv("MACD_CROSS_CONFIRM", "1")) == 1
MACD_ZERO_FILTER = int(os.getenv("MACD_ZERO_FILTER", "0")) == 1
MACD_ZERO_EPS = float(os.getenv("MACD_ZERO_EPS", "0.02"))  # 0'a yakın tolerans

# ATR / TP-SL
ATR_LEN = int(os.getenv("ATR_LEN", "14"))
ATR_SL_MULT = float(os.getenv("ATR_SL_MULT", "1.5"))
TP1_ATR_MULT = float(os.getenv("TP1_ATR_MULT", "1.0"))
TP2_ATR_MULT = float(os.getenv("TP2_ATR_MULT", "2.0"))

# PSAR
USE_PARABOLIC_SAR = int(os.getenv("USE_PARABOLIC_SAR", "1")) == 1
PSAR_AF_STEP = float(os.getenv("PSAR_AF_STEP", "0.02"))
PSAR_AF_MAX = float(os.getenv("PSAR_AF_MAX", "0.2"))

# Tarama evreni
SCAN_ALL_USDT_PERPS = int(os.getenv("SCAN_ALL_USDT_PERPS", "1")) == 1

# 24h Volume filter (default kapalı)
USE_24H_VOLUME_FILTER = int(os.getenv("USE_24H_VOLUME_FILTER", "0")) == 1
MIN_QUOTE_VOLUME_24H = float(os.getenv("MIN_QUOTE_VOLUME_24H", "3000000"))

# BTC filtresi (default kapalı)
USE_BTC_FILTER = int(os.getenv("USE_BTC_FILTER", "0")) == 1
BTC_SYMBOL = os.getenv("BTC_SYMBOL", "BTCUSDT")
BTC_TF = os.getenv("BTC_TF", "1h")
BTC_RSI_MIN = float(os.getenv("BTC_RSI_MIN", "42"))

# Multi-timeframe RSI prefilter (LONG için oversold seçimi)
# Senin kilitlediğin: 1M<=10, 1W<=20, 1D<=30, 4H<=30, 1H<=30
USE_MTF_RSI_PREFILTER = int(os.getenv("USE_MTF_RSI_PREFILTER", "1")) == 1
MTF_RSI_1M_MAX = float(os.getenv("MTF_RSI_1M_MAX", "10"))
MTF_RSI_1W_MAX = float(os.getenv("MTF_RSI_1W_MAX", "20"))
MTF_RSI_1D_MAX = float(os.getenv("MTF_RSI_1D_MAX", "30"))
MTF_RSI_4H_MAX = float(os.getenv("MTF_RSI_4H_MAX", "30"))
MTF_RSI_1H_MAX = float(os.getenv("MTF_RSI_1H_MAX", "30"))

# Cooldown / tekrar engelleme
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", "0"))  # 0 = kapalı
CLOSE_COOLDOWN_SEC = int(os.getenv("CLOSE_COOLDOWN_SEC", "0"))  # şimdilik kapalı
STORAGE_PATH = os.getenv("STORAGE_PATH", "/tmp/futures_scanner_storage.json")

# Debug
DEBUG = int(os.getenv("DEBUG", "1")) == 1
DEBUG_REJECTS = int(os.getenv("DEBUG_REJECTS", "0")) == 1
DRY_RUN = int(os.getenv("DRY_RUN", "0")) == 1

# Hız / oran limit
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "15"))
SYMBOL_SLEEP_SEC = float(os.getenv("SYMBOL_SLEEP_SEC", "0.05"))  # rate limit için
MAX_SYMBOLS_PER_CYCLE = int(os.getenv("MAX_SYMBOLS_PER_CYCLE", "0"))  # 0 = sınırsız (tüm Binance futures)

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# =========================
# HTTP Helpers
# =========================
sess = requests.Session()


def http_get(path: str, params: Optional[dict] = None):
    url = f"{BINANCE_FAPI}{path}"
    r = sess.get(url, params=params or {}, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_klines(symbol: str, interval: str, limit: int = 200):
    return http_get("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})


def get_exchange_symbols_usdt_perp() -> List[str]:
    info = http_get("/fapi/v1/exchangeInfo")
    out = []
    for s in info.get("symbols", []):
        if s.get("status") != "TRADING":
            continue
        if s.get("contractType") != "PERPETUAL":
            continue
        if s.get("quoteAsset") != "USDT":
            continue
        # Binance USDT perpetual format: XXXUSDT
        sym = s.get("symbol")
        if not sym or not sym.endswith("USDT"):
            continue
        out.append(sym)
    return out


def get_24h_quote_volumes() -> Dict[str, float]:
    # /fapi/v1/ticker/24hr returns list
    data = http_get("/fapi/v1/ticker/24hr")
    vols = {}
    for x in data:
        sym = x.get("symbol")
        if not sym:
            continue
        # quoteVolume is in quote asset (USDT)
        qv = x.get("quoteVolume")
        try:
            vols[sym] = float(qv)
        except Exception:
            continue
    return vols


# =========================
# Indicator math
# =========================
def ema(values: List[float], length: int) -> List[float]:
    if length <= 1:
        return values[:]
    out = []
    k = 2 / (length + 1)
    e = None
    for v in values:
        if e is None:
            e = v
        else:
            e = v * k + e * (1 - k)
        out.append(e)
    return out


def sma(values: List[float], length: int) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    if length <= 0:
        return [None] * len(values)
    s = 0.0
    q: List[float] = []
    for v in values:
        q.append(v)
        s += v
        if len(q) > length:
            s -= q.pop(0)
        if len(q) == length:
            out.append(s / length)
        else:
            out.append(None)
    return out


def rsi(close: List[float], length: int) -> List[Optional[float]]:
    if length <= 0 or len(close) < length + 1:
        return [None] * len(close)
    gains = [0.0]
    losses = [0.0]
    for i in range(1, len(close)):
        ch = close[i] - close[i - 1]
        gains.append(max(ch, 0.0))
        losses.append(max(-ch, 0.0))

    avg_gain = None
    avg_loss = None
    out: List[Optional[float]] = [None] * len(close)

    for i in range(1, len(close)):
        if i < length:
            continue
        if i == length:
            avg_gain = sum(gains[1:length + 1]) / length
            avg_loss = sum(losses[1:length + 1]) / length
        else:
            assert avg_gain is not None and avg_loss is not None
            avg_gain = (avg_gain * (length - 1) + gains[i]) / length
            avg_loss = (avg_loss * (length - 1) + losses[i]) / length

        if avg_loss == 0:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out


def macd(close: List[float], fast: int, slow: int, signal: int) -> Tuple[List[float], List[float], List[float]]:
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = [a - b for a, b in zip(ema_fast, ema_slow)]
    signal_line = ema(macd_line, signal)
    hist = [m - s for m, s in zip(macd_line, signal_line)]
    return macd_line, signal_line, hist


def true_range(high: List[float], low: List[float], close: List[float]) -> List[float]:
    tr = [high[0] - low[0]]
    for i in range(1, len(close)):
        tr.append(max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])))
    return tr


def atr(high: List[float], low: List[float], close: List[float], length: int) -> List[Optional[float]]:
    if len(close) < length + 1:
        return [None] * len(close)
    tr = true_range(high, low, close)
    # Wilder smoothing
    out: List[Optional[float]] = [None] * len(close)
    atr_val = None
    for i in range(len(close)):
        if i < length:
            continue
        if i == length:
            atr_val = sum(tr[1:length + 1]) / length
        else:
            assert atr_val is not None
            atr_val = (atr_val * (length - 1) + tr[i]) / length
        out[i] = atr_val
    return out


def heikin_ashi(ohlc: List[Tuple[float, float, float, float]]) -> List[Tuple[float, float, float, float]]:
    # returns HA (open, high, low, close)
    ha = []
    prev_ha_open = None
    prev_ha_close = None
    for i, (o, h, l, c) in enumerate(ohlc):
        ha_close = (o + h + l + c) / 4.0
        if prev_ha_open is None:
            ha_open = (o + c) / 2.0
        else:
            ha_open = (prev_ha_open + prev_ha_close) / 2.0
        ha_high = max(h, ha_open, ha_close)
        ha_low = min(l, ha_open, ha_close)
        ha.append((ha_open, ha_high, ha_low, ha_close))
        prev_ha_open, prev_ha_close = ha_open, ha_close
    return ha


def psar(high: List[float], low: List[float], step: float = 0.02, max_af: float = 0.2) -> List[Optional[float]]:
    # Basic Parabolic SAR implementation
    n = len(high)
    if n < 3:
        return [None] * n

    sar: List[Optional[float]] = [None] * n

    # initial trend
    up = high[1] > high[0]
    ep = high[1] if up else low[1]
    af = step
    sar[1] = low[0] if up else high[0]

    for i in range(2, n):
        prev_sar = sar[i - 1]
        if prev_sar is None:
            sar[i] = None
            continue

        new_sar = prev_sar + af * (ep - prev_sar)

        if up:
            new_sar = min(new_sar, low[i - 1], low[i - 2])
            if low[i] < new_sar:
                # switch to down
                up = False
                sar[i] = ep
                ep = low[i]
                af = step
            else:
                sar[i] = new_sar
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + step, max_af)
        else:
            new_sar = max(new_sar, high[i - 1], high[i - 2])
            if high[i] > new_sar:
                # switch to up
                up = True
                sar[i] = ep
                ep = high[i]
                af = step
            else:
                sar[i] = new_sar
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + step, max_af)

    return sar


# =========================
# Signal logic
# =========================
def _pick_closed_indexes(n: int) -> Tuple[int, int]:
    """
    Returns (prev_idx, cur_idx) for "closed candle" logic.
    Binance klines last item often is still forming.
    So we use -2 as "current closed", -3 as "previous closed".
    """
    if n < 4:
        return -2, -1
    return n - 3, n - 2


def trend_color_from_series(close_series: List[float], ma_len: int) -> List[str]:
    ma = ema(close_series, ma_len) if TREND_MA_TYPE == "EMA" else [x for x in close_series]
    colors = []
    for c, m in zip(close_series, ma):
        colors.append("green" if c >= m else "red")
    return colors


def crosses_up(prev_a: float, prev_b: float, cur_a: float, cur_b: float) -> bool:
    return prev_a <= prev_b and cur_a > cur_b


def build_message(
    symbol: str,
    tf: str,
    cur_close: float,
    sar_val: Optional[float],
    atr_val: Optional[float],
    rsi_val: float,
    macd_val: float,
    macd_sig: float,
    tp1: Optional[float],
    tp2: Optional[float],
    sl: Optional[float],
) -> str:
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        f"✅ LONG SİNYALİ",
        f"🪙 {symbol}  | TF: {tf}",
        f"⏱ {now}",
        "",
        f"Entry (close): {cur_close:.8f}".rstrip("0").rstrip("."),
    ]
    if sar_val is not None:
        lines.append(f"SAR: {sar_val:.8f}".rstrip("0").rstrip("."))
    if atr_val is not None:
        lines.append(f"ATR({ATR_LEN}): {atr_val:.8f}".rstrip("0").rstrip("."))
    lines += [
        "",
        f"RSI({RSI_LEN}): {rsi_val:.2f}",
        f"MACD: {macd_val:.6f} | Signal: {macd_sig:.6f}",
    ]
    if sl is not None:
        lines.append(f"🛡 SL (ATR×{ATR_SL_MULT:g}): {sl:.8f}".rstrip("0").rstrip("."))
    if tp1 is not None:
        lines.append(f"🎯 TP1 (ATR×{TP1_ATR_MULT:g}): {tp1:.8f}".rstrip("0").rstrip("."))
    if tp2 is not None:
        lines.append(f"🎯 TP2 (ATR×{TP2_ATR_MULT:g}): {tp2:.8f}".rstrip("0").rstrip("."))
    lines += ["", "Not: Candle close onaylıdır. (Test)"]
    return "\n".join(lines)


def check_btc_filter() -> bool:
    if not USE_BTC_FILTER:
        return True
    try:
        kl = get_klines(BTC_SYMBOL, BTC_TF, limit=max(100, RSI_LEN + 20))
        closes = [float(x[4]) for x in kl]
        rs = rsi(closes, RSI_LEN)
        prev_i, cur_i = _pick_closed_indexes(len(closes))
        cur_rsi = rs[cur_i]
        if cur_rsi is None:
            return False
        return cur_rsi >= BTC_RSI_MIN
    except Exception:
        return False


def mtf_rsi_pass(symbol: str) -> bool:
    if not USE_MTF_RSI_PREFILTER:
        return True

    checks = [
        ("1M", MTF_RSI_1M_MAX),
        ("1w", MTF_RSI_1W_MAX),
        ("1d", MTF_RSI_1D_MAX),
        ("4h", MTF_RSI_4H_MAX),
        ("1h", MTF_RSI_1H_MAX),
    ]
    for interval, mx in checks:
        try:
            kl = get_klines(symbol, interval, limit=max(100, RSI_LEN + 20))
            closes = [float(x[4]) for x in kl]
            rs = rsi(closes, RSI_LEN)
            prev_i, cur_i = _pick_closed_indexes(len(closes))
            cur_rsi = rs[cur_i]
            if cur_rsi is None:
                return False
            if cur_rsi > mx:
                return False
        except Exception:
            return False
        time.sleep(SYMBOL_SLEEP_SEC)
    return True


def evaluate_long_signal(symbol: str) -> Optional[Dict]:
    kl = get_klines(symbol, TF, limit=KLINE_LIMIT)
    if not kl or len(kl) < max(60, RSI_LEN + 10):
        return None

    o = [float(x[1]) for x in kl]
    h = [float(x[2]) for x in kl]
    l = [float(x[3]) for x in kl]
    c = [float(x[4]) for x in kl]

    # Candle close doğrulama
    prev_i, cur_i = _pick_closed_indexes(len(c)) if TREND_CONFIRM_CLOSE else (len(c) - 2, len(c) - 1)

    # Heikin Ashi
    if USE_HEIKIN_ASHI:
        ha = heikin_ashi(list(zip(o, h, l, c)))
        ha_close = [x[3] for x in ha]
        ha_high = [x[1] for x in ha]
        ha_low = [x[2] for x in ha]
        trend_series_close = ha_close
        atr_high = ha_high
        atr_low = ha_low
        atr_close = ha_close
    else:
        trend_series_close = c
        atr_high, atr_low, atr_close = h, l, c

    # Trend color change (red -> green)
    colors = trend_color_from_series(trend_series_close, TREND_MA_LEN)
    trend_flip = (colors[prev_i] == "red" and colors[cur_i] == "green")

    # RSI cross
    rs = rsi(c, RSI_LEN)
    if rs[cur_i] is None or rs[prev_i] is None:
        return None
    rsi_val_cur = float(rs[cur_i])
    rsi_vals = [x if x is not None else 0.0 for x in rs]
    rsi_ma = sma(rsi_vals, RSI_MA_LEN)

    rsi_ok = rsi_val_cur >= RSI_MIN
    rsi_cross_ok = True
    if RSI_CROSS_CONFIRM:
        if rsi_ma[cur_i] is None or rsi_ma[prev_i] is None:
            return None
        rsi_cross_ok = crosses_up(float(rsi_vals[prev_i]), float(rsi_ma[prev_i]), float(rsi_vals[cur_i]), float(rsi_ma[cur_i]))

    # MACD cross
    macd_line, sig_line, _ = macd(c, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    m_prev, m_cur = macd_line[prev_i], macd_line[cur_i]
    s_prev, s_cur = sig_line[prev_i], sig_line[cur_i]
    macd_cross_ok = True
    if MACD_CROSS_CONFIRM:
        macd_cross_ok = crosses_up(m_prev, s_prev, m_cur, s_cur)

    macd_zero_ok = True
    if MACD_ZERO_FILTER:
        macd_zero_ok = (m_cur >= -MACD_ZERO_EPS)

    # ATR / SAR
    atr_series = atr(atr_high, atr_low, atr_close, ATR_LEN)
    atr_val = atr_series[cur_i] if atr_series[cur_i] is not None else None

    sar_val = None
    if USE_PARABOLIC_SAR:
        sar_series = psar(h, l, PSAR_AF_STEP, PSAR_AF_MAX)
        sar_val = sar_series[cur_i]

    # Final decision
    all_ok = trend_flip and rsi_ok and rsi_cross_ok and macd_cross_ok and macd_zero_ok

    if not all_ok:
        if DEBUG and DEBUG_REJECTS:
            print(json.dumps({
                "symbol": symbol,
                "trend_flip": trend_flip,
                "rsi_ok": rsi_ok,
                "rsi_cross_ok": rsi_cross_ok,
                "macd_cross_ok": macd_cross_ok,
                "macd_zero_ok": macd_zero_ok,
                "rsi": rsi_val_cur,
                "macd": m_cur,
                "sig": s_cur,
                "colors_prev": colors[prev_i],
                "colors_cur": colors[cur_i],
            }, ensure_ascii=False))
        return None

    entry = float(c[cur_i])
    sl = None
    tp1 = None
    tp2 = None
    if atr_val is not None and atr_val > 0:
        sl = entry - (atr_val * ATR_SL_MULT)
        tp1 = entry + (atr_val * TP1_ATR_MULT)
        tp2 = entry + (atr_val * TP2_ATR_MULT)

    return {
        "symbol": symbol,
        "tf": TF,
        "entry": entry,
        "sar": sar_val,
        "atr": atr_val,
        "rsi": rsi_val_cur,
        "macd": m_cur,
        "macd_sig": s_cur,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
    }


# =========================
# Main loop
# =========================
def main():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("ERROR: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID eksik.")
        return

    store = Storage(STORAGE_PATH)

    print("✅ futures-scanner started")
    print(f"TF={TF} interval_sec={INTERVAL_SEC} HA={USE_HEIKIN_ASHI} trend=EMA({TREND_MA_LEN})")
    print(f"MTF prefilter={USE_MTF_RSI_PREFILTER} | BTC filter={USE_BTC_FILTER} | 24h vol filter={USE_24H_VOLUME_FILTER}")

    while True:
        try:
            if USE_BTC_FILTER and not check_btc_filter():
                if DEBUG:
                    print("BTC filter: PASS değil (RSI düşük). Bu cycle skip.")
                time.sleep(INTERVAL_SEC)
                continue

            symbols = get_exchange_symbols_usdt_perp() if SCAN_ALL_USDT_PERPS else []
            if not symbols:
                print("ERROR: Symbol list boş.")
                time.sleep(INTERVAL_SEC)
                continue

            vols = get_24h_quote_volumes() if USE_24H_VOLUME_FILTER else {}

            sent_count = 0
            scanned = 0

            for sym in symbols:
                scanned += 1
                if MAX_SYMBOLS_PER_CYCLE and scanned > MAX_SYMBOLS_PER_CYCLE:
                    break

                # 24h volume filter
                if USE_24H_VOLUME_FILTER:
                    qv = vols.get(sym, 0.0)
                    if qv < MIN_QUOTE_VOLUME_24H:
                        if DEBUG and DEBUG_REJECTS:
                            print(f"{sym} reject: low 24h quoteVolume {qv:.0f}")
                        continue

                # cooldown
                if COOLDOWN_SEC > 0 and store.is_in_cooldown(sym, "LONG", COOLDOWN_SEC):
                    continue

                # MTF RSI prefilter
                if USE_MTF_RSI_PREFILTER:
                    ok = mtf_rsi_pass(sym)
                    if not ok:
                        if DEBUG and DEBUG_REJECTS:
                            print(f"{sym} reject: MTF RSI prefilter fail")
                        continue

                # Entry conditions on TF
                sig = evaluate_long_signal(sym)
                if sig is None:
                    time.sleep(SYMBOL_SLEEP_SEC)
                    continue

                msg = build_message(
                    symbol=sig["symbol"],
                    tf=sig["tf"],
                    cur_close=sig["entry"],
                    sar_val=sig["sar"],
                    atr_val=sig["atr"],
                    rsi_val=sig["rsi"],
                    macd_val=sig["macd"],
                    macd_sig=sig["macd_sig"],
                    tp1=sig["tp1"],
                    tp2=sig["tp2"],
                    sl=sig["sl"],
                )

                if DRY_RUN:
                    print("DRY_RUN signal:\n", msg)
                else:
                    send_telegram(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, msg)

                store.mark_sent(sym, "LONG")
                sent_count += 1

                # “her bulduğu noktada kaç tane bulursa göndersin”
                # yani limit yok, buldukça bas.

                time.sleep(SYMBOL_SLEEP_SEC)

            if DEBUG:
                print(f"Cycle done. scanned={scanned} sent={sent_count}")

        except Exception as e:
            print("Loop error:", repr(e))

        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main()
