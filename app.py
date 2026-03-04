# app.py
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

# tarama
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "600"))  # 10 dk default
KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "200"))    # RSI için yeterli

# sinyal TF (kilit)
SIGNAL_TF = os.getenv("TF", "30m")  # kilit: 30m

# RSI filtreleri (kilit)
RSI_LEN = int(os.getenv("RSI_LEN", "14"))
RSI_1M_MAX = float(os.getenv("RSI_1M_MAX", "10"))
RSI_1W_MAX = float(os.getenv("RSI_1W_MAX", "20"))
RSI_1D_MAX = float(os.getenv("RSI_1D_MAX", "30"))
RSI_4H_MAX = float(os.getenv("RSI_4H_MAX", "30"))
RSI_1H_MAX = float(os.getenv("RSI_1H_MAX", "30"))

# RSI None (—) coinleri dahil etme (default 0 = kapalı)
INCLUDE_RSI_NA = int(os.getenv("INCLUDE_RSI_NA", "0"))

# Sinyal koşulları
RSI_MIN = float(os.getenv("RSI_MIN", "42"))  # kilit: >42
MACD_FAST = int(os.getenv("MACD_FAST", "12"))
MACD_SLOW = int(os.getenv("MACD_SLOW", "26"))
MACD_SIGNAL = int(os.getenv("MACD_SIGNAL", "9"))
MACD_ZERO_FILTER = int(os.getenv("MACD_ZERO_FILTER", "1"))  # 1: 0'a yakın/üstü şartı aktif
MACD_ZERO_EPS = float(os.getenv("MACD_ZERO_EPS", "0.02"))   # 0'a yakın tolerans

# Trend teyidi (TV indikatör yerine ölçülebilir trend)
TREND_MA_TYPE = os.getenv("TREND_MA_TYPE", "EMA").upper()   # EMA / SMA
TREND_MA_LEN = int(os.getenv("TREND_MA_LEN", "9"))          # senin ekrandaki 9 ile uyumlu
USE_HEIKIN_ASHI = int(os.getenv("USE_HEIKIN_ASHI", "0"))    # default kapalı
TREND_CONFIRM_CLOSE = int(os.getenv("TREND_CONFIRM_CLOSE", "1"))  # close'ta teyit

# TP/SL (ATR)
ATR_LEN = int(os.getenv("ATR_LEN", "14"))
ATR_SL_MULT = float(os.getenv("ATR_SL_MULT", "1.5"))
TP1_ATR_MULT = float(os.getenv("TP1_ATR_MULT", "1.0"))
TP2_ATR_MULT = float(os.getenv("TP2_ATR_MULT", "2.0"))

# Opsiyonel filtreler (default 0)
USE_24H_VOLUME_FILTER = int(os.getenv("USE_24H_VOLUME_FILTER", "0"))
MIN_QUOTE_VOLUME_24H = float(os.getenv("MIN_QUOTE_VOLUME_24H", "3000000"))

USE_BTC_FILTER = int(os.getenv("USE_BTC_FILTER", "0"))
BTC_SYMBOL = os.getenv("BTC_SYMBOL", "BTCUSDT")
BTC_TF = os.getenv("BTC_TF", "1h")
BTC_RSI_MIN = float(os.getenv("BTC_RSI_MIN", "42"))

# Universe
SCAN_ALL_USDT_PERPS = int(os.getenv("SCAN_ALL_USDT_PERPS", "1"))
ONLY_USDT_PERP = int(os.getenv("ONLY_USDT_PERP", "1"))

# Telegram ENV (iki isim de destekli)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TG_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("TG_CHAT_ID")

# Storage
STORAGE_PATH = os.getenv("STORAGE_PATH", "/tmp/futures_scanner_storage.json")
USE_STORAGE = int(os.getenv("USE_STORAGE", "1"))
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", "1800"))  # 30 dk cooldown
DEBUG = int(os.getenv("DEBUG", "0"))
DEBUG_REJECTS = int(os.getenv("DEBUG_REJECTS", "0"))

# =========================
# BINANCE API
# =========================
def _get_json(url: str, params: Optional[dict] = None, timeout: int = 15):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def get_exchange_info() -> dict:
    return _get_json(f"{BINANCE_FAPI}/fapi/v1/exchangeInfo")

def get_24h_tickers() -> List[dict]:
    return _get_json(f"{BINANCE_FAPI}/fapi/v1/ticker/24hr")

def get_klines(symbol: str, interval: str, limit: int) -> List[list]:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    return _get_json(f"{BINANCE_FAPI}/fapi/v1/klines", params=params)

# =========================
# INDICATORS
# =========================
def heikin_ashi(candles: List[Tuple[float,float,float,float]]) -> List[Tuple[float,float,float,float]]:
    # candles: (o,h,l,c)
    out = []
    ha_open = candles[0][0]
    ha_close = sum(candles[0]) / 4.0
    ha_high = max(candles[0][1], ha_open, ha_close)
    ha_low  = min(candles[0][2], ha_open, ha_close)
    out.append((ha_open, ha_high, ha_low, ha_close))

    for i in range(1, len(candles)):
        o,h,l,c = candles[i]
        ha_close = (o+h+l+c)/4.0
        ha_open = (out[i-1][0] + out[i-1][3]) / 2.0
        ha_high = max(h, ha_open, ha_close)
        ha_low = min(l, ha_open, ha_close)
        out.append((ha_open, ha_high, ha_low, ha_close))
    return out

def sma(values: List[float], n: int) -> Optional[float]:
    if len(values) < n:
        return None
    return sum(values[-n:]) / n

def ema_series(values: List[float], n: int) -> List[float]:
    if not values:
        return []
    k = 2 / (n + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(out[-1] + k * (v - out[-1]))
    return out

def ema(values: List[float], n: int) -> Optional[float]:
    if len(values) < n:
        return None
    return ema_series(values, n)[-1]

def rsi(values: List[float], n: int) -> Optional[float]:
    if len(values) < n + 1:
        return None
    gains = []
    losses = []
    for i in range(-n, 0):
        diff = values[i] - values[i-1]
        if diff >= 0:
            gains.append(diff)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-diff)
    avg_gain = sum(gains) / n
    avg_loss = sum(losses) / n
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def atr(candles: List[Tuple[float,float,float,float]], n: int) -> Optional[float]:
    if len(candles) < n + 1:
        return None
    trs = []
    for i in range(1, len(candles)):
        prev_close = candles[i-1][3]
        high = candles[i][1]
        low = candles[i][2]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    if len(trs) < n:
        return None
    return sum(trs[-n:]) / n

def macd(values: List[float], fast: int, slow: int, signal: int) -> Tuple[Optional[float], Optional[float]]:
    if len(values) < slow + signal:
        return (None, None)
    fast_ema = ema_series(values, fast)
    slow_ema = ema_series(values, slow)
    # align
    m = min(len(fast_ema), len(slow_ema))
    macd_line = [fast_ema[-m+i] - slow_ema[-m+i] for i in range(m)]
    sig = ema_series(macd_line, signal)
    return (macd_line[-1], sig[-1])

def crossed_up(prev_a: float, prev_b: float, cur_a: float, cur_b: float) -> bool:
    return prev_a <= prev_b and cur_a > cur_b

# =========================
# HELPERS
# =========================
def parse_ohlc(klines: List[list]) -> Tuple[List[float], List[Tuple[float,float,float,float]]]:
    closes = []
    candles = []
    for k in klines:
        o = float(k[1]); h = float(k[2]); l = float(k[3]); c = float(k[4])
        closes.append(c)
        candles.append((o,h,l,c))
    if USE_HEIKIN_ASHI:
        candles = heikin_ashi(candles)
        closes = [c[3] for c in candles]
    return closes, candles

def get_symbols_universe() -> List[str]:
    info = get_exchange_info()
    syms = []
    for s in info.get("symbols", []):
        if s.get("contractType") != "PERPETUAL":
            continue
        if ONLY_USDT_PERP and s.get("quoteAsset") != "USDT":
            continue
        if s.get("status") != "TRADING":
            continue
        sym = s.get("symbol")
        if sym:
            syms.append(sym)
    return sorted(set(syms))

def build_quote_volume_map() -> Dict[str, float]:
    out = {}
    for t in get_24h_tickers():
        sym = t.get("symbol")
        qv = t.get("quoteVolume")
        if sym and qv is not None:
            try:
                out[sym] = float(qv)
            except:
                pass
    return out

def passes_btc_filter() -> bool:
    if USE_BTC_FILTER != 1:
        return True
    try:
        kl = get_klines(BTC_SYMBOL, BTC_TF, KLINE_LIMIT)
        closes, _ = parse_ohlc(kl)
        r = rsi(closes, RSI_LEN)
        if r is None:
            return True
        return r >= BTC_RSI_MIN
    except Exception as e:
        if DEBUG:
            print("BTC filter error:", e)
        return True  # hata durumunda sistemi durdurma

def mtf_rsi_ok(symbol: str) -> Tuple[bool, Dict[str, Optional[float]]]:
    # Binance interval mapping:
    intervals = {
        "1M": "1M",
        "1W": "1w",
        "1D": "1d",
        "4H": "4h",
        "1H": "1h",
    }
    limits = {"1M": RSI_1M_MAX, "1W": RSI_1W_MAX, "1D": RSI_1D_MAX, "4H": RSI_4H_MAX, "1H": RSI_1H_MAX}
    rsis: Dict[str, Optional[float]] = {}

    for k, interval in intervals.items():
        try:
            kl = get_klines(symbol, interval, KLINE_LIMIT)
            closes, _ = parse_ohlc(kl)
            val = rsi(closes, RSI_LEN)
            rsis[k] = val
        except Exception:
            rsis[k] = None

        # filtre
        if rsis[k] is None:
            if INCLUDE_RSI_NA == 1:
                continue
            return (False, rsis)

        if rsis[k] > limits[k]:
            return (False, rsis)

    return (True, rsis)

def signal_on_30m(symbol: str) -> Tuple[bool, Dict[str, float]]:
    # 30m sinyal: close’ta kontrol edeceğiz.
    kl = get_klines(symbol, SIGNAL_TF, KLINE_LIMIT)
    closes, candles = parse_ohlc(kl)

    if len(closes) < 50:
        return (False, {})

    # Trend teyidi: MA flip (close’ta)
    if TREND_MA_TYPE == "SMA":
        ma_prev = sma(closes[:-1], TREND_MA_LEN)
        ma_cur = sma(closes, TREND_MA_LEN)
    else:
        ma_prev = ema(closes[:-1], TREND_MA_LEN)
        ma_cur = ema(closes, TREND_MA_LEN)

    if ma_prev is None or ma_cur is None:
        return (False, {})

    price_prev = closes[-2]
    price_cur = closes[-1]

    # "red->green" benzeri: fiyat MA altından üstüne geçti (trend flip)
    trend_flip = (price_prev <= ma_prev) and (price_cur > ma_cur)
    if not trend_flip:
        return (False, {"ma": ma_cur})

    # RSI > 42 ve yukarı ivme: RSI önce <=42 iken şimdi >42 (cross up)
    r_prev = rsi(closes[:-1], RSI_LEN)
    r_cur = rsi(closes, RSI_LEN)
    if r_prev is None or r_cur is None:
        return (False, {"rsi": r_cur or -1})

    if not (r_prev <= RSI_MIN and r_cur > RSI_MIN):
        return (False, {"rsi": r_cur})

    # MACD cross up
    # MACD için son iki barın macd/signal değerini hesaplamak adına kısmi yaklaşım:
    m_cur, s_cur = macd(closes, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    m_prev, s_prev = macd(closes[:-1], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    if None in (m_prev, s_prev, m_cur, s_cur):
        return (False, {})

    if not crossed_up(m_prev, s_prev, m_cur, s_cur):
        return (False, {"macd": m_cur, "signal": s_cur})

    if MACD_ZERO_FILTER == 1:
        # 0'a yakın veya üstü
        if not (m_cur >= -MACD_ZERO_EPS):
            return (False, {"macd": m_cur})

    # ATR / TP / SL
    atr_val = atr(candles, ATR_LEN)
    if atr_val is None or atr_val <= 0:
        return (False, {})

    entry = price_cur
    sl = entry - atr_val * ATR_SL_MULT
    tp1 = entry + atr_val * TP1_ATR_MULT
    tp2 = entry + atr_val * TP2_ATR_MULT

    return (True, {
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "rsi": r_cur,
        "macd": m_cur,
        "macd_signal": s_cur,
        "atr": atr_val,
        "ma": ma_cur,
    })

def fmt_signal(symbol: str, mtf: Dict[str, Optional[float]], sig: Dict[str, float]) -> str:
    def f(x):
        return "-" if x is None else f"{x:.2f}"
    return (
        f"✅ LONG SIGNAL (30m close)\n"
        f"Symbol: {symbol}\n\n"
        f"MTF RSI(14): 1M={f(mtf.get('1M'))} | 1W={f(mtf.get('1W'))} | 1D={f(mtf.get('1D'))} | 4H={f(mtf.get('4H'))} | 1H={f(mtf.get('1H'))}\n\n"
        f"Entry: {sig['entry']:.6f}\n"
        f"SL:    {sig['sl']:.6f}\n"
        f"TP1:   {sig['tp1']:.6f}\n"
        f"TP2:   {sig['tp2']:.6f}\n\n"
        f"ATR({ATR_LEN}): {sig['atr']:.6f}\n"
        f"RSI: {sig['rsi']:.2f} | MACD: {sig['macd']:.6f} / {sig['macd_signal']:.6f}\n"
    )

def main():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("ERROR: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID eksik.")
        return

    store = Storage(STORAGE_PATH) if USE_STORAGE == 1 else None

    symbols = get_symbols_universe() if SCAN_ALL_USDT_PERPS == 1 else []
    qv_map = build_quote_volume_map() if USE_24H_VOLUME_FILTER == 1 else {}

    if DEBUG:
        print(f"Universe symbols: {len(symbols)} | SIGNAL_TF={SIGNAL_TF} | INCLUDE_RSI_NA={INCLUDE_RSI_NA}")

    while True:
        try:
            if not passes_btc_filter():
                if DEBUG:
                    print("BTC filter blocked scan (BTC RSI below threshold).")
                time.sleep(INTERVAL_SEC)
                continue

            for sym in symbols:
                # volume filtresi opsiyonel
                if USE_24H_VOLUME_FILTER == 1:
                    qv = qv_map.get(sym, 0.0)
                    if qv < MIN_QUOTE_VOLUME_24H:
                        if DEBUG_REJECTS:
                            print("reject volume", sym, qv)
                        continue

                ok, mtf = mtf_rsi_ok(sym)
                if not ok:
                    if DEBUG_REJECTS:
                        print("reject mtf", sym, mtf)
                    continue

                # cooldown (spam engel)
                if store:
                    last = store.get(sym)
                    if last and (time.time() - float(last)) < COOLDOWN_SEC:
                        continue

                # sinyal
                is_sig, sig = signal_on_30m(sym)
                if not is_sig:
                    continue

                msg = fmt_signal(sym, mtf, sig)
                send_telegram(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, msg)

                if store:
                    store.set(sym, str(time.time()))

            time.sleep(INTERVAL_SEC)

        except Exception as e:
            print("loop error:", e)
            time.sleep(10)

if __name__ == "__main__":
    main()
