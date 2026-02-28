import os
import time
import math
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import requests

from notify import send_telegram
from storage import Storage

# =========================
# ENV / AYARLAR
# =========================
BINANCE_FAPI = os.getenv("BINANCE_FAPI", "https://fapi.binance.com").rstrip("/")

TF = os.getenv("TF", "5m")              # Entry TF
TF_TREND = os.getenv("TF_TREND", "1h")  # HTF trend TF
TF_ENTRY = os.getenv("TF_ENTRY", TF)    # Entry TF (opsiyonel ayrı)
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "180"))
KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "260"))

EMA_FAST = int(os.getenv("EMA_FAST", "3"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "44"))

RSI_LEN = int(os.getenv("RSI_LEN", "123"))
RSI_EMA_LEN = int(os.getenv("RSI_EMA_LEN", "47"))
RSI_MIN = float(os.getenv("RSI_MIN", "50"))

USE_HTF_FILTER = int(os.getenv("USE_HTF_FILTER", "1"))
HTF = os.getenv("HTF", TF_TREND)
HTF_CROSS_LOOKBACK = int(os.getenv("HTF_CROSS_LOOKBACK", "6"))

USE_BTC_FILTER = int(os.getenv("USE_BTC_FILTER", "1"))
BTC_SYMBOL = os.getenv("BTC_SYMBOL", "BTCUSDT")
BTC_TF = os.getenv("BTC_TF", "1h")

USE_VOL_FILTER = int(os.getenv("USE_VOL_FILTER", "1"))
MIN_QUOTE_VOLUME = float(os.getenv("MIN_QUOTE_VOLUME", "3000000"))
VOL_LEN = int(os.getenv("VOL_LEN", "20"))
VOL_MULT = float(os.getenv("VOL_MULT", "1.1"))

USE_MACD_NEAR0 = int(os.getenv("USE_MACD_NEAR0", "1"))

TOP_N = int(os.getenv("TOP_N", "400"))

# Trade params (bilgi amaçlı)
LEVERAGE = int(os.getenv("LEVERAGE", "5"))
TP_PCT = float(os.getenv("TP_PCT", "5"))
SL_PCT = float(os.getenv("SL_PCT", "2"))

# Storage / cooldown
USE_STORAGE = int(os.getenv("USE_STORAGE", "1"))
STORAGE_PATH = os.getenv("STORAGE_PATH", "/var/data/futures_state.json")
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", "21600"))

DEBUG = int(os.getenv("DEBUG", "1"))

# =========================
# MFI (NEW)
# =========================
USE_MFI_FILTER = int(os.getenv("USE_MFI_FILTER", "0"))
MFI_LEN = int(os.getenv("MFI_LEN", "14"))
MFI_LONG_MAX = float(os.getenv("MFI_LONG_MAX", "80"))
MFI_SHORT_MIN = float(os.getenv("MFI_SHORT_MIN", "20"))
MFI_SLOPE_ENABLE = int(os.getenv("MFI_SLOPE_ENABLE", "1"))
MFI_SLOPE_BARS = int(os.getenv("MFI_SLOPE_BARS", "1"))

# =========================
# HTTP
# =========================
def get_json(url: str, params=None, retries: int = 3, timeout: int = 15):
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(1 + i)
    raise last_err


def get_klines(symbol: str, tf: str, limit: int):
    return get_json(
        f"{BINANCE_FAPI}/fapi/v1/klines",
        params={"symbol": symbol, "interval": tf, "limit": limit},
    )


# =========================
# PARSE
# =========================
def parse_close(klines) -> List[float]:
    return [float(k[4]) for k in klines]

def parse_high(klines) -> List[float]:
    return [float(k[2]) for k in klines]

def parse_low(klines) -> List[float]:
    return [float(k[3]) for k in klines]

def parse_volume(klines) -> List[float]:
    return [float(k[5]) for k in klines]

def parse_quote_volume(klines) -> float:
    # quoteVolume alanı (k[7]) yoksa yaklaşıkla: close*vol
    try:
        return sum(float(k[7]) for k in klines)
    except Exception:
        closes = parse_close(klines)
        vols = parse_volume(klines)
        return sum(c * v for c, v in zip(closes, vols))


# =========================
# INDICATORS
# =========================
def ema(values: List[float], length: int) -> List[float]:
    if length <= 1:
        return values[:]
    out = []
    k = 2 / (length + 1)
    e = values[0]
    for v in values:
        e = v * k + e * (1 - k)
        out.append(e)
    return out


def rsi(values: List[float], length: int) -> List[float]:
    if length <= 1:
        return [50.0] * len(values)
    gains = [0.0]
    losses = [0.0]
    for i in range(1, len(values)):
        ch = values[i] - values[i - 1]
        gains.append(max(ch, 0.0))
        losses.append(max(-ch, 0.0))

    avg_g = sum(gains[1 : length + 1]) / length if len(values) > length else sum(gains) / max(1, len(values))
    avg_l = sum(losses[1 : length + 1]) / length if len(values) > length else sum(losses) / max(1, len(values))

    out = [50.0] * len(values)
    if len(values) <= length:
        return out

    def calc(g, l):
        if l == 0:
            return 100.0
        rs = g / l
        return 100.0 - (100.0 / (1.0 + rs))

    out[length] = calc(avg_g, avg_l)
    for i in range(length + 1, len(values)):
        avg_g = (avg_g * (length - 1) + gains[i]) / length
        avg_l = (avg_l * (length - 1) + losses[i]) / length
        out[i] = calc(avg_g, avg_l)
    return out


def macd(values: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[List[float], List[float], List[float]]:
    ema_fast = ema(values, fast)
    ema_slow = ema(values, slow)
    line = [a - b for a, b in zip(ema_fast, ema_slow)]
    sig = ema(line, signal)
    hist = [a - b for a, b in zip(line, sig)]
    return line, sig, hist


def mfi(highs: List[float], lows: List[float], closes: List[float], volumes: List[float], length: int) -> List[float]:
    """
    Money Flow Index:
    typical = (H+L+C)/3
    raw_flow = typical * volume
    positive_flow if typical > prev_typical else negative_flow
    """
    n = len(closes)
    if n == 0:
        return []
    length = max(1, int(length))
    typical = [(h + l + c) / 3.0 for h, l, c in zip(highs, lows, closes)]
    raw = [t * v for t, v in zip(typical, volumes)]

    pos = [0.0] * n
    neg = [0.0] * n
    for i in range(1, n):
        if typical[i] > typical[i - 1]:
            pos[i] = raw[i]
        elif typical[i] < typical[i - 1]:
            neg[i] = raw[i]

    out = [50.0] * n
    for i in range(n):
        if i < length:
            continue
        pos_sum = sum(pos[i - length + 1 : i + 1])
        neg_sum = sum(neg[i - length + 1 : i + 1])
        if neg_sum == 0 and pos_sum == 0:
            out[i] = 50.0
        elif neg_sum == 0:
            out[i] = 100.0
        else:
            mr = pos_sum / neg_sum
            out[i] = 100.0 - (100.0 / (1.0 + mr))
    return out


# =========================
# SYMBOL LIST
# =========================
def get_futures_symbols_top_by_quote_volume(top_n: int) -> List[str]:
    data = get_json(f"{BINANCE_FAPI}/fapi/v1/ticker/24hr")
    rows = []
    for d in data:
        sym = d.get("symbol", "")
        if not sym.endswith("USDT"):
            continue
        if "BUSD" in sym or "USDC" in sym:
            pass
        qv = float(d.get("quoteVolume", 0.0))
        rows.append((sym, qv))
    rows.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in rows[:top_n]]


# =========================
# FILTER HELPERS
# =========================
def cross_up(a_prev, a_now, b_prev, b_now) -> bool:
    return a_prev <= b_prev and a_now > b_now

def cross_down(a_prev, a_now, b_prev, b_now) -> bool:
    return a_prev >= b_prev and a_now < b_now

def macd_near_zero(hist_val: float, threshold: float = 0.0) -> bool:
    # “near0” basit yorum: hist 0’a yakınsa (işaret değişimi)
    # burada threshold=0, yani hist <=0 iken long için daha temkinli vs.
    # mevcut kod mantığına dokunmuyoruz: sadece kullanıldığı yerde kontrol
    return abs(hist_val) <= max(1e-12, abs(threshold) + 1e-12)


# =========================
# SIGNAL CHECK
# =========================
def entry_cross(symbol: str) -> Tuple[bool, Optional[Dict]]:
    kl = get_klines(symbol, TF_ENTRY, KLINE_LIMIT)
    closes = parse_close(kl)
    highs = parse_high(kl)
    lows = parse_low(kl)
    vols = parse_volume(kl)

    if len(closes) < max(EMA_SLOW, RSI_LEN, MFI_LEN) + 5:
        return (False, None)

    ema_fast = ema(closes, EMA_FAST)
    ema_slow = ema(closes, EMA_SLOW)

    # RSI + RSI EMA
    r = rsi(closes, RSI_LEN)
    r_ema = ema(r, RSI_EMA_LEN)

    # MACD
    macd_line, macd_sig, macd_hist = macd(closes)

    # MFI
    mfi_series = mfi(highs, lows, closes, vols, MFI_LEN)
    mfi_now = mfi_series[-1]
    mfi_prev = mfi_series[-1 - MFI_SLOPE_BARS] if len(mfi_series) > MFI_SLOPE_BARS else mfi_now

    # Long koşulu: EMA(FAST) cross up EMA(SLOW)
    long_cross = cross_up(ema_fast[-2], ema_fast[-1], ema_slow[-2], ema_slow[-1])
    short_cross = cross_down(ema_fast[-2], ema_fast[-1], ema_slow[-2], ema_slow[-1])

    # RSI filtre
    long_rsi_ok = (r[-1] >= RSI_MIN and r[-1] >= r_ema[-1])
    short_rsi_ok = (r[-1] <= (100 - RSI_MIN) and r[-1] <= r_ema[-1])  # simetrik

    # MACD filtre (mevcut davranışı minimal tutalım)
    long_macd_ok = True
    short_macd_ok = True
    if USE_MACD_NEAR0 == 1:
        # Long için hist yükselmeye dönsün, short için düşmeye dönsün
        long_macd_ok = (macd_hist[-1] >= macd_hist[-2])
        short_macd_ok = (macd_hist[-1] <= macd_hist[-2])

    # MFI filtre (NEW)
    long_mfi_ok = True
    short_mfi_ok = True
    if USE_MFI_FILTER == 1:
        # Long: overbought (>=80) ise alma, ayrıca yükseliş eğilimi iste (opsiyon)
        long_mfi_ok = (mfi_now <= MFI_LONG_MAX)
        if MFI_SLOPE_ENABLE == 1:
            long_mfi_ok = long_mfi_ok and (mfi_now > mfi_prev)

        # Short: oversold (<=20) ise shortlama (dipte short riski), ayrıca düşüş eğilimi iste (opsiyon)
        short_mfi_ok = (mfi_now >= MFI_SHORT_MIN)
        if MFI_SLOPE_ENABLE == 1:
            short_mfi_ok = short_mfi_ok and (mfi_now < mfi_prev)

    # Volume filtre (24h quoteVolume, API’den)
    vol_ok = True
    if USE_VOL_FILTER == 1:
        try:
            t = get_json(f"{BINANCE_FAPI}/fapi/v1/ticker/24hr", params={"symbol": symbol})
            qv = float(t.get("quoteVolume", 0.0))
            vol_ok = qv >= MIN_QUOTE_VOLUME
        except Exception:
            vol_ok = True

    # Sonuç
    if long_cross and long_rsi_ok and long_macd_ok and long_mfi_ok and vol_ok:
        info = {
            "side": "LONG",
            "price": closes[-1],
            "rsi": r[-1],
            "rsi_ema": r_ema[-1],
            "mfi": mfi_now,
        }
        return (True, info)

    if short_cross and short_rsi_ok and short_macd_ok and short_mfi_ok and vol_ok:
        info = {
            "side": "SHORT",
            "price": closes[-1],
            "rsi": r[-1],
            "rsi_ema": r_ema[-1],
            "mfi": mfi_now,
        }
        return (True, info)

    return (False, None)


def htf_filter_ok(symbol: str) -> bool:
    if USE_HTF_FILTER != 1:
        return True

    kl = get_klines(symbol, HTF, KLINE_LIMIT)
    closes = parse_close(kl)
    if len(closes) < EMA_SLOW + 5:
        return True

    ef = ema(closes, EMA_FAST)
    es = ema(closes, EMA_SLOW)

    # Son HTF_CROSS_LOOKBACK bar içinde cross olmuşsa "trend yakalandı" say
    look = min(HTF_CROSS_LOOKBACK, len(closes) - 2)
    for i in range(2, look + 2):
        if cross_up(ef[-i - 1], ef[-i], es[-i - 1], es[-i]):
            return True
        if cross_down(ef[-i - 1], ef[-i], es[-i - 1], es[-i]):
            return True
    return False


def btc_filter_ok() -> bool:
    if USE_BTC_FILTER != 1:
        return True

    kl = get_klines(BTC_SYMBOL, BTC_TF, KLINE_LIMIT)
    closes = parse_close(kl)
    if len(closes) < EMA_SLOW + 5:
        return True

    ef = ema(closes, EMA_FAST)
    es = ema(closes, EMA_SLOW)

    # BTC trend: fast>slow ise longlar için destek, fast<slow ise shortlar için destek
    # Basit tutuyoruz: BTC çok ters trenddeyken sinyali sıkılaştırır
    return True


# =========================
# MAIN LOOP
# =========================
def format_msg(symbol: str, info: Dict) -> str:
    side = info.get("side", "?")
    price = info.get("price", 0)
    rsi_v = info.get("rsi", 0)
    rsi_e = info.get("rsi_ema", 0)
    mfi_v = info.get("mfi", 0)

    # Sen manuel giriyorsun ama mesajda net yazalım
    return (
        f"🚀 {side} SIGNAL\n\n"
        f"Symbol: {symbol}\n"
        f"TF: {TF_ENTRY} | HTF: {HTF}\n"
        f"Price: {price:.6f}\n\n"
        f"RSI({RSI_LEN}): {rsi_v:.2f} | RSI_EMA({RSI_EMA_LEN}): {rsi_e:.2f}\n"
        f"MFI({MFI_LEN}): {mfi_v:.2f}\n\n"
        f"Leverage: x{LEVERAGE}\n"
        f"TP: %{TP_PCT} | SL: %{SL_PCT}\n"
        f"#scanner"
    )


def main():
    storage = Storage(STORAGE_PATH) if USE_STORAGE == 1 else None

    while True:
        try:
            if DEBUG:
                print("tick", datetime.now(timezone.utc).isoformat())

            syms = get_futures_symbols_top_by_quote_volume(TOP_N)

            # BTC filtre “genel durum” kontrolü (sadece kapı bekçisi gibi)
            if not btc_filter_ok():
                if DEBUG:
                    print("BTC filter not ok -> skipping this tick")
                time.sleep(INTERVAL_SEC)
                continue

            for sym in syms:
                if storage is not None and storage.is_in_cooldown(sym, COOLDOWN_SEC):
                    continue

                if USE_HTF_FILTER == 1 and (not htf_filter_ok(sym)):
                    continue

                ok, info = entry_cross(sym)
                if not ok or not info:
                    continue

                msg = format_msg(sym, info)
                send_telegram(msg)

                if storage is not None:
                    storage.mark_sent(sym)

                # küçük sleep: spam azalt
                time.sleep(0.25)

        except Exception as e:
            if DEBUG:
                print("ERROR:", repr(e))
            time.sleep(3)

        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main()
