# app.py
import os
import time
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests

from notify import send_telegram_chunked, send_dip_list
from storage import Storage

# =========================
# ENV / AYARLAR
# =========================
BINANCE_FAPI = os.getenv("BINANCE_FAPI", "https://fapi.binance.com").rstrip("/")

TF = os.getenv("TF", "30m")  # sinyal hesaplanan TF (örn: 30m)
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "600"))
KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "200"))

# RSI
RSI_LEN = int(os.getenv("RSI_LEN", "14"))
RSI_MIN = float(os.getenv("RSI_MIN", "42"))  # sinyalde min RSI
RSI_CROSS_CONFIRM = int(os.getenv("RSI_CROSS_CONFIRM", "1"))  # 1 ise 42 üstüne cross

# Dip (MTF RSI havuzu - UNION)
RSI_1M_MAX = float(os.getenv("RSI_1M_MAX", "10"))
RSI_1W_MAX = float(os.getenv("RSI_1W_MAX", "20"))
RSI_1D_MAX = float(os.getenv("RSI_1D_MAX", "30"))
RSI_4H_MAX = float(os.getenv("RSI_4H_MAX", "30"))
RSI_1H_MAX = float(os.getenv("RSI_1H_MAX", "30"))
INCLUDE_RSI_NA = int(os.getenv("INCLUDE_RSI_NA", "1"))  # 1 ise RSI hesaplanamayan yeni coinleri de havuza al

# BTC filtresi
USE_BTC_FILTER = int(os.getenv("USE_BTC_FILTER", "0"))
BTC_SYMBOL = os.getenv("BTC_SYMBOL", "BTCUSDT")
BTC_TF = os.getenv("BTC_TF", "1h")
BTC_RSI_MIN = float(os.getenv("BTC_RSI_MIN", "42"))

# 24h Volume filtresi
USE_24H_VOLUME_FILTER = int(os.getenv("USE_24H_VOLUME_FILTER", "1"))
MIN_QUOTE_VOLUME_24H = float(os.getenv("MIN_QUOTE_VOLUME_24H", "3000000"))  # 3m

# MACD
MACD_FAST = int(os.getenv("MACD_FAST", "12"))
MACD_SLOW = int(os.getenv("MACD_SLOW", "26"))
MACD_SIGNAL = int(os.getenv("MACD_SIGNAL", "9"))
MACD_CROSS_CONFIRM = int(os.getenv("MACD_CROSS_CONFIRM", "1"))
MACD_ZERO_FILTER = int(os.getenv("MACD_ZERO_FILTER", "0"))
MACD_ZERO_EPS = float(os.getenv("MACD_ZERO_EPS", "0.02"))

# Trend (basit)
TREND_MA_TYPE = os.getenv("TREND_MA_TYPE", "EMA").upper()  # EMA/SMA
TREND_MA_LEN = int(os.getenv("TREND_MA_LEN", "9"))
TREND_CONFIRM_CLOSE = int(os.getenv("TREND_CONFIRM_CLOSE", "1"))

# ATR TP/SL
ATR_LEN = int(os.getenv("ATR_LEN", "14"))
ATR_SL_MULT = float(os.getenv("ATR_SL_MULT", "1.5"))
TP1_ATR_MULT = float(os.getenv("TP1_ATR_MULT", "1.0"))
TP2_ATR_MULT = float(os.getenv("TP2_ATR_MULT", "2.0"))

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
DRY_RUN = int(os.getenv("DRY_RUN", "0"))
DEBUG = int(os.getenv("DEBUG", "1"))

# Storage
STORAGE_PATH = os.getenv("STORAGE_PATH", "/var/data/futures_scanner_storage.json")
storage = Storage(STORAGE_PATH)

# Dip list gönderimi (opsiyonel)
SEND_DIP_LIST = int(os.getenv("SEND_DIP_LIST", "1"))  # 1 ise dip list otomatik gönderir
DIPLIST_HOUR = int(os.getenv("DIPLIST_HOUR", "11"))   # TR saati: 11 (sen değiştirebilirsin)
DIPLIST_MINUTE = int(os.getenv("DIPLIST_MINUTE", "5"))  # 11:05
DIPLIST_TOP_N = int(os.getenv("DIPLIST_TOP_N", "40"))

# Kısıtlar
SCAN_ALL_USDT_PERPS = int(os.getenv("SCAN_ALL_USDT_PERPS", "1"))
ONLY_USDT_PERP = int(os.getenv("ONLY_USDT_PERP", "1"))

# Sinyal spam önleme
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", "0"))  # 0 kapalı
CLOSE_COOLDOWN_SEC = int(os.getenv("CLOSE_COOLDOWN_SEC", "0"))


# =========================
# BINANCE API
# =========================
def _get(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def get_exchange_info() -> dict:
    return _get(f"{BINANCE_FAPI}/fapi/v1/exchangeInfo", {})


def get_klines(symbol: str, interval: str, limit: int) -> list:
    return _get(f"{BINANCE_FAPI}/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})


def get_24h_ticker(symbol: str) -> dict:
    return _get(f"{BINANCE_FAPI}/fapi/v1/ticker/24hr", {"symbol": symbol})


# =========================
# INDICATORS
# =========================
def ema(values: List[float], length: int) -> List[float]:
    if length <= 1:
        return values[:]
    out = []
    k = 2.0 / (length + 1.0)
    prev = None
    for v in values:
        if prev is None:
            prev = v
        else:
            prev = (v * k) + (prev * (1 - k))
        out.append(prev)
    return out


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


def rsi(values: List[float], length: int) -> List[Optional[float]]:
    # Wilder RSI
    if len(values) < length + 1:
        return [None] * len(values)

    rsis: List[Optional[float]] = [None] * len(values)
    gains = 0.0
    losses = 0.0

    # initial average gain/loss
    for i in range(1, length + 1):
        diff = values[i] - values[i - 1]
        if diff >= 0:
            gains += diff
        else:
            losses += -diff

    avg_gain = gains / length
    avg_loss = losses / length
    rs = (avg_gain / avg_loss) if avg_loss != 0 else float("inf")
    rsis[length] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(length + 1, len(values)):
        diff = values[i] - values[i - 1]
        gain = diff if diff > 0 else 0.0
        loss = -diff if diff < 0 else 0.0
        avg_gain = (avg_gain * (length - 1) + gain) / length
        avg_loss = (avg_loss * (length - 1) + loss) / length
        rs = (avg_gain / avg_loss) if avg_loss != 0 else float("inf")
        rsis[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsis


def true_range(high: float, low: float, prev_close: float) -> float:
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def atr(highs: List[float], lows: List[float], closes: List[float], length: int) -> List[Optional[float]]:
    if len(closes) < length + 1:
        return [None] * len(closes)

    trs: List[float] = []
    for i in range(len(closes)):
        if i == 0:
            trs.append(highs[i] - lows[i])
        else:
            trs.append(true_range(highs[i], lows[i], closes[i - 1]))

    out: List[Optional[float]] = [None] * len(closes)

    # Wilder smoothing
    first = sum(trs[1:length + 1]) / length
    out[length] = first
    prev = first
    for i in range(length + 1, len(closes)):
        prev = (prev * (length - 1) + trs[i]) / length
        out[i] = prev
    return out


def macd(closes: List[float], fast: int, slow: int, sig: int) -> Tuple[List[float], List[float]]:
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    macd_line = [a - b for a, b in zip(ema_fast, ema_slow)]
    signal_line = ema(macd_line, sig)
    return macd_line, signal_line


# =========================
# LOG
# =========================
def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# =========================
# UNIVERSE
# =========================
def load_universe() -> List[str]:
    info = get_exchange_info()
    symbols = []
    for s in info.get("symbols", []):
        if s.get("contractType") != "PERPETUAL":
            continue
        if s.get("status") != "TRADING":
            continue
        sym = s.get("symbol", "")
        if ONLY_USDT_PERP and not sym.endswith("USDT"):
            continue
        symbols.append(sym)
    return symbols


# =========================
# DIP POOL (UNION)
# =========================
def safe_last_rsi(symbol: str, tf: str, threshold: float) -> Tuple[bool, Optional[float]]:
    """
    True => bu TF dip şartını sağlıyor (rsi<=threshold veya include_na ile NA)
    """
    try:
        kl = get_klines(symbol, tf, KLINE_LIMIT)
        closes = [float(x[4]) for x in kl]
        rsis = rsi(closes, RSI_LEN)
        last = rsis[-1]
        if last is None:
            return (INCLUDE_RSI_NA == 1), None
        return (last <= threshold), float(last)
    except Exception:
        # veri çekilemezse: NA gibi davranmayalım, false dönelim (gürültü azaltır)
        return False, None


def build_dip_pool(symbols: List[str]) -> Tuple[List[str], Dict[str, dict]]:
    """
    Dip havuzu: UNION
    """
    dip = []
    meta: Dict[str, dict] = {}

    for sym in symbols:
        ok_any = False
        details = {}

        for tf, thr, key in [
            ("1M", RSI_1M_MAX, "rsi_1M"),
            ("1w", RSI_1W_MAX, "rsi_1W"),
            ("1d", RSI_1D_MAX, "rsi_1D"),
            ("4h", RSI_4H_MAX, "rsi_4H"),
            ("1h", RSI_1H_MAX, "rsi_1H"),
        ]:
            ok, val = safe_last_rsi(sym, tf, thr)
            details[key] = val
            details[key + "_ok"] = ok
            if ok:
                ok_any = True

        if ok_any:
            dip.append(sym)
            meta[sym] = details

    # basit sıralama: "daha dip" gibi = en küçük mevcut RSI'ya göre
    def score(sym: str) -> float:
        vals = [v for k, v in meta[sym].items() if k.startswith("rsi_") and isinstance(v, (int, float))]
        if not vals:
            return -1.0  # hepsi NA ise en üste gelsin (istersen tam tersi de yaparız)
        return min(vals)

    dip_sorted = sorted(dip, key=score)
    return dip_sorted, meta


# =========================
# SIGNAL CHECK (TF)
# =========================
def trend_ma(closes: List[float]) -> List[float]:
    if TREND_MA_TYPE == "SMA":
        return sma(closes, TREND_MA_LEN)
    return ema(closes, TREND_MA_LEN)


def passes_volume(sym: str) -> bool:
    if USE_24H_VOLUME_FILTER != 1:
        return True
    try:
        t = get_24h_ticker(sym)
        qv = float(t.get("quoteVolume", "0") or "0")
        return qv >= MIN_QUOTE_VOLUME_24H
    except Exception:
        return False


def btc_filter_ok() -> bool:
    if USE_BTC_FILTER != 1:
        return True
    try:
        kl = get_klines(BTC_SYMBOL, BTC_TF, KLINE_LIMIT)
        closes = [float(x[4]) for x in kl]
        rsis = rsi(closes, RSI_LEN)
        last = rsis[-1]
        if last is None:
            return False
        return float(last) >= BTC_RSI_MIN
    except Exception:
        return False


def compute_signal(sym: str) -> Optional[dict]:
    """
    LONG sinyali: TF üzerinde:
      - Trend: close > MA (ve confirm açık ise bir önceki <= MA)
      - RSI: >= RSI_MIN (ve confirm açık ise cross)
      - MACD: (confirm açık ise cross) ve (zero filter açık ise |macd|<=eps)
      - Volume filtresi (ops)
      - BTC filtresi (ops)
    """
    if not passes_volume(sym):
        return None
    if not btc_filter_ok():
        return None

    kl = get_klines(sym, TF, KLINE_LIMIT)
    closes = [float(x[4]) for x in kl]
    highs = [float(x[2]) for x in kl]
    lows = [float(x[3]) for x in kl]

    if len(closes) < max(RSI_LEN + 2, ATR_LEN + 2, MACD_SLOW + 2, TREND_MA_LEN + 2):
        return None

    ma = trend_ma(closes)
    rsis = rsi(closes, RSI_LEN)
    macd_line, sig_line = macd(closes, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    atrs = atr(highs, lows, closes, ATR_LEN)

    c = closes[-1]
    c_prev = closes[-2]
    ma_now = ma[-1]
    ma_prev = ma[-2]

    r_now = rsis[-1]
    r_prev = rsis[-2]
    if r_now is None or r_prev is None:
        return None

    m_now = macd_line[-1]
    m_prev = macd_line[-2]
    s_now = sig_line[-1]
    s_prev = sig_line[-2]

    a_now = atrs[-1]
    if a_now is None:
        return None

    # Trend
    trend_ok = (c > ma_now)
    if TREND_CONFIRM_CLOSE == 1:
        trend_ok = trend_ok and (c_prev <= ma_prev)

    # RSI
    rsi_ok = (r_now >= RSI_MIN)
    if RSI_CROSS_CONFIRM == 1:
        rsi_ok = rsi_ok and (r_prev < RSI_MIN)

    # MACD
    macd_ok = True
    if MACD_CROSS_CONFIRM == 1:
        macd_ok = (m_now > s_now) and (m_prev <= s_prev)
    if MACD_ZERO_FILTER == 1:
        macd_ok = macd_ok and (abs(m_now) <= MACD_ZERO_EPS)

    if not (trend_ok and rsi_ok and macd_ok):
        return None

    sl = c - (a_now * ATR_SL_MULT)
    tp1 = c + (a_now * TP1_ATR_MULT)
    tp2 = c + (a_now * TP2_ATR_MULT)

    return {
        "symbol": sym,
        "tf": TF,
        "price": c,
        "rsi": float(r_now),
        "macd": float(m_now),
        "atr": float(a_now),
        "sl": float(sl),
        "tp1": float(tp1),
        "tp2": float(tp2),
    }


def can_send(key: str, cooldown_sec: int) -> bool:
    if cooldown_sec <= 0:
        return True
    last = storage.get(key, 0)
    try:
        last = float(last)
    except Exception:
        last = 0
    return (time.time() - last) >= cooldown_sec


def mark_sent(key: str) -> None:
    storage.set(key, time.time())


def maybe_send_dip_list(dip_sorted: List[str]) -> None:
    if SEND_DIP_LIST != 1:
        return
    now = datetime.now()
    if not (now.hour == DIPLIST_HOUR and now.minute == DIPLIST_MINUTE):
        return

    today = now.strftime("%Y-%m-%d")
    last_day = storage.get("diplist_last_day", "")
    if last_day == today:
        return  # bugün zaten gönderdik

    if DRY_RUN == 1:
        log(f"[DRY_RUN] DIP LIST would be sent: total={len(dip_sorted)} top={DIPLIST_TOP_N}")
        storage.set("diplist_last_day", today)
        return

    send_dip_list(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, dip_sorted, total=len(dip_sorted), top_n=DIPLIST_TOP_N)
    storage.set("diplist_last_day", today)
    log(f"DIP LIST sent. total={len(dip_sorted)}")


def send_signal(sig: dict) -> None:
    sym = sig["symbol"]
    msg = (
        f"🚀 LONG SIGNAL\n"
        f"Symbol: {sym}\n"
        f"TF: {sig['tf']}\n"
        f"Price: {sig['price']:.8f}\n"
        f"RSI: {sig['rsi']:.2f}\n"
        f"MACD: {sig['macd']:.6f}\n"
        f"ATR: {sig['atr']:.6f}\n"
        f"SL: {sig['sl']:.8f}\n"
        f"TP1: {sig['tp1']:.8f}\n"
        f"TP2: {sig['tp2']:.8f}\n"
    )

    if DRY_RUN == 1:
        log("[DRY_RUN] " + msg.replace("\n", " | "))
        return

    send_telegram_chunked(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, msg)


def main() -> None:
    log("futures-scanner worker started")

    # Telegram test
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        log("Telegram OK.")
    else:
        log("WARNING: Telegram token/chat missing.")

    # universe
    symbols = load_universe()
    log(f"Universe loaded: {len(symbols)} symbols")

    while True:
        try:
            # 1) Dip pool
            dip_sorted, _meta = build_dip_pool(symbols)
            if DEBUG == 1:
                log(f"Dip pool size: {len(dip_sorted)}")

            # 2) Dip list scheduled send
            maybe_send_dip_list(dip_sorted)

            # 3) Sinyal tarama: sadece dip havuzundakiler üzerinden tarayalım (hız + kalite)
            for sym in dip_sorted:
                # cooldown
                if not can_send(f"cooldown:{sym}", COOLDOWN_SEC):
                    continue

                sig = compute_signal(sym)
                if not sig:
                    continue

                # close cooldown (istersen farklı key)
                if not can_send(f"closecooldown:{sym}", CLOSE_COOLDOWN_SEC):
                    continue

                send_signal(sig)
                mark_sent(f"cooldown:{sym}")
                mark_sent(f"closecooldown:{sym}")

            time.sleep(INTERVAL_SEC)

        except Exception as e:
            log(f"Loop error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
