# app.py
import os
import time
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests

from notify import send_telegram
from storage import Storage

# =========================
# ENV
# =========================
BINANCE_FAPI = os.getenv("BINANCE_FAPI", "https://fapi.binance.com").rstrip("/")

# Ana tarama TF (trade setup katmanı için)
TF = os.getenv("TF", "30m")
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "600"))
KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "200"))

# RSI
RSI_LEN = int(os.getenv("RSI_LEN", "14"))

# Dip filtresi (MTF RSI)
RSI_1M_MAX = float(os.getenv("RSI_1M_MAX", "10"))
RSI_1W_MAX = float(os.getenv("RSI_1W_MAX", "20"))
RSI_1D_MAX = float(os.getenv("RSI_1D_MAX", "30"))
RSI_4H_MAX = float(os.getenv("RSI_4H_MAX", "30"))
RSI_1H_MAX = float(os.getenv("RSI_1H_MAX", "30"))

INCLUDE_RSI_NA = int(os.getenv("INCLUDE_RSI_NA", "1"))

# 24h volume filtresi
USE_24H_VOLUME_FILTER = int(os.getenv("USE_24H_VOLUME_FILTER", "1"))
MIN_QUOTE_VOLUME_24H = float(os.getenv("MIN_QUOTE_VOLUME_24H", "3000000"))
ONLY_USDT_PERP = int(os.getenv("ONLY_USDT_PERP", "1"))
SCAN_ALL_USDT_PERPS = int(os.getenv("SCAN_ALL_USDT_PERPS", "1"))

# Günlük rapor
SEND_DAILY_DIP_REPORT = int(os.getenv("SEND_DAILY_DIP_REPORT", "0"))
DIP_REPORT_HOUR = int(os.getenv("DIP_REPORT_HOUR", "11"))
DIP_REPORT_MINUTE = int(os.getenv("DIP_REPORT_MINUTE", "15"))
DIP_REPORT_TOP = int(os.getenv("DIP_REPORT_TOP", "40"))

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Storage
STORAGE_PATH = os.getenv("STORAGE_PATH", "/tmp/futures_scanner_storage.json")

# Debug
DEBUG = int(os.getenv("DEBUG", "1"))


# =========================
# Helpers
# =========================
def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def http_get(url: str, params: Optional[dict] = None, timeout: int = 15) -> dict:
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def get_exchange_info() -> dict:
    return http_get(f"{BINANCE_FAPI}/fapi/v1/exchangeInfo")


def get_24h_tickers() -> List[dict]:
    return http_get(f"{BINANCE_FAPI}/fapi/v1/ticker/24hr")


def get_klines(symbol: str, interval: str, limit: int) -> List[List]:
    return http_get(
        f"{BINANCE_FAPI}/fapi/v1/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
    )


def closes_from_klines(klines: List[List]) -> List[float]:
    # kline: [openTime, open, high, low, close, volume, ...]
    out = []
    for k in klines:
        try:
            out.append(float(k[4]))
        except Exception:
            pass
    return out


def rsi(values: List[float], length: int = 14) -> Optional[float]:
    if len(values) < length + 1:
        return None
    gains = 0.0
    losses = 0.0
    # İlk ortalama
    for i in range(1, length + 1):
        delta = values[i] - values[i - 1]
        if delta >= 0:
            gains += delta
        else:
            losses += -delta
    avg_gain = gains / length
    avg_loss = losses / length

    # Wilder smoothing
    for i in range(length + 1, len(values)):
        delta = values[i] - values[i - 1]
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        avg_gain = (avg_gain * (length - 1) + gain) / length
        avg_loss = (avg_loss * (length - 1) + loss) / length

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def safe_float(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return float(x)


# =========================
# Symbol universe
# =========================
def build_universe() -> List[str]:
    info = get_exchange_info()
    symbols = []
    for s in info.get("symbols", []):
        if s.get("contractType") != "PERPETUAL":
            continue
        if s.get("status") != "TRADING":
            continue

        sym = s.get("symbol", "")
        if ONLY_USDT_PERP == 1:
            # USDT margined perpetual
            if not sym.endswith("USDT"):
                continue

        symbols.append(sym)

    return sorted(set(symbols))


def build_quote_volume_map() -> Dict[str, float]:
    tickers = get_24h_tickers()
    m: Dict[str, float] = {}
    for t in tickers:
        sym = t.get("symbol")
        if not sym:
            continue
        # quoteVolume: USDT bazında 24h hacim (futures)
        try:
            qv = float(t.get("quoteVolume", 0.0))
        except Exception:
            qv = 0.0
        m[sym] = qv
    return m


# =========================
# Dip filter (MTF RSI)
# =========================
def pass_rsi_threshold(val: Optional[float], max_val: float, include_na: int) -> bool:
    if val is None:
        return include_na == 1
    return val <= max_val


def compute_mtf_rsi(symbol: str) -> Dict[str, Optional[float]]:
    """
    Binance interval mapping:
    1M, 1w, 1d, 4h, 1h
    """
    out: Dict[str, Optional[float]] = {}

    for tf_key, interval in [
        ("1M", "1M"),
        ("1W", "1w"),
        ("1D", "1d"),
        ("4H", "4h"),
        ("1H", "1h"),
    ]:
        try:
            kl = get_klines(symbol, interval=interval, limit=max(RSI_LEN + 2, 60))
            closes = closes_from_klines(kl)
            out[tf_key] = safe_float(rsi(closes, RSI_LEN))
        except Exception:
            out[tf_key] = None

    return out


def dip_filter_pass(mtf: Dict[str, Optional[float]]) -> bool:
    return (
        pass_rsi_threshold(mtf.get("1M"), RSI_1M_MAX, INCLUDE_RSI_NA)
        and pass_rsi_threshold(mtf.get("1W"), RSI_1W_MAX, INCLUDE_RSI_NA)
        and pass_rsi_threshold(mtf.get("1D"), RSI_1D_MAX, INCLUDE_RSI_NA)
        and pass_rsi_threshold(mtf.get("4H"), RSI_4H_MAX, INCLUDE_RSI_NA)
        and pass_rsi_threshold(mtf.get("1H"), RSI_1H_MAX, INCLUDE_RSI_NA)
    )


def dip_score(mtf: Dict[str, Optional[float]]) -> float:
    """
    Sıralama skoru:
    - Eğer tüm RSI'lar None ise yeni/NA coin → en öne gelsin diye -1
    - Aksi halde mevcut RSI'ların MIN'ini alıyoruz (daha dipte olan öne)
    """
    vals = [v for v in mtf.values() if v is not None]
    if not vals:
        return -1.0
    return min(vals)


def build_dip_pool(symbols: List[str], quote_vol_map: Dict[str, float]) -> List[Tuple[str, Dict[str, Optional[float]], float]]:
    pool: List[Tuple[str, Dict[str, Optional[float]], float]] = []

    for sym in symbols:
        if USE_24H_VOLUME_FILTER == 1:
            qv = quote_vol_map.get(sym, 0.0)
            if qv < MIN_QUOTE_VOLUME_24H:
                continue

        mtf = compute_mtf_rsi(sym)
        if dip_filter_pass(mtf):
            score = dip_score(mtf)
            pool.append((sym, mtf, score))

    # skor küçük olan önde (NA = -1 en önde)
    pool.sort(key=lambda x: x[2])
    return pool


# =========================
# Daily report scheduling
# =========================
def should_send_daily_report(now: datetime, storage: Storage) -> bool:
    if SEND_DAILY_DIP_REPORT != 1:
        return False

    # her gün 1 kez: date bazında kilitle
    today_key = now.strftime("%Y-%m-%d")
    last_sent = storage.get("last_dip_report_date", "")

    if last_sent == today_key:
        return False

    if now.hour == DIP_REPORT_HOUR and now.minute == DIP_REPORT_MINUTE:
        return True

    return False


def send_dip_report(storage: Storage, symbols: List[str]) -> None:
    quote_vol_map = build_quote_volume_map()
    pool = build_dip_pool(symbols, quote_vol_map)

    total = len(pool)
    top_n = max(1, DIP_REPORT_TOP)
    top = pool[:top_n]

    lines = []
    lines.append(f"📊 DIP FILTER SCAN (TF:{TF})")
    lines.append(f"Toplam coin: {total}")
    lines.append("")
    lines.append(f"Top {min(top_n, total)}:")

    for i, (sym, mtf, score) in enumerate(top, start=1):
        # küçük, okunur MTF RSI özeti
        r1m = mtf.get("1M")
        r1w = mtf.get("1W")
        r1d = mtf.get("1D")
        r4h = mtf.get("4H")
        r1h = mtf.get("1H")

        def fmt(v: Optional[float]) -> str:
            return "-" if v is None else f"{v:.2f}"

        lines.append(
            f"{i:02d}. {sym} | RSI 1M:{fmt(r1m)} 1W:{fmt(r1w)} 1D:{fmt(r1d)} 4H:{fmt(r4h)} 1H:{fmt(r1h)}"
        )

    extra = total - len(top)
    if extra > 0:
        lines.append("")
        lines.append(f"(+{extra} more)")

    text = "\n".join(lines)

    # Telegram gönder
    send_telegram(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, text)

    # Bugün gönderildi olarak işaretle
    storage.set("last_dip_report_date", datetime.now().strftime("%Y-%m-%d"))


# =========================
# Main loop
# =========================
def main() -> None:
    storage = Storage(STORAGE_PATH)

    # Telegram env kontrol (yoksa bot çalışır ama telegram atamaz)
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log("ERROR: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID eksik.")
        # burada exit etmiyorum; istersen exit de edebilirdik
    else:
        log("Telegram OK.")

    # Universe
    symbols = build_universe()
    log(f"Universe loaded: {len(symbols)} symbols")

    # Döngü
    while True:
        try:
            now = datetime.now()

            # Günlük dip raporu
            if should_send_daily_report(now, storage):
                log("Daily dip report time reached. Building pool...")
                try:
                    send_dip_report(storage, symbols)
                    log("Daily dip report sent.")
                except Exception as e:
                    log(f"Daily dip report send failed: {e}")

            # (Şimdilik testteyiz: trade sinyali katmanı burada yok.
            #  Dip havuzu/raporu stabil olsun, sonra sinyal katmanını ekleriz.)

        except Exception as e:
            log(f"Loop error: {e}")

        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    log("✅ futures-scanner worker started")
    main()
