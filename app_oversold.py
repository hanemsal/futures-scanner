import os
import time
from typing import List, Dict, Any, Tuple

import requests

from notify_oversold import send_telegram_oversold


# =========================
# ENV / AYARLAR
# =========================
BINANCE_FAPI = os.getenv("BINANCE_FAPI", "https://fapi.binance.com").rstrip("/")

TF = os.getenv("TF", "1h")  # 1m,5m,15m,30m,1h,4h,1d...
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "600"))  # 10 dk

TOP_N = int(os.getenv("TOP_N", "200"))  # Hacme göre ilk N
MIN_QV_USDT = float(os.getenv("MIN_QV_USDT", "1000000"))  # 24h quoteVolume USDT min
KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "200"))

RSI_LEN = int(os.getenv("RSI_LEN", "14"))
RSI_MAX = float(os.getenv("RSI_MAX", "30"))  # RSI <= 30 -> oversold
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", "3600"))  # aynı coine 1 saat içinde tekrar mesaj atma

HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "20"))


# =========================
# BASİT RSI HESABI (Wilder)
# =========================
def rsi_wilder(closes: List[float], length: int = 14) -> float:
    if len(closes) < length + 1:
        return float("nan")

    gains = []
    losses = []
    for i in range(1, length + 1):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))

    avg_gain = sum(gains) / length
    avg_loss = sum(losses) / length

    for i in range(length + 1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gain = max(diff, 0.0)
        loss = max(-diff, 0.0)
        avg_gain = (avg_gain * (length - 1) + gain) / length
        avg_loss = (avg_loss * (length - 1) + loss) / length

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


# =========================
# BINANCE API
# =========================
def http_get(path: str, params: Dict[str, Any] | None = None) -> Any:
    url = f"{BINANCE_FAPI}{path}"
    r = requests.get(url, params=params or {}, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_usdt_perp_symbols() -> Dict[str, Dict[str, Any]]:
    """
    USDT perpetual TRADING sembollerini al.
    """
    data = http_get("/fapi/v1/exchangeInfo")
    out = {}
    for s in data.get("symbols", []):
        if (
            s.get("quoteAsset") == "USDT"
            and s.get("contractType") == "PERPETUAL"
            and s.get("status") == "TRADING"
        ):
            out[s["symbol"]] = s
    return out


def get_24h_tickers() -> List[Dict[str, Any]]:
    return http_get("/fapi/v1/ticker/24hr")


def pick_top_symbols_by_quote_volume(usdt_perps: Dict[str, Any]) -> List[Tuple[str, float]]:
    """
    24h quoteVolume'a göre sıralayıp TOP_N seç.
    """
    tickers = get_24h_tickers()
    rows = []
    for t in tickers:
        sym = t.get("symbol")
        if sym not in usdt_perps:
            continue
        try:
            qv = float(t.get("quoteVolume", 0.0))
        except Exception:
            qv = 0.0
        if qv >= MIN_QV_USDT:
            rows.append((sym, qv))

    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:TOP_N]


def get_closes(symbol: str, interval: str, limit: int) -> List[float]:
    kl = http_get("/fapi/v1/klines", params={"symbol": symbol, "interval": interval, "limit": limit})
    closes = []
    for k in kl:
        closes.append(float(k[4]))  # close
    return closes


# =========================
# COOLDOWN (RAM içi)
# =========================
_last_sent: Dict[str, float] = {}


def can_send(symbol: str) -> bool:
    now = time.time()
    last = _last_sent.get(symbol, 0.0)
    return (now - last) >= COOLDOWN_SEC


def mark_sent(symbol: str) -> None:
    _last_sent[symbol] = time.time()


# =========================
# MAIN LOOP
# =========================
def scan_once() -> Tuple[int, int]:
    usdt_perps = get_usdt_perp_symbols()
    top = pick_top_symbols_by_quote_volume(usdt_perps)

    scanned = 0
    signals = 0

    for symbol, qv in top:
        scanned += 1
        try:
            closes = get_closes(symbol, TF, KLINE_LIMIT)
            val = rsi_wilder(closes, RSI_LEN)

            if val != val:  # NaN kontrol
                continue

            if val <= RSI_MAX:
                if can_send(symbol):
                    msg = (
                        "📉 OVERSOLD SCANNER\n"
                        f"Symbol: {symbol}\n"
                        f"TF: {TF}\n"
                        f"RSI({RSI_LEN}): {val:.2f}\n"
                        f"24h QuoteVol: {qv:,.0f} USDT\n"
                    )
                    send_telegram_oversold(msg)
                    mark_sent(symbol)
                    signals += 1

        except Exception as e:
            print(f"[WARN] {symbol} hata: {e}")

        # Binance rate-limit için ufak bekleme
        time.sleep(0.12)

    return scanned, signals


def main():
    send_telegram_oversold(f"✅ Oversold worker başladı | TF={TF} | TOP_N={TOP_N} | RSI<={RSI_MAX}")
    while True:
        try:
            scanned, signals = scan_once()
            print(f"[INFO] tarandı={scanned} sinyal={signals} TF={TF} TOP_N={TOP_N} MIN_QV={MIN_QV_USDT}")
        except Exception as e:
            print(f"[ERROR] scan loop: {e}")

        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main()
