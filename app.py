import os
import json
import time
import math
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests

BINANCE_FAPI_BASE = "https://fapi.binance.com"
STATE_FILE = os.getenv("STATE_FILE", "state.json")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", "20"))
EXIT_CHECK_SECONDS = int(os.getenv("EXIT_CHECK_SECONDS", "60"))
TARGET_CLOSE_PCT = float(os.getenv("TARGET_CLOSE_PCT", "3.0"))  # hedef %3 ve altıysa yakın say
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "15"))

MACD_FAST = 47
MACD_SLOW = 123
MACD_SIGNAL = 9
DAILY_EMA_LEN = 47

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

session = requests.Session()


def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {
            "last_scanned_15m_close": None,
            "open_positions": {},     # symbol -> {entry, target, entry_time, status}
            "sent_entries": {}        # symbol -> last entry candle close time
        }
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logging.exception("State okunamadı, sıfırdan başlatılıyor.")
        return {
            "last_scanned_15m_close": None,
            "open_positions": {},
            "sent_entries": {}
        }


def save_state(state: dict) -> None:
    tmp_file = f"{STATE_FILE}.tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp_file, STATE_FILE)


def send_telegram(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram env eksik, mesaj gönderilmedi.")
        logging.info("\n%s", text)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        resp = session.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception:
        logging.exception("Telegram mesajı gönderilemedi.")


def http_get_json(url: str, params: Optional[dict] = None) -> dict:
    resp = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_usdt_perpetual_symbols() -> List[str]:
    data = http_get_json(f"{BINANCE_FAPI_BASE}/fapi/v1/exchangeInfo")
    symbols = []
    for s in data.get("symbols", []):
        if (
            s.get("contractType") == "PERPETUAL"
            and s.get("status") == "TRADING"
            and s.get("quoteAsset") == "USDT"
        ):
            symbols.append(s["symbol"])
    return symbols


def get_klines(symbol: str, interval: str, limit: int = 200) -> List[list]:
    return http_get_json(
        f"{BINANCE_FAPI_BASE}/fapi/v1/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit}
    )


def get_mark_prices() -> Dict[str, float]:
    data = http_get_json(f"{BINANCE_FAPI_BASE}/fapi/v1/ticker/price")
    out = {}
    for item in data:
        try:
            out[item["symbol"]] = float(item["price"])
        except Exception:
            continue
    return out


def to_closes(klines: List[list]) -> List[float]:
    return [float(k[4]) for k in klines]


def ema(values: List[float], length: int) -> List[float]:
    if len(values) < length:
        return []

    alpha = 2 / (length + 1)
    result = [None] * len(values)

    sma = sum(values[:length]) / length
    result[length - 1] = sma

    prev = sma
    for i in range(length, len(values)):
        cur = (values[i] * alpha) + (prev * (1 - alpha))
        result[i] = cur
        prev = cur

    return result


def macd_series(closes: List[float], fast: int, slow: int, signal_len: int) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)

    macd = [None] * len(closes)
    for i in range(len(closes)):
        if i < len(ema_fast) and i < len(ema_slow) and ema_fast[i] is not None and ema_slow[i] is not None:
            macd[i] = ema_fast[i] - ema_slow[i]

    macd_vals_only = [x for x in macd if x is not None]
    signal_raw = ema(macd_vals_only, signal_len)

    signal = [None] * len(closes)
    sig_idx = 0
    for i in range(len(closes)):
        if macd[i] is not None:
            if sig_idx < len(signal_raw):
                signal[i] = signal_raw[sig_idx]
                sig_idx += 1

    return macd, signal


def bullish_macd_cross_below_zero(closes: List[float]) -> Tuple[bool, Optional[float], Optional[float]]:
    """
    Kural:
    - MACD(fast=47, slow=123, signal=9)
    - son kapanmış mumda bullish cross
    - macd line < 0
    """
    macd, signal = macd_series(closes, MACD_FAST, MACD_SLOW, MACD_SIGNAL)

    if len(closes) < max(MACD_FAST, MACD_SLOW) + MACD_SIGNAL + 5:
        return False, None, None

    i = len(closes) - 1
    prev_i = i - 1

    if any(x is None for x in [macd[prev_i], signal[prev_i], macd[i], signal[i]]):
        return False, None, None

    crossed_up = macd[prev_i] <= signal[prev_i] and macd[i] > signal[i]
    below_zero = macd[i] < 0

    return crossed_up and below_zero, macd[i], signal[i]


def get_daily_ema47(symbol: str) -> Optional[float]:
    try:
        klines = get_klines(symbol, "1d", limit=100)
        closes = to_closes(klines)
        ema47 = ema(closes, DAILY_EMA_LEN)
        if not ema47:
            return None
        return ema47[-1]
    except Exception:
        logging.exception("Daily EMA47 alınamadı: %s", symbol)
        return None


def format_price(price: float) -> str:
    if price >= 1000:
        return f"{price:,.2f}"
    if price >= 1:
        return f"{price:,.4f}"
    if price >= 0.01:
        return f"{price:,.5f}"
    return f"{price:,.6f}"


def format_pct(pct: float) -> str:
    return f"{pct:.2f}"


def ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def get_last_closed_15m_candle_time_ms() -> int:
    """
    Son TAMAMLANMIŞ 15dk mum kapanış zamanı.
    """
    now = int(time.time())
    bucket = now - (now % 900)   # içinde bulunulan 15dk başlangıcı
    last_close = bucket          # şu an kapanan / kapanmış son mumun close time
    return last_close * 1000


def scan_entries(state: dict, symbols: List[str]) -> None:
    logging.info("15dk entry taraması başladı. Sembol sayısı: %s", len(symbols))
    scanned = 0
    signaled = 0

    for symbol in symbols:
        try:
            klines = get_klines(symbol, "15m", limit=220)
            if len(klines) < 150:
                continue

            # Binance son kline çoğu zaman açık mum olabilir.
            # Bu yüzden son kapanmış mumu almak için sondan 2'yi kullanıyoruz.
            closed_klines = klines[:-1]
            closes = to_closes(closed_klines)
            signal_ok, macd_val, signal_val = bullish_macd_cross_below_zero(closes)
            scanned += 1

            if not signal_ok:
                continue

            last_closed = closed_klines[-1]
            candle_close_time_ms = int(last_closed[6])  # kline close time
            entry_price = float(last_closed[4])

            # Aynı mum için tekrar sinyal atma
            last_sent = state["sent_entries"].get(symbol)
            if last_sent == candle_close_time_ms:
                continue

            daily_target = get_daily_ema47(symbol)
            if daily_target is None:
                continue

            potential_pct = ((daily_target - entry_price) / entry_price) * 100
            status = "VALID" if potential_pct > TARGET_CLOSE_PCT else "TARGET CLOSE"

            text = (
                f"<b>COIN:</b> {symbol}\n"
                f"<b>SIGNAL:</b> LONG\n\n"
                f"<b>ENTRY:</b> {format_price(entry_price)}\n"
                f"<b>TARGET:</b> {format_price(daily_target)} (Daily EMA47)\n\n"
                f"<b>POTENTIAL:</b> %{format_pct(potential_pct)}\n\n"
                f"<b>STATUS:</b> {status}"
            )

            if status == "TARGET CLOSE":
                text += "\nManual decision recommended"

            send_telegram(text)

            state["sent_entries"][symbol] = candle_close_time_ms
            state["open_positions"][symbol] = {
                "entry": entry_price,
                "target": daily_target,
                "entry_time": candle_close_time_ms,
                "status": status,
                "macd": macd_val,
                "signal": signal_val
            }
            signaled += 1

            time.sleep(0.05)

        except Exception:
            logging.exception("Entry scan hatası: %s", symbol)
            time.sleep(0.1)

    logging.info("15dk entry taraması bitti. Taranan: %s | Sinyal: %s", scanned, signaled)


def check_exits(state: dict) -> None:
    open_positions = state.get("open_positions", {})
    if not open_positions:
        return

    try:
        prices = get_mark_prices()
    except Exception:
        logging.exception("Mark price alınamadı.")
        return

    to_remove = []

    for symbol, pos in open_positions.items():
        try:
            current_price = prices.get(symbol)
            if current_price is None:
                continue

            target = float(pos["target"])
            entry = float(pos["entry"])

            if current_price >= target:
                profit_pct = ((current_price - entry) / entry) * 100

                text = (
                    f"<b>EXIT SIGNAL</b>\n\n"
                    f"<b>COIN:</b> {symbol}\n"
                    f"<b>ENTRY:</b> {format_price(entry)}\n"
                    f"<b>EXIT:</b> {format_price(current_price)}\n\n"
                    f"<b>PROFIT:</b> %{format_pct(profit_pct)}"
                )
                send_telegram(text)
                to_remove.append(symbol)

        except Exception:
            logging.exception("Exit kontrol hatası: %s", symbol)

    for symbol in to_remove:
        state["open_positions"].pop(symbol, None)


def main() -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("TELEGRAM_BOT_TOKEN veya TELEGRAM_CHAT_ID eksik.")

    send_telegram("Worker started: MACD47/123 + Daily EMA47 scanner aktif.")

    state = load_state()
    symbols = get_usdt_perpetual_symbols()
    logging.info("USDT perpetual futures sembolleri yüklendi: %s", len(symbols))

    while True:
        try:
            last_closed_15m = get_last_closed_15m_candle_time_ms()

            if state.get("last_scanned_15m_close") != last_closed_15m:
                # yeni 15dk kapanışı geldiyse entry scan çalıştır
                scan_entries(state, symbols)
                state["last_scanned_15m_close"] = last_closed_15m
                save_state(state)

            # exit kontrolleri
            check_exits(state)
            save_state(state)

        except Exception:
            logging.exception("Ana döngü hatası.")

        time.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
