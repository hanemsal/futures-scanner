import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import requests

from notify import send_telegram
from storage import Storage

# ---------- Ayarlar ----------
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "900"))  # 15 dk
TF = "1h"

EMA_LEN = int(os.getenv("EMA_LEN", "123"))
RSI_LEN = int(os.getenv("RSI_LEN", "123"))

# İSTEK: RSI > 51
RSI_THRESHOLD = float(os.getenv("RSI_THRESHOLD", "51"))

# İSTEK: EMA üstüne mesafe yok (buffer kaldır)
# İstersen ENV'den tamamen 0 yap: BUFFER_PCT=0
BUFFER_PCT = float(os.getenv("BUFFER_PCT", "0"))

# Fake breakout filtresi (gövde oranı kalsın dedin)
BODY_RATIO_MIN = float(os.getenv("BODY_RATIO_MIN", "0.45"))

# EMA eğim filtresi (istersen 0 yapıp kapatabilirsin)
EMA_SLOPE_LOOKBACK = int(os.getenv("EMA_SLOPE_LOOKBACK", "3"))  # 3 mum önce ile kıyas
USE_EMA_SLOPE = os.getenv("USE_EMA_SLOPE", "1") == "1"

USE_STORAGE = os.getenv("USE_STORAGE", "1") == "1"
STORAGE_PATH = os.getenv("STORAGE_PATH", "state.json")

BINANCE_FAPI = "https://fapi.binance.com"

# Telegram limit: 4096 char. Güvende kalalım.
TG_CHUNK_LIMIT = int(os.getenv("TG_CHUNK_LIMIT", "3500"))

st = Storage(STORAGE_PATH) if USE_STORAGE else None


# ---------- Yardımcılar ----------
def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def fetch_usdt_perp_symbols() -> List[str]:
    """
    USDT perpetual, TRADING durumunda olan sözleşmeler.
    API'de .P yok, örn: POWERUSDT
    """
    url = f"{BINANCE_FAPI}/fapi/v1/exchangeInfo"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()

    out = []
    for s in data.get("symbols", []):
        if s.get("quoteAsset") != "USDT":
            continue
        if s.get("contractType") != "PERPETUAL":
            continue
        if s.get("status") != "TRADING":
            continue
        sym = s.get("symbol")
        if sym:
            out.append(sym)
    return out


def fetch_klines(symbol: str, interval: str, limit: int = 300) -> List[list]:
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def ema_series(values: List[float], length: int) -> List[Optional[float]]:
    if len(values) < length:
        return [None] * len(values)

    out: List[Optional[float]] = [None] * len(values)
    alpha = 2.0 / (length + 1.0)

    sma = sum(values[:length]) / length
    out[length - 1] = sma

    prev = sma
    for i in range(length, len(values)):
        prev = alpha * values[i] + (1 - alpha) * prev
        out[i] = prev
    return out


def rsi_series(values: List[float], length: int) -> List[Optional[float]]:
    if len(values) < length + 1:
        return [None] * len(values)

    out: List[Optional[float]] = [None] * len(values)

    gains = 0.0
    losses = 0.0
    for i in range(1, length + 1):
        ch = values[i] - values[i - 1]
        if ch >= 0:
            gains += ch
        else:
            losses += (-ch)

    avg_gain = gains / length
    avg_loss = losses / length

    def calc_rsi(ag: float, al: float) -> float:
        if al == 0:
            return 100.0
        rs = ag / al
        return 100.0 - (100.0 / (1.0 + rs))

    out[length] = calc_rsi(avg_gain, avg_loss)

    for i in range(length + 1, len(values)):
        ch = values[i] - values[i - 1]
        gain = ch if ch > 0 else 0.0
        loss = (-ch) if ch < 0 else 0.0

        avg_gain = (avg_gain * (length - 1) + gain) / length
        avg_loss = (avg_loss * (length - 1) + loss) / length
        out[i] = calc_rsi(avg_gain, avg_loss)

    return out


def body_ratio(open_: float, high: float, low: float, close: float) -> float:
    rng = max(high - low, 1e-12)
    body = abs(close - open_)
    return body / rng


def chunk_messages(text: str, limit: int) -> List[str]:
    if len(text) <= limit:
        return [text]
    parts = []
    buf = ""
    for line in text.split("\n"):
        if len(buf) + len(line) + 1 > limit:
            parts.append(buf.rstrip())
            buf = ""
        buf += line + "\n"
    if buf.strip():
        parts.append(buf.rstrip())
    return parts


# ---------- Sinyal Kontrol ----------
def check_long_signal(symbol: str) -> Tuple[bool, Optional[Dict]]:
    """
    1H son KAPANAN mumda:
      - Gövde kesişimi: open_last <= ema_last AND close_last > ema_last
        (tercihen yeşil kapanış)
      - RSI(RSI_LEN) > RSI_THRESHOLD
      - Buffer yok (BUFFER_PCT default 0)
      - Body_ratio >= BODY_RATIO_MIN
      - EMA slope (+) opsiyonel
    """
    kl = fetch_klines(symbol, TF, limit=320)
    if not kl or len(kl) < (max(EMA_LEN, RSI_LEN) + 10):
        return (False, None)

    # Binance klines: son eleman çoğu zaman açık mumdur
    # KAPANAN son mum = -2
    last_closed = kl[-2]
    last_close_time = int(last_closed[6])  # ms (close time)

    if USE_STORAGE and st:
        if last_close_time <= st.get_last_close_time(symbol):
            return (False, None)

    opens = [safe_float(x[1]) for x in kl][:-1]   # açık mum çıkar
    highs = [safe_float(x[2]) for x in kl][:-1]
    lows = [safe_float(x[3]) for x in kl][:-1]
    closes = [safe_float(x[4]) for x in kl][:-1]

    ema = ema_series(closes, EMA_LEN)
    rsi = rsi_series(closes, RSI_LEN)

    if ema[-1] is None or rsi[-1] is None:
        return (False, None)

    open_last = opens[-1]
    high_last = highs[-1]
    low_last = lows[-1]
    close_last = closes[-1]
    ema_last = float(ema[-1])
    rsi_last = float(rsi[-1])

    # 1) Gövde kesişimi (candle crosses EMA)
    # İstersen daha sert: open_last < ema_last AND close_last > ema_last AND close_last > open_last
    candle_cross = (open_last <= ema_last) and (close_last > ema_last) and (close_last > open_last)

    # 2) RSI
    rsi_ok = rsi_last > RSI_THRESHOLD

    # 3) Buffer (kapalı / default 0)
    buffer_mult = 1.0 + (BUFFER_PCT / 100.0)
    buffer_ok = close_last > ema_last * buffer_mult

    # 4) Body ratio
    br = body_ratio(open_last, high_last, low_last, close_last)
    body_ok = br >= BODY_RATIO_MIN

    # 5) EMA slope (opsiyonel)
    slope_ok = True
    ema_lb = None
    if USE_EMA_SLOPE and EMA_SLOPE_LOOKBACK > 0:
        lb = EMA_SLOPE_LOOKBACK
        if len(ema) >= (lb + 2) and ema[-(lb + 1)] is not None:
            ema_lb = float(ema[-(lb + 1)])
            slope_ok = ema_last > ema_lb

    ok = candle_cross and rsi_ok and buffer_ok and body_ok and slope_ok

    info = {
        "symbol": symbol,
        "close_time_ms": last_close_time,
        "open": open_last,
        "close": close_last,
        "ema": ema_last,
        "rsi": rsi_last,
        "ema_dist_pct": (close_last / ema_last - 1.0) * 100.0,
        "body_ratio": br,
        "ema_lb": ema_lb,
        "flags": {
            "candle_cross": candle_cross,
            "rsi_ok": rsi_ok,
            "buffer_ok": buffer_ok,
            "body_ok": body_ok,
            "slope_ok": slope_ok,
        },
    }
    return (ok, info)


def format_telegram(signals: List[Dict]) -> str:
    signals_sorted = sorted(signals, key=lambda x: x.get("ema_dist_pct", 0.0), reverse=True)

    header = (
        f"🟢 <b>EMA{EMA_LEN} Breakout LONG</b>\n"
        f"<b>TF:</b> 1H | <b>RSI({RSI_LEN}):</b> > {RSI_THRESHOLD}\n"
        f"<b>Koşul:</b> gövde EMA üstü kapanış | body≥{BODY_RATIO_MIN:.2f}"
        + (f" | slope(+)" if USE_EMA_SLOPE else "")
        + (f" | buffer {BUFFER_PCT:.2f}%" if BUFFER_PCT > 0 else " | buffer yok")
        + "\n"
        f"<b>Uyan Coin:</b> {len(signals_sorted)}\n"
        f"<i>{utc_now_str()}</i>\n"
        f"────────────────────"
    )

    lines = [header]
    for s in signals_sorted:
        sym = s["symbol"] + ".P"  # TradingView uyumu için gösterimde .P
        rsi = s["rsi"]
        dist = s["ema_dist_pct"]
        br = s["body_ratio"]
        close = s["close"]
        ema = s["ema"]
        lines.append(
            f"• <b>{sym}</b> | RSI: {rsi:.2f} | Close: {close} | EMA{EMA_LEN}: {ema} | EMA üstü: {dist:+.2f}% | BR: {br:.2f}"
        )

    return "\n".join(lines)


def main():
    print("Scanner started...", flush=True)
    try:
        send_telegram("✅ Futures Scanner başladı. 1H kapanış bekleniyor.")
    except Exception as e:
        print("Telegram startup ping failed:", str(e), flush=True)

    while True:
        loop_started = utc_now_str()
        try:
            symbols = fetch_usdt_perp_symbols()

            signals: List[Dict] = []
            updated_symbols: List[Tuple[str, int]] = []

            for sym in symbols:
                try:
                    ok, info = check_long_signal(sym)

                    if info and "close_time_ms" in info:
                        updated_symbols.append((sym, int(info["close_time_ms"])))

                    if ok and info:
                        signals.append(info)

                except requests.HTTPError:
                    continue
                except Exception:
                    continue

            # Storage update
            if USE_STORAGE and st:
                for sym, ct in updated_symbols:
                    last = st.get_last_close_time(sym)
                    if ct > last:
                        st.set_last_close_time(sym, ct)

            if signals:
                msg = format_telegram(signals)
                parts = chunk_messages(msg, TG_CHUNK_LIMIT)
                for i, part in enumerate(parts, start=1):
                    if len(parts) > 1:
                        part = f"<b>({i}/{len(parts)})</b>\n" + part
                    send_telegram(part)

            print(
                f"[LOOP] scanned={len(symbols)} new1h={len(updated_symbols)} signals={len(signals)} time={loop_started}",
                flush=True
            )

        except Exception as e:
            print("Main error:", str(e), flush=True)

        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main()
