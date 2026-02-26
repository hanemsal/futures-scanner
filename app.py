import os
import time
import math
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional

import requests

from notify import send_telegram
from storage import Storage

# =========================
# ENV
# =========================
BINANCE_FAPI = os.getenv("BINANCE_FAPI", "https://fapi.binance.com").rstrip("/")
TF = os.getenv("TF", "1h")  # "1h" / "30m"
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "600"))
KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "300"))

EMA_FAST = int(os.getenv("EMA_FAST", "3"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "44"))

RSI_LEN = int(os.getenv("RSI_LEN", "123"))
RSI_MIN = float(os.getenv("RSI_MIN", "45"))

TOP_N = int(os.getenv("TOP_N", "30"))
MIN_QUOTE_VOLUME = float(os.getenv("MIN_QUOTE_VOLUME", "15000000"))

COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", "3600"))
USE_STORAGE = int(os.getenv("USE_STORAGE", "1"))
STORAGE_PATH = os.getenv("STORAGE_PATH", "/tmp/futures_scanner_storage.json")

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "")
TG_CHUNK_LIMIT = int(os.getenv("TG_CHUNK_LIMIT", "3500"))

# BTC filter
USE_BTC_FILTER = int(os.getenv("USE_BTC_FILTER", "0"))
BTC_SYMBOL = os.getenv("BTC_SYMBOL", "BTCUSDT")
BTC_RSI_MIN = float(os.getenv("BTC_RSI_MIN", "0"))  # örn 45

# Premium filters
USE_SMA_FILTER = int(os.getenv("USE_SMA_FILTER", "1"))
SMA_LEN = int(os.getenv("SMA_LEN", "47"))

USE_VOL_FILTER = int(os.getenv("USE_VOL_FILTER", "1"))
VOL_LEN = int(os.getenv("VOL_LEN", "20"))
VOL_MULT = float(os.getenv("VOL_MULT", "1.3"))

USE_MACD_NEAR0 = int(os.getenv("USE_MACD_NEAR0", "1"))

ATR_LEN = int(os.getenv("ATR_LEN", "14"))
SWING_LEN = int(os.getenv("SWING_LEN", "10"))

# =========================
# HTTP helpers
# =========================
def http_get(path: str, params: Optional[dict] = None, timeout: int = 20) -> Any:
    url = f"{BINANCE_FAPI}{path}"
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

# =========================
# Indicators (pure python)
# =========================
def ema(values: List[float], length: int) -> List[float]:
    k = 2 / (length + 1)
    out = []
    prev = None
    for v in values:
        prev = v if prev is None else (v * k + prev * (1 - k))
        out.append(prev)
    return out

def sma(values: List[float], length: int) -> List[float]:
    out = []
    s = 0.0
    for i, v in enumerate(values):
        s += v
        if i >= length:
            s -= values[i - length]
        out.append(s / length if i >= length - 1 else float("nan"))
    return out

def rsi(values: List[float], length: int) -> List[float]:
    gains = [0.0]
    losses = [0.0]
    for i in range(1, len(values)):
        ch = values[i] - values[i - 1]
        gains.append(max(ch, 0.0))
        losses.append(max(-ch, 0.0))

    out = [float("nan")] * len(values)
    avg_g = 0.0
    avg_l = 0.0
    for i in range(1, len(values)):
        g, l = gains[i], losses[i]
        if i == length:
            avg_g = sum(gains[1:length + 1]) / length
            avg_l = sum(losses[1:length + 1]) / length
        elif i > length:
            avg_g = (avg_g * (length - 1) + g) / length
            avg_l = (avg_l * (length - 1) + l) / length

        if i >= length:
            rs = (avg_g / avg_l) if avg_l != 0 else math.inf
            out[i] = 100 - (100 / (1 + rs))
    return out

def macd(values: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[List[float], List[float], List[float]]:
    ef = ema(values, fast)
    es = ema(values, slow)
    macd_line = [a - b for a, b in zip(ef, es)]
    sig = ema(macd_line, signal)
    hist = [m - s for m, s in zip(macd_line, sig)]
    return macd_line, sig, hist

def stdev(values: List[float], length: int) -> List[float]:
    out = []
    for i in range(len(values)):
        if i < length - 1:
            out.append(float("nan"))
            continue
        w = values[i - length + 1:i + 1]
        mu = sum(w) / length
        var = sum((x - mu) ** 2 for x in w) / length
        out.append(math.sqrt(var))
    return out

def atr(highs: List[float], lows: List[float], closes: List[float], length: int) -> List[float]:
    tr = [0.0]
    for i in range(1, len(closes)):
        tr_i = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        tr.append(tr_i)
    return ema(tr, length)

# =========================
# Binance data
# =========================
def get_symbols_usdt_perp() -> List[str]:
    info = http_get("/fapi/v1/exchangeInfo")
    out = []
    for s in info.get("symbols", []):
        if s.get("quoteAsset") == "USDT" and s.get("contractType") == "PERPETUAL" and s.get("status") == "TRADING":
            out.append(s["symbol"])
    return out

def get_24h_tickers() -> Dict[str, Dict[str, Any]]:
    tickers = http_get("/fapi/v1/ticker/24hr")
    mp = {}
    for t in tickers:
        mp[t["symbol"]] = t
    return mp

def get_klines(symbol: str, interval: str, limit: int) -> List[List[Any]]:
    return http_get("/fapi/v1/klines", params={"symbol": symbol, "interval": interval, "limit": limit})

# =========================
# Signal logic + plan
# =========================
def compute_long_signal(symbol: str, kl: List[List[Any]]) -> Optional[Dict[str, Any]]:
    # Use last CLOSED candle: index -2
    closes = [float(x[4]) for x in kl]
    highs  = [float(x[2]) for x in kl]
    lows   = [float(x[3]) for x in kl]
    vols   = [float(x[5]) for x in kl]
    close_times = [int(x[6]) for x in kl]  # close time ms

    if len(closes) < max(EMA_SLOW, RSI_LEN, 200, ATR_LEN, SWING_LEN) + 5:
        return None

    i = len(closes) - 2
    if i < 5:
        return None

    ema_fast = ema(closes, EMA_FAST)
    ema_slow = ema(closes, EMA_SLOW)

    cross_up = (ema_fast[i - 1] <= ema_slow[i - 1]) and (ema_fast[i] > ema_slow[i])

    r = rsi(closes, RSI_LEN)
    rsi_ok = (not math.isnan(r[i])) and (r[i] >= RSI_MIN) and (r[i] > r[i - 1])

    sma47 = sma(closes, SMA_LEN)
    trend_ok = True
    if USE_SMA_FILTER == 1 and not math.isnan(sma47[i]):
        trend_ok = closes[i] > sma47[i]

    vol_sma = sma(vols, VOL_LEN)
    vol_ok = True
    vol_ratio = None
    if USE_VOL_FILTER == 1 and not math.isnan(vol_sma[i]) and vol_sma[i] > 0:
        vol_ratio = vols[i] / vol_sma[i]
        vol_ok = vols[i] > vol_sma[i] * VOL_MULT
    elif not math.isnan(vol_sma[i]) and vol_sma[i] > 0:
        vol_ratio = vols[i] / vol_sma[i]

    macd_line, sig_line, hist = macd(closes, 12, 26, 9)
    hist_up = hist[i] > hist[i - 1]

    macd_std = stdev(macd_line, 200)
    macd_near0 = True
    if USE_MACD_NEAR0 == 1 and not math.isnan(macd_std[i]):
        near0_th = macd_std[i] * 0.25
        macd_near0 = abs(macd_line[i]) <= near0_th

    macd_ok = hist_up and (macd_near0 if USE_MACD_NEAR0 == 1 else (macd_line[i] > sig_line[i]))

    long_signal = cross_up and rsi_ok and trend_ok and vol_ok and macd_ok
    if not long_signal:
        return None

    # ---- Risk plan (ENTRY/SL/TP)
    atr14 = atr(highs, lows, closes, ATR_LEN)
    atr_i = atr14[i] if not math.isnan(atr14[i]) else 0.0

    signal_close = closes[i]
    signal_low = lows[i]

    swing_start = max(0, i - SWING_LEN + 1)
    swing_low = min(lows[swing_start:i + 1])

    sl = min(signal_low, swing_low) - (0.10 * atr_i)
    # avoid weird negative SL on tiny coins
    sl = max(sl, 0.0)

    r_val = signal_close - sl
    if r_val <= 0:
        return None

    tp1 = signal_close + 1.0 * r_val
    tp2 = signal_close + 2.0 * r_val
    tp3 = signal_close + 3.0 * r_val

    # time
    ts = datetime.fromtimestamp(close_times[i] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return {
        "symbol": symbol,
        "tf": TF,
        "time": ts,
        "entry": signal_close,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "rsi": r[i],
        "rsi_up": r[i] > r[i - 1],
        "macd_hist": hist[i],
        "macd_hist_up": hist_up,
        "macd_near0": macd_near0,
        "vol_ratio": vol_ratio,
        "ema_fast": ema_fast[i],
        "ema_slow": ema_slow[i],
    }

def format_signal_msg(sig: Dict[str, Any]) -> str:
    def f(x: float) -> str:
        # pretty format for large/small numbers
        if x == 0:
            return "0"
        if x >= 100:
            return f"{x:.2f}"
        if x >= 1:
            return f"{x:.4f}"
        return f"{x:.6f}"

    vol_ratio = sig["vol_ratio"]
    vol_txt = "n/a" if vol_ratio is None else f"{vol_ratio:.2f}x"

    msg = []
    msg.append("🚀 <b>LONG BREAKOUT</b> (EMA{}↑EMA{})".format(EMA_FAST, EMA_SLOW))
    msg.append(f"<b>SYMBOL:</b> {sig['symbol']}   <b>TF:</b> {sig['tf']}")
    msg.append(f"<b>Time (closed candle):</b> {sig['time']}")
    msg.append("")
    msg.append(f"<b>ENTRY:</b> {f(sig['entry'])}")
    msg.append(f"<b>SL:</b>    {f(sig['sl'])}  (SwingLow-ATR)")
    msg.append(f"<b>TP1:</b>   {f(sig['tp1'])}  (1R)")
    msg.append(f"<b>TP2:</b>   {f(sig['tp2'])}  (2R)")
    msg.append(f"<b>TP3:</b>   {f(sig['tp3'])}  (3R)")
    msg.append("")
    msg.append(f"<b>RSI({RSI_LEN}):</b> {sig['rsi']:.2f} {'(↑)' if sig['rsi_up'] else '(↓)'}")
    msg.append(f"<b>MACD hist:</b> {sig['macd_hist']:+.6f} {'(↑)' if sig['macd_hist_up'] else '(↓)'} | near0: {'✅' if sig['macd_near0'] else '❌'}")
    msg.append(f"<b>Vol:</b> {vol_txt} (vs SMA{VOL_LEN})")
    msg.append("")
    msg.append("<b>Plan:</b>")
    msg.append("- TP1 görünce SL = ENTRY (break-even)")
    msg.append(f"- Kapanış EMA{EMA_SLOW} altına inerse çık (trail)")
    msg.append("")
    msg.append("#scanner")
    return "\n".join(msg)

def chunk_messages(messages: List[str], limit: int) -> List[str]:
    chunks = []
    cur = ""
    for m in messages:
        if not cur:
            cur = m
            continue
        if len(cur) + 2 + len(m) <= limit:
            cur += "\n\n" + m
        else:
            chunks.append(cur)
            cur = m
    if cur:
        chunks.append(cur)
    return chunks

# =========================
# BTC filter
# =========================
def btc_filter_pass() -> Tuple[bool, str]:
    if USE_BTC_FILTER != 1:
        return True, "BTC filter: OFF"

    try:
        kl = get_klines(BTC_SYMBOL, TF, max(KLINE_LIMIT, RSI_LEN + 5))
        closes = [float(x[4]) for x in kl]
        r = rsi(closes, RSI_LEN)
        i = len(closes) - 2
        if i < 0 or math.isnan(r[i]):
            return False, "BTC filter: NO DATA"
        ok = r[i] >= BTC_RSI_MIN
        return ok, f"BTC filter: RSI({RSI_LEN})={r[i]:.2f} >= {BTC_RSI_MIN}"
    except Exception:
        return False, "BTC filter: ERROR"

# =========================
# Main loop
# =========================
def main():
    storage = Storage(STORAGE_PATH) if USE_STORAGE == 1 else None

    while True:
        try:
            btc_ok, btc_reason = btc_filter_pass()
            if not btc_ok:
                send_telegram(TG_BOT_TOKEN, TG_CHAT_ID, f"⛔️ {btc_reason}\nSinyaller pas geçildi.", timeout=20)
                time.sleep(INTERVAL_SEC)
                continue

            symbols = get_symbols_usdt_perp()
            tickers = get_24h_tickers()

            # volume filter + topN by quoteVolume
            candidates = []
            for sym in symbols:
                t = tickers.get(sym)
                if not t:
                    continue
                try:
                    qv = float(t.get("quoteVolume", 0.0))
                except Exception:
                    qv = 0.0
                if qv >= MIN_QUOTE_VOLUME:
                    candidates.append((sym, qv))

            candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = candidates[:TOP_N]

            signals = []
            for sym, _qv in candidates:
                # cooldown
                key = f"{sym}:{TF}:LONG"
                if storage and not storage.can_send(key, COOLDOWN_SEC):
                    continue

                try:
                    kl = get_klines(sym, TF, KLINE_LIMIT)
                    sig = compute_long_signal(sym, kl)
                    if sig:
                        signals.append(sig)
                        if storage:
                            storage.mark_sent(key)
                except Exception:
                    continue

            if signals:
                msgs = [format_signal_msg(s) for s in signals]
                chunks = chunk_messages(msgs, TG_CHUNK_LIMIT)
                header = f"✅ {len(signals)} sinyal | {btc_reason}"
                send_telegram(TG_BOT_TOKEN, TG_CHAT_ID, header, timeout=20)
                for ch in chunks:
                    send_telegram(TG_BOT_TOKEN, TG_CHAT_ID, ch, timeout=20)
            else:
                # istersen sessiz kalsın; debug istersen açarız
                pass

        except Exception as e:
            send_telegram(TG_BOT_TOKEN, TG_CHAT_ID, f"⚠️ Scanner error: {e}", timeout=20)

        time.sleep(INTERVAL_SEC)

if __name__ == "__main__":
    main()
