import os
import time
import requests
from datetime import datetime, timezone
from typing import List, Dict

# =========================
# ENV CONFIG
# =========================

INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "600"))
TF = os.getenv("TF", "30m")

EMA_FAST = int(os.getenv("EMA_FAST", "3"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "41"))
RSI_LEN = int(os.getenv("RSI_LEN", "14"))

USE_BTC_FILTER = os.getenv("USE_BTC_FILTER", "1") == "1"
BTC_SYMBOL = os.getenv("BTC_SYMBOL", "BTCUSDT")
BTC_RSI_MIN = float(os.getenv("BTC_RSI_MIN", "42"))

MIN_QUOTE_VOLUME = float(os.getenv("MIN_QUOTE_VOLUME", "15000000"))
TOP_N = int(os.getenv("TOP_N", "30"))
KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "200"))
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", "3600"))

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
TG_CHAT_ID = os.getenv("TG_CHAT_ID")

BASE_URL = "https://fapi.binance.com"

last_sent = {}

# =========================
# HELPERS
# =========================

def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def send_telegram(msg: str):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        print("Telegram config missing")
        return

    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": TG_CHAT_ID,
        "text": msg,
        "parse_mode": "HTML"
    }
    requests.post(url, data=data, timeout=10)

def fetch_symbols():
    r = requests.get(f"{BASE_URL}/fapi/v1/exchangeInfo", timeout=10)
    data = r.json()
    return [
        s["symbol"]
        for s in data["symbols"]
        if s["contractType"] == "PERPETUAL"
        and s["quoteAsset"] == "USDT"
        and s["status"] == "TRADING"
    ]

def fetch_klines(symbol: str):
    r = requests.get(
        f"{BASE_URL}/fapi/v1/klines",
        params={"symbol": symbol, "interval": TF, "limit": KLINE_LIMIT},
        timeout=10,
    )
    return r.json()

def fetch_24h(symbol: str):
    r = requests.get(
        f"{BASE_URL}/fapi/v1/ticker/24hr",
        params={"symbol": symbol},
        timeout=10,
    )
    return r.json()

# =========================
# INDICATORS
# =========================

def ema(values: List[float], period: int):
    k = 2 / (period + 1)
    ema_vals = [values[0]]
    for v in values[1:]:
        ema_vals.append(v * k + ema_vals[-1] * (1 - k))
    return ema_vals

def rsi(values: List[float], period: int):
    gains = []
    losses = []
    for i in range(1, len(values)):
        diff = values[i] - values[i-1]
        gains.append(max(diff, 0))
        losses.append(abs(min(diff, 0)))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    rsis = []
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period-1) + gains[i]) / period
        avg_loss = (avg_loss * (period-1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsis.append(100 - (100 / (1 + rs)))
    return rsis

# =========================
# SCORING
# =========================

def score_symbol(rsi_val, ema_gap_pct, volume_m):
    return (rsi_val * 0.4) + (ema_gap_pct * 30 * 0.4) + (volume_m * 0.00001 * 0.2)

# =========================
# MAIN SCAN
# =========================

def btc_gate():
    if not USE_BTC_FILTER:
        return True, "BTC filter OFF"

    kl = fetch_klines(BTC_SYMBOL)
    closes = [float(k[4]) for k in kl[:-1]]

    ema_f = ema(closes, EMA_FAST)
    ema_s = ema(closes, EMA_SLOW)
    rsi_vals = rsi(closes, RSI_LEN)

    cond = ema_f[-1] > ema_s[-1] and rsi_vals[-1] > BTC_RSI_MIN

    msg = f"BTC gate: EMA{EMA_FAST}>{EMA_SLOW}={ema_f[-1]>ema_s[-1} | RSI={rsi_vals[-1]:.2f} > {BTC_RSI_MIN}"

    return cond, msg

def scan():
    symbols = fetch_symbols()
    rows = []

    for sym in symbols:
        try:
            ticker = fetch_24h(sym)
            qv = float(ticker["quoteVolume"])
            if qv < MIN_QUOTE_VOLUME:
                continue

            kl = fetch_klines(sym)
            closes = [float(k[4]) for k in kl[:-1]]

            ema_f = ema(closes, EMA_FAST)
            ema_s = ema(closes, EMA_SLOW)

            if len(ema_f) < 3:
                continue

            fresh_cross = ema_f[-2] < ema_s[-2] and ema_f[-1] > ema_s[-1]
            trend_up = ema_f[-1] > ema_s[-1]

            if not trend_up:
                continue

            rsi_vals = rsi(closes, RSI_LEN)
            rsi_val = rsi_vals[-1]

            gap_pct = (ema_f[-1] - ema_s[-1]) / ema_s[-1]
            volume_m = qv / 1_000_000

            sc = score_symbol(rsi_val, gap_pct, volume_m)

            rows.append({
                "symbol": sym,
                "fresh": fresh_cross,
                "score": sc,
                "rsi": rsi_val,
                "gap": gap_pct,
                "vol": volume_m,
                "close": closes[-1]
            })

        except:
            continue

    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows

def format_message(rows, btc_msg):
    header = f"📡 Futures Scanner (TF={TF}) | {now_utc()}\n{btc_msg}\n"

    fresh = [r for r in rows if r["fresh"]]
    trend = [r for r in rows if not r["fresh"]]

    msg = header + "\n🚀 FRESH CROSS\n"
    for r in fresh[:TOP_N]:
        msg += f"{r['symbol']} | score={r['score']:.2f} | RSI={r['rsi']:.2f} | Vol={r['vol']:.1f}M\n"

    msg += "\n📈 TREND DEVAM\n"
    for r in trend[:TOP_N]:
        msg += f"{r['symbol']} | score={r['score']:.2f} | RSI={r['rsi']:.2f} | Vol={r['vol']:.1f}M\n"

    return msg

# =========================
# LOOP
# =========================

while True:
    try:
        gate, btc_msg = btc_gate()

        if not gate:
            send_telegram(f"📡 Futures Scanner (TF={TF}) | {now_utc()}\n{btc_msg}\n\nSinyal yok.")
            time.sleep(INTERVAL_SEC)
            continue

        rows = scan()
        msg = format_message(rows, btc_msg)

        send_telegram(msg)

        print(f"[LOOP] scanned={len(rows)} sent=True time={now_utc()}")

    except Exception as e:
        print("ERROR:", e)

    time.sleep(INTERVAL_SEC)
