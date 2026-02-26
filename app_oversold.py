import os
import time
import math
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional

import requests

from notify import send_telegram
from storage import Storage

# =====================================================
# ENV
# =====================================================
BINANCE_FAPI = os.getenv("BINANCE_FAPI", "https://fapi.binance.com").rstrip("/")

# Scan cadence
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC_OVERSOLD", os.getenv("INTERVAL_SEC", "900")))  # default 15m
KLINE_LIMIT = int(os.getenv("KLINE_LIMIT_OVERSOLD", os.getenv("KLINE_LIMIT", "300")))

# Universe controls
TOP_N = int(os.getenv("TOP_N_OVERSOLD", os.getenv("TOP_N", "9999")))
MIN_QUOTE_VOLUME = float(os.getenv("MIN_QUOTE_VOLUME_OVERSOLD", os.getenv("MIN_QUOTE_VOLUME", "1500000")))

# Strategy thresholds
RSI_LEN = int(os.getenv("RSI_LEN_OVERSOLD", os.getenv("RSI_LEN", "14")))
RSI_HTF_MAX = float(os.getenv("RSI_HTF_MAX", "20"))     # 1M & 1w RSI <= this
RSI_4H_MAX = float(os.getenv("RSI_4H_MAX", "40"))       # 4h RSI still low while turning up
STOCH_MAX = float(os.getenv("STOCH_MAX", "30"))         # 4h StochK <= this while turning up
BOTTOM_LOOKBACK = int(os.getenv("BOTTOM_LOOKBACK", "12"))  # 4h local bottom window

# Cooldown / storage
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC_OVERSOLD", "21600"))  # 6h
USE_STORAGE = int(os.getenv("USE_STORAGE_OVERSOLD", os.getenv("USE_STORAGE", "1")))
STORAGE_PATH = os.getenv("STORAGE_PATH_OVERSOLD", os.getenv("STORAGE_PATH", "/tmp/state_oversold.json"))

# Telegram
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TG_CHAT_ID = os.getenv("TG_CHAT_ID_OVERSOLD", os.getenv("TG_CHAT_ID", ""))
TG_CHUNK_LIMIT = int(os.getenv("TG_CHUNK_LIMIT_OVERSOLD", os.getenv("TG_CHUNK_LIMIT", "3500")))

# Debug / heartbeat
DEBUG = int(os.getenv("DEBUG_OVERSOLD", os.getenv("DEBUG", "0")))
HEARTBEAT_SEC = int(os.getenv("HEARTBEAT_SEC_OVERSOLD", os.getenv("HEARTBEAT_SEC", str(INTERVAL_SEC))))

# =====================================================
# HTTP
# =====================================================
def http_get(path: str, params: Optional[dict] = None, timeout: int = 25) -> Any:
    url = f"{BINANCE_FAPI}{path}"
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

# =====================================================
# Indicators
# =====================================================
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

def stochastic_kd(highs: List[float], lows: List[float], closes: List[float],
                  k_len: int = 14, k_smooth: int = 3, d_len: int = 3) -> Tuple[List[float], List[float]]:
    raw_k = [float("nan")] * len(closes)
    for i in range(len(closes)):
        if i < k_len - 1:
            continue
        hh = max(highs[i - k_len + 1:i + 1])
        ll = min(lows[i - k_len + 1:i + 1])
        raw_k[i] = 0.0 if hh == ll else 100.0 * (closes[i] - ll) / (hh - ll)

    k_sm = sma([0.0 if math.isnan(x) else x for x in raw_k], k_smooth)
    d = sma([0.0 if math.isnan(x) else x for x in k_sm], d_len)
    return k_sm, d

def last_closed_index(n: int) -> int:
    return n - 2  # last CLOSED candle

# =====================================================
# Binance data
# =====================================================
def get_symbols_usdt_perp() -> List[str]:
    info = http_get("/fapi/v1/exchangeInfo")
    out = []
    for s in info.get("symbols", []):
        if s.get("quoteAsset") == "USDT" and s.get("contractType") == "PERPETUAL" and s.get("status") == "TRADING":
            out.append(s["symbol"])
    return out

def get_24h_tickers() -> Dict[str, Dict[str, Any]]:
    tickers = http_get("/fapi/v1/ticker/24hr")
    return {t["symbol"]: t for t in tickers}

def get_klines(symbol: str, interval: str, limit: int) -> List[List[Any]]:
    return http_get("/fapi/v1/klines", params={"symbol": symbol, "interval": interval, "limit": limit})

# =====================================================
# Strategy
# =====================================================
def htf_rsi_under(symbol: str, interval: str, rsi_max: float) -> Optional[float]:
    kl = get_klines(symbol, interval, max(KLINE_LIMIT, RSI_LEN + 80))
    closes = [float(x[4]) for x in kl]
    if len(closes) < RSI_LEN + 10:
        return None
    r = rsi(closes, RSI_LEN)
    i = last_closed_index(len(closes))
    if i < 0 or math.isnan(r[i]):
        return None
    return r[i] if r[i] <= rsi_max else None

def ltf_turn_4h(symbol: str) -> Optional[Dict[str, float]]:
    kl = get_klines(symbol, "4h", max(KLINE_LIMIT, 220))
    highs = [float(x[2]) for x in kl]
    lows  = [float(x[3]) for x in kl]
    closes= [float(x[4]) for x in kl]

    need = max(RSI_LEN + 10, 14 + 10, BOTTOM_LOOKBACK + 10)
    if len(closes) < need:
        return None

    i = last_closed_index(len(closes))
    r = rsi(closes, RSI_LEN)
    if i < 2 or math.isnan(r[i]) or math.isnan(r[i-1]):
        return None

    k, d = stochastic_kd(highs, lows, closes, 14, 3, 3)
    if math.isnan(k[i]) or math.isnan(k[i-1]) or math.isnan(d[i]):
        return None

    # RSI "dipten yukarı": local min civarında iken yukarı dönsün
    win = r[i - BOTTOM_LOOKBACK + 1:i + 1]
    local_min = min(win)
    rsi_reversal = (r[i] > r[i-1]) and (r[i-1] <= local_min + 0.8) and (r[i] <= RSI_4H_MAX)

    # Stoch "0'dan yukarı": düşük bölgede yukarı
    stoch_reversal = (k[i] > k[i-1]) and (k[i] <= STOCH_MAX)

    if not (rsi_reversal and stoch_reversal):
        return None

    return {"rsi4h": r[i], "k": k[i], "d": d[i]}

def chunk_lines(lines: List[str], limit: int) -> List[str]:
    chunks = []
    cur = ""
    for line in lines:
        if not cur:
            cur = line
            continue
        if len(cur) + 1 + len(line) <= limit:
            cur += "\n" + line
        else:
            chunks.append(cur)
            cur = line
    if cur:
        chunks.append(cur)
    return chunks

# =====================================================
# Main loop
# =====================================================
def main():
    storage = Storage(STORAGE_PATH) if USE_STORAGE == 1 else None
    last_hb = 0

    if DEBUG == 1:
        send_telegram(
            TG_BOT_TOKEN,
            TG_CHAT_ID,
            (
                "✅ <b>Oversold scanner started</b>\n"
                f"RSI_LEN={RSI_LEN} | HTF(1M&1w)≤{RSI_HTF_MAX} | 4H_RSI≤{RSI_4H_MAX} | StochK≤{STOCH_MAX}\n"
                f"MIN_QV={int(MIN_QUOTE_VOLUME)} | TOP_N={TOP_N}"
            ),
            timeout=20
        )

    while True:
        base_cnt = 0
        final_cnt = 0

        try:
            symbols = get_symbols_usdt_perp()
            tickers = get_24h_tickers()

            # Liquidity prefilter + sort by 24h quoteVolume
            liquid = []
            for sym in symbols:
                t = tickers.get(sym)
                if not t:
                    continue
                try:
                    qv = float(t.get("quoteVolume", 0.0))
                except Exception:
                    qv = 0.0
                if qv >= MIN_QUOTE_VOLUME:
                    liquid.append((sym, qv))

            liquid.sort(key=lambda x: x[1], reverse=True)
            if TOP_N > 0:
                liquid = liquid[:TOP_N]

            monthly: Dict[str, float] = {}
            weekly: Dict[str, float] = {}

            for sym, _qv in liquid:
                # Cooldown: same symbol shouldn't spam
                key = f"{sym}:OVERSOLD_HTF"
                if storage and not storage.can_send(key, COOLDOWN_SEC):
                    continue

                m = htf_rsi_under(sym, "1M", RSI_HTF_MAX)
                if m is not None:
                    monthly[sym] = m

                w = htf_rsi_under(sym, "1w", RSI_HTF_MAX)
                if w is not None:
                    weekly[sym] = w

            base = [sym for sym in monthly.keys() if sym in weekly]
            base_cnt = len(base)

            results = []
            for sym in base:
                turn = ltf_turn_4h(sym)
                if not turn:
                    continue
                qv = float(tickers.get(sym, {}).get("quoteVolume", 0.0) or 0.0)
                results.append({
                    "symbol": sym,
                    "qv": qv,
                    "rsi1m": monthly[sym],
                    "rsi1w": weekly[sym],
                    "rsi4h": turn["rsi4h"],
                    "k": turn["k"],
                    "d": turn["d"],
                })
                if storage:
                    storage.mark_sent(f"{sym}:OVERSOLD_HTF")

            results.sort(key=lambda x: x["qv"], reverse=True)
            final_cnt = len(results)

            if final_cnt > 0:
                ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                header = (
                    f"🧲 <b>OVERSOLD WATCHLIST</b>\n"
                    f"<b>Kriter:</b> RSI(1M)&RSI(1w) ≤ {RSI_HTF_MAX}  +  4H RSI dipten ↑  &  StochK 0'dan ↑\n"
                    f"<b>Time:</b> {ts}\n"
                    f"<b>Bulunan:</b> {final_cnt}  (base={base_cnt})\n"
                    f"<b>Sıralama:</b> 24h quoteVolume ↓"
                )
                send_telegram(TG_BOT_TOKEN, TG_CHAT_ID, header, timeout=20)

                lines = []
                for idx, r in enumerate(results[:60], start=1):
                    lines.append(
                        f"{idx:02d}) <b>{r['symbol']}</b> | "
                        f"RSI1M={r['rsi1m']:.2f} RSI1W={r['rsi1w']:.2f} | "
                        f"RSI4H={r['rsi4h']:.2f} | StochK={r['k']:.1f} | "
                        f"QV24h={int(r['qv'])}"
                    )

                for ch in chunk_lines(lines, TG_CHUNK_LIMIT):
                    send_telegram(TG_BOT_TOKEN, TG_CHAT_ID, ch, timeout=20)

        except Exception as e:
            if DEBUG == 1:
                send_telegram(TG_BOT_TOKEN, TG_CHAT_ID, f"⚠️ Oversold scanner error: {e}", timeout=20)

        # Heartbeat
        now = int(time.time())
        if DEBUG == 1 and (now - last_hb) >= HEARTBEAT_SEC:
            last_hb = now
            send_telegram(
                TG_BOT_TOKEN,
                TG_CHAT_ID,
                f"🫀 HB(oversold) | MIN_QV={int(MIN_QUOTE_VOLUME)} | liquid_top={TOP_N} | base(1M∩1w)={base_cnt} | final(4Hturn)={final_cnt}",
                timeout=20
            )

        time.sleep(INTERVAL_SEC)

if __name__ == "__main__":
    main()
