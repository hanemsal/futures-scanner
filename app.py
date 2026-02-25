# app.py  (FULL - FINAL)
import os
import time
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional

import requests

from notify import send_telegram
from storage import Storage


# =========================
# ENV / AYARLAR
# =========================
BINANCE_FAPI = "https://fapi.binance.com"

TF = os.getenv("TF", "30m")  # 30m
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "600"))  # 10 dk
KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "200"))  # EMA41+RSI14 için yeterli

EMA_FAST = int(os.getenv("EMA_FAST", "3"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "41"))
RSI_LEN = int(os.getenv("RSI_LEN", "14"))

TOP_N = int(os.getenv("TOP_N", "30"))  # telegramda listelenecek max coin

MIN_QUOTE_VOLUME = float(os.getenv("MIN_QUOTE_VOLUME", "15000000"))  # 15M (USDT)
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", "3600"))  # aynı coin mesajını 1 saat kilitle

# BTC gate
USE_BTC_FILTER = os.getenv("USE_BTC_FILTER", "1") == "1"
BTC_SYMBOL = os.getenv("BTC_SYMBOL", "BTCUSDT")  # Binance'te BTCUSDT olur. TradingView'deki .P değil.
BTC_RSI_MIN = float(os.getenv("BTC_RSI_MIN", "42"))

# Storage
USE_STORAGE = os.getenv("USE_STORAGE", "1") == "1"
STORAGE_PATH = os.getenv("STORAGE_PATH", "state.json")

# Telegram chunk limiti
TG_CHUNK_LIMIT = int(os.getenv("TG_CHUNK_LIMIT", "3500"))


# =========================
# UTILS
# =========================
def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


# =========================
# TECH INDICATORS
# =========================
def ema(series: List[float], length: int) -> List[float]:
    if length <= 1:
        return series[:]
    out = []
    k = 2.0 / (length + 1.0)
    ema_prev = series[0]
    out.append(ema_prev)
    for v in series[1:]:
        ema_prev = (v * k) + (ema_prev * (1 - k))
        out.append(ema_prev)
    return out


def rsi(series: List[float], length: int) -> List[float]:
    # Wilder RSI
    if len(series) < length + 2:
        return [50.0] * len(series)

    deltas = [series[i] - series[i - 1] for i in range(1, len(series))]
    gains = [max(d, 0.0) for d in deltas]
    losses = [abs(min(d, 0.0)) for d in deltas]

    avg_gain = sum(gains[:length]) / length
    avg_loss = sum(losses[:length]) / length

    rsis = [50.0] * (length)  # pad

    for i in range(length, len(deltas)):
        avg_gain = (avg_gain * (length - 1) + gains[i]) / length
        avg_loss = (avg_loss * (length - 1) + losses[i]) / length
        if avg_loss == 0:
            rs = 999999
        else:
            rs = avg_gain / avg_loss
        rsi_val = 100 - (100 / (1 + rs))
        rsis.append(rsi_val)

    # rsis length = len(series)-1, pad 1 to match series
    rsis = [rsis[0]] + rsis
    if len(rsis) < len(series):
        rsis += [rsis[-1]] * (len(series) - len(rsis))
    return rsis


# =========================
# BINANCE FETCH
# =========================
def fetch_exchange_info() -> Dict:
    url = f"{BINANCE_FAPI}/fapi/v1/exchangeInfo"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


def fetch_ticker_24h() -> List[Dict]:
    url = f"{BINANCE_FAPI}/fapi/v1/ticker/24hr"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


def fetch_klines(symbol: str) -> List[List]:
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": TF, "limit": KLINE_LIMIT}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def get_usdt_perp_symbols() -> List[str]:
    info = fetch_exchange_info()
    symbols = []
    for s in info.get("symbols", []):
        # USDT perpetual + trading
        if (
            s.get("contractType") == "PERPETUAL"
            and s.get("quoteAsset") == "USDT"
            and s.get("status") == "TRADING"
        ):
            symbols.append(s["symbol"])
    return symbols


def build_quote_volume_map() -> Dict[str, float]:
    # quoteVolume: 24h quote asset volume (USDT)
    tickers = fetch_ticker_24h()
    m = {}
    for t in tickers:
        sym = t.get("symbol")
        qv = safe_float(t.get("quoteVolume", 0))
        if sym:
            m[sym] = qv
    return m


# =========================
# SIGNAL LOGIC
# =========================
def last_closed_cross(closes: List[float]) -> Tuple[bool, bool, float, float, float]:
    """
    closes: already excluding last open candle
    Returns:
      fresh_cross: EMAfast crossed above EMAslow on the last closed candle
      trend_ok: EMAfast > EMAslow currently (trend continues)
      ef: last EMAfast
      es: last EMAslow
      gap_pct: (ef-es)/es * 100
    """
    ef_series = ema(closes, EMA_FAST)
    es_series = ema(closes, EMA_SLOW)

    ef_now, es_now = ef_series[-1], es_series[-1]
    ef_prev, es_prev = ef_series[-2], es_series[-2]

    trend_ok = ef_now > es_now
    fresh_cross = (ef_prev <= es_prev) and (ef_now > es_now)

    gap_pct = 0.0
    if es_now != 0:
        gap_pct = (ef_now - es_now) / es_now * 100.0

    return fresh_cross, trend_ok, ef_now, es_now, gap_pct


def score_coin(rsi_val: float, gap_pct: float, vol: float) -> float:
    """
    Basit skor:
    - RSI 45-65 aralığı daha "sağlıklı" trend -> merkez 55'e yakınsa artar
    - EMA gap çok şiştiyse biraz düşürür (çok şişince fake pullback olur)
    - Volume yüksekse artar
    """
    # RSI skor (55'e yakın iyi)
    rsi_score = max(0.0, 1.0 - abs(rsi_val - 55.0) / 25.0)  # 0..1

    # gap skoru (0-2% iyi, 6%+ şişkin)
    if gap_pct < 0:
        gap_score = 0.0
    elif gap_pct <= 2:
        gap_score = 1.0
    elif gap_pct <= 6:
        gap_score = 1.0 - (gap_pct - 2.0) / 4.0  # 1 -> 0
    else:
        gap_score = 0.0

    # volume skoru (15M referans)
    vol_score = min(1.0, vol / max(MIN_QUOTE_VOLUME, 1.0))

    # ağırlıklar
    return (rsi_score * 0.45) + (gap_score * 0.35) + (vol_score * 0.20)


def btc_gate() -> Tuple[bool, str]:
    if not USE_BTC_FILTER:
        return True, "BTC filter OFF"

    kl = fetch_klines(BTC_SYMBOL)
    closes = [float(k[4]) for k in kl[:-1]]  # sadece kapanmış mumlar

    ef = ema(closes, EMA_FAST)
    es = ema(closes, EMA_SLOW)
    r = rsi(closes, RSI_LEN)

    ema_condition = ef[-1] > es[-1]
    rsi_condition = r[-1] > BTC_RSI_MIN
    cond = ema_condition and rsi_condition

    msg = (
        f"BTC gate: EMA{EMA_FAST}>{EMA_SLOW}={ema_condition} | "
        f"RSI={r[-1]:.2f} > {BTC_RSI_MIN}"
    )
    return cond, msg


# =========================
# TELEGRAM FORMAT
# =========================
def chunk_text(s: str, limit: int) -> List[str]:
    if len(s) <= limit:
        return [s]
    parts = []
    cur = ""
    for line in s.split("\n"):
        if len(cur) + len(line) + 1 > limit:
            parts.append(cur.rstrip())
            cur = ""
        cur += line + "\n"
    if cur.strip():
        parts.append(cur.rstrip())
    return parts


def fmt_coin_line(sym: str, rsi_val: float, close: float, ef: float, es: float, gap_pct: float, vol: float, sc: float) -> str:
    return (
        f"• <b>{sym}</b> | RSI: {rsi_val:.2f} | Close: {close:.6g} | "
        f"EMA{EMA_FAST}:{ef:.6g} > EMA{EMA_SLOW}:{es:.6g} | "
        f"Gap: {gap_pct:.2f}% | Vol(24h): {vol/1e6:.1f}M | Score: {sc:.3f}"
    )


def build_message(fresh: List[Dict], cont: List[Dict], btc_msg: str) -> str:
    header = f"📊 <b>Futures Scanner</b> (TF={TF}) | <i>{utc_now_str()}</i>\n{btc_msg}\n"
    out = [header]

    if fresh:
        out.append("\n🟢 <b>FRESH CROSS (EMA fast yeni yukarı kesti)</b>")
        for x in fresh[:TOP_N]:
            out.append(fmt_coin_line(**x))
    else:
        out.append("\n🟢 <b>FRESH CROSS</b>\n— Yok")

    if cont:
        out.append("\n🟠 <b>TREND DEVAM (EMA fast hâlâ üstte)</b>")
        for x in cont[:TOP_N]:
            out.append(fmt_coin_line(**x))
    else:
        out.append("\n🟠 <b>TREND DEVAM</b>\n— Yok")

    return "\n".join(out)


# =========================
# MAIN LOOP
# =========================
def main():
    storage = Storage(STORAGE_PATH) if USE_STORAGE else None

    while True:
        t0 = time.time()

        try:
            gate_ok, gate_msg = btc_gate()
        except Exception as e:
            gate_ok, gate_msg = False, f"BTC gate error: {e}"

        if not gate_ok:
            # sadece BTC durumu yazıp çıkma (spam olmasın)
            print(f"[LOOP] BTC gate FALSE | {gate_msg} | {utc_now_str()}")
            time.sleep(INTERVAL_SEC)
            continue

        try:
            symbols = get_usdt_perp_symbols()
            qv_map = build_quote_volume_map()

            fresh_list = []
            cont_list = []

            scanned = 0
            for sym in symbols:
                scanned += 1
                vol = qv_map.get(sym, 0.0)
                if vol < MIN_QUOTE_VOLUME:
                    continue

                # cooldown
                if storage:
                    last_ts = storage.get(sym, 0)
                    if last_ts and (time.time() - last_ts) < COOLDOWN_SEC:
                        continue

                try:
                    kl = fetch_klines(sym)
                    closes = [float(k[4]) for k in kl[:-1]]  # sadece kapanmış mum
                    if len(closes) < max(EMA_SLOW + 5, RSI_LEN + 5):
                        continue

                    fresh_cross, trend_ok, ef, es, gap_pct = last_closed_cross(closes)
                    if not trend_ok:
                        continue

                    r = rsi(closes, RSI_LEN)
                    rsi_val = r[-1]
                    close = closes[-1]

                    sc = score_coin(rsi_val, gap_pct, vol)

                    item = dict(
                        sym=sym,
                        rsi_val=rsi_val,
                        close=close,
                        ef=ef,
                        es=es,
                        gap_pct=gap_pct,
                        vol=vol,
                        sc=sc
                    )

                    if fresh_cross:
                        fresh_list.append(item)
                    else:
                        cont_list.append(item)

                except Exception:
                    continue

            # skorla sırala
            fresh_list.sort(key=lambda x: x["sc"], reverse=True)
            cont_list.sort(key=lambda x: x["sc"], reverse=True)

            # mesaj gönderme koşulu: liste varsa
            if fresh_list or cont_list:
                msg = build_message(fresh_list, cont_list, btc_msg=gate_msg)

                chunks = chunk_text(msg, TG_CHUNK_LIMIT)
                for c in chunks:
                    send_telegram(c)

                if storage:
                    # mesaj attıklarımızı cooldown'a al
                    now = int(time.time())
                    for x in fresh_list[:TOP_N]:
                        storage.set(x["sym"], now)
                    for x in cont_list[:TOP_N]:
                        storage.set(x["sym"], now)
                    storage.save()

                signals = len(fresh_list) + len(cont_list)
                print(f"[LOOP] scanned={scanned} symbols | fresh={len(fresh_list)} cont={len(cont_list)} total={signals} | {utc_now_str()}")
            else:
                print(f"[LOOP] scanned={len(symbols)} symbols | no candidates | {utc_now_str()}")

        except Exception as e:
            print(f"[ERROR] {e} | {utc_now_str()}")

        # uyku
        dt = time.time() - t0
        sleep_for = max(3, INTERVAL_SEC - int(dt))
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
