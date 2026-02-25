import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import requests

from notify import send_telegram
from storage import Storage

# ======================
# ENV AYARLARI
# ======================
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "600"))  # 5-10 dk öneri (600 = 10 dk)
TF = os.getenv("TF", "30m")  # "30m" veya "1h" vb.

EMA_FAST = int(os.getenv("EMA_FAST", "3"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "41"))

RSI_LEN = int(os.getenv("RSI_LEN", "14"))

# BTC filter
USE_BTC_FILTER = os.getenv("USE_BTC_FILTER", "1") == "1"
BTC_RSI_MIN = float(os.getenv("BTC_RSI_MIN", "42"))
BTC_SYMBOL = os.getenv("BTC_SYMBOL", "BTCUSDT")

# Volume filter (24h quote volume)
MIN_QUOTE_VOLUME = float(os.getenv("MIN_QUOTE_VOLUME", "15000000"))  # 15M

# Liste boyutu
TOP_N = int(os.getenv("TOP_N", "30"))

# Kline limit (EMA/RSI için yeterli olmalı)
KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "200"))

# Spam önleme: aynı kesişimi tekrar yollama
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", "3600"))  # 1 saat

BINANCE_FAPI = "https://fapi.binance.com"

# ======================
# HELPERS
# ======================

def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def fetch_json(url: str, params: dict = None, timeout: int = 20):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def fetch_usdt_perp_symbols() -> List[str]:
    """Binance USDT perpetual symbols"""
    data = fetch_json(f"{BINANCE_FAPI}/fapi/v1/exchangeInfo")
    out = []
    for s in data.get("symbols", []):
        if s.get("contractType") != "PERPETUAL":
            continue
        if s.get("quoteAsset") != "USDT":
            continue
        if s.get("status") != "TRADING":
            continue
        out.append(s["symbol"])
    return out

def fetch_24h_quote_volumes() -> Dict[str, float]:
    """24h ticker -> quoteVolume"""
    data = fetch_json(f"{BINANCE_FAPI}/fapi/v1/ticker/24hr")
    vols = {}
    for row in data:
        sym = row.get("symbol")
        vols[sym] = safe_float(row.get("quoteVolume", 0.0), 0.0)
    return vols

def fetch_klines(symbol: str, interval: str, limit: int) -> List[list]:
    return fetch_json(
        f"{BINANCE_FAPI}/fapi/v1/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=30
    )

def close_series_from_klines(klines: List[list]) -> List[float]:
    # kline format: [openTime, open, high, low, close, volume, closeTime, quoteVol, ...]
    return [safe_float(k[4]) for k in klines]

def ema_series(values: List[float], length: int) -> List[float]:
    if length <= 1:
        return values[:]
    out = []
    k = 2.0 / (length + 1.0)
    ema = None
    for v in values:
        if ema is None:
            ema = v
        else:
            ema = (v - ema) * k + ema
        out.append(ema)
    return out

def calc_rsi(values: List[float], length: int) -> float:
    # classic Wilder RSI on last value
    if len(values) < length + 2:
        return 50.0
    gains = 0.0
    losses = 0.0
    for i in range(-length, 0):
        diff = values[i] - values[i - 1]
        if diff >= 0:
            gains += diff
        else:
            losses += -diff
    if losses == 0:
        return 100.0
    rs = (gains / length) / (losses / length)
    return 100.0 - (100.0 / (1.0 + rs))

def calc_macd_hist(values: List[float], fast=12, slow=26, signal=9) -> float:
    if len(values) < slow + signal + 5:
        return 0.0
    ema_fast = ema_series(values, fast)
    ema_slow = ema_series(values, slow)
    macd = [a - b for a, b in zip(ema_fast, ema_slow)]
    sig = ema_series(macd, signal)
    hist = macd[-1] - sig[-1]
    return hist

def last_closed_index(klines: List[list]) -> int:
    # Binance son kline genelde “aktif” olabilir; güvenli yöntem: -2
    if len(klines) < 3:
        return -1
    return -2

def crossed_up(fast_prev: float, slow_prev: float, fast_now: float, slow_now: float) -> bool:
    return (fast_prev <= slow_prev) and (fast_now > slow_now)

def score_symbol(
    close_now: float,
    ema_fast_now: float,
    ema_slow_now: float,
    ema_fast_prev: float,
    ema_slow_prev: float,
    rsi_now: float,
    macd_hist: float,
    quote_vol_24h: float,
) -> float:
    """
    Skor: trend gücü + hacim + momentum
    - EMA ayrışması (%)
    - MACD hist pozitifse bonus
    - RSI 45-70 arası daha sağlıklı (aşırıya kaçınca biraz kırp)
    - Hacim log etkisi gibi davranır
    """
    if close_now <= 0:
        return 0.0

    ema_gap_pct = ((ema_fast_now - ema_slow_now) / close_now) * 100.0  # pozitif iyidir
    ema_gap_pct = max(-5.0, min(5.0, ema_gap_pct))  # kırp

    # trend devam: ema_fast yükseliyor mu?
    slope = (ema_fast_now - ema_fast_prev) / close_now * 100.0

    # RSI shaping
    rsi_bonus = 0.0
    if 45 <= rsi_now <= 70:
        rsi_bonus = (rsi_now - 45) * 0.03  # max ~0.75
    elif rsi_now > 70:
        rsi_bonus = 0.3  # aşırıysa az bonus
    else:
        rsi_bonus = -0.2

    # MACD
    macd_bonus = 0.5 if macd_hist > 0 else -0.2

    # volume bonus (15M üstü +)
    vol_bonus = 0.0
    if quote_vol_24h > 0:
        vol_bonus = min(2.0, (quote_vol_24h / 15_000_000.0) * 0.4)  # 15M => 0.4, 75M => 2.0 cap

    score = (ema_gap_pct * 1.8) + (slope * 1.2) + rsi_bonus + macd_bonus + vol_bonus
    return score

def btc_gate_ok() -> Tuple[bool, str]:
    if not USE_BTC_FILTER:
        return True, "BTC filter OFF"

    try:
        kl = fetch_klines(BTC_SYMBOL, TF, max(KLINE_LIMIT, 120))
        idx = last_closed_index(kl)
        closes = close_series_from_klines(kl)

        closes_closed = closes[: idx + 1]  # last closed dahil
        ef = ema_series(closes_closed, EMA_FAST)
        es = ema_series(closes_closed, EMA_SLOW)

        rsi = calc_rsi(closes_closed, RSI_LEN)

        ok = (ef[-1] > es[-1]) and (rsi > BTC_RSI_MIN)
        msg = f"BTC gate: EMA{EMA_FAST}>{EMA_SLOW}={ef[-1] > es[-1]} | RSI{RSI_LEN}={rsi:.2f} > {BTC_RSI_MIN}"
        return ok, msg
    except Exception as e:
        return False, f"BTC gate error: {e}"

def format_ranked_message(rows: List[dict], btc_msg: str) -> str:
    header = f"📡 Futures Scanner (TF={TF}) | {utc_now_str()}\n{btc_msg}\n"
    if not rows:
        return header + "\nSinyal yok."

    lines = [header, f"✅ Sıralı Liste (Top {min(TOP_N, len(rows))})\n"]
    for i, r in enumerate(rows[:TOP_N], start=1):
        # boşluklu format
        lines.append(
            f"{i}) {r['symbol']}  | skor={r['score']:.2f}\n"
            f"   cross={r['fresh_cross']} | RSI={r['rsi']:.2f} | MACDh={r['macd_hist']:.4f}\n"
            f"   24hQVol={r['qv_m']:.1f}M | close={r['close']:.6f}\n"
        )
    return "\n".join(lines)

# ======================
# MAIN LOOP
# ======================

def main():
    st = Storage(os.getenv("STORAGE_PATH", "state.json"))

    while True:
        t0 = time.time()
        try:
            symbols = fetch_usdt_perp_symbols()
            vols = fetch_24h_quote_volumes()

            btc_ok, btc_msg = btc_gate_ok()
            if not btc_ok:
                # BTC gate kapalıysa sinyal üretmeyelim ama bilgi atalım (spam olmaması için cooldown)
                last_btc_note = st.get("last_btc_note_ts", 0)
                now = int(time.time())
                if now - last_btc_note > 900:  # 15 dk’da bir bilgilendir
                    send_telegram(f"⛔ BTC Filtresi Uymuyor\n{btc_msg}\n{utc_now_str()}")
                    st.set("last_btc_note_ts", now)
                    st.save()

                print("[BTC] gate NOT OK:", btc_msg)
                time.sleep(INTERVAL_SEC)
                continue

            candidates = []
            scanned = 0
            for sym in symbols:
                qv = vols.get(sym, 0.0)
                if qv < MIN_QUOTE_VOLUME:
                    continue

                try:
                    kl = fetch_klines(sym, TF, KLINE_LIMIT)
                    idx = last_closed_index(kl)
                    if idx < 2:
                        continue

                    closes = close_series_from_klines(kl)
                    closes_closed = closes[: idx + 1]  # sadece kapananlar

                    close_now = closes_closed[-1]
                    close_prev = closes_closed[-2]

                    ef = ema_series(closes_closed, EMA_FAST)
                    es = ema_series(closes_closed, EMA_SLOW)

                    ema_fast_now = ef[-1]
                    ema_slow_now = es[-1]
                    ema_fast_prev = ef[-2]
                    ema_slow_prev = es[-2]

                    fresh = crossed_up(ema_fast_prev, ema_slow_prev, ema_fast_now, ema_slow_now)

                    rsi = calc_rsi(closes_closed, RSI_LEN)
                    macd_hist = calc_macd_hist(closes_closed)

                    sc = score_symbol(
                        close_now, ema_fast_now, ema_slow_now,
                        ema_fast_prev, ema_slow_prev, rsi, macd_hist, qv
                    )

                    # spam engeli: fresh cross olduğunda bir kere
                    if fresh:
                        key = f"last_cross_{sym}"
                        last_ts = int(st.get(key, 0))
                        now = int(time.time())
                        if now - last_ts < COOLDOWN_SEC:
                            # cooldown içindeyse yine de sıralamaya alabiliriz ama mesajı şişirmeyelim:
                            pass
                        else:
                            st.set(key, now)

                    candidates.append({
                        "symbol": sym,
                        "score": sc,
                        "fresh_cross": "YES" if fresh else "no",
                        "rsi": rsi,
                        "macd_hist": macd_hist,
                        "qv_m": qv / 1_000_000.0,
                        "close": close_now,
                        "ema_fast": ema_fast_now,
                        "ema_slow": ema_slow_now,
                    })
                    scanned += 1

                except Exception:
                    continue

            # Sırala: skor desc
            candidates.sort(key=lambda x: x["score"], reverse=True)

            # Mesaj spam kontrolü: aynı listeyi sürekli yollamasın
            # "fingerprint" top 10 symbol list
            fp = ",".join([c["symbol"] for c in candidates[:10]])
            last_fp = st.get("last_fp", "")
            now = int(time.time())

            send_ok = (fp != last_fp) or (now - int(st.get("last_fp_ts", 0)) > 1800)  # 30 dk’da bir yine at
            if send_ok:
                msg = format_ranked_message(candidates, btc_msg)
                send_telegram(msg)
                st.set("last_fp", fp)
                st.set("last_fp_ts", now)
                st.save()

            dt = time.time() - t0
            print(f"[LOOP] scanned={scanned} candidates={len(candidates)} sent={send_ok} time={dt:.1f}s")

        except Exception as e:
            print("[ERROR]", e)

        # döngü bekleme
        elapsed = time.time() - t0
        sleep_for = max(1, INTERVAL_SEC - int(elapsed))
        time.sleep(sleep_for)

if __name__ == "__main__":
    main()
