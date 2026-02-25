import os
import time
import math
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import requests

from notify import send_telegram
from storage import Storage

# =========================
# ENV / AYARLAR
# =========================
BINANCE_FAPI = os.getenv("BINANCE_FAPI", "https://fapi.binance.com").rstrip("/")

TF = os.getenv("TF", "30m")                       # 30m / 1h vs
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "600"))  # 10 dk
KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "200"))

EMA_FAST = int(os.getenv("EMA_FAST", "3"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "41"))
RSI_LEN = int(os.getenv("RSI_LEN", "14"))

TOP_N = int(os.getenv("TOP_N", "30"))
MIN_QUOTE_VOLUME = float(os.getenv("MIN_QUOTE_VOLUME", "15000000"))  # 15M
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", "3600"))                # 1 saat

USE_BTC_FILTER = os.getenv("USE_BTC_FILTER", "1") == "1"
BTC_SYMBOL = os.getenv("BTC_SYMBOL", "BTCUSDT").strip().upper()
BTC_RSI_MIN = float(os.getenv("BTC_RSI_MIN", "42"))

USE_STORAGE = os.getenv("USE_STORAGE", "1") == "1"
STORAGE_PATH = os.getenv("STORAGE_PATH", "state.json")

TG_CHUNK_LIMIT = int(os.getenv("TG_CHUNK_LIMIT", "3500"))  # 4096 limitine güvenli

# =========================
# UTILS
# =========================
def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def chunk_text(text: str, limit: int) -> List[str]:
    """Telegram limit için satır bazlı güvenli parçalama."""
    if len(text) <= limit:
        return [text]
    lines = text.split("\n")
    chunks, buf, size = [], [], 0
    for ln in lines:
        add = len(ln) + 1
        if size + add > limit and buf:
            chunks.append("\n".join(buf))
            buf = [ln]
            size = len(ln) + 1
        else:
            buf.append(ln)
            size += add
    if buf:
        chunks.append("\n".join(buf))
    return chunks

# =========================
# INDICATORS
# =========================
def ema_series(values: List[float], length: int) -> List[float]:
    if length <= 1:
        return values[:]
    k = 2 / (length + 1.0)
    out = []
    ema = None
    for v in values:
        if ema is None:
            ema = v
        else:
            ema = v * k + ema * (1 - k)
        out.append(ema)
    return out

def rsi_series(values: List[float], length: int) -> List[float]:
    # Wilder RSI
    if length <= 1 or len(values) < length + 2:
        return [50.0] * len(values)

    gains = [0.0]
    losses = [0.0]
    for i in range(1, len(values)):
        ch = values[i] - values[i - 1]
        gains.append(max(ch, 0.0))
        losses.append(max(-ch, 0.0))

    avg_gain = sum(gains[1:length+1]) / length
    avg_loss = sum(losses[1:length+1]) / length

    out = [50.0] * (length)

    def rs_to_rsi(ag, al):
        if al == 0:
            return 100.0
        rs = ag / al
        return 100.0 - (100.0 / (1.0 + rs))

    out.append(rs_to_rsi(avg_gain, avg_loss))

    for i in range(length + 1, len(values)):
        avg_gain = (avg_gain * (length - 1) + gains[i]) / length
        avg_loss = (avg_loss * (length - 1) + losses[i]) / length
        out.append(rs_to_rsi(avg_gain, avg_loss))

    if len(out) < len(values):
        out += [out[-1]] * (len(values) - len(out))
    return out[:len(values)]

# =========================
# BINANCE FETCH
# =========================
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "futures-scanner/1.0"})

def fetch_exchange_symbols() -> List[str]:
    url = f"{BINANCE_FAPI}/fapi/v1/exchangeInfo"
    r = SESSION.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()

    symbols = []
    for s in data.get("symbols", []):
        try:
            if s.get("status") != "TRADING":
                continue
            if s.get("contractType") != "PERPETUAL":
                continue
            if s.get("quoteAsset") != "USDT":
                continue
            sym = s.get("symbol")
            if sym and sym.endswith("USDT"):
                symbols.append(sym)
        except Exception:
            continue
    return symbols

def fetch_24h_tickers() -> Dict[str, Dict]:
    url = f"{BINANCE_FAPI}/fapi/v1/ticker/24hr"
    r = SESSION.get(url, timeout=25)
    r.raise_for_status()
    arr = r.json()
    out = {}
    for it in arr:
        sym = it.get("symbol")
        if sym:
            out[sym] = it
    return out

def fetch_klines(symbol: str, interval: str, limit: int) -> List[List]:
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = SESSION.get(url, params=params, timeout=25)
    r.raise_for_status()
    return r.json()

def drop_unclosed_kline(klines: List[List]) -> List[List]:
    """Sadece kapanmış mumları değerlendir."""
    if not klines:
        return klines
    last = klines[-1]
    close_time_ms = int(last[6])
    now_ms = int(time.time() * 1000)
    if now_ms < close_time_ms:
        return klines[:-1]
    return klines

# =========================
# SIGNAL LOGIC
# =========================
def crossed_up(fast: List[float], slow: List[float], lookback_bars: int = 2) -> bool:
    """
    Fresh cross: son lookback_bars kapanmış mum içinde EMA_FAST, EMA_SLOW'u aşağıdan yukarı kesmiş mi?
    lookback_bars=2 => son 1-2 kapanmış mum geçişi kontrol.
    """
    n = min(len(fast), len(slow))
    if n < 3:
        return False

    for i in range(1, lookback_bars + 1):
        a = n - 1 - i
        b = n - i
        if a < 0 or b < 0:
            continue
        prev = fast[a] - slow[a]
        cur = fast[b] - slow[b]
        if prev <= 0 and cur > 0:
            return True
    return False

def calc_score(rsi: float, gap_pct: float, vol_24h: float, is_fresh: bool) -> float:
    # 0..1 arası stabil skor
    rsi_component = max(0.0, min(1.0, (rsi - 40.0) / 40.0))     # 40->0, 80->1
    gap_component = max(0.0, min(1.0, gap_pct / 6.0))           # 0-6% normalize
    vol_component = max(0.0, min(1.0, math.log10(max(vol_24h, 1.0)) / 8.0))
    fresh_bonus = 0.08 if is_fresh else 0.0

    score = 0.45 * gap_component + 0.35 * rsi_component + 0.20 * vol_component + fresh_bonus
    return max(0.0, min(1.0, score))

def should_cooldown(storage: Optional[Storage], key: str) -> bool:
    if not USE_STORAGE or storage is None:
        return False
    last = storage.get(key, 0)
    now = int(time.time())
    return (now - int(last)) < COOLDOWN_SEC

def mark_cooldown(storage: Optional[Storage], key: str) -> None:
    if not USE_STORAGE or storage is None:
        return
    storage.set(key, int(time.time()))

# =========================
# BTC GATE
# =========================
def btc_gate(tf: str) -> Tuple[bool, str]:
    try:
        kl = fetch_klines(BTC_SYMBOL, tf, max(120, KLINE_LIMIT))
        kl = drop_unclosed_kline(kl)
        if len(kl) < 60:
            return True, f"🧩 BTC Gate: <i>yetersiz veri</i> (skip) ✅"

        closes = [safe_float(x[4]) for x in kl]
        ema_f = ema_series(closes, EMA_FAST)
        ema_s = ema_series(closes, EMA_SLOW)
        rsi_v = rsi_series(closes, RSI_LEN)

        cond_ema = ema_f[-1] > ema_s[-1]
        cond_rsi = rsi_v[-1] >= BTC_RSI_MIN
        ok = cond_ema and cond_rsi

        msg = f"🧩 BTC Gate: EMA{EMA_FAST}>{EMA_SLOW} = <b>{str(cond_ema)}</b> | RSI{RSI_LEN} = <b>{rsi_v[-1]:.2f}</b> ≥ {BTC_RSI_MIN}"
        return ok, msg

    except Exception as e:
        return True, f"🧩 BTC Gate: <i>hata</i> (skip) ✅ | {e}"

# =========================
# TELEGRAM FORMAT (Premium)
# =========================
def format_coin_line(rank: int, sym: str, score: float, gap_pct: float, rsi_v: float, vol_24h: float, close: float) -> str:
    vol_m = vol_24h / 1_000_000.0
    # 2 satır: ilk satır okunaklı, ikinci satır close
    return (
        f"• <b>{rank:02d}) {sym}</b> | Sc:<b>{score:.3f}</b> | Gap:<b>{gap_pct:.2f}%</b> | RSI:<b>{rsi_v:.1f}</b> | Vol:<b>{vol_m:.1f}M</b>\n"
        f"   └ C:{close:.6g}"
    )

def build_message(now_utc: str, btc_gate_msg: str,
                  symbols_total: int, eligible_total: int,
                  fresh_lines: List[str], trend_lines: List[str]) -> str:

    header = [
        "🧠 <b>Futures Scanner</b>",
        f"🕒 <i>{now_utc} UTC</i> | TF: <b>{TF}</b>",
        "",
        f"🧩 {btc_gate_msg}",
        f"🧷 Symbols: <b>{symbols_total}</b> | Eligible(Vol≥{int(MIN_QUOTE_VOLUME/1_000_000)}M): <b>{eligible_total}</b>",
        "",
        "────────────────────",
        ""
    ]

    body = []
    if fresh_lines:
        body += [
            "🟢 <b>FRESH CROSS</b> <i>(son 1–2 kapanmış mum)</i>",
            f"EMA{EMA_FAST} ↑ EMA{EMA_SLOW}",
            ""
        ]
        body += fresh_lines
        body += ["", "────────────────────", ""]

    if trend_lines:
        body += [
            "🟣 <b>TREND DEVAM</b> <i>(EMA üstü korunuyor)</i>",
            ""
        ]
        body += trend_lines
        body += ["", "────────────────────"]

    if not fresh_lines and not trend_lines:
        body += ["❌ <b>Sinyal yok.</b>", "────────────────────"]

    return "\n".join(header + body)

# =========================
# MAIN SCAN
# =========================
def scan_once(storage: Optional[Storage]) -> None:
    now = utc_now_str()

    # BTC filter
    if USE_BTC_FILTER:
        ok, btc_msg = btc_gate(TF)
        if not ok:
            # Gate false ise status basıp çıkar
            msg = build_message(now, btc_msg, 0, 0, [], [])
            for chunk in chunk_text(msg, TG_CHUNK_LIMIT):
                send_telegram(chunk)
            return
    else:
        btc_msg = "BTC Gate: <i>kapalı</i> ✅"

    symbols = fetch_exchange_symbols()
    tickers = fetch_24h_tickers()

    fresh_candidates = []  # (score, gap, rsi, vol, close, sym, cd_key)
    trend_candidates = []

    eligible = 0

    for sym in symbols:
        t = tickers.get(sym, {})
        qv = safe_float(t.get("quoteVolume", 0.0))
        if qv < MIN_QUOTE_VOLUME:
            continue

        eligible += 1

        try:
            kl = fetch_klines(sym, TF, KLINE_LIMIT)
            kl = drop_unclosed_kline(kl)
            if len(kl) < max(EMA_SLOW + 5, RSI_LEN + 5, 60):
                continue

            closes = [safe_float(x[4]) for x in kl]
            ema_f = ema_series(closes, EMA_FAST)
            ema_s = ema_series(closes, EMA_SLOW)
            rsi_v = rsi_series(closes, RSI_LEN)

            is_trend = ema_f[-1] > ema_s[-1]
            if not is_trend:
                continue

            is_fresh = crossed_up(ema_f, ema_s, lookback_bars=2)
            gap_pct = ((ema_f[-1] - ema_s[-1]) / ema_s[-1]) * 100.0 if ema_s[-1] != 0 else 0.0
            score = calc_score(rsi_v[-1], gap_pct, qv, is_fresh)

            bucket = "fresh" if is_fresh else "trend"
            cd_key = f"{bucket}:{TF}:{sym}"
            if should_cooldown(storage, cd_key):
                continue

            item = (score, gap_pct, rsi_v[-1], qv, closes[-1], sym, cd_key)
            if is_fresh:
                fresh_candidates.append(item)
            else:
                trend_candidates.append(item)

        except Exception:
            continue

    fresh_candidates.sort(key=lambda x: x[0], reverse=True)
    trend_candidates.sort(key=lambda x: x[0], reverse=True)

    fresh_top = fresh_candidates[:TOP_N]
    trend_top = trend_candidates[:TOP_N]

    fresh_lines = []
    for i, (sc, gap, rsi, vol, close, sym, cd_key) in enumerate(fresh_top, start=1):
        fresh_lines.append(format_coin_line(i, sym, sc, gap, rsi, vol, close))
        mark_cooldown(storage, cd_key)

    trend_lines = []
    for i, (sc, gap, rsi, vol, close, sym, cd_key) in enumerate(trend_top, start=1):
        trend_lines.append(format_coin_line(i, sym, sc, gap, rsi, vol, close))
        mark_cooldown(storage, cd_key)

    if USE_STORAGE and storage is not None:
        storage.save()

    msg = build_message(now, btc_msg, len(symbols), eligible, fresh_lines, trend_lines)
    for chunk in chunk_text(msg, TG_CHUNK_LIMIT):
        send_telegram(chunk)

    print(f"[LOOP] symbols={len(symbols)} eligible={eligible} fresh={len(fresh_lines)} trend={len(trend_lines)} time={now} UTC")

def main() -> None:
    storage = Storage(STORAGE_PATH) if USE_STORAGE else None
    print("==> futures-scanner started")
    print(f"TF={TF} | EMA={EMA_FAST}/{EMA_SLOW} | RSI={RSI_LEN} | MIN_QUOTE_VOLUME={MIN_QUOTE_VOLUME} | TOP_N={TOP_N}")
    print(f"BTC_FILTER={USE_BTC_FILTER} BTC_SYMBOL={BTC_SYMBOL} BTC_RSI_MIN={BTC_RSI_MIN}")
    print(f"INTERVAL_SEC={INTERVAL_SEC} | KLINE_LIMIT={KLINE_LIMIT} | COOLDOWN_SEC={COOLDOWN_SEC}")

    while True:
        try:
            scan_once(storage)
        except Exception as e:
            print("[WARN] loop exception:", e)
        time.sleep(INTERVAL_SEC)

if __name__ == "__main__":
    main()
