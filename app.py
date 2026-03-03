import os
import time
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests

from notify import send_telegram
from storage import Storage

# =========================
# CONFIG
# =========================
BINANCE_FAPI = os.getenv("BINANCE_FAPI", "https://fapi.binance.com").rstrip("/")
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "12"))

TF_ENTRY = os.getenv("TF_ENTRY", os.getenv("TF", "1h"))  # compat
KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "260"))
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "600"))
HEARTBEAT_SEC = int(os.getenv("HEARTBEAT_SEC", "900"))
TEST_ONCE = int(os.getenv("TEST_ONCE", "0"))

# Entry EMA cross
EMA_FAST = int(os.getenv("EMA_FAST", "3"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "44"))
LOOKBACK = int(os.getenv("LOOKBACK", "6"))

# RSI
RSI_LEN = int(os.getenv("RSI_LEN", "21"))
RSI_MIN = float(os.getenv("RSI_MIN", "42"))

# Market universe
TOP_N = int(os.getenv("TOP_N", "200"))
MIN_QUOTE_VOLUME = float(os.getenv("MIN_QUOTE_VOLUME", "3000000"))
ONLY_USDT_PERP = int(os.getenv("ONLY_USDT_PERP", "1"))

# Cooldown
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", "21600"))

# Toggles
DEBUG = int(os.getenv("DEBUG", "1"))
DEBUG_REJECTS = int(os.getenv("DEBUG_REJECTS", "0"))
DRY_RUN = int(os.getenv("DRY_RUN", "0"))

USE_STORAGE = int(os.getenv("USE_STORAGE", "1"))
STORAGE_PATH = os.getenv("STORAGE_PATH", "/var/data/futures_state.json")

USE_BTC_FILTER = int(os.getenv("USE_BTC_FILTER", "0"))
BTC_SYMBOL = os.getenv("BTC_SYMBOL", "BTCUSDT")
BTC_TF = os.getenv("BTC_TF", "4h")

USE_HTF_FILTER = int(os.getenv("USE_HTF_FILTER", "0"))
HTF = os.getenv("HTF", "4h")
HTF_CROSS_LOOKBACK = int(os.getenv("HTF_CROSS_LOOKBACK", "6"))
HTF_STRICT_CROSS = int(os.getenv("HTF_STRICT_CROSS", "0"))  # 0: "cross within lookback", 1: "must be currently above too"

USE_VOL_FILTER = int(os.getenv("USE_VOL_FILTER", "0"))
VOL_LEN = int(os.getenv("VOL_LEN", "20"))
VOL_MULT = float(os.getenv("VOL_MULT", "1.1"))
VOL_USE_QUOTE = int(os.getenv("VOL_USE_QUOTE", "1"))

# MFI (opsiyonel) - TV setup'a yakın istiyorsan USE_MFI_FILTER=0
USE_MFI_FILTER = int(os.getenv("USE_MFI_FILTER", "0"))
MFI_LEN = int(os.getenv("MFI_LEN", "14"))
MFI_LONG_MIN = float(os.getenv("MFI_LONG_MIN", "40"))
MFI_LONG_MAX = float(os.getenv("MFI_LONG_MAX", "85"))
MFI_SLOPE_ENABLE = int(os.getenv("MFI_SLOPE_ENABLE", "1"))
MFI_SLOPE_BARS = int(os.getenv("MFI_SLOPE_BARS", "1"))

# WaveTrend + Stoch RSI
USE_WT = int(os.getenv("USE_WT", "1"))
USE_STOCH_RSI = int(os.getenv("USE_STOCH_RSI", "1"))

WT_CH_LEN = int(os.getenv("WT_CH_LEN", "9"))
WT_AVG_LEN = int(os.getenv("WT_AVG_LEN", "12"))
WT_CH = int(os.getenv("WT_CH", "12"))          # compat (kullanılmıyor ama envde dursun)
WT_AVG = int(os.getenv("WT_AVG", "12"))        # compat (kullanılmıyor ama envde dursun)

WT_OB1 = float(os.getenv("WT_OB1", "60"))
WT_OB2 = float(os.getenv("WT_OB2", "53"))
WT_OS1 = float(os.getenv("WT_OS1", "-60"))
WT_OS2 = float(os.getenv("WT_OS2", "-53"))

USE_WT_DIP = int(os.getenv("USE_WT_DIP", "1"))
USE_WT_CONTINUATION = int(os.getenv("USE_WT_CONTINUATION", "0"))

STOCH_RSI_LEN = int(os.getenv("STOCH_RSI_LEN", "14"))
STOCH_K = int(os.getenv("STOCH_K", "5"))
STOCH_D = int(os.getenv("STOCH_D", "5"))

# Candle selection:
# 0 = last CLOSED candle (önerilen, repaint yok)
# 1 = last candle (canlı, repaint riski var)
USE_LAST_CANDLE = int(os.getenv("USE_LAST_CANDLE", "0"))

# Trend EMA (opsiyonel, mesajda gösteriyoruz)
EMA_TREND = int(os.getenv("EMA_TREND", "123"))

# Telegram
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "")

LONG_ONLY = int(os.getenv("LONG_ONLY", "1"))

# Exit alert (opsiyonel)
USE_EXIT_ALERT = int(os.getenv("USE_EXIT_ALERT", "0"))
MAX_HOLD_SEC = int(os.getenv("MAX_HOLD_SEC", "172800"))  # 2 gün default


# =========================
# DATA STRUCTURES
# =========================
@dataclass
class Candle:
    o: float
    h: float
    l: float
    c: float
    v: float


# =========================
# HELPERS
# =========================
def _now() -> int:
    return int(time.time())


def _http_get(path: str, params: Optional[dict] = None) -> dict:
    url = f"{BINANCE_FAPI}{path}"
    r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_usdt_perp_symbols() -> List[str]:
    info = _http_get("/fapi/v1/exchangeInfo")
    out = []
    for s in info.get("symbols", []):
        if s.get("status") != "TRADING":
            continue
        if ONLY_USDT_PERP:
            if s.get("contractType") != "PERPETUAL":
                continue
            if s.get("quoteAsset") != "USDT":
                continue
        out.append(s["symbol"])
    return out


def get_top_symbols_by_volume(symbols: List[str]) -> List[Tuple[str, float]]:
    tickers = _http_get("/fapi/v1/ticker/24hr")
    vol_map = {}
    for t in tickers:
        sym = t.get("symbol")
        if sym in symbols:
            # quoteVolume = son 24 saatte USDT bazlı hacim
            try:
                qv = float(t.get("quoteVolume", 0.0))
            except Exception:
                qv = 0.0
            vol_map[sym] = qv

    items = [(s, vol_map.get(s, 0.0)) for s in symbols]
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:TOP_N]


def get_klines(symbol: str, interval: str, limit: int) -> List[Candle]:
    raw = _http_get("/fapi/v1/klines", params={"symbol": symbol, "interval": interval, "limit": limit})
    out: List[Candle] = []
    for k in raw:
        out.append(
            Candle(
                o=float(k[1]),
                h=float(k[2]),
                l=float(k[3]),
                c=float(k[4]),
                v=float(k[5]),
            )
        )
    return out


def sma(values: List[float], length: int) -> List[float]:
    if length <= 1:
        return values[:]
    out = []
    s = 0.0
    q = []
    for x in values:
        q.append(x)
        s += x
        if len(q) > length:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def ema(values: List[float], length: int) -> List[float]:
    if length <= 1:
        return values[:]
    out = []
    k = 2.0 / (length + 1.0)
    e = values[0]
    for x in values:
        e = x * k + e * (1.0 - k)
        out.append(e)
    return out


def rsi(closes: List[float], length: int) -> List[float]:
    if length <= 1:
        return [50.0] * len(closes)
    gains = [0.0]
    losses = [0.0]
    for i in range(1, len(closes)):
        ch = closes[i] - closes[i - 1]
        gains.append(max(ch, 0.0))
        losses.append(max(-ch, 0.0))

    avg_g = ema(gains, length)
    avg_l = ema(losses, length)

    out = []
    for g, l in zip(avg_g, avg_l):
        if l == 0:
            out.append(100.0)
        else:
            rs = g / l
            out.append(100.0 - (100.0 / (1.0 + rs)))
    return out


def stoch_rsi(closes: List[float], rsi_len: int, stoch_len: int, k_len: int, d_len: int) -> Tuple[List[float], List[float]]:
    r = rsi(closes, rsi_len)
    st = []
    for i in range(len(r)):
        start = max(0, i - stoch_len + 1)
        window = r[start : i + 1]
        lo = min(window)
        hi = max(window)
        if hi - lo == 0:
            st.append(0.0)
        else:
            st.append(100.0 * (r[i] - lo) / (hi - lo))
    k = sma(st, k_len)
    d = sma(k, d_len)
    return k, d


def wavetrend(candles: List[Candle], ch_len: int, avg_len: int) -> Tuple[List[float], List[float]]:
    # LazyBear WT (yakın implementasyon)
    ap = [(c.h + c.l + c.c) / 3.0 for c in candles]
    esa = ema(ap, ch_len)
    dev = ema([abs(ap[i] - esa[i]) for i in range(len(ap))], ch_len)

    ci = []
    for i in range(len(ap)):
        d = dev[i]
        if d == 0:
            ci.append(0.0)
        else:
            ci.append((ap[i] - esa[i]) / (0.015 * d))

    tci = ema(ci, avg_len)
    wt1 = tci
    wt2 = sma(wt1, 4)
    return wt1, wt2


def mfi(candles: List[Candle], length: int) -> List[float]:
    tp = [(c.h + c.l + c.c) / 3.0 for c in candles]
    rmf = [tp[i] * candles[i].v for i in range(len(candles))]
    pos = [0.0]
    neg = [0.0]
    for i in range(1, len(tp)):
        if tp[i] > tp[i - 1]:
            pos.append(rmf[i])
            neg.append(0.0)
        elif tp[i] < tp[i - 1]:
            pos.append(0.0)
            neg.append(rmf[i])
        else:
            pos.append(0.0)
            neg.append(0.0)

    out = []
    for i in range(len(tp)):
        start = max(0, i - length + 1)
        ps = sum(pos[start : i + 1])
        ns = sum(neg[start : i + 1])
        if ns == 0:
            out.append(100.0)
        else:
            mr = ps / ns
            out.append(100.0 - (100.0 / (1.0 + mr)))
    return out


def crossed_up(a: List[float], b: List[float], i: int) -> bool:
    if i <= 0:
        return False
    return a[i - 1] <= b[i - 1] and a[i] > b[i]


def crossed_down(a: List[float], b: List[float], i: int) -> bool:
    if i <= 0:
        return False
    return a[i - 1] >= b[i - 1] and a[i] < b[i]


def pick_index(n: int) -> int:
    # last closed candle index = -2 (çünkü -1 canlı mum)
    if n < 3:
        return n - 1
    return -1 if USE_LAST_CANDLE else -2


# =========================
# FILTERS / SIGNAL LOGIC
# =========================
def ema_cross_ok(closes: List[float], fast_len: int, slow_len: int, lookback: int, idx: int) -> bool:
    ef = ema(closes, fast_len)
    es = ema(closes, slow_len)

    # ŞART 1: şu an (seçilen mumda) fast > slow
    if not (ef[idx] > es[idx]):
        return False

    # ŞART 2: lookback içinde "cross up" olmuş olmalı
    # (idx negatif olabilir; gerçek indexe çevirelim)
    N = len(closes)
    cur = idx if idx >= 0 else N + idx
    start = max(1, cur - lookback + 1)
    for i in range(start, cur + 1):
        if crossed_up(ef, es, i):
            return True
    return False


def htf_filter_ok(symbol: str) -> bool:
    if not USE_HTF_FILTER:
        return True
    candles = get_klines(symbol, HTF, min(KLINE_LIMIT, 260))
    closes = [c.c for c in candles]
    idx = pick_index(len(closes))
    ok = ema_cross_ok(closes, EMA_FAST, EMA_SLOW, HTF_CROSS_LOOKBACK, idx)
    if HTF_STRICT_CROSS:
        ef = ema(closes, EMA_FAST)
        es = ema(closes, EMA_SLOW)
        ok = ok and (ef[idx] > es[idx])
    return ok


def btc_filter_ok() -> bool:
    if not USE_BTC_FILTER:
        return True
    candles = get_klines(BTC_SYMBOL, BTC_TF, min(KLINE_LIMIT, 260))
    closes = [c.c for c in candles]
    idx = pick_index(len(closes))
    # BTC trend: EMA_FAST > EMA_SLOW şartı
    ef = ema(closes, EMA_FAST)
    es = ema(closes, EMA_SLOW)
    return ef[idx] > es[idx]


def vol_filter_ok(candles: List[Candle], idx: int) -> bool:
    if not USE_VOL_FILTER:
        return True
    vols = [c.v for c in candles]
    if VOL_USE_QUOTE:
        vols = [candles[i].v * candles[i].c for i in range(len(candles))]
    N = len(vols)
    cur = idx if idx >= 0 else N + idx
    if cur <= 0:
        return False
    start = max(0, cur - VOL_LEN)
    baseline = sum(vols[start:cur]) / max(1, (cur - start))
    return vols[cur] >= baseline * VOL_MULT


def mfi_filter_ok(candles: List[Candle], idx: int) -> bool:
    if not USE_MFI_FILTER:
        return True
    mf = mfi(candles, MFI_LEN)
    val = mf[idx]
    if not (MFI_LONG_MIN <= val <= MFI_LONG_MAX):
        return False
    if MFI_SLOPE_ENABLE and MFI_SLOPE_BARS > 0:
        # son N barda yükseliyor mu
        N = len(mf)
        cur = idx if idx >= 0 else N + idx
        prev = max(0, cur - MFI_SLOPE_BARS)
        return mf[cur] >= mf[prev]
    return True


def build_signal(symbol: str, candles: List[Candle]) -> Optional[Dict]:
    closes = [c.c for c in candles]
    idx = pick_index(len(closes))

    # RSI
    r = rsi(closes, RSI_LEN)
    if r[idx] < RSI_MIN:
        return None

    # EMA cross on entry TF
    if not ema_cross_ok(closes, EMA_FAST, EMA_SLOW, LOOKBACK, idx):
        return None

    # Optional filters
    if not vol_filter_ok(candles, idx):
        return None
    if not mfi_filter_ok(candles, idx):
        return None
    if not htf_filter_ok(symbol):
        return None
    if not btc_filter_ok():
        return None

    # Indicators for message / hybrid decision
    ef = ema(closes, EMA_FAST)
    es = ema(closes, EMA_SLOW)
    et = ema(closes, EMA_TREND) if EMA_TREND > 1 else closes[:]

    st_k, st_d = stoch_rsi(closes, RSI_LEN, STOCH_RSI_LEN, STOCH_K, STOCH_D) if USE_STOCH_RSI else ([50.0]*len(closes), [50.0]*len(closes))
    wt1, wt2 = wavetrend(candles, WT_CH_LEN, WT_AVG_LEN) if USE_WT else ([0.0]*len(closes), [0.0]*len(closes))

    # Decide signal type
    sig_type = None

    # DIP reversal: WT çok negatifken (dip) yukarı döndürme
    if USE_WT and USE_WT_DIP:
        dip_zone = wt1[idx] <= WT_OS2
        st_ok = (st_k[idx] < 20 and st_d[idx] < 20 and st_k[idx] >= st_k[idx - 1]) if (idx != 0) else False
        wt_turn = crossed_up(wt1, wt2, (idx if idx >= 0 else len(wt1) + idx)) if len(wt1) > 2 else False
        if dip_zone and (wt_turn or st_ok):
            sig_type = "WT_DIP"

    # CONTINUATION: trend devamı (0 üstünde WT güçlenmesi)
    if USE_WT and USE_WT_CONTINUATION and sig_type is None:
        N = len(wt1)
        cur = idx if idx >= 0 else N + idx
        cont_cross = crossed_up(wt1, wt2, cur)
        cont_zone = wt1[idx] > 0 and wt1[idx] < WT_OB2
        if cont_cross and cont_zone:
            sig_type = "WT_CONT"

    # Eğer WT kapalıysa yine de EMA cross + RSI ile sinyal üretebilirsin (ama sen WT istiyorsun)
    if sig_type is None and USE_WT:
        # WT açık ama dip/cont seçilmediyse: sinyal yok
        return None
    if sig_type is None and not USE_WT:
        sig_type = "EMA_CROSS"

    return {
        "type": sig_type,
        "symbol": symbol,
        "price": closes[idx],
        "ema_fast": ef[idx],
        "ema_slow": es[idx],
        "ema_trend": et[idx] if len(et) == len(closes) else None,
        "rsi": r[idx],
        "st_k": st_k[idx],
        "st_d": st_d[idx],
        "wt1": wt1[idx],
        "wt2": wt2[idx],
        "tf_entry": TF_ENTRY,
        "htf": HTF,
    }


def should_exit(signal: Dict, candles: List[Candle]) -> bool:
    # Basit WT exit kuralı: WT1>WT_OB2 iken WT1 aşağı keserse
    if not USE_WT:
        return False
    closes = [c.c for c in candles]
    idx = pick_index(len(closes))
    wt1, wt2 = wavetrend(candles, WT_CH_LEN, WT_AVG_LEN)
    N = len(wt1)
    cur = idx if idx >= 0 else N + idx
    return wt1[idx] > WT_OB2 and crossed_down(wt1, wt2, cur)


def format_signal_msg(sig: Dict) -> str:
    title = f"🚀 LONG SIGNAL [{sig['type']}]"
    lines = [
        title,
        f"Symbol: {sig['symbol']}",
        f"TF: {sig['tf_entry']} | HTF: {sig['htf']}",
        f"Price: {sig['price']:.6f}".rstrip("0").rstrip("."),
        "",
        f"EMA{EMA_FAST}: {sig['ema_fast']:.6f} | EMA{EMA_SLOW}: {sig['ema_slow']:.6f}",
        f"RSI({RSI_LEN}): {sig['rsi']:.2f}",
    ]
    if USE_STOCH_RSI:
        lines.append(f"StochRSI K/D (K={STOCH_K},D={STOCH_D}): {sig['st_k']:.2f}/{sig['st_d']:.2f}")
    if USE_WT:
        lines.append(f"WT_LB (ch={WT_CH_LEN},avg={WT_AVG_LEN}) WT1/WT2: {sig['wt1']:.2f}/{sig['wt2']:.2f}")

    lines += [
        "",
        "Exit plan (manual):",
        f"- TP1: +{float(os.getenv('TP_PCT','8')):.1f}% (suggestion)",
        f"- SL: -{float(os.getenv('SL_PCT','2')):.1f}% (suggestion)",
        f"- WT exit: if WT1 crosses DOWN WT2 while WT1>{WT_OB2:.0f} consider close/trim",
        f"- WT warning: if WT1>{WT_OB1:.0f} and turns down -> tighten stop",
    ]
    return "\n".join(lines)


def format_exit_msg(symbol: str, price: float) -> str:
    return "\n".join([
        "🟠 EXIT WARNING [WT_EXIT]",
        f"Symbol: {symbol}",
        f"Price: {price:.6f}".rstrip("0").rstrip("."),
        f"Rule: WT1>{WT_OB2:.0f} and crossed DOWN WT2",
    ])


def log(msg: str):
    if DEBUG:
        print(msg, flush=True)


# =========================
# MAIN LOOP
# =========================
def main():
    storage = Storage(STORAGE_PATH) if USE_STORAGE else None

    hb_next = _now() + HEARTBEAT_SEC

    log(f"[BOOT] Futures scanner started")
    log(f"[CFG] TF_ENTRY={TF_ENTRY} EMA={EMA_FAST}/{EMA_SLOW} LOOKBACK={LOOKBACK} RSI_LEN={RSI_LEN} RSI_MIN={RSI_MIN}")
    log(f"[CFG] WT={USE_WT} DIP={USE_WT_DIP} CONT={USE_WT_CONTINUATION} STOCH_RSI={USE_STOCH_RSI}")
    log(f"[CFG] TOP_N={TOP_N} MIN_QUOTE_VOLUME={MIN_QUOTE_VOLUME} COOLDOWN_SEC={COOLDOWN_SEC} DRY_RUN={DRY_RUN} USE_LAST_CANDLE={USE_LAST_CANDLE}")
    log(f"[CFG] STORAGE_PATH={STORAGE_PATH}")

    # Heartbeat
    send_telegram(TG_BOT_TOKEN, TG_CHAT_ID,
                  f"✅ worker alive | TF={TF_ENTRY} TOP_N={TOP_N} DIP={USE_WT_DIP} CONT={USE_WT_CONTINUATION}",
                  dry_run=bool(DRY_RUN))

    # Universe
    all_syms = get_usdt_perp_symbols()
    top = get_top_symbols_by_volume(all_syms)

    # Pre-filter min volume
    top = [(s, qv) for (s, qv) in top if qv >= MIN_QUOTE_VOLUME]

    if DEBUG:
        log(f"[INFO] universe size={len(all_syms)} top(after min vol)={len(top)}")

    while True:
        t0 = _now()

        # heartbeat
        if t0 >= hb_next:
            send_telegram(TG_BOT_TOKEN, TG_CHAT_ID,
                          f"✅ worker alive | TF={TF_ENTRY} TOP_N={TOP_N} DIP={USE_WT_DIP} CONT={USE_WT_CONTINUATION}",
                          dry_run=bool(DRY_RUN))
            hb_next = t0 + HEARTBEAT_SEC

        for sym, qv in top:
            try:
                # cooldown
                if storage and storage.in_cooldown(sym, COOLDOWN_SEC):
                    continue

                candles = get_klines(sym, TF_ENTRY, KLINE_LIMIT)

                sig = build_signal(sym, candles)
                if sig:
                    msg = format_signal_msg(sig)
                    send_telegram(TG_BOT_TOKEN, TG_CHAT_ID, msg, dry_run=bool(DRY_RUN))

                    if storage:
                        storage.mark_event(sym, "entry")
                    continue

                # Optional exit alert for recently signaled symbols
                if USE_EXIT_ALERT and storage:
                    last_entry = storage.get_last(sym, "entry")
                    if last_entry and (_now() - last_entry) <= MAX_HOLD_SEC:
                        if storage.in_cooldown(sym + ":exit", 900):  # exit spam koruması
                            continue
                        idx = pick_index(len(candles))
                        if should_exit({"symbol": sym}, candles):
                            price = candles[idx].c
                            send_telegram(TG_BOT_TOKEN, TG_CHAT_ID, format_exit_msg(sym, price), dry_run=bool(DRY_RUN))
                            storage.mark_event(sym + ":exit", "exit")

            except Exception as e:
                if DEBUG:
                    log(f"[ERR] {sym}: {type(e).__name__}: {e}")

        if TEST_ONCE:
            log("[INFO] TEST_ONCE=1 -> exit")
            return

        # loop pacing
        dt = _now() - t0
        sleep_sec = max(1, INTERVAL_SEC - dt)
        time.sleep(sleep_sec)


if __name__ == "__main__":
    main()
