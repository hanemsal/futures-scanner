import os
import time
import math
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import requests

from notify import send_telegram
from storage import Storage

BINANCE_FAPI = os.getenv("BINANCE_FAPI", "https://fapi.binance.com").rstrip("/")

# ===== Telegram =====
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "")
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "12"))

# ===== Core =====
TF_ENTRY = os.getenv("TF_ENTRY", "1h")           # entry timeframe
KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "260"))
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "600"))
TOP_N = int(os.getenv("TOP_N", "200"))
MIN_QUOTE_VOLUME = float(os.getenv("MIN_QUOTE_VOLUME", "3000000"))
ONLY_USDT_PERP = int(os.getenv("ONLY_USDT_PERP", "1"))

# ===== EMA / RSI =====
EMA_FAST = int(os.getenv("EMA_FAST", "3"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "44"))
LOOKBACK = int(os.getenv("LOOKBACK", "6"))
RSI_LEN = int(os.getenv("RSI_LEN", "21"))
RSI_MIN = float(os.getenv("RSI_MIN", "42"))

# Eğer 1 yaparsan sadece "son kapanan mum" üstünden sinyal üretir
# (kapanmamış son mumu yok sayar)
USE_LAST_CANDLE = int(os.getenv("USE_LAST_CANDLE", "0"))

# ===== Feature toggles =====
DEBUG = int(os.getenv("DEBUG", "1"))
DEBUG_REJECTS = int(os.getenv("DEBUG_REJECTS", "0"))
DRY_RUN = int(os.getenv("DRY_RUN", "0"))
TEST_ONCE = int(os.getenv("TEST_ONCE", "0"))

USE_STORAGE = int(os.getenv("USE_STORAGE", "1"))
STORAGE_PATH = os.getenv("STORAGE_PATH", "/var/data/futures_state.json")
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", "21600"))  # 6 saat

HEARTBEAT_SEC = int(os.getenv("HEARTBEAT_SEC", "900"))

# ===== Optional filters =====
USE_HTF_FILTER = int(os.getenv("USE_HTF_FILTER", "0"))
HTF = os.getenv("HTF", "4h")
EMA_TREND = int(os.getenv("EMA_TREND", "123"))

USE_BTC_FILTER = int(os.getenv("USE_BTC_FILTER", "0"))
BTC_SYMBOL = os.getenv("BTC_SYMBOL", "BTCUSDT")
BTC_TF = os.getenv("BTC_TF", "4h")

USE_VOL_FILTER = int(os.getenv("USE_VOL_FILTER", "0"))
VOL_LEN = int(os.getenv("VOL_LEN", "20"))
VOL_MULT = float(os.getenv("VOL_MULT", "1.1"))
VOL_USE_QUOTE = int(os.getenv("VOL_USE_QUOTE", "1"))

USE_MFI_FILTER = int(os.getenv("USE_MFI_FILTER", "0"))
MFI_LEN = int(os.getenv("MFI_LEN", "14"))
MFI_LONG_MIN = float(os.getenv("MFI_LONG_MIN", "40"))
MFI_LONG_MAX = float(os.getenv("MFI_LONG_MAX", "85"))
MFI_SLOPE_ENABLE = int(os.getenv("MFI_SLOPE_ENABLE", "1"))
MFI_SLOPE_BARS = int(os.getenv("MFI_SLOPE_BARS", "1"))

# ===== Stoch RSI =====
USE_STOCH_RSI = int(os.getenv("USE_STOCH_RSI", "1"))
STOCH_RSI_LEN = int(os.getenv("STOCH_RSI_LEN", "14"))
STOCH_K = int(os.getenv("STOCH_K", "5"))
STOCH_D = int(os.getenv("STOCH_D", "5"))

# ===== WaveTrend =====
USE_WT = int(os.getenv("USE_WT", "1"))
WT_CH = int(os.getenv("WT_CH", "12"))
WT_AVG = int(os.getenv("WT_AVG", "12"))
WT_CH_LEN = int(os.getenv("WT_CH_LEN", "9"))      # EMA for wt1
WT_AVG_LEN = int(os.getenv("WT_AVG_LEN", "12"))   # SMA for wt2

WT_OS1 = float(os.getenv("WT_OS1", "-60"))
WT_OS2 = float(os.getenv("WT_OS2", "-53"))
WT_OB1 = float(os.getenv("WT_OB1", "60"))
WT_OB2 = float(os.getenv("WT_OB2", "53"))

# Signal modes
USE_WT_DIP = int(os.getenv("USE_WT_DIP", "1"))               # dip reversal
USE_WT_CONTINUATION = int(os.getenv("USE_WT_CONTINUATION", "1"))  # continuation

# Close alerts
USE_CLOSE_ALERT = int(os.getenv("USE_CLOSE_ALERT", "1"))     # exit signal göndermek
TP_PCT = float(os.getenv("TP_PCT", "8"))
SL_PCT = float(os.getenv("SL_PCT", "2"))

LONG_ONLY = int(os.getenv("LONG_ONLY", "1"))  # şimdilik sadece long

# =========================
# Helpers
# =========================
def log(msg: str) -> None:
    if DEBUG:
        print(msg, flush=True)

def reject(rejects: Dict[str, int], reason: str) -> None:
    rejects[reason] = rejects.get(reason, 0) + 1

def ts_now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S")

def http_get(path: str, params: dict) -> dict:
    url = f"{BINANCE_FAPI}{path}"
    r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()

def get_futures_symbols() -> List[str]:
    info = http_get("/fapi/v1/exchangeInfo", {})
    symbols = []
    for s in info.get("symbols", []):
        if s.get("status") != "TRADING":
            continue
        if ONLY_USDT_PERP:
            if s.get("quoteAsset") != "USDT":
                continue
            if s.get("contractType") != "PERPETUAL":
                continue
        symbols.append(s["symbol"])
    return symbols

def get_24h_tickers() -> List[dict]:
    return http_get("/fapi/v1/ticker/24hr", {})

def top_by_quote_volume(symbols: List[str], top_n: int) -> List[Tuple[str, float]]:
    tickers = get_24h_tickers()
    m = {t["symbol"]: float(t.get("quoteVolume", 0) or 0) for t in tickers if "symbol" in t}
    items = [(sym, m.get(sym, 0.0)) for sym in symbols]
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:top_n]

def get_klines(symbol: str, interval: str, limit: int) -> List[List]:
    return http_get("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})

def closes_from_klines(kl: List[List]) -> Tuple[List[float], List[float], List[float]]:
    # returns close, high, low
    close = [float(x[4]) for x in kl]
    high = [float(x[2]) for x in kl]
    low = [float(x[3]) for x in kl]
    return close, high, low

def volumes_from_klines(kl: List[List]) -> List[float]:
    return [float(x[5]) for x in kl]

def ema(series: List[float], length: int) -> List[float]:
    if length <= 1:
        return series[:]
    k = 2 / (length + 1)
    out = []
    e = series[0]
    for v in series:
        e = v * k + e * (1 - k)
        out.append(e)
    return out

def sma(series: List[float], length: int) -> List[float]:
    out = []
    s = 0.0
    for i, v in enumerate(series):
        s += v
        if i >= length:
            s -= series[i - length]
        if i + 1 < length:
            out.append(float("nan"))
        else:
            out.append(s / length)
    return out

def rsi(series: List[float], length: int) -> List[float]:
    if length <= 1:
        return [50.0] * len(series)
    gains = [0.0]
    losses = [0.0]
    for i in range(1, len(series)):
        ch = series[i] - series[i - 1]
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

def stoch_rsi(rsi_vals: List[float], length: int, k_len: int, d_len: int) -> Tuple[List[float], List[float]]:
    # Stoch RSI 0-100
    st = []
    for i in range(len(rsi_vals)):
        start = max(0, i - length + 1)
        window = rsi_vals[start:i + 1]
        lo = min(window)
        hi = max(window)
        if hi - lo == 0:
            st.append(0.0)
        else:
            st.append(100.0 * (rsi_vals[i] - lo) / (hi - lo))
    k_raw = sma(st, k_len)
    d = sma([0.0 if math.isnan(x) else x for x in k_raw], d_len)
    k = [0.0 if math.isnan(x) else x for x in k_raw]
    return k, d

def wavetrend(close: List[float], channel_len: int, average_len: int, wt1_len: int, wt2_len: int) -> Tuple[List[float], List[float]]:
    # Classic WT (approx):
    # esa = EMA(price, channel_len)
    # de  = EMA(|price-esa|, channel_len)
    # ci  = (price-esa)/(0.015*de)
    # wt1 = EMA(ci, average_len)
    # wt2 = SMA(wt1, wt2_len)
    price = close
    esa = ema(price, channel_len)
    abs_dev = [abs(p - e) for p, e in zip(price, esa)]
    de = ema(abs_dev, channel_len)
    ci = []
    for p, e, d in zip(price, esa, de):
        denom = 0.015 * d if d != 0 else 1e-9
        ci.append((p - e) / denom)

    wt1 = ema(ci, wt1_len)
    wt2 = sma(wt1, wt2_len)
    wt2 = [0.0 if math.isnan(x) else x for x in wt2]
    return wt1, wt2

def mfi(high: List[float], low: List[float], close: List[float], vol: List[float], length: int) -> List[float]:
    tp = [(h + l + c) / 3.0 for h, l, c in zip(high, low, close)]
    rmf = [tp[i] * vol[i] for i in range(len(tp))]
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
        ps = sum(pos[start:i + 1])
        ns = sum(neg[start:i + 1])
        if ns == 0:
            out.append(100.0)
        else:
            mr = ps / ns
            out.append(100.0 - (100.0 / (1.0 + mr)))
    return out

def ema_cross_within(ema_f: List[float], ema_s: List[float], lookback: int, last_only: bool) -> bool:
    # last_only: sadece son kapanan mumda kesişim (i = last_index)
    # Binance klines son eleman bazen kapanmamış olur; USE_LAST_CANDLE=1 ise -2 kullanıyoruz.
    idx_last = -2 if last_only else -1
    if len(ema_f) < 3:
        return False

    if last_only:
        i = len(ema_f) + idx_last
        if i <= 0:
            return False
        return ema_f[i - 1] <= ema_s[i - 1] and ema_f[i] > ema_s[i]

    # lookback içinde herhangi bir kesişim
    end = len(ema_f) - 1  # son eleman
    start = max(1, end - lookback + 1)
    for i in range(start, end + 1):
        if ema_f[i - 1] <= ema_s[i - 1] and ema_f[i] > ema_s[i]:
            return True
    return False

def last_index_for_signals() -> int:
    return -2 if USE_LAST_CANDLE else -1

# =========================
# Filters
# =========================
def btc_ok() -> bool:
    if not USE_BTC_FILTER:
        return True
    kl = get_klines(BTC_SYMBOL, BTC_TF, KLINE_LIMIT)
    c, _, _ = closes_from_klines(kl)
    e = ema(c, EMA_TREND)
    i = last_index_for_signals()
    return c[i] > e[i]

def htf_ok(symbol: str) -> bool:
    if not USE_HTF_FILTER:
        return True
    kl = get_klines(symbol, HTF, KLINE_LIMIT)
    c, _, _ = closes_from_klines(kl)
    e = ema(c, EMA_TREND)
    i = last_index_for_signals()
    return c[i] > e[i]

def vol_ok(symbol: str, kl: List[List]) -> bool:
    if not USE_VOL_FILTER:
        return True
    c, _, _ = closes_from_klines(kl)
    v = volumes_from_klines(kl)
    vol_series = []
    for i in range(len(v)):
        vol_series.append(v[i] * c[i] if VOL_USE_QUOTE else v[i])
    ma = sma(vol_series, VOL_LEN)
    i = last_index_for_signals()
    if math.isnan(ma[i]) or ma[i] == 0:
        return False
    return vol_series[i] >= ma[i] * VOL_MULT

def mfi_ok(kl: List[List]) -> bool:
    if not USE_MFI_FILTER:
        return True
    c, h, l = closes_from_klines(kl)
    v = volumes_from_klines(kl)
    m = mfi(h, l, c, v, MFI_LEN)
    i = last_index_for_signals()
    if not (MFI_LONG_MIN <= m[i] <= MFI_LONG_MAX):
        return False
    if MFI_SLOPE_ENABLE:
        b = max(1, MFI_SLOPE_BARS)
        j = i - b
        if j < 0:
            return False
        if m[i] <= m[j]:
            return False
    return True

# =========================
# Signal logic
# =========================
def entry_signal(symbol: str, kl: List[List]) -> Optional[Tuple[str, str]]:
    """
    Returns (kind, text) or None
    kind: "WT_DIP" / "WT_CONT"
    """
    c, h, l = closes_from_klines(kl)

    # indicators
    ef = ema(c, EMA_FAST)
    es = ema(c, EMA_SLOW)
    r = rsi(c, RSI_LEN)

    k = d = None
    if USE_STOCH_RSI:
        k, d = stoch_rsi(r, STOCH_RSI_LEN, STOCH_K, STOCH_D)

    wt1 = wt2 = None
    if USE_WT:
        wt1, wt2 = wavetrend(c, WT_CH, WT_AVG, WT_CH_LEN, WT_AVG_LEN)

    i = last_index_for_signals()
    ip = i - 1

    # base filters
    if r[i] < RSI_MIN:
        return None

    # EMA cross condition: kesin TV setup yaklaşımı
    crossed = ema_cross_within(ef, es, LOOKBACK, last_only=bool(USE_LAST_CANDLE))
    if not crossed:
        return None

    # Optional filters
    if not vol_ok(symbol, kl):
        return None
    if not mfi_ok(kl):
        return None
    if not htf_ok(symbol):
        return None

    # WT + Stoch conditions
    stoch_ok = True
    if USE_STOCH_RSI and k is not None and d is not None:
        # Dip için: K düşük ve yukarı dönüyor (K > D)
        # Continuation için: K>20 ve momentum var (K > D)
        stoch_ok = True  # aşağıda moda göre daha sıkı yapacağız

    # --- WT DIP (dip reversal) ---
    if USE_WT and USE_WT_DIP and wt1 is not None and wt2 is not None:
        # deep oversold bölgesi + wt1 cross up wt2 + wt1 yükseliyor
        dip_zone = (wt1[i] <= WT_OS2) or (wt1[ip] <= WT_OS2) or (wt1[i] <= WT_OS1)
        cross_up = wt1[ip] < wt2[ip] and wt1[i] > wt2[i]
        rising = wt1[i] > wt1[ip]

        dip_stoch = True
        if USE_STOCH_RSI and k is not None and d is not None:
            dip_stoch = (k[i] < 25.0) and (k[i] > d[i])  # dipte dönüş

        if dip_zone and cross_up and rising and dip_stoch:
            return "WT_DIP", format_signal(symbol, c[i], ef[i], es[i], r[i], k[i] if k else None, d[i] if d else None, wt1[i], wt2[i])

    # --- WT CONTINUATION (trend continuation) ---
    if USE_WT and USE_WT_CONTINUATION and wt1 is not None and wt2 is not None:
        # wt1 > wt2 ve yükseliyor; oversold değil (0 altına çok gömülmesin)
        cont_ok = (wt1[i] > wt2[i]) and (wt1[i] > wt1[ip]) and (wt1[i] > -5.0)
        cont_stoch = True
        if USE_STOCH_RSI and k is not None and d is not None:
            cont_stoch = (k[i] > 20.0) and (k[i] > d[i])

        if cont_ok and cont_stoch:
            return "WT_CONT", format_signal(symbol, c[i], ef[i], es[i], r[i], k[i] if k else None, d[i] if d else None, wt1[i], wt2[i])

    return None

def exit_signal(symbol: str, kl: List[List]) -> Optional[str]:
    if not (USE_WT and USE_CLOSE_ALERT):
        return None
    c, _, _ = closes_from_klines(kl)
    wt1, wt2 = wavetrend(c, WT_CH, WT_AVG, WT_CH_LEN, WT_AVG_LEN)
    i = last_index_for_signals()
    ip = i - 1

    # exit: WT1 crosses DOWN WT2 while WT1 > WT_OB2
    if wt1[ip] > wt2[ip] and wt1[i] < wt2[i] and wt1[ip] > WT_OB2:
        return format_close(symbol, c[i], wt1[i], wt2[i])
    return None

def format_signal(symbol: str, price: float, ema3: float, ema44: float, rsi_v: float,
                  k: Optional[float], d: Optional[float], wt1: float, wt2: float) -> str:
    stoch_line = ""
    if k is not None and d is not None:
        stoch_line = f"\nStochRSI K/D (K={STOCH_K},D={STOCH_D}): {k:.2f}/{d:.2f}"

    msg = (
        f"🚀 <b>LONG SIGNAL</b>\n"
        f"Symbol: <b>{symbol}</b>\n"
        f"TF: {TF_ENTRY}"
        f"\nPrice: {price:.6f}\n\n"
        f"EMA{EMA_FAST}: {ema3:.6f} | EMA{EMA_SLOW}: {ema44:.6f}\n"
        f"RSI({RSI_LEN}): {rsi_v:.2f}"
        f"{stoch_line}\n"
        f"WT (ch={WT_CH},avg={WT_AVG}) WT1/WT2: {wt1:.2f}/{wt2:.2f}\n\n"
        f"<b>Exit plan (manual)</b>\n"
        f"- TP1: +{TP_PCT:.1f}% (suggestion)\n"
        f"- SL: -{SL_PCT:.1f}% (suggestion)\n"
        f"- WT exit: if WT1 crosses DOWN WT2 while WT1>{WT_OB2:.0f} consider close/trim\n"
        f"- WT warning: if WT1>{WT_OB1:.0f} and turns down -> tighten stop"
    )
    return msg

def format_close(symbol: str, price: float, wt1: float, wt2: float) -> str:
    return (
        f"🧯 <b>CLOSE SIGNAL</b>\n"
        f"Symbol: <b>{symbol}</b>\n"
        f"TF: {TF_ENTRY}\n"
        f"Price: {price:.6f}\n\n"
        f"WT1/WT2: {wt1:.2f}/{wt2:.2f}\n"
        f"Reason: WT1 crossed DOWN WT2 while WT1>{WT_OB2:.0f}"
    )

def main() -> None:
    storage = Storage(STORAGE_PATH) if USE_STORAGE else None

    log(f"{ts_now()} [BOOT] Futures scanner started")
    log(f"{ts_now()} [CFG] TF_ENTRY={TF_ENTRY} EMA={EMA_FAST}/{EMA_SLOW} LOOKBACK={LOOKBACK} RSI_LEN={RSI_LEN} RSI_MIN={RSI_MIN} WT={USE_WT} DIP={USE_WT_DIP} CONT={USE_WT_CONTINUATION} STOCH_RSI={USE_STOCH_RSI}")
    log(f"{ts_now()} [CFG] TOP_N={TOP_N} MIN_QUOTE_VOLUME={MIN_QUOTE_VOLUME} COOLDOWN_SEC={COOLDOWN_SEC} DRY_RUN={DRY_RUN} USE_LAST_CANDLE={USE_LAST_CANDLE}")
    log(f"{ts_now()} [CFG] STORAGE_PATH={STORAGE_PATH}")

    last_heartbeat = 0

    # preload symbols list
    all_symbols = get_futures_symbols()
    top_symbols = top_by_quote_volume(all_symbols, TOP_N)

    while True:
        now = int(time.time())

        # heartbeat
        if now - last_heartbeat >= HEARTBEAT_SEC:
            hb = f"✅ worker alive | TF={TF_ENTRY} TOP_N={TOP_N} DIP={USE_WT_DIP} CONT={USE_WT_CONTINUATION}"
            send_telegram(TG_BOT_TOKEN, TG_CHAT_ID, hb, dry_run=bool(DRY_RUN), timeout=HTTP_TIMEOUT)
            last_heartbeat = now

        # BTC filter once per cycle (cheap)
        if not btc_ok():
            if DEBUG_REJECTS:
                log(f"{ts_now()} [BTC] filter reject")
            time.sleep(INTERVAL_SEC)
            if TEST_ONCE:
                break
            continue

        rejects: Dict[str, int] = {}

        # open positions exit scan
        if storage and USE_CLOSE_ALERT:
            open_map = storage.get_open()
            for sym in list(open_map.keys()):
                try:
                    kl = get_klines(sym, TF_ENTRY, KLINE_LIMIT)
                    ex = exit_signal(sym, kl)
                    if ex:
                        key = f"{sym}:CLOSE"
                        if not storage.is_cooldown(key, int(COOLDOWN_SEC / 3)):
                            send_telegram(TG_BOT_TOKEN, TG_CHAT_ID, ex, dry_run=bool(DRY_RUN), timeout=HTTP_TIMEOUT)
                            storage.mark_sent(key)
                        storage.clear_open(sym)
                except Exception as e:
                    log(f"{ts_now()} [EXIT_ERR] {sym}: {e}")

        # entry scan
        for sym, qv in top_symbols:
            try:
                if qv < MIN_QUOTE_VOLUME:
                    reject(rejects, "LOW_QUOTE_VOL")
                    continue

                kl = get_klines(sym, TF_ENTRY, KLINE_LIMIT)

                sig = entry_signal(sym, kl)
                if not sig:
                    reject(rejects, "NO_SIGNAL")
                    continue

                kind, msg = sig
                # başlığa sinyal tipi ekleyelim
                msg = msg.replace("🚀 <b>LONG SIGNAL</b>", f"🚀 <b>LONG SIGNAL ({kind})</b>")

                # cooldown key
                key = f"{sym}:{kind}"
                if storage and storage.is_cooldown(key, COOLDOWN_SEC):
                    reject(rejects, "COOLDOWN")
                    continue

                send_telegram(TG_BOT_TOKEN, TG_CHAT_ID, msg, dry_run=bool(DRY_RUN), timeout=HTTP_TIMEOUT)

                if storage:
                    storage.mark_sent(key)
                    storage.set_open(sym, kind)

            except Exception as e:
                log(f"{ts_now()} [ERR] {sym}: {e}")

        if DEBUG_REJECTS and rejects:
            log(f"{ts_now()} [REJECTS] {rejects}")

        if TEST_ONCE:
            break

        time.sleep(INTERVAL_SEC)

if __name__ == "__main__":
    main()
