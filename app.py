# app.py
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

TF = os.getenv("TF", "30m")
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "60"))
KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "200"))

# Trend Indicator A (v2.3) - basit yaklaşım:
# MA Type + Length ile trend belirleme (close > MA => bullish)
TREND_MA_TYPE = os.getenv("TREND_MA_TYPE", "EMA").upper()  # EMA/SMA
TREND_MA_LEN = int(os.getenv("TREND_MA_LEN", "9"))
TREND_CONFIRM_CLOSE = int(os.getenv("TREND_CONFIRM_CLOSE", "1"))  # candle close ile teyit

# RSI
RSI_LEN = int(os.getenv("RSI_LEN", "14"))
RSI_MIN = float(os.getenv("RSI_MIN", "42"))  # RSI > 42
RSI_MA_LEN = int(os.getenv("RSI_MA_LEN", "14"))  # RSI sarı çizgi için EMA
RSI_CROSS_CONFIRM = int(os.getenv("RSI_CROSS_CONFIRM", "1"))  # RSI mor, sarıyı aşağıdan yukarı kessin

# MACD
MACD_FAST = int(os.getenv("MACD_FAST", "12"))
MACD_SLOW = int(os.getenv("MACD_SLOW", "26"))
MACD_SIGNAL = int(os.getenv("MACD_SIGNAL", "9"))
MACD_CROSS_CONFIRM = int(os.getenv("MACD_CROSS_CONFIRM", "1"))
MACD_ZERO_FILTER = int(os.getenv("MACD_ZERO_FILTER", "0"))  # 0/1
MACD_ZERO_EPS = float(os.getenv("MACD_ZERO_EPS", "0.0000"))  # 0’a yakın tolerans

# ATR / TP / SL
ATR_LEN = int(os.getenv("ATR_LEN", "14"))
ATR_SL_MULT = float(os.getenv("ATR_SL_MULT", "1.5"))
TP1_ATR_MULT = float(os.getenv("TP1_ATR_MULT", "1.0"))
TP2_ATR_MULT = float(os.getenv("TP2_ATR_MULT", "2.0"))

# Parabolic SAR (opsiyonel)
USE_PARABOLIC_SAR = int(os.getenv("USE_PARABOLIC_SAR", "0"))  # 0/1
SAR_STEP = float(os.getenv("SAR_STEP", "0.02"))
SAR_MAX = float(os.getenv("SAR_MAX", "0.2"))

# Heikin Ashi (opsiyonel)
USE_HEIKIN_ASHI = int(os.getenv("USE_HEIKIN_ASHI", "0"))  # 0/1

# Filtreler
SCAN_ALL_USDT_PERPS = int(os.getenv("SCAN_ALL_USDT_PERPS", "1"))  # 0/1
ONLY_USDT_PERP = int(os.getenv("ONLY_USDT_PERP", "1"))  # 0/1 (genelde 1 kalsın)

USE_24H_VOLUME_FILTER = int(os.getenv("USE_24H_VOLUME_FILTER", "0"))  # 0/1 (default pasif)
MIN_QUOTE_VOLUME_24H = float(os.getenv("MIN_QUOTE_VOLUME_24H", "3000000"))  # 24h quote volume eşiği

USE_BTC_FILTER = int(os.getenv("USE_BTC_FILTER", "0"))  # 0/1 (default pasif)
BTC_SYMBOL = os.getenv("BTC_SYMBOL", "BTCUSDT")
BTC_TF = os.getenv("BTC_TF", TF)
BTC_RSI_MIN = float(os.getenv("BTC_RSI_MIN", "42"))  # BTC RSI < 42 ise longları kapat (riskli dönem)

# Spam / tekrar kontrol
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", "1800"))  # aynı coin sinyal tekrar süresi
CLOSE_COOLDOWN_SEC = int(os.getenv("CLOSE_COOLDOWN_SEC", "0"))  # ekstra (istersen kullan)

DEBUG = int(os.getenv("DEBUG", "1"))
DEBUG_REJECTS = int(os.getenv("DEBUG_REJECTS", "0"))
DRY_RUN = int(os.getenv("DRY_RUN", "0"))

STORAGE_PATH = os.getenv("STORAGE_PATH", "/tmp/futures_scanner_storage.json")

# Telegram ENV alias (kritik fix)
TG_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TG_BOT_TOKEN")
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("TG_CHAT_ID")

# Eğer telegram yoksa servis çökmesin:
if not TG_BOT_TOKEN or not TG_CHAT_ID:
    if DRY_RUN == 0:
        DRY_RUN = 1
    print("WARN: TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID (veya TG_*) eksik. DRY_RUN aktif edildi (servis düşmez).")


# =========================
# HTTP yardımcı
# =========================
_session = requests.Session()
_session.headers.update({"User-Agent": "futures-scanner/1.0"})


def _get_json(url: str, params: Optional[dict] = None, timeout: int = 15):
    r = _session.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def get_exchange_symbols() -> List[str]:
    """
    Binance Futures USDT-P perpetual trading sembolleri
    """
    data = _get_json(f"{BINANCE_FAPI}/fapi/v1/exchangeInfo")
    symbols = []
    for s in data.get("symbols", []):
        if s.get("status") != "TRADING":
            continue
        if ONLY_USDT_PERP and s.get("quoteAsset") != "USDT":
            continue
        if s.get("contractType") != "PERPETUAL":
            continue
        symbols.append(s["symbol"])
    return symbols


def get_24h_quote_volumes() -> Dict[str, float]:
    """
    /ticker/24hr endpointinden quoteVolume alır.
    """
    data = _get_json(f"{BINANCE_FAPI}/fapi/v1/ticker/24hr")
    out = {}
    for it in data:
        sym = it.get("symbol")
        qv = it.get("quoteVolume")
        if sym and qv is not None:
            try:
                out[sym] = float(qv)
            except Exception:
                pass
    return out


def get_klines(symbol: str, interval: str, limit: int) -> List[List]:
    return _get_json(
        f"{BINANCE_FAPI}/fapi/v1/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=20,
    )


# =========================
# İndikatör hesapları
# =========================
def sma(values: List[float], length: int) -> List[Optional[float]]:
    out = [None] * len(values)
    if length <= 0:
        return out
    s = 0.0
    for i, v in enumerate(values):
        s += v
        if i >= length:
            s -= values[i - length]
        if i >= length - 1:
            out[i] = s / length
    return out


def ema(values: List[float], length: int) -> List[Optional[float]]:
    out = [None] * len(values)
    if length <= 0:
        return out
    k = 2 / (length + 1)
    ema_prev = None
    for i, v in enumerate(values):
        if ema_prev is None:
            ema_prev = v
        else:
            ema_prev = v * k + ema_prev * (1 - k)
        out[i] = ema_prev
    return out


def rsi(values: List[float], length: int) -> List[Optional[float]]:
    out = [None] * len(values)
    if length <= 0 or len(values) < 2:
        return out
    gains = 0.0
    losses = 0.0
    for i in range(1, len(values)):
        ch = values[i] - values[i - 1]
        g = max(ch, 0.0)
        l = max(-ch, 0.0)
        if i <= length:
            gains += g
            losses += l
            if i == length:
                avg_g = gains / length
                avg_l = losses / length
                rs = (avg_g / avg_l) if avg_l != 0 else 999999
                out[i] = 100 - (100 / (1 + rs))
        else:
            prev = out[i - 1]
            # Wilder smoothing
            avg_g = (avg_g * (length - 1) + g) / length
            avg_l = (avg_l * (length - 1) + l) / length
            rs = (avg_g / avg_l) if avg_l != 0 else 999999
            out[i] = 100 - (100 / (1 + rs))
    return out


def macd(values: List[float], fast: int, slow: int, signal: int):
    ema_fast = ema(values, fast)
    ema_slow = ema(values, slow)
    macd_line = [None] * len(values)
    for i in range(len(values)):
        if ema_fast[i] is None or ema_slow[i] is None:
            macd_line[i] = None
        else:
            macd_line[i] = ema_fast[i] - ema_slow[i]
    # signal EMA over macd_line with None handling
    macd_vals = [v if v is not None else 0.0 for v in macd_line]
    sig = ema(macd_vals, signal)
    return macd_line, sig


def atr(high: List[float], low: List[float], close: List[float], length: int) -> List[Optional[float]]:
    out = [None] * len(close)
    if len(close) < 2 or length <= 0:
        return out
    trs = []
    for i in range(len(close)):
        if i == 0:
            trs.append(high[i] - low[i])
        else:
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )
            trs.append(tr)
    # Wilder smoothing ATR
    atr_prev = None
    for i in range(len(trs)):
        if i < length:
            # ilk length içinde ortalama
            if i == length - 1:
                atr_prev = sum(trs[:length]) / length
                out[i] = atr_prev
        else:
            atr_prev = (atr_prev * (length - 1) + trs[i]) / length
            out[i] = atr_prev
    return out


def heikin_ashi(ohlc: List[Tuple[float, float, float, float]]) -> List[Tuple[float, float, float, float]]:
    """
    OHLC -> Heikin Ashi OHLC
    """
    ha = []
    ha_open = None
    ha_close = None
    for i, (o, h, l, c) in enumerate(ohlc):
        hc = (o + h + l + c) / 4.0
        if i == 0:
            ho = (o + c) / 2.0
        else:
            ho = (ha_open + ha_close) / 2.0
        hh = max(h, ho, hc)
        hl = min(l, ho, hc)
        ha_open, ha_close = ho, hc
        ha.append((ho, hh, hl, hc))
    return ha


def parabolic_sar(high: List[float], low: List[float], step: float = 0.02, max_af: float = 0.2) -> List[Optional[float]]:
    """
    Basit Parabolic SAR implementasyonu.
    """
    n = len(high)
    out = [None] * n
    if n < 3:
        return out

    # initial trend: use first two closes-like approximation
    uptrend = True
    ep = high[0]
    sar = low[0]
    af = step

    for i in range(1, n):
        sar = sar + af * (ep - sar)

        # clamp sar
        if uptrend:
            sar = min(sar, low[i - 1], low[i] if i >= 2 else low[i - 1])
        else:
            sar = max(sar, high[i - 1], high[i] if i >= 2 else high[i - 1])

        # switch?
        if uptrend:
            if low[i] < sar:
                uptrend = False
                sar = ep
                ep = low[i]
                af = step
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(max_af, af + step)
        else:
            if high[i] > sar:
                uptrend = True
                sar = ep
                ep = high[i]
                af = step
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(max_af, af + step)

        out[i] = sar
    return out


def crossed_up(prev_a: float, prev_b: float, cur_a: float, cur_b: float) -> bool:
    return prev_a <= prev_b and cur_a > cur_b


# =========================
# Sinyal mantığı
# =========================
def compute_signal(symbol: str, klines: List[List]) -> Optional[dict]:
    """
    Long sinyali üretir. Sinyal yoksa None.
    """
    o = [float(k[1]) for k in klines]
    h = [float(k[2]) for k in klines]
    l = [float(k[3]) for k in klines]
    c = [float(k[4]) for k in klines]
    t_close = [int(k[6]) for k in klines]  # close time ms

    # Heikin Ashi opsiyonel
    if USE_HEIKIN_ASHI:
        ha = heikin_ashi(list(zip(o, h, l, c)))
        o = [x[0] for x in ha]
        h = [x[1] for x in ha]
        l = [x[2] for x in ha]
        c = [x[3] for x in ha]

    # candle close teyidi (son mum kapanmış mı?)
    # Binance klines: son candle kapanış zamanı gelecekte olabilir -> kapanmadı
    now_ms = int(time.time() * 1000)
    last_closed_index = len(c) - 1
    if TREND_CONFIRM_CLOSE:
        if t_close[-1] > now_ms:
            last_closed_index = len(c) - 2
    i = last_closed_index
    if i < 5:
        return None

    # Trend MA
    if TREND_MA_TYPE == "SMA":
        ma = sma(c, TREND_MA_LEN)
    else:
        ma = ema(c, TREND_MA_LEN)

    # trend r->g: önceki bar bearish (close < ma), şimdi bullish (close > ma)
    if ma[i] is None or ma[i - 1] is None:
        return None
    prev_trend_bull = c[i - 1] > ma[i - 1]
    cur_trend_bull = c[i] > ma[i]
    trend_flip = (not prev_trend_bull) and cur_trend_bull
    if not trend_flip:
        if DEBUG_REJECTS:
            print(symbol, "reject: trend not flip")
        return None

    # RSI ve RSI-MA
    r = rsi(c, RSI_LEN)
    if r[i] is None or r[i - 1] is None:
        return None
    r_ma = ema([x if x is not None else 0.0 for x in r], RSI_MA_LEN)

    if r[i] <= RSI_MIN:
        if DEBUG_REJECTS:
            print(symbol, "reject: rsi <= min", r[i])
        return None

    if RSI_CROSS_CONFIRM:
        if r_ma[i] is None or r_ma[i - 1] is None:
            return None
        if not crossed_up(r[i - 1], r_ma[i - 1], r[i], r_ma[i]):
            if DEBUG_REJECTS:
                print(symbol, "reject: rsi cross not ok")
            return None

    # MACD
    macd_line, sig = macd(c, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    if macd_line[i] is None or macd_line[i - 1] is None or sig[i] is None or sig[i - 1] is None:
        return None

    if MACD_CROSS_CONFIRM:
        if not crossed_up(macd_line[i - 1], sig[i - 1], macd_line[i], sig[i]):
            if DEBUG_REJECTS:
                print(symbol, "reject: macd cross not ok")
            return None

    if MACD_ZERO_FILTER:
        # 0'a doğru/üstü: macd_line >= -eps
        if macd_line[i] < -abs(MACD_ZERO_EPS):
            if DEBUG_REJECTS:
                print(symbol, "reject: macd below zero filter", macd_line[i])
            return None

    # ATR / SL / TP
    a = atr(h, l, c, ATR_LEN)
    if a[i] is None:
        return None
    entry = c[i]
    sl = entry - a[i] * ATR_SL_MULT
    tp1 = entry + a[i] * TP1_ATR_MULT
    tp2 = entry + a[i] * TP2_ATR_MULT

    # SAR opsiyonel
    sar_val = None
    if USE_PARABOLIC_SAR:
        sar_series = parabolic_sar(h, l, SAR_STEP, SAR_MAX)
        sar_val = sar_series[i]

    # zaman
    ts = datetime.fromtimestamp(t_close[i] / 1000, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "symbol": symbol,
        "tf": TF,
        "time": ts,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "atr": a[i],
        "sar": sar_val,
        "rsi": r[i],
        "rsi_ma": r_ma[i] if r_ma[i] is not None else None,
        "macd": macd_line[i],
        "macd_sig": sig[i],
    }


def passes_btc_filter() -> bool:
    """
    USE_BTC_FILTER=1 ise: BTC RSI < BTC_RSI_MIN olduğunda long sinyalleri kapat
    """
    if not USE_BTC_FILTER:
        return True
    try:
        kl = get_klines(BTC_SYMBOL, BTC_TF, KLINE_LIMIT)
        c = [float(k[4]) for k in kl]
        r = rsi(c, RSI_LEN)
        # son kapanmış bar
        i = len(c) - 2
        if r[i] is None:
            return True
        # BTC RSI düşükse riskli -> sinyal üretme
        if r[i] < BTC_RSI_MIN:
            if DEBUG:
                print(f"BTC FILTER: BTC RSI {r[i]:.2f} < {BTC_RSI_MIN} -> BLOCK")
            return False
        return True
    except Exception as e:
        print("WARN: BTC filter check failed:", e)
        return True


def format_message(sig: dict) -> str:
    lines = []
    lines.append(f"🚀 LONG SİNYAL | {sig['symbol']} | TF {sig['tf']}")
    lines.append(f"⏱️ Zaman: {sig['time']}")
    lines.append("")
    lines.append(f"Entry: {sig['entry']:.8f}")
    lines.append(f"SL (ATR): {sig['sl']:.8f}  | ATR: {sig['atr']:.8f}  (x{ATR_SL_MULT})")
    lines.append(f"TP1 (ATR): {sig['tp1']:.8f} (x{TP1_ATR_MULT})")
    lines.append(f"TP2 (ATR): {sig['tp2']:.8f} (x{TP2_ATR_MULT})")
    if sig.get("sar") is not None:
        lines.append(f"SAR: {sig['sar']:.8f}")
    lines.append("")
    lines.append(f"RSI: {sig['rsi']:.2f} | RSI-MA: {sig.get('rsi_ma', 0):.2f}")
    lines.append(f"MACD: {sig['macd']:.8f} | SIG: {sig['macd_sig']:.8f}")
    return "\n".join(lines)


def main():
    store = Storage(STORAGE_PATH)
    last_sent = store.get("last_sent", {})  # symbol -> ts

    if SCAN_ALL_USDT_PERPS:
        symbols = get_exchange_symbols()
    else:
        # istersen manuel liste
        symbols = [os.getenv("SYMBOL", "BTCUSDT")]

    volumes_24h = {}
    if USE_24H_VOLUME_FILTER:
        volumes_24h = get_24h_quote_volumes()

    if DEBUG:
        print(f"Scanner started | TF={TF} symbols={len(symbols)} DRY_RUN={DRY_RUN}")

    while True:
        try:
            # BTC filtresi
            if not passes_btc_filter():
                time.sleep(INTERVAL_SEC)
                continue

            # 24h volume filtresi için güncelle (arada bir)
            if USE_24H_VOLUME_FILTER:
                # her turda çekmek ağır olabilir, ama starter için idare
                volumes_24h = get_24h_quote_volumes()

            now = int(time.time())
            found = 0

            for sym in symbols:
                # volume filtresi
                if USE_24H_VOLUME_FILTER:
                    qv = volumes_24h.get(sym, 0.0)
                    if qv < MIN_QUOTE_VOLUME_24H:
                        continue

                # cooldown
                last = int(last_sent.get(sym, 0))
                if COOLDOWN_SEC > 0 and now - last < COOLDOWN_SEC:
                    continue

                try:
                    kl = get_klines(sym, TF, KLINE_LIMIT)
                    sig = compute_signal(sym, kl)
                    if not sig:
                        continue

                    msg = format_message(sig)
                    found += 1

                    if DRY_RUN:
                        print("\n" + msg + "\n")
                    else:
                        send_telegram(TG_BOT_TOKEN, TG_CHAT_ID, msg)

                    last_sent[sym] = now
                    store.set("last_sent", last_sent)
                    store.save()

                except Exception as e:
                    if DEBUG:
                        print(f"WARN: {sym} failed:", e)

            if DEBUG:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] scan done | found={found}")

        except Exception as e:
            print("ERROR main loop:", e)

        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main()
