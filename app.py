# app.py
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests

from notify import send_telegram
from storage import Storage

BINANCE_FAPI = os.getenv("BINANCE_FAPI", "https://fapi.binance.com").rstrip("/")

# ---- Mod / Zaman ----
TF = os.getenv("TF", "30m")  # Ana çalışma TF (ATR ve entry bu TF'den)
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "60"))  # worker döngü sıklığı
KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "200"))

DIRECTION = os.getenv("DIRECTION", "LONG").upper()  # LONG / SHORT (şimdilik LONG)

# ---- RSI Multi Timeframe eşikleri (kilit) ----
RSI_LEN = int(os.getenv("RSI_LEN", "14"))

RSI_1M_MAX = float(os.getenv("RSI_1M_MAX", "10"))
RSI_1W_MAX = float(os.getenv("RSI_1W_MAX", "20"))
RSI_1D_MAX = float(os.getenv("RSI_1D_MAX", "30"))
RSI_4H_MAX = float(os.getenv("RSI_4H_MAX", "30"))
RSI_1H_MAX = float(os.getenv("RSI_1H_MAX", "30"))

# RSI'si "-" (None) olan coinleri ne yapalım?
# 0: dahil et (None => efektif 0 say)
# 1: None olanı ele (fake artarsa aç)
REQUIRE_RSI_VALUES = int(os.getenv("REQUIRE_RSI_VALUES", "0"))

# ---- Mum kapanışı ----
TREND_CONFIRM_CLOSE = int(os.getenv("TREND_CONFIRM_CLOSE", "1"))  # 1: sadece kapanmış mumla çalış

# ---- ATR TP/SL ----
ATR_LEN = int(os.getenv("ATR_LEN", "14"))
ATR_SL_MULT = float(os.getenv("ATR_SL_MULT", "1.0"))
TP1_ATR_MULT = float(os.getenv("TP1_ATR_MULT", "1.0"))
TP2_ATR_MULT = float(os.getenv("TP2_ATR_MULT", "2.0"))

# ---- Filtreler (kapalı default) ----
USE_24H_VOLUME_FILTER = int(os.getenv("USE_24H_VOLUME_FILTER", "0"))
MIN_QUOTE_VOLUME_24H = float(os.getenv("MIN_QUOTE_VOLUME_24H", "0"))

USE_BTC_FILTER = int(os.getenv("USE_BTC_FILTER", "0"))
BTC_SYMBOL = os.getenv("BTC_SYMBOL", "BTCUSDT")
BTC_TF = os.getenv("BTC_TF", "1h")
BTC_RSI_MIN = float(os.getenv("BTC_RSI_MIN", "42"))

# ---- Cooldown ----
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", "3600"))

# ---- Telegram ENV (iki isim de destek) ----
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
TG_CHAT_ID = os.getenv("TG_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID")

# ---- Storage ----
STORAGE_PATH = os.getenv("STORAGE_PATH", "/tmp/futures_scanner_storage.json")


def now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def get_exchange_info() -> dict:
    url = f"{BINANCE_FAPI}/fapi/v1/exchangeInfo"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


def get_all_usdt_perp_symbols() -> List[str]:
    info = get_exchange_info()
    out = []
    for s in info.get("symbols", []):
        try:
            if s.get("contractType") != "PERPETUAL":
                continue
            if s.get("quoteAsset") != "USDT":
                continue
            if s.get("status") != "TRADING":
                continue
            out.append(s["symbol"])
        except Exception:
            continue
    return out


def get_24h_quote_volume_map() -> Dict[str, float]:
    url = f"{BINANCE_FAPI}/fapi/v1/ticker/24hr"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()
    vol = {}
    for row in data:
        sym = row.get("symbol")
        if not sym:
            continue
        try:
            vol[sym] = float(row.get("quoteVolume", 0.0))
        except Exception:
            vol[sym] = 0.0
    return vol


def get_klines(symbol: str, interval: str, limit: int) -> List[List]:
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def closes_from_klines(kl: List[List]) -> List[float]:
    # kline close index = 4
    return [float(x[4]) for x in kl]


def rsi(values: List[float], length: int) -> Optional[float]:
    # klasik RSI (Wilder smoothing basit versiyon)
    if len(values) < length + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(1, length + 1):
        ch = values[i] - values[i - 1]
        if ch >= 0:
            gains += ch
        else:
            losses -= ch
    avg_gain = gains / length
    avg_loss = losses / length
    for i in range(length + 1, len(values)):
        ch = values[i] - values[i - 1]
        gain = ch if ch > 0 else 0.0
        loss = (-ch) if ch < 0 else 0.0
        avg_gain = (avg_gain * (length - 1) + gain) / length
        avg_loss = (avg_loss * (length - 1) + loss) / length
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def atr(highs: List[float], lows: List[float], closes: List[float], length: int) -> Optional[float]:
    if len(closes) < length + 1 or len(highs) != len(lows) or len(lows) != len(closes):
        return None
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    if len(trs) < length:
        return None
    # Wilder smoothing
    a = sum(trs[:length]) / length
    for i in range(length, len(trs)):
        a = (a * (length - 1) + trs[i]) / length
    return a


def highs_lows_closes(kl: List[List]) -> Tuple[List[float], List[float], List[float]]:
    highs = [float(x[2]) for x in kl]
    lows = [float(x[3]) for x in kl]
    closes = [float(x[4]) for x in kl]
    return highs, lows, closes


def last_closed_index(kl: List[List]) -> int:
    # Binance kline[6] close time (ms)
    # son kline bazen hala açık olabilir; TREND_CONFIRM_CLOSE=1 ise sondan bir önceyi kullanırız
    if len(kl) < 2:
        return len(kl) - 1
    return -2 if TREND_CONFIRM_CLOSE == 1 else -1


def passes_rsi_gate(v: Optional[float], maxv: float) -> bool:
    if v is None:
        return REQUIRE_RSI_VALUES == 0  # 0 ise geçsin, 1 ise elensin
    return v <= maxv


def effective_rsi(v: Optional[float]) -> float:
    # None ise 0 say (REQUIRE_RSI_VALUES=0 modunda)
    return 0.0 if v is None else float(v)


def fmt_rsi(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.2f}"


def check_btc_filter() -> bool:
    if USE_BTC_FILTER != 1:
        return True
    try:
        kl = get_klines(BTC_SYMBOL, BTC_TF, KLINE_LIMIT)
        closes = closes_from_klines(kl)
        val = rsi(closes, RSI_LEN)
        if val is None:
            # BTC RSI yoksa güvenli tarafta kal: sinyal basma
            print(f"[{now_utc_str()}] BTC filter: RSI None -> block")
            return False
        ok = val >= BTC_RSI_MIN
        print(f"[{now_utc_str()}] BTC RSI({BTC_TF})={val:.2f} >= {BTC_RSI_MIN} ? {ok}")
        return ok
    except Exception as e:
        print(f"[{now_utc_str()}] BTC filter error: {e}")
        return False


def build_signal_message(symbol: str,
                         entry: float,
                         sl: float,
                         tp1: float,
                         tp2: float,
                         rsi_map: Dict[str, Optional[float]],
                         qvol24: Optional[float]) -> str:
    lines = []
    lines.append(f"📌 {DIRECTION} Sinyal (RSI MultiTF) — {symbol}")
    lines.append(f"TF: {TF} | CandleClose: {TREND_CONFIRM_CLOSE}")
    lines.append("")
    lines.append(f"Entry: {entry:.6g}")
    lines.append(f"SL:    {sl:.6g}")
    lines.append(f"TP1:   {tp1:.6g}")
    lines.append(f"TP2:   {tp2:.6g}")
    lines.append("")
    lines.append("RSI(14): " +
                 f"1M {fmt_rsi(rsi_map['1M'])} | "
                 f"1W {fmt_rsi(rsi_map['1W'])} | "
                 f"1D {fmt_rsi(rsi_map['1D'])} | "
                 f"4H {fmt_rsi(rsi_map['4H'])} | "
                 f"1H {fmt_rsi(rsi_map['1H'])}")
    if qvol24 is not None:
        lines.append(f"24h QuoteVol: {qvol24:,.0f}")
    lines.append(f"Time: {now_utc_str()}")
    return "\n".join(lines)


def main_loop() -> None:
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        raise SystemExit("ERROR: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID eksik.")

    storage = Storage(STORAGE_PATH)

    symbols = get_all_usdt_perp_symbols()
    print(f"[{now_utc_str()}] Loaded {len(symbols)} USDT-PERP symbols.")

    vol_map = {}
    if USE_24H_VOLUME_FILTER == 1:
        vol_map = get_24h_quote_volume_map()
        print(f"[{now_utc_str()}] 24h volume map loaded.")

    while True:
        try:
            if not check_btc_filter():
                time.sleep(INTERVAL_SEC)
                continue

            found = 0

            # vol map’i ara ara yenile
            if USE_24H_VOLUME_FILTER == 1 and (int(time.time()) % 600 < INTERVAL_SEC):
                vol_map = get_24h_quote_volume_map()

            for sym in symbols:
                # 24h volume filtresi
                qv = None
                if USE_24H_VOLUME_FILTER == 1:
                    qv = vol_map.get(sym, 0.0)
                    if qv < MIN_QUOTE_VOLUME_24H:
                        continue

                key = f"{DIRECTION}:{sym}"
                last_ts = storage.get_last_sent(key)
                if last_ts is not None and (time.time() - last_ts) < COOLDOWN_SEC:
                    continue

                # RSI’lar
                rsi_map: Dict[str, Optional[float]] = {"1M": None, "1W": None, "1D": None, "4H": None, "1H": None}
                try:
                    for tf, k in [("1M", "1M"), ("1w", "1W"), ("1d", "1D"), ("4h", "4H"), ("1h", "1H")]:
                        kl = get_klines(sym, tf, KLINE_LIMIT)
                        closes = closes_from_klines(kl)
                        rsi_map[k] = rsi(closes, RSI_LEN)

                    # Gate: (None ise REQUIRE_RSI_VALUES=0 modunda geçebilir)
                    if not passes_rsi_gate(rsi_map["1M"], RSI_1M_MAX):
                        continue
                    if not passes_rsi_gate(rsi_map["1W"], RSI_1W_MAX):
                        continue
                    if not passes_rsi_gate(rsi_map["1D"], RSI_1D_MAX):
                        continue
                    if not passes_rsi_gate(rsi_map["4H"], RSI_4H_MAX):
                        continue
                    if not passes_rsi_gate(rsi_map["1H"], RSI_1H_MAX):
                        continue

                    # Entry/ATR: ana TF (30m)
                    kl_tf = get_klines(sym, TF, KLINE_LIMIT)
                    idx = last_closed_index(kl_tf)
                    highs, lows, closes = highs_lows_closes(kl_tf)

                    a = atr(highs, lows, closes, ATR_LEN)
                    if a is None:
                        # ATR yoksa bu coini pas geçelim
                        continue

                    entry = closes[idx]

                    if DIRECTION == "LONG":
                        sl = entry - a * ATR_SL_MULT
                        tp1 = entry + a * TP1_ATR_MULT
                        tp2 = entry + a * TP2_ATR_MULT
                    else:
                        # (Şimdilik long kuruyoruz, ama short için de hazır)
                        sl = entry + a * ATR_SL_MULT
                        tp1 = entry - a * TP1_ATR_MULT
                        tp2 = entry - a * TP2_ATR_MULT

                    msg = build_signal_message(sym, entry, sl, tp1, tp2, rsi_map, qv)
                    send_telegram(TG_BOT_TOKEN, TG_CHAT_ID, msg)
                    storage.mark_sent(key)
                    found += 1

                except Exception as e:
                    # tek coin patlarsa tüm worker ölmesin
                    print(f"[{now_utc_str()}] {sym} error: {e}")
                    continue

            if found > 0:
                print(f"[{now_utc_str()}] Sent {found} signals.")
            time.sleep(INTERVAL_SEC)

        except Exception as e:
            print(f"[{now_utc_str()}] LOOP error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main_loop()
