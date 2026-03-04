#!/usr/bin/env python3
import os
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import requests

BINANCE_FAPI = os.getenv("BINANCE_FAPI", "https://fapi.binance.com").rstrip("/")

# =========================
# GENERAL
# =========================
TF_ENTRY = os.getenv("TF_ENTRY", "1h")          # entry timeframe
HTF = os.getenv("HTF", "4h")                    # higher timeframe (optional, currently only used in message header)
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "600"))
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "12"))
KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "260"))
TOP_N = int(os.getenv("TOP_N", "200"))
MIN_QUOTE_VOLUME = float(os.getenv("MIN_QUOTE_VOLUME", "3000000"))
ONLY_USDT_PERP = int(os.getenv("ONLY_USDT_PERP", "1"))
LONG_ONLY = int(os.getenv("LONG_ONLY", "1"))

DEBUG = int(os.getenv("DEBUG", "1"))
DEBUG_REJECTS = int(os.getenv("DEBUG_REJECTS", "0"))
TEST_ONCE = int(os.getenv("TEST_ONCE", "0"))
DRY_RUN = int(os.getenv("DRY_RUN", "0"))
HEARTBEAT_SEC = int(os.getenv("HEARTBEAT_SEC", "900"))

# NEW: refresh watchlist periodically (in seconds). 1800 = 30 minutes
WATCH_REFRESH_SEC = int(os.getenv("WATCH_REFRESH_SEC", "1800"))

# If 1 -> evaluate signals on the LAST CLOSED candle (safer)
USE_LAST_CANDLE = int(os.getenv("USE_LAST_CANDLE", "1"))

# =========================
# TELEGRAM
# =========================
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "").strip()
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "").strip()

def send_telegram(text: str) -> None:
    """Minimal telegram sender (no external notify.py dependency)."""
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        if DEBUG:
            print("[WARN] TG_BOT_TOKEN / TG_CHAT_ID missing; message skipped.")
        return
    if DRY_RUN:
        print("[DRY_RUN] Would send:\n", text)
        return
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TG_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, json=payload, timeout=HTTP_TIMEOUT)
        if r.status_code != 200:
            print("[ERR] Telegram send failed:", r.status_code, r.text[:500])
    except Exception as e:
        print("[ERR] Telegram exception:", repr(e))

# =========================
# STORAGE (cooldown + dedupe)
# =========================
STORAGE_PATH = os.getenv("STORAGE_PATH", "/var/data/futures_state.json")
USE_STORAGE = int(os.getenv("USE_STORAGE", "1"))
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", "21600"))  # 6 hours default

@dataclass
class Storage:
    path: str
    data: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f) or {}
        except FileNotFoundError:
            self.data = {}
        except Exception:
            self.data = {}

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        except Exception:
            pass
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f)
        except Exception:
            pass

    def _get(self, sym: str) -> Dict[str, float]:
        if sym not in self.data or not isinstance(self.data.get(sym), dict):
            self.data[sym] = {}
        return self.data[sym]

    def is_cooldown(self, sym: str, key: str, cooldown_sec: int) -> bool:
        """key: 'entry' or 'close' etc."""
        now = time.time()
        d = self._get(sym)
        last = float(d.get(f"last_{key}_ts", 0.0))
        return (now - last) < cooldown_sec

    def touch(self, sym: str, key: str) -> None:
        d = self._get(sym)
        d[f"last_{key}_ts"] = time.time()

# =========================
# INDICATORS
# =========================
def ema(values: List[float], length: int) -> List[float]:
    if length <= 1:
        return values[:]
    k = 2 / (length + 1)
    out = []
    prev = values[0]
    out.append(prev)
    for v in values[1:]:
        prev = v * k + prev * (1 - k)
        out.append(prev)
    return out

def rsi(values: List[float], length: int) -> List[float]:
    if length <= 1 or len(values) < length + 1:
        return [50.0] * len(values)
    gains = [0.0]
    losses = [0.0]
    for i in range(1, len(values)):
        ch = values[i] - values[i - 1]
        gains.append(max(ch, 0.0))
        losses.append(max(-ch, 0.0))
    avg_gain = sum(gains[1:length+1]) / length
    avg_loss = sum(losses[1:length+1]) / length
    out = [50.0] * (length)  # seed
    rs = (avg_gain / avg_loss) if avg_loss != 0 else 999999.0
    out.append(100 - (100 / (1 + rs)))
    for i in range(length + 1, len(values)):
        avg_gain = (avg_gain * (length - 1) + gains[i]) / length
        avg_loss = (avg_loss * (length - 1) + losses[i]) / length
        rs = (avg_gain / avg_loss) if avg_loss != 0 else 999999.0
        out.append(100 - (100 / (1 + rs)))
    if len(out) < len(values):
        out = out + [out[-1]] * (len(values) - len(out))
    return out[:len(values)]

def sma(values: List[float], length: int) -> List[float]:
    if length <= 1:
        return values[:]
    out = []
    s = 0.0
    for i, v in enumerate(values):
        s += v
        if i >= length:
            s -= values[i - length]
        if i >= length - 1:
            out.append(s / length)
        else:
            out.append(v)
    return out

def stoch_rsi(values: List[float], rsi_len: int, stoch_len: int, k: int, d: int) -> Tuple[List[float], List[float]]:
    r = rsi(values, rsi_len)
    stoch = []
    for i in range(len(r)):
        lo = min(r[max(0, i - stoch_len + 1): i + 1])
        hi = max(r[max(0, i - stoch_len + 1): i + 1])
        if hi - lo == 0:
            stoch.append(50.0)
        else:
            stoch.append(100 * (r[i] - lo) / (hi - lo))
    k_line = sma(stoch, k)
    d_line = sma(k_line, d)
    return k_line, d_line

def wavetrend(hlc3: List[float], ch_len: int, avg_len: int) -> Tuple[List[float], List[float]]:
    esa = ema(hlc3, ch_len)
    abs_diff = [abs(hlc3[i] - esa[i]) for i in range(len(hlc3))]
    d = ema(abs_diff, ch_len)
    ci = []
    for i in range(len(hlc3)):
        denom = 0.015 * d[i] if d[i] != 0 else 1e-9
        ci.append((hlc3[i] - esa[i]) / denom)
    wt1 = ema(ci, avg_len)
    wt2 = sma(wt1, 4)
    return wt1, wt2

def cross_up(a_prev: float, a: float, b_prev: float, b: float) -> bool:
    return a_prev <= b_prev and a > b

def cross_down(a_prev: float, a: float, b_prev: float, b: float) -> bool:
    return a_prev >= b_prev and a < b

# =========================
# STRATEGY PARAMS
# =========================
EMA_FAST = int(os.getenv("EMA_FAST", "3"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "44"))
EMA_TOL_PCT = float(os.getenv("EMA_TOL_PCT", "0.002"))  # 0.2% tolerance

RSI_LEN = int(os.getenv("RSI_LEN", "21"))
RSI_MIN = float(os.getenv("RSI_MIN", "42"))

USE_STOCH_RSI = int(os.getenv("USE_STOCH_RSI", "1"))
STOCH_RSI_LEN = int(os.getenv("STOCH_RSI_LEN", "14"))
STOCH_K = int(os.getenv("STOCH_K", "5"))
STOCH_D = int(os.getenv("STOCH_D", "5"))
STOCH_OS = float(os.getenv("STOCH_OS", "20"))
STOCH_OB = float(os.getenv("STOCH_OB", "80"))

USE_WT = int(os.getenv("USE_WT", "1"))
WT_CH = int(os.getenv("WT_CH", "12"))
WT_AVG = int(os.getenv("WT_AVG", "12"))
WT_OS1 = float(os.getenv("WT_OS1", "-60"))
WT_OS2 = float(os.getenv("WT_OS2", "-53"))
WT_OB1 = float(os.getenv("WT_OB1", "60"))
WT_OB2 = float(os.getenv("WT_OB2", "53"))

LOOKBACK = int(os.getenv("LOOKBACK", "6"))

USE_WT_DIP = int(os.getenv("USE_WT_DIP", "1"))
USE_WT_CONTINUATION = int(os.getenv("USE_WT_CONTINUATION", "1"))

WT_CONT_WT2_MAX = float(os.getenv("WT_CONT_WT2_MAX", "-35"))
WT_CONT_STOCH_MIN = float(os.getenv("WT_CONT_STOCH_MIN", "60"))
WT_CONT_RSI_MIN = float(os.getenv("WT_CONT_RSI_MIN", "50"))

ENABLE_CLOSE = int(os.getenv("ENABLE_CLOSE", "1"))
CLOSE_STRICT_OB1 = int(os.getenv("CLOSE_STRICT_OB1", "0"))
CLOSE_COOLDOWN_SEC = int(os.getenv("CLOSE_COOLDOWN_SEC", str(COOLDOWN_SEC)))

TP_PCT = float(os.getenv("TP_PCT", "8"))
SL_PCT = float(os.getenv("SL_PCT", "2"))

# =========================
# BINANCE API
# =========================
def http_get(path: str, params: Optional[Dict] = None):
    url = f"{BINANCE_FAPI}{path}"
    r = requests.get(url, params=params or {}, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()

def get_usdt_perp_symbols() -> List[str]:
    info = http_get("/fapi/v1/exchangeInfo")
    out = []
    for s in info.get("symbols", []):
        if s.get("status") != "TRADING":
            continue
        if ONLY_USDT_PERP:
            if s.get("quoteAsset") != "USDT":
                continue
            if s.get("contractType") != "PERPETUAL":
                continue
        out.append(s.get("symbol"))
    return out

def get_top_symbols_by_quote_volume(symbols: List[str], top_n: int) -> List[str]:
    tickers = http_get("/fapi/v1/ticker/24hr")
    vol_map = {}
    for t in tickers:
        sym = t.get("symbol")
        if sym not in symbols:
            continue
        try:
            qv = float(t.get("quoteVolume", 0.0))
        except Exception:
            qv = 0.0
        vol_map[sym] = qv
    filt = [(s, vol_map.get(s, 0.0)) for s in symbols if vol_map.get(s, 0.0) >= MIN_QUOTE_VOLUME]
    filt.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in filt[:top_n]]

def get_klines(symbol: str, interval: str, limit: int) -> List[List]:
    return http_get("/fapi/v1/klines", params={"symbol": symbol, "interval": interval, "limit": limit})

def klines_to_ohlc(kl: List[List]) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    o, h, l, c, v = [], [], [], [], []
    for k in kl:
        o.append(float(k[1])); h.append(float(k[2])); l.append(float(k[3])); c.append(float(k[4])); v.append(float(k[5]))
    return o, h, l, c, v

# =========================
# SIGNAL LOGIC
# =========================
def ema_ok(ema_fast_val: float, ema_slow_val: float) -> bool:
    return ema_fast_val >= ema_slow_val * (1 - EMA_TOL_PCT)

def pick_idx(n: int) -> int:
    if n < 3:
        return -1
    return -2 if USE_LAST_CANDLE else -1

def build_long_signal(symbol: str, closes: List[float], highs: List[float], lows: List[float]) -> Optional[Dict]:
    idx = pick_idx(len(closes))
    ema_f = ema(closes, EMA_FAST)
    ema_s = ema(closes, EMA_SLOW)
    r = rsi(closes, RSI_LEN)

    hlc3 = [(highs[i] + lows[i] + closes[i]) / 3 for i in range(len(closes))]
    wt1, wt2 = wavetrend(hlc3, WT_CH, WT_AVG) if USE_WT else ([0.0]*len(closes), [0.0]*len(closes))

    k_line, d_line = stoch_rsi(closes, RSI_LEN, STOCH_RSI_LEN, STOCH_K, STOCH_D) if USE_STOCH_RSI else ([50.0]*len(closes), [50.0]*len(closes))

    i = idx
    ip = idx - 1
    price = closes[i]

    ema3 = ema_f[i]; ema44 = ema_s[i]
    rsi_v = r[i]
    st_k = k_line[i]; st_d = d_line[i]
    wt1_v = wt1[i]; wt2_v = wt2[i]
    wt1_p = wt1[ip]; wt2_p = wt2[ip]

    if LONG_ONLY != 1:
        return None
    if rsi_v < RSI_MIN:
        if DEBUG_REJECTS:
            print(f"[REJ] {symbol} rsi {rsi_v:.2f} < {RSI_MIN}")
        return None
    if not ema_ok(ema3, ema44):
        if DEBUG_REJECTS:
            print(f"[REJ] {symbol} ema3 {ema3:.6f} < ema44 {ema44:.6f} (tol {EMA_TOL_PCT})")
        return None

    dip_ok = False
    if USE_WT and USE_WT_DIP:
        dip_ok = (
            (wt1_v <= WT_OS1) and
            cross_up(wt1_p, wt1_v, wt2_p, wt2_v) and
            (st_k <= STOCH_OS and st_d <= STOCH_OS)
        )

    cont_ok = False
    if USE_WT and USE_WT_CONTINUATION:
        cont_ok = (
            cross_up(wt1_p, wt1_v, wt2_p, wt2_v) and
            (wt2_v <= WT_CONT_WT2_MAX) and
            (st_k >= WT_CONT_STOCH_MIN) and
            (rsi_v >= WT_CONT_RSI_MIN)
        )

    if not (dip_ok or cont_ok):
        return None

    sig_type = "WT_DIP" if dip_ok else "WT_CONT"
    return {
        "type": sig_type,
        "symbol": symbol,
        "price": price,
        "ema_fast": ema3,
        "ema_slow": ema44,
        "rsi": rsi_v,
        "stoch_k": st_k,
        "stoch_d": st_d,
        "wt1": wt1_v,
        "wt2": wt2_v,
    }

def build_close_signal(symbol: str, closes: List[float], highs: List[float], lows: List[float]) -> Optional[Dict]:
    if not (USE_WT and ENABLE_CLOSE):
        return None
    idx = pick_idx(len(closes))
    hlc3 = [(highs[i] + lows[i] + closes[i]) / 3 for i in range(len(closes))]
    wt1, wt2 = wavetrend(hlc3, WT_CH, WT_AVG)
    i = idx
    ip = idx - 1
    wt1_v = wt1[i]; wt2_v = wt2[i]
    wt1_p = wt1[ip]; wt2_p = wt2[ip]

    ob_thr = WT_OB1 if CLOSE_STRICT_OB1 else WT_OB2
    if cross_down(wt1_p, wt1_v, wt2_p, wt2_v) and (wt1_p >= ob_thr):
        return {
            "symbol": symbol,
            "price": closes[i],
            "wt1": wt1_v,
            "wt2": wt2_v,
            "wt1_prev": wt1_p,
            "wt2_prev": wt2_p,
            "ob_thr": ob_thr,
        }
    return None

def fmt_long_message(sig: Dict) -> str:
    title = f"🚀 <b>LONG SIGNAL</b> <code>({sig['type']})</code>"
    s = [
        title,
        f"Symbol: <b>{sig['symbol']}</b>",
        f"TF: <b>{TF_ENTRY}</b> | HTF: <b>{HTF}</b>",
        f"Price: <b>{sig['price']:.6f}</b>" if sig["price"] < 10 else f"Price: <b>{sig['price']:.4f}</b>",
        "",
        f"EMA{EMA_FAST}: {sig['ema_fast']:.6f} | EMA{EMA_SLOW}: {sig['ema_slow']:.6f}",
        f"RSI({RSI_LEN}): {sig['rsi']:.2f}",
        f"StochRSI K/D (K={STOCH_K},D={STOCH_D}): {sig['stoch_k']:.2f}/{sig['stoch_d']:.2f}",
        f"WT (ch={WT_CH},avg={WT_AVG}) WT1/WT2: {sig['wt1']:.2f}/{sig['wt2']:.2f}",
        "",
        "<b>Exit plan (manual):</b>",
        f"- TP1: +{TP_PCT:.1f}% (suggestion)",
        f"- SL: -{SL_PCT:.1f}% (suggestion)",
        f"- WT exit: if WT1 crosses DOWN WT2 while WT1>{WT_OB2:.0f} consider close/trim",
        f"- WT warning: if WT1>{WT_OB1:.0f} and turns down -> tighten stop",
    ]
    return "\n".join(s)

def fmt_close_message(sig: Dict) -> str:
    return "\n".join([
        "🧯 <b>CLOSE SIGNAL</b>",
        f"Symbol: <b>{sig['symbol']}</b>",
        f"TF: <b>{TF_ENTRY}</b>",
        f"Price: <b>{sig['price']:.6f}</b>" if sig["price"] < 10 else f"Price: <b>{sig['price']:.4f}</b>",
        "",
        f"WT1/WT2: {sig['wt1']:.2f}/{sig['wt2']:.2f}",
        f"Reason: WT1 crossed DOWN WT2 and previous WT1 ≥ {sig['ob_thr']:.0f}",
    ])

# =========================
# MAIN LOOP
# =========================
def main() -> None:
    storage = Storage(STORAGE_PATH) if USE_STORAGE else None
    if storage:
        storage.load()

    last_heartbeat = 0.0

    if DEBUG:
        print("[BOOT] Futures scanner started")
        print(f"[CFG] TF_ENTRY={TF_ENTRY} EMA={EMA_FAST}/{EMA_SLOW} LOOKBACK={LOOKBACK} RSI_LEN={RSI_LEN} RSI_MIN={RSI_MIN} WT={USE_WT} DIP={USE_WT_DIP} CONT={USE_WT_CONTINUATION} STOCH_RSI={USE_STOCH_RSI}")
        print(f"[CFG] TOP_N={TOP_N} MIN_QUOTE_VOLUME={MIN_QUOTE_VOLUME} COOLDOWN_SEC={COOLDOWN_SEC} DRY_RUN={DRY_RUN} USE_LAST_CANDLE={USE_LAST_CANDLE}")
        print(f"[CFG] WATCH_REFRESH_SEC={WATCH_REFRESH_SEC} INTERVAL_SEC={INTERVAL_SEC} HEARTBEAT_SEC={HEARTBEAT_SEC}")
        print(f"[CFG] STORAGE_PATH={STORAGE_PATH}")

    # initial universe + watchlist
    all_syms = get_usdt_perp_symbols()
    watch = get_top_symbols_by_quote_volume(all_syms, TOP_N)
    last_watch_refresh = time.time()

    if DEBUG:
        print(f"[INFO] symbols in universe: {len(all_syms)} | watching: {len(watch)}")

    while True:
        now = time.time()

        # heartbeat
        if (now - last_heartbeat) >= HEARTBEAT_SEC:
            send_telegram(f"✅ worker alive | TF={TF_ENTRY} TOP_N={TOP_N} DIP={USE_WT_DIP} CONT={USE_WT_CONTINUATION}")
            last_heartbeat = now

        # NEW: refresh watchlist periodically (does NOT change strategy, only updates universe)
        if WATCH_REFRESH_SEC > 0 and (now - last_watch_refresh) >= WATCH_REFRESH_SEC:
            try:
                all_syms = get_usdt_perp_symbols()
                watch = get_top_symbols_by_quote_volume(all_syms, TOP_N)
                last_watch_refresh = now
                if DEBUG:
                    print(f"[INFO] watchlist refreshed | universe={len(all_syms)} watching={len(watch)} TOP_N={TOP_N}")
                send_telegram(f"🔄 watchlist refreshed | TOP_N={TOP_N} watching={len(watch)} TF={TF_ENTRY}")
            except Exception as e:
                if DEBUG:
                    print("[ERR] watchlist refresh failed:", repr(e))

        # scan
        for sym in watch:
            try:
                kl = get_klines(sym, TF_ENTRY, KLINE_LIMIT)
                if not kl or len(kl) < 50:
                    continue
                o, h, l, c, v = klines_to_ohlc(kl)

                sig = build_long_signal(sym, c, h, l)
                if sig:
                    if storage and storage.is_cooldown(sym, "entry", COOLDOWN_SEC):
                        continue
                    send_telegram(fmt_long_message(sig))
                    if storage:
                        storage.touch(sym, "entry")
                        storage.save()

                cs = build_close_signal(sym, c, h, l)
                if cs:
                    if storage and storage.is_cooldown(sym, "close", CLOSE_COOLDOWN_SEC):
                        continue
                    send_telegram(fmt_close_message(cs))
                    if storage:
                        storage.touch(sym, "close")
                        storage.save()

            except Exception as e:
                if DEBUG:
                    print(f"[ERR] {sym} -> {repr(e)}")
                continue

        if TEST_ONCE:
            break
        time.sleep(INTERVAL_SEC)

if __name__ == "__main__":
    main()
