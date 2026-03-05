import math
import time
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple

import requests

BINANCE_FAPI = os.getenv("BINANCE_FAPI", "https://fapi.binance.com").rstrip("/")
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "60"))
KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "200"))

# Model-1 eşikleri (senin sayılar)
RSI_1M_MAX = float(os.getenv("RSI_1M_MAX", "10"))
RSI_1W_MAX = float(os.getenv("RSI_1W_MAX", "20"))
RSI_1D_MAX = float(os.getenv("RSI_1D_MAX", "30"))
RSI_4H_MAX = float(os.getenv("RSI_4H_MAX", "30"))
RSI_1H_MAX = float(os.getenv("RSI_1H_MAX", "30"))

# 0 => tüm coinler
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "0"))

# DIPLIST_MODE:
# - "UNION": herhangi bir timeframe koşulu tutarsa listeye girer (senin "hepsi listelenecek" isteğin)
# - "ALL": hepsi aynı anda tutarsa listeye girer (eski davranış)
DIPLIST_MODE = os.getenv("DIPLIST_MODE", "UNION").strip().upper()


@dataclass
class DipItem:
    symbol: str
    tv_symbol: str
    rsi_1m: Optional[float]
    rsi_1w: Optional[float]
    rsi_1d: Optional[float]
    rsi_4h: Optional[float]
    rsi_1h: Optional[float]
    triggers: List[str]   # hangi timeframe tetikledi (örn: ["1H", "4H"])
    reasons: List[str]    # metin açıklama


def _http_get(url: str, params: Dict[str, Any]) -> Any:
    last_err = None
    for attempt in range(7):
        try:
            r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
            if r.status_code in (418, 429):
                time.sleep(1.0 + attempt * 1.7)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(0.9 + attempt * 1.4)
    raise last_err


def get_usdt_perp_symbols() -> List[str]:
    url = f"{BINANCE_FAPI}/fapi/v1/exchangeInfo"
    data = _http_get(url, {})
    out: List[str] = []
    for s in data.get("symbols", []):
        if s.get("status") != "TRADING":
            continue
        if s.get("contractType") != "PERPETUAL":
            continue
        if s.get("quoteAsset") != "USDT":
            continue
        sym = s.get("symbol")
        if sym:
            out.append(sym)

    out.sort()

    # MAX_SYMBOLS=0 => sınırlama yok
    if MAX_SYMBOLS and len(out) > MAX_SYMBOLS:
        out = out[:MAX_SYMBOLS]

    return out


def get_closes(symbol: str, interval: str, limit: int = KLINE_LIMIT) -> List[float]:
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    klines = _http_get(url, params)
    closes: List[float] = []
    for k in klines:
        try:
            closes.append(float(k[4]))
        except Exception:
            continue
    return closes


def rsi_wilder(closes: List[float], period: int = 14) -> Optional[float]:
    if closes is None or len(closes) < period + 1:
        return None

    gains = []
    losses = []
    for i in range(1, period + 1):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    for i in range(period + 1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    if math.isnan(rsi) or math.isinf(rsi):
        return None
    return round(rsi, 2)


def compute_rsi(symbol: str, interval: str) -> Optional[float]:
    closes = get_closes(symbol, interval=interval, limit=KLINE_LIMIT)
    return rsi_wilder(closes, period=14)


def build_diplist() -> Tuple[List[DipItem], Dict[str, Any]]:
    symbols = get_usdt_perp_symbols()
    t0 = time.time()

    items: List[DipItem] = []
    errors = 0

    # PASS debug sayaçları (koşulları tek tek geçen kaç coin var)
    c_1m = 0
    c_1w = 0
    c_1d = 0
    c_4h = 0
    c_1h = 0
    c_all = 0

    # UNION sayacı (en az 1 tetikleyen)
    c_union = 0

    for sym in symbols:
        tv = f"{sym}.P"

        rsi_1h = rsi_4h = rsi_1d = rsi_1w = rsi_1m = None

        try:
            rsi_1h = compute_rsi(sym, "1h")
            rsi_4h = compute_rsi(sym, "4h")
            rsi_1d = compute_rsi(sym, "1d")
            rsi_1w = compute_rsi(sym, "1w")
            rsi_1m = compute_rsi(sym, "1M")
        except Exception:
            errors += 1
            continue

        # Senin tarifin:
        # 1M: 0-10 + değer almayan
        # 1W: 0-20 + değer almayan
        # 1D: 0-30
        # 4H: 0-30
        # 1H: 0-30

        cond_1m = (rsi_1m is None) or (rsi_1m <= RSI_1M_MAX)
        cond_1w = (rsi_1w is None) or (rsi_1w <= RSI_1W_MAX)
        cond_1d = (rsi_1d is not None) and (rsi_1d <= RSI_1D_MAX)
        cond_4h = (rsi_4h is not None) and (rsi_4h <= RSI_4H_MAX)
        cond_1h = (rsi_1h is not None) and (rsi_1h <= RSI_1H_MAX)

        # PASS sayaçları
        if cond_1m:
            c_1m += 1
        if cond_1w:
            c_1w += 1
        if cond_1d:
            c_1d += 1
        if cond_4h:
            c_4h += 1
        if cond_1h:
            c_1h += 1
        if cond_1m and cond_1w and cond_1d and cond_4h and cond_1h:
            c_all += 1

        triggers: List[str] = []
        reasons: List[str] = []

        # UNION mantığı: herhangi biri tetiklerse listede göster
        if cond_1m:
            triggers.append("1M")
            reasons.append("1M=NaN" if rsi_1m is None else f"1M<= {RSI_1M_MAX} ({rsi_1m})")
        if cond_1w:
            triggers.append("1W")
            reasons.append("1W=NaN" if rsi_1w is None else f"1W<= {RSI_1W_MAX} ({rsi_1w})")
        if cond_1d:
            triggers.append("1D")
            reasons.append(f"1D<= {RSI_1D_MAX} ({rsi_1d})")
        if cond_4h:
            triggers.append("4H")
            reasons.append(f"4H<= {RSI_4H_MAX} ({rsi_4h})")
        if cond_1h:
            triggers.append("1H")
            reasons.append(f"1H<= {RSI_1H_MAX} ({rsi_1h})")

        if DIPLIST_MODE == "ALL":
            # eski davranış: hepsi aynı anda
            if not (cond_1m and cond_1w and cond_1d and cond_4h and cond_1h):
                continue
        else:
            # yeni davranış: UNION (senin "hepsi listelenecek")
            if not triggers:
                continue
            c_union += 1

        items.append(
            DipItem(
                symbol=sym,
                tv_symbol=tv,
                rsi_1m=rsi_1m,
                rsi_1w=rsi_1w,
                rsi_1d=rsi_1d,
                rsi_4h=rsi_4h,
                rsi_1h=rsi_1h,
                triggers=triggers,
                reasons=reasons,
            )
        )

    elapsed = round(time.time() - t0, 2)
    meta = {
        "mode": DIPLIST_MODE,
        "symbols_scanned": len(symbols),
        "dip_count": len(items),
        "errors": errors,
        "elapsed_sec": elapsed,
        # PASS DEBUG
        "pass_1m": c_1m,
        "pass_1w": c_1w,
        "pass_1d": c_1d,
        "pass_4h": c_4h,
        "pass_1h": c_1h,
        "pass_all": c_all,
        "pass_union": c_union,
    }
    return items, meta


def save_diplist(path: str, items: List[DipItem], meta: Dict[str, Any]) -> None:
    # /tmp gibi dizinlerde dirname("") problem olmasın
    dirn = os.path.dirname(path) or "."
    os.makedirs(dirn, exist_ok=True)

    payload = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "items": [asdict(x) for x in items],
        "meta": meta,
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_diplist(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
