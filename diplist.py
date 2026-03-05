import math
import time
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple

import requests

BINANCE_FAPI = os.getenv("BINANCE_FAPI", "https://fapi.binance.com").rstrip("/")
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "10"))
KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "200"))

# RSI eşikleri
RSI_1M_MAX = float(os.getenv("RSI_1M_MAX", "10"))
RSI_1W_MAX = float(os.getenv("RSI_1W_MAX", "20"))
RSI_1D_MAX = float(os.getenv("RSI_1D_MAX", "30"))
RSI_4H_MAX = float(os.getenv("RSI_4H_MAX", "30"))
RSI_1H_MAX = float(os.getenv("RSI_1H_MAX", "30"))

# Performans için ilk testte sınır koyabilirsin
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "0"))  # 0 = sınırsız


@dataclass
class DipItem:
    symbol: str          # Binance symbol, örn BTCUSDT
    tv_symbol: str       # TradingView display, örn BTCUSDT.P
    rsi_1m: Optional[float]
    rsi_1w: Optional[float]
    rsi_1d: Optional[float]
    rsi_4h: Optional[float]
    rsi_1h: Optional[float]
    reasons: List[str]


def _http_get(url: str, params: Dict[str, Any]) -> Any:
    r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_usdt_perp_symbols() -> List[str]:
    """
    Binance Futures USDT perpetual, TRADING olan tüm semboller.
    """
    url = f"{BINANCE_FAPI}/fapi/v1/exchangeInfo"
    data = _http_get(url, {})
    out = []
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
    if MAX_SYMBOLS and len(out) > MAX_SYMBOLS:
        out = out[:MAX_SYMBOLS]
    return out


def get_closes(symbol: str, interval: str, limit: int = KLINE_LIMIT) -> List[float]:
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    klines = _http_get(url, params)
    closes = []
    for k in klines:
        # k[4] close
        try:
            closes.append(float(k[4]))
        except Exception:
            continue
    return closes


def rsi_wilder(closes: List[float], period: int = 14) -> Optional[float]:
    """
    RSI(14) Wilder smoothing.
    Yeterli mum yoksa None döner (değer almamış).
    """
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

    # Wilder smoothing
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


def qualifies(item: DipItem) -> bool:
    """
    Union: herhangi bir koşul tutarsa listede.
    1M: 0-10 veya None
    1W: 0-20 veya None
    1D: 0-30
    4H: 0-30
    1H: 0-30
    """
    # reasons zaten dolduruluyor; burada sadece hızlı true/false
    return len(item.reasons) > 0


def build_diplist() -> Tuple[List[DipItem], Dict[str, Any]]:
    """
    Dip list üretir. Performans için:
    Önce 1H/4H/1D bakar, tutmazsa 1W/1M'e geçer.
    """
    symbols = get_usdt_perp_symbols()
    t0 = time.time()

    items: List[DipItem] = []
    errors = 0

    for idx, sym in enumerate(symbols, start=1):
        tv = f"{sym}.P"
        reasons: List[str] = []

        rsi_1h = rsi_4h = rsi_1d = rsi_1w = rsi_1m = None

        try:
            rsi_1h = compute_rsi(sym, "1h")
            if rsi_1h is not None and rsi_1h <= RSI_1H_MAX:
                reasons.append(f"1H<= {RSI_1H_MAX} ({rsi_1h})")

            rsi_4h = compute_rsi(sym, "4h")
            if rsi_4h is not None and rsi_4h <= RSI_4H_MAX:
                reasons.append(f"4H<= {RSI_4H_MAX} ({rsi_4h})")

            rsi_1d = compute_rsi(sym, "1d")
            if rsi_1d is not None and rsi_1d <= RSI_1D_MAX:
                reasons.append(f"1D<= {RSI_1D_MAX} ({rsi_1d})")

            # 1W / 1M sadece lazım olduğunda (ya zaten dipte, ya da NaN avı)
            # Ama 1W/1M NaN olanları da yakalamak istediğin için, burada yine kontrol ediyoruz:
            rsi_1w = compute_rsi(sym, "1w")
            if rsi_1w is None:
                reasons.append("1W=NaN")
            elif rsi_1w <= RSI_1W_MAX:
                reasons.append(f"1W<= {RSI_1W_MAX} ({rsi_1w})")

            rsi_1m = compute_rsi(sym, "1M")
            if rsi_1m is None:
                reasons.append("1M=NaN")
            elif rsi_1m <= RSI_1M_MAX:
                reasons.append(f"1M<= {RSI_1M_MAX} ({rsi_1m})")

        except Exception:
            errors += 1

        item = DipItem(
            symbol=sym,
            tv_symbol=tv,
            rsi_1m=rsi_1m,
            rsi_1w=rsi_1w,
            rsi_1d=rsi_1d,
            rsi_4h=rsi_4h,
            rsi_1h=rsi_1h,
            reasons=reasons,
        )

        if qualifies(item):
            items.append(item)

    elapsed = round(time.time() - t0, 2)
    meta = {
        "symbols_scanned": len(symbols),
        "dip_count": len(items),
        "errors": errors,
        "elapsed_sec": elapsed,
    }
    return items, meta


def save_diplist(path: str, items: List[DipItem], meta: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
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
