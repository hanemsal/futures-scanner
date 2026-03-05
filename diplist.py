import math
import time
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

BINANCE_FAPI = os.getenv("BINANCE_FAPI", "https://fapi.binance.com").rstrip("/")
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "120"))
KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "120"))

SCAN_THREADS = int(os.getenv("SCAN_THREADS", "25"))

RSI_1M_MAX = float(os.getenv("RSI_1M_MAX", "10"))
RSI_1W_MAX = float(os.getenv("RSI_1W_MAX", "20"))
RSI_1D_MAX = float(os.getenv("RSI_1D_MAX", "30"))
RSI_4H_MAX = float(os.getenv("RSI_4H_MAX", "30"))
RSI_1H_MAX = float(os.getenv("RSI_1H_MAX", "30"))

MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "0"))

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
    triggers: List[str]
    reasons: List[str]


def http_get(url: str, params: Dict[str, Any]):

    for _ in range(5):
        try:
            r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except:
            time.sleep(1)
    return None


def get_usdt_perp_symbols():

    url = f"{BINANCE_FAPI}/fapi/v1/exchangeInfo"
    data = http_get(url, {})

    out = []

    for s in data["symbols"]:

        if s["contractType"] != "PERPETUAL":
            continue

        if s["quoteAsset"] != "USDT":
            continue

        if s["status"] != "TRADING":
            continue

        out.append(s["symbol"])

    out.sort()

    if MAX_SYMBOLS and len(out) > MAX_SYMBOLS:
        out = out[:MAX_SYMBOLS]

    return out


def get_closes(symbol, interval):

    url = f"{BINANCE_FAPI}/fapi/v1/klines"

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": KLINE_LIMIT
    }

    data = http_get(url, params)

    if not data:
        return []

    return [float(x[4]) for x in data]


def rsi_wilder(closes, period=14):

    if len(closes) < period + 1:
        return None

    gains = []
    losses = []

    for i in range(1, period + 1):

        diff = closes[i] - closes[i-1]

        gains.append(max(diff,0))
        losses.append(max(-diff,0))

    avg_gain = sum(gains)/period
    avg_loss = sum(losses)/period

    for i in range(period + 1, len(closes)):

        diff = closes[i] - closes[i-1]

        gain = max(diff,0)
        loss = max(-diff,0)

        avg_gain = (avg_gain*(period-1)+gain)/period
        avg_loss = (avg_loss*(period-1)+loss)/period

    if avg_loss == 0:
        return 100

    rs = avg_gain/avg_loss

    rsi = 100-(100/(1+rs))

    return round(rsi,2)


def compute_rsi(symbol, interval):

    closes = get_closes(symbol, interval)

    if not closes:
        return None

    return rsi_wilder(closes)


def scan_symbol(sym):

    tv = f"{sym}.P"

    try:

        rsi_1h = compute_rsi(sym,"1h")
        rsi_4h = compute_rsi(sym,"4h")
        rsi_1d = compute_rsi(sym,"1d")
        rsi_1w = compute_rsi(sym,"1w")
        rsi_1m = compute_rsi(sym,"1M")

    except:
        return None

    cond_1m = (rsi_1m is None) or (rsi_1m <= RSI_1M_MAX)
    cond_1w = (rsi_1w is None) or (rsi_1w <= RSI_1W_MAX)
    cond_1d = (rsi_1d is not None and rsi_1d <= RSI_1D_MAX)
    cond_4h = (rsi_4h is not None and rsi_4h <= RSI_4H_MAX)
    cond_1h = (rsi_1h is not None and rsi_1h <= RSI_1H_MAX)

    triggers=[]
    reasons=[]

    if cond_1m:
        triggers.append("1M")

    if cond_1w:
        triggers.append("1W")

    if cond_1d:
        triggers.append("1D")

    if cond_4h:
        triggers.append("4H")

    if cond_1h:
        triggers.append("1H")

    if DIPLIST_MODE == "ALL":

        if not (cond_1m and cond_1w and cond_1d and cond_4h and cond_1h):
            return None

    else:

        if not triggers:
            return None

    reasons = triggers

    return DipItem(
        sym,
        tv,
        rsi_1m,
        rsi_1w,
        rsi_1d,
        rsi_4h,
        rsi_1h,
        triggers,
        reasons
    )


def build_diplist():

    symbols = get_usdt_perp_symbols()

    start=time.time()

    items=[]
    errors=0

    with ThreadPoolExecutor(max_workers=SCAN_THREADS) as executor:

        futures = [executor.submit(scan_symbol,s) for s in symbols]

        for future in as_completed(futures):

            res = future.result()

            if res:
                items.append(res)

    elapsed=round(time.time()-start,2)

    meta = {

        "symbols_scanned":len(symbols),
        "dip_count":len(items),
        "elapsed_sec":elapsed,
        "threads":SCAN_THREADS
    }

    return items,meta


def save_diplist(path,items,meta):

    data={
        "generated_at_utc":time.strftime("%Y-%m-%d %H:%M:%S"),
        "items":[asdict(x) for x in items],
        "meta":meta
    }

    with open(path,"w") as f:

        json.dump(data,f)


def load_diplist(path):

    if not os.path.exists(path):
        return None

    with open(path) as f:

        return json.load(f)
