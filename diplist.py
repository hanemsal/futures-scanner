import os
import json
import asyncio
import aiohttp
import time
import requests

BINANCE_FAPI = "https://fapi.binance.com"

KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "120"))
SCAN_THREADS = int(os.getenv("SCAN_THREADS", "25"))

RSI_1M_MAX = 10
RSI_1W_MAX = 20
RSI_1D_MAX = 30
RSI_4H_MAX = 30
RSI_1H_MAX = 30


INTERVALS = {
    "1M": "1M",
    "1W": "1w",
    "1D": "1d",
    "4H": "4h",
    "1H": "1h",
}


# =========================
# RSI CALCULATION
# =========================
def calc_rsi(closes, period=14):

    if len(closes) < period + 1:
        return None

    gains = []
    losses = []

    for i in range(1, len(closes)):

        diff = closes[i] - closes[i - 1]

        if diff >= 0:
            gains.append(diff)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(diff))

    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    return round(rsi, 2)


# =========================
# FETCH KLINES
# =========================
async def get_klines(session, symbol, interval):

    url = f"{BINANCE_FAPI}/fapi/v1/klines"

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": KLINE_LIMIT,
    }

    try:

        async with session.get(url, params=params, timeout=15) as resp:

            data = await resp.json()

            closes = [float(x[4]) for x in data]

            if len(closes) < 20:
                return None

            return calc_rsi(closes)

    except:
        return None


# =========================
# SCAN ONE SYMBOL
# =========================
async def scan_symbol(session, symbol):

    rsi = {}

    for k, v in INTERVALS.items():

        rsi[k] = await get_klines(session, symbol, v)

    triggers = []

    # 1M
    if rsi["1M"] is None or rsi["1M"] <= RSI_1M_MAX:
        triggers.append("1M")

    # 1W
    if rsi["1W"] is None or rsi["1W"] <= RSI_1W_MAX:
        triggers.append("1W")

    # 1D
    if rsi["1D"] is not None and rsi["1D"] <= RSI_1D_MAX:
        triggers.append("1D")

    # 4H
    if rsi["4H"] is not None and rsi["4H"] <= RSI_4H_MAX:
        triggers.append("4H")

    # 1H
    if rsi["1H"] is not None and rsi["1H"] <= RSI_1H_MAX:
        triggers.append("1H")

    if not triggers:
        return None

    return {
        "symbol": symbol,
        "tv_symbol": f"{symbol}.P",
        "rsi_1m": rsi["1M"],
        "rsi_1w": rsi["1W"],
        "rsi_1d": rsi["1D"],
        "rsi_4h": rsi["4H"],
        "rsi_1h": rsi["1H"],
        "triggers": triggers,
        "reasons": triggers,
    }


# =========================
# SCAN ALL SYMBOLS
# =========================
async def scan_all(symbols):

    connector = aiohttp.TCPConnector(limit=SCAN_THREADS)

    async with aiohttp.ClientSession(connector=connector) as session:

        tasks = []

        for s in symbols:
            tasks.append(scan_symbol(session, s))

        results = await asyncio.gather(*tasks)

    return [x for x in results if x]


# =========================
# GET BINANCE SYMBOLS
# =========================
def get_symbols():

    url = f"{BINANCE_FAPI}/fapi/v1/exchangeInfo"

    data = requests.get(url).json()

    symbols = []

    for s in data["symbols"]:

        if s["contractType"] != "PERPETUAL":
            continue

        if s["quoteAsset"] != "USDT":
            continue

        if s["status"] != "TRADING":
            continue

        symbols.append(s["symbol"])

    return sorted(symbols)


# =========================
# BUILD DIPLIST
# =========================
def build_diplist():

    start = time.time()

    symbols = get_symbols()

    results = asyncio.run(scan_all(symbols))

    elapsed = round(time.time() - start, 2)

    meta = {
        "symbols_scanned": len(symbols),
        "dip_count": len(results),
        "elapsed_sec": elapsed,
        "threads": SCAN_THREADS,
    }

    return results, meta


# =========================
# SAVE DIPLIST
# =========================
def save_diplist(path, items, meta):

    data = {
        "generated_at_utc": time.strftime("%Y-%m-%d %H:%M:%S"),
        "items": items,
        "meta": meta,
    }

    with open(path, "w") as f:
        json.dump(data, f)


# =========================
# LOAD DIPLIST
# =========================
def load_diplist(path):

    if not os.path.exists(path):
        return None

    with open(path) as f:
        return json.load(f)
