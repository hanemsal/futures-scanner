import os
import time
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple

import requests
import aiohttp

BINANCE_FAPI = os.getenv("BINANCE_FAPI", "https://fapi.binance.com").rstrip("/")

KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "120"))
SCAN_THREADS = int(os.getenv("SCAN_THREADS", "20"))  # Render için 20 stabil
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "8"))
RETRY = int(os.getenv("REQUEST_RETRY", "2"))
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "0"))  # 0 => hepsi

INTERVALS = {
    "1M": "1M",
    "1W": "1w",
    "1D": "1d",
    "4H": "4h",
    "1H": "1h",
}

# Eşikler (senin Model 1)
THRESH = {"1M": 10.0, "1W": 20.0, "1D": 30.0, "4H": 30.0, "1H": 30.0}


def calc_rsi(closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 2:
        return None

    gains = []
    losses = []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        if diff >= 0:
            gains.append(diff)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(diff))

    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def get_symbols() -> List[str]:
    url = f"{BINANCE_FAPI}/fapi/v1/exchangeInfo"
    data = requests.get(url, timeout=20).json()

    out = []
    for s in data.get("symbols", []):
        if s.get("contractType") != "PERPETUAL":
            continue
        if s.get("quoteAsset") != "USDT":
            continue
        if s.get("status") != "TRADING":
            continue
        out.append(s["symbol"])

    if MAX_SYMBOLS and MAX_SYMBOLS > 0:
        out = out[:MAX_SYMBOLS]
    return out


async def _fetch_klines(session: aiohttp.ClientSession, symbol: str, interval: str) -> Optional[List[float]]:
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": KLINE_LIMIT}

    last_err = None
    for attempt in range(RETRY + 1):
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as resp:
                if resp.status != 200:
                    last_err = f"HTTP {resp.status}"
                    await resp.release()
                    await asyncio.sleep(0.2 * (attempt + 1))
                    continue
                data = await resp.json()
                closes = [float(x[4]) for x in data]
                return closes
        except Exception as e:
            last_err = str(e)
            await asyncio.sleep(0.2 * (attempt + 1))
            continue

    # timeout / network fail => None
    return None


async def _scan_symbol(session: aiohttp.ClientSession, symbol: str) -> Optional[Dict[str, Any]]:
    rsi_map: Dict[str, Optional[float]] = {}

    # 5 timeframe ardışık (aynı symbol için), ama global concurrency pool içinde hızlı çalışır
    for key, interval in INTERVALS.items():
        closes = await _fetch_klines(session, symbol, interval)
        if not closes:
            rsi_map[key] = None
            continue
        rsi_map[key] = calc_rsi(closes, 14)

    triggers: List[str] = []

    # “değer almayan” (None/NaN) olanlar DAHİL edilsin istiyorsun:
    # -> TradingView’de “—” gördüğün şey bizde None döner.
    # -> Koşul: (rsi <= eşik) veya (rsi is None) ise trigger.
    for tf, limit in THRESH.items():
        v = rsi_map.get(tf)
        if v is None:
            triggers.append(tf)  # değer almayan dahil
        else:
            if v <= limit:
                triggers.append(tf)

    if not triggers:
        return None

    return {
        "symbol": symbol,
        "rsi_1m": rsi_map.get("1M"),
        "rsi_1w": rsi_map.get("1W"),
        "rsi_1d": rsi_map.get("1D"),
        "rsi_4h": rsi_map.get("4H"),
        "rsi_1h": rsi_map.get("1H"),
        "triggers": triggers,
    }


async def _scan_all_async(symbols: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    passed = {"1M": 0, "1W": 0, "1D": 0, "4H": 0, "1H": 0, "ALL": 0}
    results: List[Dict[str, Any]] = []

    connector = aiohttp.TCPConnector(
        limit=SCAN_THREADS,
        limit_per_host=SCAN_THREADS,
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
    )

    async with aiohttp.ClientSession(connector=connector) as session:
        # bounded worker pool: 544 task’i aynı anda yaratıp RAM şişirmiyoruz
        q: asyncio.Queue[str] = asyncio.Queue()
        for s in symbols:
            q.put_nowait(s)

        async def worker() -> None:
            while True:
                try:
                    sym = q.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    res = await _scan_symbol(session, sym)
                    if res:
                        results.append(res)
                        for tf in res["triggers"]:
                            if tf in passed:
                                passed[tf] += 1
                        if set(res["triggers"]) == set(["1M", "1W", "1D", "4H", "1H"]):
                            passed["ALL"] += 1
                finally:
                    q.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(SCAN_THREADS)]
        await q.join()
        for w in workers:
            w.cancel()

    return results, passed


def build_diplist() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    start = time.time()
    symbols = get_symbols()

    results, passed = asyncio.run(_scan_all_async(symbols))

    elapsed = round(time.time() - start, 2)
    meta = {
        "symbols_scanned": len(symbols),
        "dip_count": len(results),
        "elapsed_sec": elapsed,
        "passed": passed,
        "utc_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    import json
    import os

    STORAGE_PATH = os.getenv("STORAGE_PATH", "/tmp/diplist.json")

    with open(STORAGE_PATH, "w") as f:
        json.dump(results, f)
    
    return results, meta


def fmt_rsi(v: Optional[float]) -> str:
    if v is None:
        return "None"
    try:
        return f"{float(v):.2f}"
    except Exception:
        return "None"


def render_text(results: List[Dict[str, Any]], meta: Dict[str, Any], top_n: int = 40) -> str:
    lines = []
    lines.append(f"✅ DipList hazır")
    lines.append(f"Scanned: {meta['symbols_scanned']}")
    lines.append(f"Dip: {meta['dip_count']}")
    lines.append(f"Time: {meta['elapsed_sec']}s")
    p = meta.get("passed", {})
    lines.append(f"Pass: 1M={p.get('1M',0)} 1W={p.get('1W',0)} 1D={p.get('1D',0)} 4H={p.get('4H',0)} 1H={p.get('1H',0)} ALL={p.get('ALL',0)}")
    lines.append("")

    # Stable sıralama: en çok trigger alan üstte
    results_sorted = sorted(results, key=lambda x: (-len(x.get("triggers", [])), x["symbol"]))

    show = results_sorted[:top_n]
    for r in show:
        lines.append(
            f"{r['symbol']} | "
            f"1M:{fmt_rsi(r['rsi_1m'])} 1W:{fmt_rsi(r['rsi_1w'])} 1D:{fmt_rsi(r['rsi_1d'])} 4H:{fmt_rsi(r['rsi_4h'])} 1H:{fmt_rsi(r['rsi_1h'])} | "
            f"TRIG:{','.join(r['triggers'])}"
        )

    lines.append("")
    lines.append(f"... toplam {len(results_sorted)} coin. (Top {top_n} gösterildi)")
    return "\n".join(lines)


def save_diplist(path: str, results: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    payload = {"meta": meta, "results": results}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def load_diplist(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
