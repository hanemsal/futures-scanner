import os
import time
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple

import requests
import aiohttp

BINANCE_FAPI = os.getenv("BINANCE_FAPI", "https://fapi.binance.com").rstrip("/")

KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "120"))
SCAN_THREADS = int(os.getenv("SCAN_THREADS", "20"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "8"))
RETRY = int(os.getenv("REQUEST_RETRY", "2"))
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "0"))
STORAGE_PATH = os.getenv("STORAGE_PATH", "/tmp/diplist.json")

INTERVALS = {
    "1M": "1M",
    "1W": "1w",
    "1D": "1d",
    "4H": "4h",
    "1H": "1h",
}

THRESH = {
    "1M": 10.0,
    "1W": 20.0,
    "1D": 30.0,
    "4H": 30.0,
    "1H": 30.0,
}


def calc_rsi(closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 2:
        return None

    gains: List[float] = []
    losses: List[float] = []

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
    return round(100.0 - (100.0 / (1.0 + rs)), 2)


def get_symbols() -> List[str]:
    url = f"{BINANCE_FAPI}/fapi/v1/exchangeInfo"

    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print("get_symbols error:", e)
        return []

    out: List[str] = []

    for s in data.get("symbols", []):
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

    out = sorted(out)

    if MAX_SYMBOLS > 0:
        out = out[:MAX_SYMBOLS]

    return out


async def _fetch_klines(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
) -> Optional[List[float]]:
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": KLINE_LIMIT}

    for attempt in range(RETRY + 1):
        try:
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
            async with session.get(url, params=params, timeout=timeout) as resp:
                if resp.status != 200:
                    await asyncio.sleep(0.25 * (attempt + 1))
                    continue

                data = await resp.json()

                closes: List[float] = []
                for row in data:
                    try:
                        closes.append(float(row[4]))
                    except Exception:
                        continue

                return closes
        except Exception:
            await asyncio.sleep(0.25 * (attempt + 1))

    return None


async def _scan_symbol(
    session: aiohttp.ClientSession,
    symbol: str,
) -> Optional[Dict[str, Any]]:
    rsi_map: Dict[str, Optional[float]] = {}

    for key, interval in INTERVALS.items():
        closes = await _fetch_klines(session, symbol, interval)
        if not closes:
            rsi_map[key] = None
            continue
        rsi_map[key] = calc_rsi(closes, 14)

    triggers: List[str] = []

    for tf, limit in THRESH.items():
        value = rsi_map.get(tf)
        if value is None:
            triggers.append(tf)  # değer almayan coinleri dahil et
        elif value <= limit:
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
        q: asyncio.Queue[str] = asyncio.Queue()

        for sym in symbols:
            q.put_nowait(sym)

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

                        if set(res["triggers"]) == {"1M", "1W", "1D", "4H", "1H"}:
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

    # web.py bunu doğrudan okuyacak
    try:
        with open(STORAGE_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)
    except Exception as e:
        print("write STORAGE_PATH error:", e)

    return results, meta


def fmt_rsi(v: Optional[float]) -> str:
    if v is None:
        return "None"
    try:
        return f"{float(v):.2f}"
    except Exception:
        return "None"


def render_text(results: List[Dict[str, Any]], meta: Dict[str, Any], top_n: int = 40) -> str:
    lines: List[str] = []
    lines.append("✅ DipList hazır")
    lines.append(f"Scanned: {meta['symbols_scanned']}")
    lines.append(f"Dip: {meta['dip_count']}")
    lines.append(f"Time: {meta['elapsed_sec']}s")

    p = meta.get("passed", {})
    lines.append(
        f"Pass: 1M={p.get('1M', 0)} 1W={p.get('1W', 0)} 1D={p.get('1D', 0)} "
        f"4H={p.get('4H', 0)} 1H={p.get('1H', 0)} ALL={p.get('ALL', 0)}"
    )
    lines.append("")

    results_sorted = sorted(
        results,
        key=lambda x: (-len(x.get("triggers", [])), x.get("symbol", "")),
    )

    for r in results_sorted[:top_n]:
        lines.append(
            f"{r['symbol']} | "
            f"1M:{fmt_rsi(r.get('rsi_1m'))} "
            f"1W:{fmt_rsi(r.get('rsi_1w'))} "
            f"1D:{fmt_rsi(r.get('rsi_1d'))} "
            f"4H:{fmt_rsi(r.get('rsi_4h'))} "
            f"1H:{fmt_rsi(r.get('rsi_1h'))} | "
            f"TRIG:{','.join(r.get('triggers', []))}"
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
