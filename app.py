import time
import requests
import pandas as pd
from ta.trend import EMAIndicator, MACD

from notify import send_telegram

BINANCE = "https://fapi.binance.com"
TF = "1h"
INTERVAL_SEC = 300
KLINE_LIMIT = 200


def get_symbols():
    url = f"{BINANCE}/fapi/v1/exchangeInfo"

    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise ValueError(f"exchangeInfo request failed: {e}")

    if not isinstance(data, dict):
        raise ValueError(f"exchangeInfo dict dönmedi: {data}")

    if "symbols" not in data:
        raise ValueError(f"exchangeInfo içinde symbols yok: {data}")

    symbols = []
    for s in data["symbols"]:
        try:
            if (
                s.get("contractType") == "PERPETUAL"
                and s.get("quoteAsset") == "USDT"
                and s.get("status") == "TRADING"
            ):
                symbols.append(s["symbol"])
        except Exception:
            continue

    return symbols


def get_klines(symbol):
    url = f"{BINANCE}/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": TF,
        "limit": KLINE_LIMIT,
    }

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    if not isinstance(data, list):
        raise ValueError(f"{symbol} kline hatası: {data}")

    df = pd.DataFrame(data)
    df = df.iloc[:, 0:6]
    df.columns = ["time", "open", "high", "low", "close", "volume"]

    df["time"] = df["time"].astype("int64")
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)

    return df


def kama(series, length=13):
    change = series.diff(length).abs()
    volatility = series.diff().abs().rolling(length).sum()

    er = change / volatility.replace(0, pd.NA)

    fast = 2 / (2 + 1)
    slow = 2 / (30 + 1)

    sc = (er * (fast - slow) + slow) ** 2
    sc = sc.fillna(0)

    kama_values = [series.iloc[0]]

    for i in range(1, len(series)):
        prev = kama_values[i - 1]
        current = prev + sc.iloc[i] * (series.iloc[i] - prev)
        kama_values.append(current)

    return pd.Series(kama_values, index=series.index)


def format_price(price):
    if price >= 100:
        return f"{price:.2f}"
    if price >= 1:
        return f"{price:.4f}"
    return f"{price:.6f}"


def check_signal(symbol):
    df = get_klines(symbol)

    close = df["close"]

    df["ema123"] = EMAIndicator(close=close, window=123).ema_indicator()
    df["kama13"] = kama(close, 13)

    macd_obj = MACD(close=close)
    df["macd"] = macd_obj.macd()

    df["vol_ma"] = df["volume"].rolling(20).mean()

    last = df.iloc[-1]
    prev = df.iloc[-2]

    if pd.isna(last["ema123"]) or pd.isna(last["kama13"]) or pd.isna(prev["kama13"]) or pd.isna(last["macd"]) or pd.isna(last["vol_ma"]):
        return False

    long_ok = (
        last["close"] > last["ema123"]
        and last["kama13"] > prev["kama13"]
        and last["macd"] >= 0
        and last["volume"] >= last["vol_ma"]
    )

    if long_ok:
        message = (
            f"🟢 LONG SIGNAL\n\n"
            f"Coin: {symbol}\n"
            f"TF: 1H\n"
            f"Price: {format_price(last['close'])}\n\n"
            f"Şartlar:\n"
            f"• Price > EMA123\n"
            f"• KAMA13 slope ↑\n"
            f"• MACD ≥ 0\n"
            f"• Volume ≥ Volume MA"
        )
        send_telegram(message)
        print(f"SIGNAL: {symbol} @ {format_price(last['close'])}")
        return True

    return False


def scan():
    try:
        symbols = get_symbols()
        print("Total symbols:", len(symbols))
    except Exception as e:
        print("get_symbols error:", e)
        return

    signal_count = 0

    for symbol in symbols:
        try:
            ok = check_signal(symbol)
            if ok:
                signal_count += 1
        except Exception as e:
            print(f"{symbol} signal error: {e}")

    print(f"Scan finished. Signals found: {signal_count}")


if __name__ == "__main__":
    print("Scanner started")
    send_telegram("🚀 Scanner aktif")

    while True:
        try:
            scan()
        except Exception as e:
            print("scan loop error:", e)

        time.sleep(INTERVAL_SEC)
