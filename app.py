import time
import requests
import pandas as pd
from ta.trend import EMAIndicator, MACD
from ta.momentum import StochRSIIndicator

from notify import send_telegram

BINANCE = "https://fapi.binance.com"
TF = "1h"
INTERVAL_SEC = 300
KLINE_LIMIT = 200

# aktif trade takibi
active_trades = {}


def get_symbols():
    url = f"{BINANCE}/fapi/v1/exchangeInfo"

    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()

    symbols = []

    for s in data["symbols"]:
        if (
            s.get("contractType") == "PERPETUAL"
            and s.get("quoteAsset") == "USDT"
            and s.get("status") == "TRADING"
        ):
            symbols.append(s["symbol"])

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

    df = pd.DataFrame(data)
    df = df.iloc[:, 0:6]

    df.columns = ["time", "open", "high", "low", "close", "volume"]

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


def process_symbol(symbol):

    global active_trades

    df = get_klines(symbol)

    close = df["close"]

    df["ema123"] = EMAIndicator(close=close, window=123).ema_indicator()

    df["kama13"] = kama(close, 13)

    macd = MACD(close=close)

    df["macd"] = macd.macd()

    stoch = StochRSIIndicator(close=close)

    df["stoch"] = stoch.stochrsi_k() * 100

    df["vol_ma"] = df["volume"].rolling(20).mean()

    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    # LONG koşulu
    long_ok = (
        last["close"] > last["ema123"]
        and last["kama13"] > prev["kama13"]
        and last["macd"] >= 0
        and last["volume"] >= last["vol_ma"]
    )

    # EXIT koşulu
    exit_ok = (
        last["kama13"] < prev["kama13"]
        and prev["kama13"] < prev2["kama13"]
        and last["stoch"] < 80
    )

    # LONG
    if long_ok and symbol not in active_trades:

        price = last["close"]

        active_trades[symbol] = price

        msg = (
            f"🟢 LONG SIGNAL\n\n"
            f"Coin: {symbol}\n"
            f"TF: 1H\n"
            f"Price: {format_price(price)}\n\n"
            f"Şartlar:\n"
            f"• Price > EMA123\n"
            f"• KAMA13 slope ↑\n"
            f"• MACD ≥ 0\n"
            f"• Volume ≥ Volume MA"
        )

        send_telegram(msg)

        print(f"LONG: {symbol} @ {format_price(price)}", flush=True)

    # EXIT
    if symbol in active_trades and exit_ok:

        entry = active_trades[symbol]
        exit_price = last["close"]

        pnl = (exit_price - entry) / entry * 100

        msg = (
            f"🔴 EXIT SIGNAL\n\n"
            f"Coin: {symbol}\n"
            f"TF: 1H\n\n"
            f"Entry: {format_price(entry)}\n"
            f"Exit: {format_price(exit_price)}\n"
            f"PnL: {pnl:.2f}%\n\n"
            f"Reason:\n"
            f"KAMA13 slope ↓ 2 candle\n"
            f"Stoch < 80"
        )

        send_telegram(msg)

        print(f"EXIT: {symbol}", flush=True)

        del active_trades[symbol]


def scan():

    symbols = get_symbols()

    print(f"Total symbols: {len(symbols)}", flush=True)

    for symbol in symbols:

        try:

            process_symbol(symbol)

        except Exception as e:

            print(f"{symbol} error: {e}", flush=True)


if __name__ == "__main__":

    print("Scanner started", flush=True)

    send_telegram("🚀 Scanner aktif")

    while True:

        try:

            scan()

        except Exception as e:

            print("scan error:", e, flush=True)

        time.sleep(INTERVAL_SEC)
