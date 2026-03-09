import requests
import pandas as pd
import numpy as np
import time
from notify import send_telegram
from ta.trend import EMAIndicator, MACD
from ta.momentum import StochRSIIndicator

BINANCE = "https://fapi.binance.com"

TF = "1h"
INTERVAL = 300


def get_symbols():

    url = f"{BINANCE}/fapi/v1/exchangeInfo"
    data = requests.get(url).json()

    symbols = []

    for s in data["symbols"]:

        if s["contractType"] == "PERPETUAL" and s["quoteAsset"] == "USDT":

            symbols.append(s["symbol"])

    return symbols


def get_klines(symbol):

    url = f"{BINANCE}/fapi/v1/klines"

    params = {
        "symbol": symbol,
        "interval": TF,
        "limit": 200
    }

    data = requests.get(url, params=params).json()

    df = pd.DataFrame(data)

    df = df.iloc[:,0:6]

    df.columns = ["time","open","high","low","close","volume"]

    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)

    return df


def kama(series, length=13):

    change = abs(series.diff(length))
    volatility = abs(series.diff()).rolling(length).sum()

    er = change / volatility

    fast = 2/(2+1)
    slow = 2/(30+1)

    sc = (er*(fast-slow)+slow)**2

    kama = [series.iloc[0]]

    for i in range(1,len(series)):

        kama.append(kama[i-1]+sc.iloc[i]*(series.iloc[i]-kama[i-1]))

    return pd.Series(kama,index=series.index)


def check_signal(symbol):

    df = get_klines(symbol)

    close = df["close"]

    df["ema123"] = EMAIndicator(close,123).ema_indicator()

    df["kama13"] = kama(close,13)

    macd = MACD(close)

    df["macd"] = macd.macd()

    stoch = StochRSIIndicator(close)

    df["stoch"] = stoch.stochrsi_k()

    df["vol_ma"] = df["volume"].rolling(20).mean()

    last = df.iloc[-1]
    prev = df.iloc[-2]

    if (
        last["close"] > last["ema123"]
        and last["kama13"] > prev["kama13"]
        and last["macd"] >= 0
        and last["volume"] > last["vol_ma"]
    ):

        price = round(last["close"],5)

        message = f"""🟢 LONG SIGNAL

Coin: {symbol}
TF: 1H

Price: {price}

Trend: EMA123 üstü
KAMA slope: ↑
MACD: pozitif
"""

        send_telegram(message)


def scan():

    symbols = get_symbols()

    print("Total symbols:",len(symbols))

    for s in symbols:

        try:

            check_signal(s)

        except:

            pass


send_telegram("🚀 Scanner aktif")

while True:

    scan()

    time.sleep(INTERVAL)
