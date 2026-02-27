import os
import time
import math
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional

import requests

from notify import send_telegram
from storage import Storage


# =========================
# ENV
# =========================

BINANCE_FAPI = os.getenv("BINANCE_FAPI", "https://fapi.binance.com").rstrip("/")

DEBUG = int(os.getenv("DEBUG", "1"))

INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "60"))
HEARTBEAT_SEC = int(os.getenv("HEARTBEAT_SEC", "600"))

KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "260"))

TOP_N = int(os.getenv("TOP_N", "80"))
MIN_QUOTE_VOLUME = float(os.getenv("MIN_QUOTE_VOLUME", "3000000"))

EMA_FAST = int(os.getenv("EMA_FAST", "3"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "44"))

RSI_LEN = int(os.getenv("RSI_LEN", "123"))
RSI_EMA_LEN = int(os.getenv("RSI_EMA_LEN", "47"))
RSI_MIN = float(os.getenv("RSI_MIN", "50"))

USE_STORAGE = int(os.getenv("USE_STORAGE", "1"))
STORAGE_PATH = os.getenv("STORAGE_PATH", "/tmp/futures_storage.json")
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", "3600"))

TF_1H = "1h"
TF_30M = "30m"
TF_5M = "5m"


# =========================
# HTTP
# =========================

def get_json(url: str, params=None):

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def get_klines(symbol, tf, limit):

    return get_json(

        f"{BINANCE_FAPI}/fapi/v1/klines",

        params=dict(

            symbol=symbol,

            interval=tf,

            limit=limit

        )

    )


# =========================
# PARSE
# =========================

def parse_ohlcv(kl):

    close_time = [int(x[6]) for x in kl]

    open = [float(x[1]) for x in kl]
    high = [float(x[2]) for x in kl]
    low = [float(x[3]) for x in kl]
    close = [float(x[4]) for x in kl]
    vol = [float(x[5]) for x in kl]

    return close_time, open, high, low, close, vol


# =========================
# INDICATORS
# =========================

def ema(data, length):

    k = 2/(length+1)

    out=[data[0]]

    for i in range(1,len(data)):

        out.append(

            data[i]*k + out[-1]*(1-k)

        )

    return out


def crossed_up(prev_a, prev_b, now_a, now_b):

    return prev_a <= prev_b and now_a > now_b


def rsi(data, length):

    rsis=[None]*len(data)

    for i in range(length,len(data)):

        gain=0
        loss=0

        for j in range(i-length+1,i+1):

            diff=data[j]-data[j-1]

            if diff>=0:
                gain+=diff
            else:
                loss+=abs(diff)

        if loss==0:
            rsis[i]=100
        else:
            rs=gain/loss
            rsis[i]=100-(100/(1+rs))

    return rsis


# =========================
# SYMBOL LIST
# =========================

def get_symbols():

    data=get_json(

        f"{BINANCE_FAPI}/fapi/v1/ticker/24hr"
    )

    rows=[]

    for x in data:

        s=x["symbol"]

        if not s.endswith("USDT"):
            continue

        vol=float(x["quoteVolume"])

        if vol>=MIN_QUOTE_VOLUME:

            rows.append(

                (s,vol)

            )

    rows.sort(

        key=lambda x:x[1],

        reverse=True

    )

    return [x[0] for x in rows[:TOP_N]]


# =========================
# SIGNAL
# =========================

def check_signal(symbol):

    kl5=get_klines(symbol,TF_5M,KLINE_LIMIT)

    _,_,_,_,close5,_=parse_ohlcv(kl5)

    ema_fast=ema(close5,EMA_FAST)
    ema_slow=ema(close5,EMA_SLOW)

    if len(ema_fast)<3:
        return None

    cross=crossed_up(

        ema_fast[-2],
        ema_slow[-2],

        ema_fast[-1],
        ema_slow[-1]
    )

    if not cross:
        return None

    return dict(

        symbol=symbol,

        price=close5[-1]

    )


# =========================
# MESSAGE
# =========================

def build_msg(sig):

    return f"""🚀 LONG SIGNAL

Symbol: {sig['symbol']}

Price: {sig['price']}

#scanner
"""


# =========================
# MAIN LOOP
# =========================

def main():

    storage = Storage(

        STORAGE_PATH,

        enabled=(USE_STORAGE == 1),

        cooldown_sec=COOLDOWN_SEC

    )

    print("BOT STARTED")

    last_hb=time.time()

    while True:

        try:

            symbols=get_symbols()

            sent=0

            for sym in symbols:

                sig=check_signal(sym)

                if not sig:
                    continue

                key=f"{sym}_LONG"

                if storage.should_send(key):

                    msg=build_msg(sig)

                    send_telegram(msg)

                    storage.mark_sent(key)

                    print("SIGNAL SENT:",sym)

                    sent+=1

                    time.sleep(1)


            if DEBUG:

                print("Cycle done. Sent:",sent)

            if time.time()-last_hb>HEARTBEAT_SEC:

                print("HEARTBEAT OK")

                last_hb=time.time()


        except Exception as e:

            print("ERROR:",e)

        time.sleep(INTERVAL_SEC)


# =========================

if __name__=="__main__":

    main()
