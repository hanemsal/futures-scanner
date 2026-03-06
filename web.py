import os
import time
import json
import threading
import requests
from flask import Flask

app = Flask(__name__)

STORAGE_PATH = os.getenv("STORAGE_PATH", "/tmp/diplist.json")

BINANCE_FAPI = "https://fapi.binance.com"

COINS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT"
]

TIMEFRAMES = {
    "1M": "1M",
    "1W": "1w",
    "1D": "1d",
    "4H": "4h",
    "1H": "1h"
}


# -----------------------------
# RSI HESAPLAMA
# -----------------------------
def calc_rsi(closes, period=14):

    gains = []
    losses = []

    for i in range(1, len(closes)):
        diff = closes[i] - closes[i-1]

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


# -----------------------------
# KLINE ÇEK
# -----------------------------
def get_klines(symbol, interval):

    url = f"{BINANCE_FAPI}/fapi/v1/klines"

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": 100
    }

    r = requests.get(url, params=params, timeout=10)

    data = r.json()

    closes = [float(x[4]) for x in data]

    return closes


# -----------------------------
# FİYAT ÇEK
# -----------------------------
def get_price(symbol):

    url = f"{BINANCE_FAPI}/fapi/v1/ticker/price"

    params = {"symbol": symbol}

    r = requests.get(url, params=params, timeout=10)

    return float(r.json()["price"])


# -----------------------------
# SCANNER
# -----------------------------
def scanner_loop():

    while True:

        results = []

        for coin in COINS:

            try:

                price = get_price(coin)

                closes = get_klines(coin, "1h")

                rsi = calc_rsi(closes)

                tf_result = {}

                for tf_name, tf_val in TIMEFRAMES.items():

                    closes_tf = get_klines(coin, tf_val)

                    rsi_tf = calc_rsi(closes_tf)

                    tf_result[tf_name] = rsi_tf < 30

                results.append({
                    "coin": coin,
                    "price": price,
                    "rsi": rsi,
                    "tf": tf_result
                })

            except Exception as e:
                print("ERROR", coin, e)

        with open(STORAGE_PATH, "w") as f:
            json.dump(results, f)

        print("scanner updated")

        time.sleep(300)


# -----------------------------
# HOME
# -----------------------------
@app.route("/")
def home():

    return "Scanner running"


# -----------------------------
# PANEL
# -----------------------------
@app.route("/diplist")
def diplist():

    if not os.path.exists(STORAGE_PATH):
        return "Diplist not generated yet"

    with open(STORAGE_PATH) as f:
        data = json.load(f)

    html = """

    <html>
    <head>

    <title>Futures Dip Scanner</title>

    <style>

    body{
        background:#0f0f0f;
        color:white;
        font-family:Arial;
    }

    h2{
        text-align:center;
    }

    table{
        border-collapse:collapse;
        margin:auto;
        width:90%;
    }

    th,td{
        border:1px solid #333;
        padding:8px;
        text-align:center;
    }

    th{
        background:#1a1a1a;
    }

    tr:nth-child(even){
        background:#161616;
    }

    .ok{
        color:#00ff88;
        font-weight:bold;
    }

    .no{
        color:#ff5555;
    }

    a{
        color:#00c3ff;
        text-decoration:none;
    }

    </style>

    </head>

    <body>

    <h2>🚀 Futures Dip Scanner</h2>

    <table>

    <tr>
        <th>Coin</th>
        <th>Price</th>
        <th>RSI</th>
        <th>1M</th>
        <th>1W</th>
        <th>1D</th>
        <th>4H</th>
        <th>1H</th>
        <th>Chart</th>
    </tr>

    """

    for row in data:

        def mark(v):
            return '<span class="ok">✓</span>' if v else '<span class="no">✗</span>'

        html += f"""

        <tr>

        <td>{row["coin"]}</td>

        <td>{row["price"]}</td>

        <td>{row["rsi"]}</td>

        <td>{mark(row["tf"]["1M"])}</td>
        <td>{mark(row["tf"]["1W"])}</td>
        <td>{mark(row["tf"]["1D"])}</td>
        <td>{mark(row["tf"]["4H"])}</td>
        <td>{mark(row["tf"]["1H"])}</td>

        <td>
        <a href="https://www.binance.com/en/futures/{row["coin"]}" target="_blank">
        Chart
        </a>
        </td>

        </tr>

        """

    html += "</table></body></html>"

    return html


# -----------------------------
# START
# -----------------------------
if __name__ == "__main__":

    t = threading.Thread(target=scanner_loop)
    t.daemon = True
    t.start()

    app.run(host="0.0.0.0", port=10000)
