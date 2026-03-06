import os
import json
import time
import threading
from flask import Flask

app = Flask(__name__)

STORAGE_PATH = os.getenv("STORAGE_PATH", "/tmp/diplist.json")


# -------------------------------
# TEST SCANNER (yerine gerçek scanner gelecek)
# -------------------------------

def scanner_loop():

    while True:

        coins = [
            "BTCUSDT",
            "ETHUSDT",
            "SOLUSDT",
            "BNBUSDT",
            "XRPUSDT"
        ]

        data = {
            "coins": coins,
            "time": time.time()
        }

        with open(STORAGE_PATH, "w") as f:
            json.dump(data, f)

        print("Diplist updated")

        time.sleep(60)


# -------------------------------
# HOME
# -------------------------------

@app.route("/")
def home():

    return "Futures Scanner Running"


# -------------------------------
# DIPLIST PANEL
# -------------------------------

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
        width:80%;
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
        <th>RSI</th>
        <th>1M</th>
        <th>1W</th>
        <th>1D</th>
        <th>4H</th>
        <th>1H</th>
        <th>Chart</th>
    </tr>

    """

    for coin in data["coins"]:

        html += f"""

        <tr>

        <td>{coin}</td>

        <td>--</td>

        <td class="ok">✓</td>
        <td class="ok">✓</td>
        <td class="ok">✓</td>
        <td class="ok">✓</td>
        <td class="ok">✓</td>

        <td>
        <a href="https://www.binance.com/en/futures/{coin}" target="_blank">
        Chart
        </a>
        </td>

        </tr>

        """

    html += """

    </table>

    </body>

    </html>

    """

    return html


# -------------------------------
# START THREAD
# -------------------------------

if __name__ == "__main__":

    t = threading.Thread(target=scanner_loop)
    t.daemon = True
    t.start()

    app.run(host="0.0.0.0", port=10000)
