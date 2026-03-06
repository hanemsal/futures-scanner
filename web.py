import os
import json
from flask import Flask

app = Flask(__name__)

STORAGE_PATH = os.getenv("STORAGE_PATH", "/tmp/diplist.json")


@app.route("/")
def home():
    return "Futures Scanner Running"


@app.route("/diplist")
def diplist():
    if not os.path.exists(STORAGE_PATH):
        return "Diplist not generated yet"

    with open(STORAGE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # diplist.py şu an results listesini direkt yazıyor
    # örnek item:
    # {
    #   "symbol": "BTCUSDT",
    #   "rsi_1m": ...,
    #   "rsi_1w": ...,
    #   "rsi_1d": ...,
    #   "rsi_4h": ...,
    #   "rsi_1h": ...,
    #   "triggers": [...]
    # }

    def fmt(v):
        if v is None:
            return "--"
        try:
            return f"{float(v):.2f}"
        except Exception:
            return "--"

    def mark(v):
        return '<span class="ok">✓</span>' if v else '<span class="no">✗</span>'

    def is_dip(v, limit):
        if v is None:
            return True
        try:
            return float(v) <= limit
        except Exception:
            return False

    # en dipten yukarı: önce 1H RSI, o yoksa büyük sayı
    data = sorted(
        data,
        key=lambda x: (
            9999 if x.get("rsi_1h") is None else float(x.get("rsi_1h")),
            x.get("symbol", "")
        )
    )

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
        width:92%;
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
        font-weight:bold;
    }
    a{
        color:#00c3ff;
        text-decoration:none;
    }
    .muted{
        color:#bbb;
        font-size:12px;
        text-align:center;
        margin-bottom:16px;
    }
    </style>
    </head>
    <body>
    <h2>🚀 Futures Dip Scanner</h2>
    <div class="muted">RSI en düşük coinler üstte listelenir</div>
    <table>
    <tr>
        <th>Coin</th>
        <th>1M RSI</th>
        <th>1W RSI</th>
        <th>1D RSI</th>
        <th>4H RSI</th>
        <th>1H RSI</th>
        <th>1M</th>
        <th>1W</th>
        <th>1D</th>
        <th>4H</th>
        <th>1H</th>
        <th>Chart</th>
    </tr>
    """

    for row in data:
        symbol = row.get("symbol", "")
        r1m = row.get("rsi_1m")
        r1w = row.get("rsi_1w")
        r1d = row.get("rsi_1d")
        r4h = row.get("rsi_4h")
        r1h = row.get("rsi_1h")

        html += f"""
        <tr>
            <td>{symbol}</td>
            <td>{fmt(r1m)}</td>
            <td>{fmt(r1w)}</td>
            <td>{fmt(r1d)}</td>
            <td>{fmt(r4h)}</td>
            <td>{fmt(r1h)}</td>
            <td>{mark(is_dip(r1m, 10))}</td>
            <td>{mark(is_dip(r1w, 20))}</td>
            <td>{mark(is_dip(r1d, 30))}</td>
            <td>{mark(is_dip(r4h, 30))}</td>
            <td>{mark(is_dip(r1h, 30))}</td>
            <td>
                <a href="https://www.binance.com/en/futures/{symbol}" target="_blank">
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


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
