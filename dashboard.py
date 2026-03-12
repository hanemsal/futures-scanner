import os
import sqlite3
from flask import Flask

DB_FILE = os.getenv("DB_FILE", "signals.db")

app = Flask(__name__)


def q(sql, params=()):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()
    return rows


@app.route("/")
def home():
    total_signals = q("SELECT COUNT(*) AS c FROM signals")[0]["c"]
    total_exits = q("SELECT COUNT(*) AS c FROM exits")[0]["c"]

    below_zero = q("SELECT COUNT(*) AS c FROM signals WHERE signal_type='BELOW_ZERO_LONG'")[0]["c"]
    above_zero = q("SELECT COUNT(*) AS c FROM signals WHERE signal_type='ABOVE_ZERO_LONG'")[0]["c"]

    valid_signals = q("SELECT COUNT(*) AS c FROM signals WHERE status='VALID'")[0]["c"]
    target_close = q("SELECT COUNT(*) AS c FROM signals WHERE status='TARGET CLOSE'")[0]["c"]

    avg_potential = q("SELECT AVG(potential_pct) AS v FROM signals")[0]["v"]
    avg_profit = q("SELECT AVG(profit_pct) AS v FROM exits")[0]["v"]

    recent_signals = q("""
        SELECT symbol, signal_type, entry, target, potential_pct, status, created_at
        FROM signals
        ORDER BY id DESC
        LIMIT 30
    """)

    recent_exits = q("""
        SELECT symbol, entry, exit, target, profit_pct, created_at
        FROM exits
        ORDER BY id DESC
        LIMIT 30
    """)

    html = f"""
    <html>
    <head>
        <title>Futures Scanner Dashboard</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: #0f1117;
                color: #fff;
                padding: 24px;
            }}
            .grid {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 16px;
                margin-bottom: 24px;
            }}
            .card {{
                background: #1a1f2b;
                padding: 16px;
                border-radius: 14px;
                box-shadow: 0 0 10px rgba(0,0,0,0.25);
            }}
            h1, h2 {{
                margin-top: 0;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 12px;
                background: #1a1f2b;
                border-radius: 12px;
                overflow: hidden;
            }}
            th, td {{
                padding: 10px;
                border-bottom: 1px solid #2a3142;
                text-align: left;
                font-size: 14px;
            }}
            th {{
                background: #232a3b;
            }}
            .section {{
                margin-top: 28px;
            }}
            .green {{ color: #49d17d; }}
            .yellow {{ color: #ffd166; }}
            .blue {{ color: #67b7ff; }}
        </style>
    </head>
    <body>
        <h1>Futures Scanner Dashboard</h1>

        <div class="grid">
            <div class="card"><h2>{total_signals}</h2><div>Total Signals</div></div>
            <div class="card"><h2>{total_exits}</h2><div>Total Exits</div></div>
            <div class="card"><h2 class="green">{below_zero}</h2><div>Below Zero Long</div></div>
            <div class="card"><h2 class="blue">{above_zero}</h2><div>Above Zero Long</div></div>
            <div class="card"><h2>{valid_signals}</h2><div>VALID</div></div>
            <div class="card"><h2 class="yellow">{target_close}</h2><div>TARGET CLOSE</div></div>
            <div class="card"><h2>{round(avg_potential or 0, 2)}%</h2><div>Avg Potential</div></div>
            <div class="card"><h2>{round(avg_profit or 0, 2)}%</h2><div>Avg Exit Profit</div></div>
        </div>

        <div class="section">
            <h2>Recent Signals</h2>
            <table>
                <tr>
                    <th>Symbol</th>
                    <th>Type</th>
                    <th>Entry</th>
                    <th>Target</th>
                    <th>Potential %</th>
                    <th>Status</th>
                    <th>Time</th>
                </tr>
    """

    for r in recent_signals:
        html += f"""
            <tr>
                <td>{r['symbol']}</td>
                <td>{r['signal_type']}</td>
                <td>{round(r['entry'], 8)}</td>
                <td>{round(r['target'], 8)}</td>
                <td>{round(r['potential_pct'], 2)}%</td>
                <td>{r['status']}</td>
                <td>{r['created_at']}</td>
            </tr>
        """

    html += """
            </table>
        </div>

        <div class="section">
            <h2>Recent Exits</h2>
            <table>
                <tr>
                    <th>Symbol</th>
                    <th>Entry</th>
                    <th>Exit</th>
                    <th>Target</th>
                    <th>Profit %</th>
                    <th>Time</th>
                </tr>
    """

    for r in recent_exits:
        html += f"""
            <tr>
                <td>{r['symbol']}</td>
                <td>{round(r['entry'], 8)}</td>
                <td>{round(r['exit'], 8)}</td>
                <td>{round(r['target'], 8)}</td>
                <td>{round(r['profit_pct'], 2)}%</td>
                <td>{r['created_at']}</td>
            </tr>
        """

    html += """
            </table>
        </div>
    </body>
    </html>
    """

    return html


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
