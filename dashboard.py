import os
import psycopg
from flask import Flask

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

app = Flask(__name__)


def q(sql, params=()):
    conn = psycopg.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


@app.route("/")
def home():

    total_signals = q("SELECT COUNT(*) FROM signals")[0][0]
    total_exits = q("SELECT COUNT(*) FROM exits")[0][0]

    below_zero = q("SELECT COUNT(*) FROM signals WHERE signal_type='BELOW_ZERO_LONG'")[0][0]
    above_zero = q("SELECT COUNT(*) FROM signals WHERE signal_type='ABOVE_ZERO_LONG'")[0][0]

    valid_signals = q("SELECT COUNT(*) FROM signals WHERE status='VALID'")[0][0]

    avg_profit = q("SELECT COALESCE(AVG(profit_pct),0) FROM exits")[0][0]
    total_pnl = q("SELECT COALESCE(SUM(profit_pct),0) FROM exits")[0][0]

    win_rate = q("""
    SELECT COALESCE(
        100.0 * SUM(CASE WHEN profit_pct > 0 THEN 1 ELSE 0 END) /
        NULLIF(COUNT(*),0),
    0)
    FROM exits
    """)[0][0]

    best_coin = q("""
    SELECT symbol, AVG(profit_pct)
    FROM exits
    GROUP BY symbol
    ORDER BY AVG(profit_pct) DESC
    LIMIT 1
    """)

    worst_coin = q("""
    SELECT symbol, AVG(profit_pct)
    FROM exits
    GROUP BY symbol
    ORDER BY AVG(profit_pct) ASC
    LIMIT 1
    """)

    best_coin = best_coin[0] if best_coin else ("-",0)
    worst_coin = worst_coin[0] if worst_coin else ("-",0)

    coin_ranking = q("""
    SELECT symbol,
    COUNT(*) as trades,
    ROUND(AVG(profit_pct)::numeric,2)
    FROM exits
    GROUP BY symbol
    ORDER BY AVG(profit_pct) DESC
    LIMIT 15
    """)

    zone_stats = q("""
    SELECT signal_type,
    COUNT(*),
    ROUND(AVG(profit_pct)::numeric,2)
    FROM exits
    GROUP BY signal_type
    """)

    daily_signals = q("""
    SELECT DATE(created_at), COUNT(*)
    FROM signals
    GROUP BY DATE(created_at)
    ORDER BY DATE(created_at)
    LIMIT 30
    """)

    daily_exits = q("""
    SELECT DATE(created_at), COUNT(*)
    FROM exits
    GROUP BY DATE(created_at)
    ORDER BY DATE(created_at)
    LIMIT 30
    """)

    recent_signals = q("""
    SELECT symbol, signal_type, entry, target, potential_pct, status, created_at
    FROM signals
    ORDER BY id DESC
    LIMIT 30
    """)

    recent_exits = q("""
    SELECT symbol, signal_type, entry, exit, target, profit_pct, created_at
    FROM exits
    ORDER BY id DESC
    LIMIT 30
    """)

    signal_labels=[str(r[0]) for r in daily_signals]
    signal_vals=[int(r[1]) for r in daily_signals]

    exit_labels=[str(r[0]) for r in daily_exits]
    exit_vals=[int(r[1]) for r in daily_exits]

    html=f"""
    <html>
    <head>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>

body{{background:#0f1117;color:white;font-family:Arial;padding:24px}}

.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:15px}}

.card{{background:#1a1f2b;padding:15px;border-radius:12px}}

table{{width:100%;border-collapse:collapse;margin-top:10px}}

td,th{{padding:8px;border-bottom:1px solid #333}}

</style>

</head>

<body>

<h1>Futures Scanner Dashboard</h1>

<div class="grid">

<div class="card"><h2>{total_signals}</h2>Total Signals</div>
<div class="card"><h2>{total_exits}</h2>Total Exits</div>
<div class="card"><h2>{below_zero}</h2>Below Zero</div>
<div class="card"><h2>{above_zero}</h2>Above Zero</div>

<div class="card"><h2>{valid_signals}</h2>VALID</div>
<div class="card"><h2>{round(win_rate,2)}%</h2>Win Rate</div>
<div class="card"><h2>{round(total_pnl,2)}%</h2>Total PnL</div>
<div class="card"><h2>{round(avg_profit,2)}%</h2>Avg Profit</div>

</div>

<h2>Best Coin</h2>
{best_coin[0]} ({round(best_coin[1],2)}%)

<h2>Worst Coin</h2>
{worst_coin[0]} ({round(worst_coin[1],2)}%)

<h2>Coin Ranking</h2>

<table>

<tr>
<th>Coin</th>
<th>Trades</th>
<th>Avg Profit %</th>
</tr>

"""

    for r in coin_ranking:

        html+=f"""
<tr>
<td>{r[0]}</td>
<td>{r[1]}</td>
<td>{r[2]}%</td>
</tr>
"""

    html+="""
</table>
"""

    html+="""

<h2>MACD Zone Performance</h2>

<table>

<tr>
<th>Zone</th>
<th>Trades</th>
<th>Avg Profit</th>
</tr>
"""

    for r in zone_stats:

        html+=f"""
<tr>
<td>{r[0]}</td>
<td>{r[1]}</td>
<td>{r[2]}%</td>
</tr>
"""

    html+="""
</table>
"""

    html+=f"""

<h2>Daily Signals</h2>
<canvas id="signals"></canvas>

<h2>Daily Exits</h2>
<canvas id="exits"></canvas>

<script>

new Chart(document.getElementById('signals'),{{
type:'bar',
data:{{labels:{signal_labels},
datasets:[{{label:'Signals',data:{signal_vals}}}]}}
}})

new Chart(document.getElementById('exits'),{{
type:'line',
data:{{labels:{exit_labels},
datasets:[{{label:'Exits',data:{exit_vals}}}]}}
}})

</script>

</body>
</html>
"""

    return html


if __name__=="__main__":
    app.run(host="0.0.0.0",port=10000)
