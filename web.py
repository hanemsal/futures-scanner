import os
import json
from flask import Flask, jsonify, render_template_string

STORAGE_PATH = os.getenv("STORAGE_PATH", "/tmp/diplist.json")

app = Flask(__name__)

HTML = """
<html>
<head>
<title>DipList</title>
<style>
body {background:#111;color:#eee;font-family:Arial}
table {border-collapse:collapse;width:100%}
td,th {border:1px solid #333;padding:6px}
th {background:#222}
</style>
</head>
<body>

<h2>DipList</h2>

<table>
<tr>
<th>Symbol</th>
<th>1M</th>
<th>1W</th>
<th>1D</th>
<th>4H</th>
<th>1H</th>
<th>Triggers</th>
</tr>

{% for c in coins %}
<tr>
<td>{{c.symbol}}</td>
<td>{{c.rsi_1m}}</td>
<td>{{c.rsi_1w}}</td>
<td>{{c.rsi_1d}}</td>
<td>{{c.rsi_4h}}</td>
<td>{{c.rsi_1h}}</td>
<td>{{c.triggers}}</td>
</tr>
{% endfor %}

</table>

</body>
</html>
"""

@app.route("/diplist")
def diplist():

    if not os.path.exists(STORAGE_PATH):
        return "Diplist not generated yet"

    with open(STORAGE_PATH) as f:
        data = json.load(f)

    coins = data["results"]

    return render_template_string(HTML, coins=coins)

@app.route("/")
def home():
    return "Scanner running"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
