import os
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from diplist import build_diplist, save_diplist, load_diplist
from telegram_bot import send_message, get_updates

STORAGE_PATH = os.getenv("STORAGE_PATH", "/tmp/diplist.json")
HTML_PATH = "/tmp/diplist.html"

PORT = int(os.getenv("PORT", "10000"))

SCHEDULE_HOURS = [2, 14, 20]


# =========================
# HTML RAPOR
# =========================
def generate_html(data):

    rows = []

    for x in data["items"]:

        rows.append(f"""
<tr>
<td>{x['symbol']}</td>
<td>{x['rsi_1m']}</td>
<td>{x['rsi_1w']}</td>
<td>{x['rsi_1d']}</td>
<td>{x['rsi_4h']}</td>
<td>{x['rsi_1h']}</td>
<td>{",".join(x['triggers'])}</td>
</tr>
""")

    html = f"""
<html>
<head>
<meta charset="utf-8">
<title>DipList</title>

<style>

body {{
background:#111;
color:#eee;
font-family:Arial
}}

table {{
border-collapse: collapse;
width:100%
}}

td,th {{
border:1px solid #333;
padding:6px
}}

th {{
background:#222
}}

</style>

</head>

<body>

<h2>DipList</h2>

<p>Toplam Coin: {len(data["items"])}</p>

<table>

<tr>
<th>Symbol</th>
<th>1M</th>
<th>1W</th>
<th>1D</th>
<th>4H</th>
<th>1H</th>
<th>Trigger</th>
</tr>

{''.join(rows)}

</table>

</body>
</html>
"""

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)


# =========================
# WEB SERVER
# =========================
class Handler(BaseHTTPRequestHandler):

    def do_GET(self):

        if self.path == "/diplist":

            if not os.path.exists(HTML_PATH):

                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"No diplist yet")
                return

            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()

            with open(HTML_PATH, "rb") as f:
                self.wfile.write(f.read())

        else:

            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Futures Scanner Running")


def run_web():

    server = HTTPServer(("0.0.0.0", PORT), Handler)
    server.serve_forever()


# =========================
# DIPLIST CALCULATE
# =========================
def run_diplist(manual=False):

    if manual:
        send_message("⏳ DipList başlıyor (MANUEL)...")

    items, meta = build_diplist()

    save_diplist(STORAGE_PATH, items, meta)

    data = load_diplist(STORAGE_PATH)

    generate_html(data)

    url = f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME')}/diplist"

    send_message(
f"""✅ DipList hazır

Scanned: {meta['symbols_scanned']}
Dip: {meta['dip_count']}
Time: {meta['elapsed_sec']}s

İncele:
{url}
"""
    )


# =========================
# SCHEDULER
# =========================
def scheduler_loop():

    last_run = None

    while True:

        now = time.localtime()

        if now.tm_hour in SCHEDULE_HOURS and now.tm_min == 0:

            key = f"{now.tm_year}-{now.tm_yday}-{now.tm_hour}"

            if key != last_run:

                run_diplist(False)

                last_run = key

        time.sleep(30)


# =========================
# TELEGRAM
# =========================
def telegram_loop():

    offset = None

    while True:

        try:

            data = get_updates(offset)

            for u in data.get("result", []):

                offset = u["update_id"] + 1

                text = u["message"].get("text", "")

                if text.startswith("/diplist now"):

                    run_diplist(True)

                elif text.startswith("/diplist"):

                    data = load_diplist(STORAGE_PATH)

                    if not data:
                        send_message("Diplist bulunamadı.")
                        continue

                    url = f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME')}/diplist"

                    send_message(
f"""Son DipList

Toplam Coin: {len(data["items"])}

İncele:
{url}
"""
                    )

        except Exception as e:

            print("Telegram hata:", e)

        time.sleep(2)


# =========================
# MAIN
# =========================
def main():

    send_message("🤖 Worker başladı. Komutlar: /diplist | /diplist now")

    send_message("⏰ Schedule aktif (Europe/Istanbul) -> 02:00,14:00,20:00")

    threading.Thread(target=run_web, daemon=True).start()
    threading.Thread(target=scheduler_loop, daemon=True).start()

    telegram_loop()


if __name__ == "__main__":
    main()
