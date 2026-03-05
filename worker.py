import os
import time
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from diplist import build_diplist, save_diplist, load_diplist, render_text
from telegram_bot import send_message, get_updates

STORAGE_PATH = os.getenv("STORAGE_PATH", "/tmp/diplist.json")
PORT = int(os.getenv("PORT", "10000"))

HTML_PATH = "/tmp/diplist.html"

SCAN_LOCK = False


# =========================
# HTML OLUŞTUR
# =========================

def generate_html(data):

    rows = []

    for x in data["results"]:

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
font-family:Arial;
padding:20px
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

<p>Total Coins: {len(data["results"])}</p>

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
                self.wfile.write(b"Diplist not generated yet")
                return

            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()

            with open(HTML_PATH, "rb") as f:
                self.wfile.write(f.read())

        else:

            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Scanner Running")


def start_web():

    server = HTTPServer(("0.0.0.0", PORT), Handler)
    server.serve_forever()


# =========================
# DIPLIST
# =========================

def run_diplist(manual=False):

    global SCAN_LOCK

    if SCAN_LOCK:
        send_message("⏳ DipList zaten çalışıyor...")
        return

    SCAN_LOCK = True

    try:

        send_message("⏳ DipList başlıyor...")

        results, meta = build_diplist()

        save_diplist(STORAGE_PATH, results, meta)

        data = load_diplist(STORAGE_PATH)

        generate_html(data)

        service = os.getenv("RENDER_SERVICE_NAME")

        url = f"https://{service}.onrender.com/diplist"

        msg = render_text(results, meta, top_n=40)

        msg += f"\n\nTüm liste:\n{url}"

        send_message(msg)

    except Exception as e:

        send_message(f"HATA: {e}")

    finally:

        SCAN_LOCK = False


# =========================
# TELEGRAM LOOP
# =========================

def telegram_loop():

    offset = None

    send_message("🤖 Scanner başladı\nKomutlar: /diplist now")

    while True:

        try:

            data = get_updates(offset)

            for u in data.get("result", []):

                offset = u["update_id"] + 1

                text = u["message"].get("text", "")

                if text == "/diplist now":

                    run_diplist(True)

                elif text == "/diplist":

                    data = load_diplist(STORAGE_PATH)

                    if not data:
                        send_message("Diplist yok.")
                        continue

                    service = os.getenv("RENDER_SERVICE_NAME")

                    url = f"https://{service}.onrender.com/diplist"

                    send_message(f"Son diplist:\n{url}")

        except Exception as e:

            print(e)

        time.sleep(2)


# =========================
# MAIN
# =========================

def main():

    threading.Thread(target=start_web, daemon=True).start()

    telegram_loop()


if __name__ == "__main__":
    main()
