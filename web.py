import os
import json
import threading
import time
from flask import Flask, Response

STORAGE_PATH = os.getenv("STORAGE_PATH", "/tmp/diplist.json")

app = Flask(__name__)


def scanner_loop():
    while True:
        # burada diplist üretilecek
        diplist = {
            "coins": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            "time": time.time()
        }

        with open(STORAGE_PATH, "w") as f:
            json.dump(diplist, f)

        time.sleep(60)


@app.route("/")
def home():
    return "Scanner running"


@app.route("/diplist")
def diplist():
    if not os.path.exists(STORAGE_PATH):
        return "Diplist not generated yet"

    with open(STORAGE_PATH) as f:
        data = json.load(f)

    text = ""

    for c in data["coins"]:
        text += c + "\n"

    return Response(text, mimetype="text/plain")


if __name__ == "__main__":
    t = threading.Thread(target=scanner_loop)
    t.start()

    app.run(host="0.0.0.0", port=10000)
