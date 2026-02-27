import os
import time
import json

from notify_channel import send_channel

STORAGE_PATH = os.getenv("STORAGE_PATH", "/tmp/futures_scanner_storage.json")
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "60"))

sent_cache = set()

def main():
    send_channel("✅ Channel mirror worker started")

    while True:
        try:
            if os.path.exists(STORAGE_PATH):
                with open(STORAGE_PATH, "r") as f:
                    data = json.load(f)

                for key, ts in data.items():

                    if key in sent_cache:
                        continue

                    symbol, tf, side = key.split(":")

                    msg = f"""🚀 LONG SIGNAL

SYMBOL: {symbol}
TF: {tf}
SIDE: {side}

#scanner"""

                    send_channel(msg)

                    sent_cache.add(key)

        except Exception as e:
            send_channel(f"⚠️ Channel worker error: {e}")

        time.sleep(INTERVAL_SEC)

if __name__ == "__main__":
    main()
