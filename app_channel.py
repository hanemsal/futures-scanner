import os
import time
from notify_channel import send_channel

INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "600"))

def main():
    send_channel("✅ Channel worker started")

    while True:
        # Test heartbeat
        send_channel("🫀 Channel worker aktif")
        time.sleep(INTERVAL_SEC)

if __name__ == "__main__":
    main()
