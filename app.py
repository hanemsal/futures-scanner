import time
from notify import send_telegram

print("Bot started")

send_telegram("✅ Futures Scanner Başladı")

while True:

    time.sleep(3600)
