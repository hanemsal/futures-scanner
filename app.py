import requests
import pandas as pd
import time
import schedule
import os

# Telegram function
def send_telegram(text: str):
    token = os.getenv("TG_BOT_TOKEN")
    chat_id = os.getenv("TG_CHAT_ID")

    if not token or not chat_id:
        print("[WARN] Telegram env eksik", flush=True)
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": text})
        print("Telegram mesaj gönderildi.", flush=True)
    except Exception as e:
        print(f"Telegram hata: {e}", flush=True)


BINANCE_FUTURES_24H = "https://fapi.binance.com/fapi/v1/ticker/24hr"

MIN_VOLUME = 25_000_000
MIN_CHANGE = 8
SCORE_THRESHOLD = 20
TOP_N = 7


def fetch_24h_data():
    response = requests.get(BINANCE_FUTURES_24H)
    data = response.json()
    df = pd.DataFrame(data)
    df = df[df["symbol"].str.endswith("USDT")]
    df["quoteVolume"] = df["quoteVolume"].astype(float)
    df["priceChangePercent"] = df["priceChangePercent"].astype(float)
    return df


def calculate_score(row):
    score = 0

    if row["quoteVolume"] > MIN_VOLUME:
        score += 15

    if abs(row["priceChangePercent"]) >= MIN_CHANGE:
        score += 10

    return score


def run_scanner():
    print("Scanner çalışıyor...", flush=True)

    df = fetch_24h_data()
    df["score"] = df.apply(calculate_score, axis=1)
    df = df[df["score"] >= SCORE_THRESHOLD]

    if df.empty:
        print("Uygun coin yok.", flush=True)
        return

    top_coins = df.sort_values("score", ascending=False).head(TOP_N)

    message_lines = ["📊 FUTURES TOP COINS"]

    for _, row in top_coins.iterrows():
        direction = "LONG" if row["priceChangePercent"] > 0 else "SHORT"

        line = (
            f"{row['symbol']} | {direction} | "
            f"Score:{int(row['score'])} | "
            f"Change:{row['priceChangePercent']:.2f}% | "
            f"Vol:{row['quoteVolume']:.0f}"
        )

        print(line, flush=True)
        message_lines.append(line)

    send_telegram("\n".join(message_lines))


def job():
    run_scanner()


schedule.every(15).minutes.do(job)

if __name__ == "__main__":
    print("Futures Scanner Başladı...", flush=True)
    run_scanner()

    while True:
        schedule.run_pending()
        time.sleep(1)
