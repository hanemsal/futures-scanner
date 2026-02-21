import requests
import pandas as pd
import numpy as np
import time
import schedule

BINANCE_FUTURES_24H = "https://fapi.binance.com/fapi/v1/ticker/24hr"
BINANCE_KLINES = "https://fapi.binance.com/fapi/v1/klines"

MIN_VOLUME = 25_000_000
MIN_CHANGE = 8

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

    # Volume score
    if row["quoteVolume"] > MIN_VOLUME:
        score += 15

    # Daily change score
    if abs(row["priceChangePercent"]) >= MIN_CHANGE:
        score += 10

    return score

def run_scanner():
    print("Scanner çalışıyor...")
    df = fetch_24h_data()

    df["score"] = df.apply(calculate_score, axis=1)
    df = df[df["score"] >= 20]  # Basit filtre (ilk versiyon)

    top_coins = df.sort_values("score", ascending=False).head(7)

    print("----- TOP COINS -----")
    for _, row in top_coins.iterrows():
        direction = "LONG" if row["priceChangePercent"] > 0 else "SHORT"
        print(f"{row['symbol']} | {direction} | Score: {row['score']}")

def job():
    run_scanner()

schedule.every(15).minutes.do(job)

if __name__ == "__main__":
    print("Futures Scanner Başladı...")
    run_scanner()
    while True:
        schedule.run_pending()
        time.sleep(1)
