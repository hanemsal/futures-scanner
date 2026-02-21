import requests
import pandas as pd
import time
import schedule

BINANCE_FUTURES_24H = "https://fapi.binance.com/fapi/v1/ticker/24hr"

MIN_VOLUME = 25_000_000
MIN_CHANGE = 8
SCORE_THRESHOLD = 20
TOP_N = 7

def fetch_24h_data():
    try:
        r = requests.get(BINANCE_FUTURES_24H, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[ERROR] 24h verisi çekilemedi: {e}", flush=True)
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if df.empty:
        print("[WARN] Binance 24h endpoint boş döndü.", flush=True)
        return df

    # Sadece USDT perpetual (genel filtre)
    df = df[df["symbol"].astype(str).str.endswith("USDT")].copy()

    # Tip dönüşümleri
    for col in ["quoteVolume", "priceChangePercent"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["quoteVolume", "priceChangePercent"])
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
    print("\n==============================", flush=True)
    print(f"Scanner çalışıyor... ({time.strftime('%Y-%m-%d %H:%M:%S')})", flush=True)

    df = fetch_24h_data()
    if df.empty:
        print("[WARN] Veri yok. 15 dk sonra tekrar denenecek.", flush=True)
        return

    df["score"] = df.apply(calculate_score, axis=1)
    df = df[df["score"] >= SCORE_THRESHOLD].copy()

    if df.empty:
        print("[INFO] Eşik üstü coin bulunamadı (score >= 20).", flush=True)
        return

    top_coins = df.sort_values("score", ascending=False).head(TOP_N)

    print("----- TOP COINS -----", flush=True)
    for _, row in top_coins.iterrows():
        direction = "LONG" if row["priceChangePercent"] > 0 else "SHORT"
        print(
            f"{row['symbol']} | {direction} | Score: {int(row['score'])} | "
            f"Change%: {row['priceChangePercent']:.2f} | Vol: {row['quoteVolume']:.0f}",
            flush=True
        )

def job():
    try:
        run_scanner()
    except Exception as e:
        print(f"[FATAL] job() hata: {e}", flush=True)

# 15 dakikada bir çalıştır
schedule.every(15).minutes.do(job)

if __name__ == "__main__":
    print("Futures Scanner Başladı...", flush=True)
    run_scanner()

    while True:
        schedule.run_pending()
        time.sleep(1)
