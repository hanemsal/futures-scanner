# futures-scanner (LONG-only, 1H Swing)

Binance Futures USDT-PERP scanner + Telegram bot.
Strategy is designed to match TradingView setup:

- TF_ENTRY: 1h
- EMA Ribbon: EMA(3) / EMA(44) cross (close confirmed)
- RSI (Wilder): RSI(21) >= 42
- MFI(14): 40..85 (optional)
- Volume spike: last bar >= avg(20) * 1.1 (optional)
- BTC filter: BTC EMA(3) >= EMA(44) (optional)
- HTF trend filter (default 4h): EMA(3) >= EMA(44) (optional)

It **only sends LONG signals** (LONG_ONLY=1).

---

## ENV Vars (Render)

### Required
- TG_BOT_TOKEN
- TG_CHAT_ID

### Core
- BINANCE_FAPI=https://fapi.binance.com
- TF_ENTRY=1h
- EMA_FAST=3
- EMA_SLOW=44
- KLINE_LIMIT=260
- INTERVAL_SEC=600

### Universe / Liquidity
- TOP_N=200
- MIN_QUOTE_VOLUME=3000000
- ONLY_USDT_PERP=1

### RSI
- RSI_LEN=21
- RSI_MIN=42

### MFI (optional)
- USE_MFI_FILTER=1
- MFI_LEN=14
- MFI_LONG_MIN=40
- MFI_LONG_MAX=85

### Volume spike (optional)
- USE_VOL_FILTER=1
- VOL_LEN=20
- VOL_MULT=1.1
- VOL_USE_QUOTE=1

### BTC Filter (optional)
- USE_BTC_FILTER=1
- BTC_SYMBOL=BTCUSDT
- BTC_TF=1h

### HTF Trend Filter (optional)
- USE_HTF_FILTER=1
- HTF=4h
- HTF_STRICT_CROSS=0

### Signal control
- COOLDOWN_SEC=21600
- USE_STORAGE=1
- STORAGE_PATH=/var/data/futures_state.json
- LONG_ONLY=1

### Debug
- DEBUG=1
- DEBUG_REJECTS=1
- TEST_ONCE=0

### Message TP suggestions (manual trading)
- TP1_PCT=8
- TP2_PCT=12
- TP3_PCT=15
- SL_PCT_SUGGEST=4

---

## Run locally
```bash
pip install -r requirements.txt
export TG_BOT_TOKEN="..."
export TG_CHAT_ID="..."
python app.py


---

## Render’da “sıfırdan ENV” için en temiz set
Sen “env’leri siler yeniden eklerim” dedin ya, aynen şöyle gir:

**Minimum (çalışsın):**
- `TG_BOT_TOKEN`
- `TG_CHAT_ID`
- `TF_ENTRY=1h`
- `EMA_FAST=3`
- `EMA_SLOW=44`
- `INTERVAL_SEC=600`
- `KLINE_LIMIT=260`
- `TOP_N=200`
- `MIN_QUOTE_VOLUME=3000000`
- `COOLDOWN_SEC=21600`
- `USE_STORAGE=1`
- `STORAGE_PATH=/var/data/futures_state.json`
- `LONG_ONLY=1`
- `DEBUG=1`
- `DEBUG_REJECTS=1`

**Önerdiğim filtreler (swing kalite):**
- `HTF=4h`
- `USE_HTF_FILTER=1`
- `USE_BTC_FILTER=1`
- `BTC_TF=1h`

---

İstersen bir sonraki mesajında şunu söyle:
- “Render’da disk var mı / var-data mount var mı?” (senin ekranda `/var/data/...` görünüyor iyi)  
Ben de sana **Render deploy kontrol listesi** (build command / start command / logs’ta ne görmen lazım) + test için “TEST_ONCE=1” ile tek tur çalıştırma adımlarını net yazayım.
