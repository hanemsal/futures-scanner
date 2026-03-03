# futures-scanner

Binance Futures USDT-PERP scanner + Telegram alerts (LONG only).

## What it does
Entry (TF_ENTRY, default 1h):
- EMA(FAST) crossed above EMA(SLOW) within LOOKBACK bars AND currently EMA_FAST > EMA_SLOW
- RSI(RSI_LEN) >= RSI_MIN
- Optional: WaveTrend + StochRSI filters
- Optional: BTC filter, HTF filter, Volume spike, MFI

Signal types:
- WT_DIP: WT1 <= WT_OS2 and turning up (dip reversal)
- WT_CONT: WT1 > 0 and crossed up WT2 under WT_OB2 (trend continuation)

## Render (Background Worker)
Build command:
- pip install -r requirements.txt

Start command:
- python app.py

Disk:
- Mount: /var/data (recommended)
- STORAGE_PATH: /var/data/futures_state.json

## Key ENV
Required:
- TG_BOT_TOKEN
- TG_CHAT_ID

Recommended defaults:
- TF_ENTRY=1h
- EMA_FAST=3
- EMA_SLOW=44
- LOOKBACK=6
- RSI_LEN=21
- RSI_MIN=42
- TOP_N=200
- MIN_QUOTE_VOLUME=3000000
- COOLDOWN_SEC=21600
- USE_STORAGE=1
- STORAGE_PATH=/var/data/futures_state.json

WT/Stoch:
- USE_WT=1
- USE_STOCH_RSI=1
- USE_WT_DIP=1
- USE_WT_CONTINUATION=0 (enable for hybrid)
- USE_LAST_CANDLE=0 (no repaint)
