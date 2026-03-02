# futures-scanner

Binance Futures USDT-PERP scanner + Telegram alerts (LONG only).

## Default logic (TradingView-like)
- TF_ENTRY: 1h
- Signal when:
  - EMA(3) crosses ABOVE EMA(44) within LOOKBACK bars (on TF_ENTRY)
  - RSI(21) >= RSI_MIN
  - Stoch RSI (K=5, D=5) computed for info / optional gating
  - WaveTrend (LazyBear-ish) computed
  - WT Dip mode (optional): oversold dip reversal (WT1 cross up WT2 from OS zone)

## Render (Background Worker)
- Build: `pip install -r requirements.txt`
- Start: `python app.py`
- Disk mount path: `/var/data`
- Recommended: `STORAGE_PATH=/var/data/futures_state.json`

## ENV (minimum)
- TG_BOT_TOKEN
- TG_CHAT_ID
- TF_ENTRY=1h
- EMA_FAST=3
- EMA_SLOW=44
- RSI_LEN=21
- RSI_MIN=42
- LOOKBACK=6
- TOP_N=200
- MIN_QUOTE_VOLUME=3000000
- COOLDOWN_SEC=21600
- USE_STORAGE=1
- STORAGE_PATH=/var/data/futures_state.json
- USE_WT=1
- WT_CH_LEN=9
- WT_AVG_LEN=12
- WT_OB1=60 WT_OB2=53 WT_OS1=-60 WT_OS2=-53
- USE_WT_DIP=1
- USE_WT_CONTINUATION=0
- USE_STOCH_RSI=1
- STOCH_K=5 STOCH_D=5 STOCH_RSI_LEN=14
