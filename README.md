# futures-scanner

Binance Futures USDT-PERP scanner + Telegram alerts (LONG only).

## Default strategy (current)
**Entry TF:** `TF_ENTRY=1h`  
**Trend TF:** `HTF=4h` (optional filter)  

**Long signal:**
- EMA(3) crosses above EMA(44) within last `LOOKBACK` bars on entry TF
- RSI >= `RSI_MIN`
- Stoch RSI K>D
- WaveTrend Hybrid confirm:
  - Dip Reversal: WT1 dipped <= -53 recently AND crosses up WT2 + rising
  - OR Strong continuation: WT1>0 & <53, rising, WT1>WT2, RSI>55, Stoch K>D
- Optional filters: BTC trend filter, HTF trend filter, Volume spike

## Render (Background Worker)
Build command:
`pip install -r requirements.txt`

Start command:
`python app.py`

### Disk mount
Mount path: `/var/data`

Set:
`STORAGE_PATH=/var/data/futures_state.json`

## ENV Vars (minimum)
- `TG_BOT_TOKEN`
- `TG_CHAT_ID`

## ENV Vars (recommended)
- `TF_ENTRY=1h`
- `HTF=4h`
- `INTERVAL_SEC=600`
- `TOP_N=200`
- `MIN_QUOTE_VOLUME=3000000`
- `COOLDOWN_SEC=21600`
- `EMA_FAST=3`
- `EMA_SLOW=44`
- `LOOKBACK=6`
- `RSI_LEN=21`
- `RSI_MIN=42`
- `USE_WT=1`
- `USE_STOCH_RSI=1`

## Disable filters to match "pure TradingView setup"
Set:
- `USE_MFI_FILTER=0`
- `USE_BTC_FILTER=0`
- `USE_VOL_FILTER=0`
- `USE_HTF_FILTER=0`
