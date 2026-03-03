# futures-scanner

Binance Futures USDT-PERP scanner + Telegram alerts (LONG only).

## What it does
- Scans TOP_N USDT perpetual symbols (by 24h quoteVolume)
- Entry TF: TF_ENTRY (default 1h)
- Entry requires:
  - EMA_FAST crosses above EMA_SLOW (within LOOKBACK bars)
  - RSI >= RSI_MIN
  - Optional WaveTrend modes:
    - WT_DIP (dip reversal)
    - WT_CONT (continuation)
  - Optional Stoch RSI confirmation
  - Optional filters: BTC trend, HTF trend, Volume spike, MFI

- Optional CLOSE alert:
  - WT1 crosses DOWN WT2 while WT1 > WT_OB2

## Run
Render Background Worker:
- Build: `pip install -r requirements.txt`
- Start: `python app.py`

## Persistent state
Set disk mount to `/var/data` and use:
- STORAGE_PATH=/var/data/futures_state.json

This stores cooldown + open/close tracking.
