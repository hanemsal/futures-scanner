# futures-scanner

Binance Futures USDT-PERP scanner + Telegram alerts (LONG only).

## How it works (default)
- Entry TF: 1h
- HTF trend: 4h
- Long signal:
  - EMA(3) crosses above EMA(44) within last `HTF_CROSS_LOOKBACK` bars (on TF_ENTRY)
  - RSI >= RSI_MIN
  - MFI in range [MFI_LONG_MIN..MFI_LONG_MAX] and rising (optional)
  - Volume spike (optional)
  - HTF trend filter (optional)
  - BTC trend filter (optional)
- Cooldown prevents repeat alerts per symbol.

## Render (Background Worker)
Start command:
```bash
python app.py
