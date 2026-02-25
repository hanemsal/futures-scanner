# futures-scanner

Binance Futures USDT-PERP scanner + Telegram bot.

## ENV Vars (Render)
- TG_BOT_TOKEN
- TG_CHAT_ID
- TF (default 30m)
- INTERVAL_SEC (default 600)
- KLINE_LIMIT (default 200)
- EMA_FAST (default 3)
- EMA_SLOW (default 41)
- RSI_LEN (default 14)
- TOP_N (default 30)
- MIN_QUOTE_VOLUME (default 15000000)
- COOLDOWN_SEC (default 3600)

BTC Filter:
- USE_BTC_FILTER (1/0)
- BTC_SYMBOL (default BTCUSDT)
- BTC_RSI_MIN (default 42)

Storage:
- USE_STORAGE (1/0)
- STORAGE_PATH (default state.json)
