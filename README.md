# futures-scanner

Binance USDT-PERP Futures scanner + Telegram alerts (LONG only).

## Strategy (TradingView-aligned)
Entry TF: 1h  
HTF Trend: 4h

Long signal:
- EMA(3) crosses above EMA(44) within last `HTF_CROSS_LOOKBACK` bars (on TF_ENTRY)
- RSI >= RSI_MIN
- StochRSI (K=5, D=5): K > D (optional)
- WaveTrend (LazyBear WT_LB): confirm momentum / reversal (optional)
- MFI in range [MFI_LONG_MIN..MFI_LONG_MAX] and rising (optional)
- Volume spike (optional)
- HTF trend: price above EMA(123) (optional)
- BTC trend: BTC price above EMA(123) (optional)

Cooldown prevents repeat alerts for the same symbol.

## Render (Background Worker)
Build command:
`pip install -r requirements.txt`

Start command:
`python app.py`

Disk mount:
`/var/data`

## ENV Vars (minimum)
Telegram:
- TG_BOT_TOKEN
- TG_CHAT_ID

Core:
- TF_ENTRY=1h
- HTF=4h
- INTERVAL_SEC=600
- TOP_N=200
- MIN_QUOTE_VOLUME=3000000
- COOLDOWN_SEC=21600

EMA:
- EMA_FAST=3
- EMA_SLOW=44
- EMA_TREND=123
- HTF_CROSS_LOOKBACK=6

RSI:
- RSI_LEN=21
- RSI_MIN=42

WaveTrend (LazyBear):
- USE_WT=1
- WT_CH=9
- WT_AVG=12
- WT_OB1=60
- WT_OB2=53
- WT_OS1=-60
- WT_OS2=-53

Stoch RSI:
- USE_STOCH_RSI=1
- STOCH_RSI_LEN=14
- STOCH_K=5
- STOCH_D=5

Filters:
- USE_MFI_FILTER=1
- MFI_LEN=14
- MFI_LONG_MIN=40
- MFI_LONG_MAX=85
- MFI_SLOPE_ENABLE=1
- MFI_SLOPE_BARS=1

- USE_VOL_FILTER=1
- VOL_LEN=20
- VOL_MULT=1.1
- VOL_USE_QUOTE=1

BTC/HTF:
- USE_BTC_FILTER=1
- BTC_SYMBOL=BTCUSDT
- BTC_TF=4h
- USE_HTF_FILTER=1

Ops:
- DEBUG=1
- DEBUG_REJECTS=0
- DRY_RUN=0
- TEST_ONCE=0
