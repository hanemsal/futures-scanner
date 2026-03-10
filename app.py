import time
import requests
import pandas as pd
from ta.trend import EMAIndicator, MACD
from ta.momentum import StochRSIIndicator

from notify import send_telegram

BINANCE = "https://fapi.binance.com"
TF = "1h"
INTERVAL_SEC = 300
KLINE_LIMIT = 200

# Trend filtresi EMA periyodu
TREND_EMA_LEN = 55

# aktif trade takibi
active_trades = {}

# aynı kapanmış mumda tekrar sinyal atmamak için
last_long_candle_sent = {}
last_exit_candle_sent = {}


def get_symbols():
    url = f"{BINANCE}/fapi/v1/exchangeInfo"

    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()

    symbols = []

    for s in data["symbols"]:
        if (
            s.get("contractType") == "PERPETUAL"
            and s.get("quoteAsset") == "USDT"
            and s.get("status") == "TRADING"
        ):
            symbols.append(s["symbol"])

    return symbols


def get_klines(symbol):
    url = f"{BINANCE}/fapi/v1/klines"

    params = {
        "symbol": symbol,
        "interval": TF,
        "limit": KLINE_LIMIT,
    }

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()

    data = r.json()

    df = pd.DataFrame(data)
    df = df.iloc[:, 0:6]
    df.columns = ["time", "open", "high", "low", "close", "volume"]

    df["time"] = df["time"].astype("int64")
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)

    return df


def kama(series, length=13):
    change = series.diff(length).abs()
    volatility = series.diff().abs().rolling(length).sum()

    er = change / volatility.replace(0, pd.NA)

    fast = 2 / (2 + 1)
    slow = 2 / (30 + 1)

    sc = (er * (fast - slow) + slow) ** 2
    sc = sc.fillna(0)

    kama_values = [series.iloc[0]]

    for i in range(1, len(series)):
        prev = kama_values[i - 1]
        current = prev + sc.iloc[i] * (series.iloc[i] - prev)
        kama_values.append(current)

    return pd.Series(kama_values, index=series.index)


def format_price(price):
    if price >= 100:
        return f"{price:.2f}"
    if price >= 1:
        return f"{price:.4f}"
    return f"{price:.6f}"


def get_btc_filter():
    """
    BTC filtresi sadece son kapanmış mumla çalışır.
    BTC close[-2] > BTC EMA55[-2]
    """
    df = get_klines("BTCUSDT")

    close = df["close"]
    df["trend_ema"] = EMAIndicator(close=close, window=TREND_EMA_LEN).ema_indicator()

    signal_candle = df.iloc[-2]  # son kapanmış mum

    if pd.isna(signal_candle["trend_ema"]):
        return False

    btc_ok = signal_candle["close"] > signal_candle["trend_ema"]

    print(
        f"BTC FILTER | close: {format_price(signal_candle['close'])} | "
        f"ema{TREND_EMA_LEN}: {format_price(signal_candle['trend_ema'])} | pass: {btc_ok}",
        flush=True
    )

    return btc_ok


def process_symbol(symbol):
    global active_trades, last_long_candle_sent, last_exit_candle_sent

    df = get_klines(symbol)

    close = df["close"]

    df["trend_ema"] = EMAIndicator(close=close, window=TREND_EMA_LEN).ema_indicator()
    df["kama13"] = kama(close, 13)

    macd = MACD(close=close)
    df["macd"] = macd.macd()

    stoch = StochRSIIndicator(close=close)
    df["stoch"] = stoch.stochrsi_k() * 100

    df["vol_ma"] = df["volume"].rolling(20).mean()

    # SADECE kapanmış mumlar
    signal_candle = df.iloc[-2]   # son kapanmış mum
    prev = df.iloc[-3]            # bir önceki kapanmış mum
    prev2 = df.iloc[-4]           # iki önceki kapanmış mum

    signal_time = str(int(signal_candle["time"]))

    # LONG koşulu - kapanmış mum bazlı
    long_ok = (
        signal_candle["close"] > signal_candle["trend_ema"]
        and signal_candle["kama13"] > prev["kama13"]
        and signal_candle["macd"] >= 0
        and signal_candle["volume"] >= signal_candle["vol_ma"]
        and signal_candle["close"] > signal_candle["open"]   # kırmızı mumda long verme
    )

    # EXIT koşulu - kapanmış mum bazlı
    exit_ok = (
        signal_candle["kama13"] < prev["kama13"]
        and prev["kama13"] < prev2["kama13"]
        and signal_candle["stoch"] < 80
    )

    # LONG
    if long_ok and symbol not in active_trades:
        if last_long_candle_sent.get(symbol) == signal_time:
            return

        price = signal_candle["close"]
        active_trades[symbol] = price
        last_long_candle_sent[symbol] = signal_time

        msg = (
            f"🟢 LONG SIGNAL\n\n"
            f"Coin: {symbol}\n"
            f"TF: 1H\n"
            f"Price: {format_price(price)}\n\n"
            f"Şartlar:\n"
            f"• BTC > EMA{TREND_EMA_LEN}\n"
            f"• Price > EMA{TREND_EMA_LEN}\n"
            f"• KAMA13 slope ↑\n"
            f"• MACD ≥ 0\n"
            f"• Volume ≥ Volume MA\n"
            f"• Green candle"
        )

        send_telegram(msg)
        print(f"LONG: {symbol} @ {format_price(price)} | candle: {signal_time}", flush=True)

    # EXIT
    if symbol in active_trades and exit_ok:
        if last_exit_candle_sent.get(symbol) == signal_time:
            return

        entry = active_trades[symbol]
        exit_price = signal_candle["close"]
        pnl = (exit_price - entry) / entry * 100

        last_exit_candle_sent[symbol] = signal_time

        msg = (
            f"🔴 EXIT SIGNAL\n\n"
            f"Coin: {symbol}\n"
            f"TF: 1H\n\n"
            f"Entry: {format_price(entry)}\n"
            f"Exit: {format_price(exit_price)}\n"
            f"PnL: {pnl:.2f}%\n\n"
            f"Reason:\n"
            f"KAMA13 slope ↓ 2 candle\n"
            f"Stoch < 80"
        )

        send_telegram(msg)
        print(f"EXIT: {symbol} @ {format_price(exit_price)} | candle: {signal_time}", flush=True)

        del active_trades[symbol]


def scan():
    symbols = get_symbols()
    print(f"Total symbols: {len(symbols)}", flush=True)

    btc_ok = get_btc_filter()

    if not btc_ok:
        print(f"BTC filter failed: BTC <= EMA{TREND_EMA_LEN} | LONG taraması pas geçildi", flush=True)
        return

    print(f"BTC filter passed: BTC > EMA{TREND_EMA_LEN} | LONG taraması devam ediyor", flush=True)

    for symbol in symbols:
        try:
            process_symbol(symbol)
        except Exception as e:
            print(f"{symbol} error: {e}", flush=True)


if __name__ == "__main__":
    print("Scanner started", flush=True)
    send_telegram("🚀 Scanner aktif")

    while True:
        try:
            scan()
        except Exception as e:
            print("scan error:", e, flush=True)

        time.sleep(INTERVAL_SEC)
