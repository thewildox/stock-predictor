import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Helper indicator functions
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def compute_stochastic(df, k_window=14, d_window=3):
    low_min = df["Low"].rolling(k_window).min()
    high_max = df["High"].rolling(k_window).max()
    stoch_k = 100 * (df["Close"] - low_min) / (high_max - low_min)
    stoch_d = stoch_k.rolling(d_window).mean()
    return stoch_k, stoch_d


def compute_cci(df, window=20):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    sma = tp.rolling(window).mean()
    mad = tp.rolling(window).apply(lambda x: np.fabs(x - x.mean()).mean())
    return (tp - sma) / (0.015 * mad)


def compute_atr(df, window=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def build_features(df, ticker):
    logger.info(f"Building features for {ticker} | Rows: {len(df)}")

    # Ensure proper date sorting
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Existing returns (target)
    df["return_1d"] = df["Close"].pct_change(1)
    df["return_3d"] = df["Close"].pct_change(3)
    df["return_7d"] = df["Close"].pct_change(7)

    # Additional returns (Features)
    df["return_2d"] = df["Close"].pct_change(2)
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_10d"] = df["Close"].pct_change(10)

    # Moving averages
    df["SMA_5"] = df["Close"].rolling(5).mean()
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()

    # Volatility features
    df["daily_range"] = df["High"] - df["Low"]
    df["candle_body"] = (df["Close"] - df["Open"]).abs()
    df["vol_ratio"] = df["candle_body"] / df["daily_range"]
    df["vol_5"] = df["return_1d"].rolling(5).std()
    df["vol_10"] = df["return_1d"].rolling(10).std()
    df["vol_20"] = df["return_1d"].rolling(20).std()
    df["ATR_14"] = compute_atr(df, window=14)

    # MACD indicators
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = compute_macd(df["Close"])

    # RSI
    df["RSI_14"] = compute_rsi(df["Close"], window=14)

    # Bollinger Bands
    sma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    df["BB_upper"] = sma20 + 2 * std20
    df["BB_lower"] = sma20 - 2 * std20
    df["BB_width"] = df["BB_upper"] - df["BB_lower"]
    df["BB_percent"] = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])

    # Stochastic + CCI
    df["Stoch_K"], df["Stoch_D"] = compute_stochastic(df)
    df["CCI_20"] = compute_cci(df)

    # Candlestick structural features
    df["upper_shadow"] = df["High"] - df[["Open", "Close"]].max(axis=1)
    df["lower_shadow"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
    df["is_bull"] = (df["Close"] > df["Open"]).astype(int)
    df["big_move"] = (df["candle_body"] > df["ATR_14"]).astype(int)

    # Day-of-week features
    df["Monday"] = (df["Date"].dt.dayofweek == 0).astype(int)
    df["Tuesday"] = (df["Date"].dt.dayofweek == 1).astype(int)
    df["Wednesday"] = (df["Date"].dt.dayofweek == 2).astype(int)
    df["Thursday"] = (df["Date"].dt.dayofweek == 3).astype(int)
    df["Friday"] = (df["Date"].dt.dayofweek == 4).astype(int)

    # Drop rows lost due to rolling windows
    initial_len = len(df)
    df.dropna(inplace=True)
    logger.info(f"Dropped {initial_len - len(df)} rows due to indicator warm-up")

    # Automatically detect project root (folder containing 'data')
    current_folder = os.getcwd()
    root_folder = current_folder
    while not os.path.exists(os.path.join(root_folder, "data")):
        parent = os.path.dirname(root_folder)
        if parent == root_folder:  # Reached filesystem root
            raise FileNotFoundError("Could not find 'data' folder in any parent directory.")
        root_folder = parent

    # Save processed file in main data/processed folder
    processed_folder = os.path.join(root_folder, "data", "processed")
    os.makedirs(processed_folder, exist_ok=True)
    processed_path = os.path.join(processed_folder, f"{ticker}_features.csv")
    df.to_csv(processed_path, index=False)

    logger.info(f"Saved processed features for {ticker} | Rows: {len(df)} | Path: {processed_path}")

    return df