import pandas as pd
import os
import logging

def build_features(df, ticker="TSLA"):
    """
    Input: raw DataFrame with columns ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    Output: DataFrame with added features, saved in data/processed/
    """
    logging.info(f"Building features for {ticker} | Rows: {len(df)}")
    
    feature_df = df.copy()
    
    # Ensure Date is datetime
    feature_df['Date'] = pd.to_datetime(feature_df['Date'])
    
    # Sort by date ascending
    feature_df.sort_values('Date', inplace=True)

    #Compute price returns
    # 1-day return
    feature_df['return_1d'] = feature_df['Close'].pct_change(1)
    # 3-day return
    feature_df['return_3d'] = feature_df['Close'].pct_change(3)
    # 7-day return
    feature_df['return_7d'] = feature_df['Close'].pct_change(7)

    #Compute rolling statistics
    # Rolling 5-day SMA (simple moving average)
    feature_df['SMA_5'] = feature_df['Close'].rolling(window=5).mean()
    # Rolling 10-day SMA
    feature_df['SMA_10'] = feature_df['Close'].rolling(window=10).mean()
    # 5-day rolling volatility
    feature_df['vol_5'] = feature_df['Close'].rolling(window=5).std()
    # 10-day rolling volatility
    feature_df['vol_10'] = feature_df['Close'].rolling(window=10).std()

    #Compute RSI (Relative Strength Index)
    delta = feature_df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    RS = avg_gain / avg_loss
    feature_df['RSI_14'] = 100 - (100 / (1 + RS))

    #Compute MACD (Moving Average Convergence Divergence)
    EMA_12 = feature_df['Close'].ewm(span=12, adjust=False).mean()
    EMA_26 = feature_df['Close'].ewm(span=26, adjust=False).mean()
    feature_df['MACD'] = EMA_12 - EMA_26
    feature_df['MACD_signal'] = feature_df['MACD'].ewm(span=9, adjust=False).mean()

    #Compute day of the week
    feature_df['day_of_week'] = feature_df['Date'].dt.day_name()
    dummies = pd.get_dummies(feature_df['day_of_week'])
    feature_df = pd.concat([feature_df, dummies], axis=1)
    feature_df.drop('day_of_week', axis=1, inplace=True)

    # Save processed features to CSV
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    processed_folder = os.path.join(project_root, "data", "processed")
    os.makedirs(processed_folder, exist_ok=True)
    
    # Save features to CSV
    csv_path = os.path.join(processed_folder, f"{ticker}_features.csv")
    feature_df.to_csv(csv_path, index=False)
    
    logging.info(f"Saved processed features for {ticker} | Rows: {len(feature_df)} | Path: {csv_path}")
    
    return feature_df
