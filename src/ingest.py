# Imports
import yfinance as yf
import pandas as pd
import logging
import time
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def fetch_data(ticker, start_date=None, end_date=None, interval="1d"):
    """
    Fetch historical stock data from yfinance and save as CSV in project/data/raw/.
    
    Inputs:
        ticker: string
        start_date: string in 'YYYY-MM-DD' format or None
        end_date: string in 'YYYY-MM-DD' format or None
        interval: string (default '1d')
    
    Returns:
        pandas DataFrame
    """

    max_retries = 3
    attempt = 0
    df = None
    
    while attempt < max_retries:
        try:
            logging.info(f"Fetching {ticker} data (attempt {attempt + 1}/{max_retries})")
            df = yf.download(ticker, start=start_date, end=end_date, interval=interval, auto_adjust=True)
            if df.empty:
                raise ValueError("No data returned from yfinance")
            break
        except Exception as e:
            logging.error(f"Error fetching {ticker}: {e}")
            attempt += 1
            time.sleep(2)
    else:
        logging.error(f"Failed to fetch {ticker} after {max_retries} attempts")
        return None

    # Flatten MultiIndex columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Reset index to make Date a column
    index_name = df.index.name
    df.reset_index(inplace=True)
    
    # Rename datetime column to 'Date'
    if index_name and index_name in df.columns:
        df.rename(columns={index_name: "Date"}, inplace=True)
    else:
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                if col != "Date":
                    df.rename(columns={col: "Date"}, inplace=True)
                break
    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    # Ensure key columns exist
    key_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing_cols = [col for col in key_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"Missing columns in {ticker} data: {missing_cols}")
        return None

    # Drop nulls
    null_rows = df[key_cols].isnull().any(axis=1).sum()
    if null_rows > 0:
        logging.warning(f"Dropping {null_rows} rows with nulls in key columns for {ticker}")
        df.dropna(subset=key_cols, inplace=True)

    # Drop duplicates
    dup_count = df.duplicated(subset="Date").sum()
    if dup_count > 0:
        logging.info(f"Dropping {dup_count} duplicate rows for {ticker}")
        df.drop_duplicates(subset="Date", inplace=True)

    # Sort ascending and ensure column order
    df.sort_values(by="Date", inplace=True)
    df = df[key_cols]

    # Get project root: go up one level from src/ directory
    script_dir = os.path.dirname(os.path.abspath(__file__))  # src/ directory
    project_root = os.path.dirname(script_dir)  # Project root (Stock Predictor folder)
    raw_folder = os.path.join(project_root, "data", "raw")
    os.makedirs(raw_folder, exist_ok=True)

    # Save CSV in project root's data/raw folder
    csv_path = os.path.join(raw_folder, f"{ticker}.csv")
    df.to_csv(csv_path, index=False)

    # Logging
    logging.info(f"Saved {ticker} data to {csv_path}")
    logging.info(f"{ticker} data range: {df['Date'].min()} to {df['Date'].max()} | Rows: {len(df)}")
    
    return df
