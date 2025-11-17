# Stock Predictor

A Python-based stock prediction system that fetches historical stock data, engineers features, and prepares data for machine learning models.

## Features

- **Data Ingestion**: Fetches historical stock data from Yahoo Finance using `yfinance`
- **Feature Engineering**: Computes technical indicators including:
  - Returns (1-day, 3-day, 7-day)
  - Moving averages (SMA_5, SMA_10)
  - Volatility measures (vol_5, vol_10)
  - Technical indicators (RSI, MACD)
  - Day-of-week features
- **Data Processing**: Automated data cleaning and feature engineering pipeline

## Project Structure

```
Stock Predictor/
├── src/                 # Source code modules
│   ├── ingest.py       # Data fetching functionality
│   ├── features.py     # Feature engineering
│   ├── train_models.py # Model training (to be implemented)
│   ├── backtest.py     # Backtesting (to be implemented)
│   └── utils.py        # Utility functions
├── notebooks/          # Jupyter notebooks for testing
│   └── test_ingest.ipynb
├── data/               # Data storage
│   ├── raw/           # Raw stock data (CSV files)
│   └── processed/     # Processed features (CSV files)
├── tests/             # Unit tests
└── venv/              # Virtual environment (not tracked in git)
```

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd "Stock Predictor"
```

### 2. Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

### 3. Install dependencies

```bash
pip install pandas yfinance numpy
```

For Jupyter notebook support:
```bash
pip install jupyter notebook ipykernel
```

### 4. Setup Jupyter kernel (optional)

```bash
python -m ipykernel install --user --name stock-predictor --display-name "Python (Stock Predictor)"
```

## Usage

### Fetch stock data

```python
from src.ingest import fetch_data

# Fetch TSLA data
df = fetch_data("TSLA")
```

### Build features

```python
from src.features import build_features

# Build features from raw data
df_features = build_features(df, ticker="TSLA")
```

### Using Jupyter Notebooks

1. Open `notebooks/test_ingest.ipynb`
2. Select the kernel "Python (Stock Predictor)"
3. Run the cells

## Data Storage

- **Raw data**: Saved to `data/raw/{ticker}.csv`
- **Processed features**: Saved to `data/processed/{ticker}_features.csv`

## Dependencies

- Python 3.9+
- pandas
- yfinance
- numpy

## License

Private project - All rights reserved

