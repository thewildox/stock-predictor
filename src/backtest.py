import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def find_project_root(marker="data"):
    folder = os.getcwd()
    while True:
        if os.path.isdir(os.path.join(folder, marker)):
            return folder
        parent = os.path.dirname(folder)
        if parent == folder:
            raise FileNotFoundError(f"Could not find a directory containing '{marker}'")
        folder = parent

def _load_features(ticker):
    root = find_project_root("data")
    path = os.path.join(root, "data", "processed", f"{ticker}_features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Features CSV not found: {path}")

    df = pd.read_csv(path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def _infer_feature_columns(model, df):
    """
    Determines which feature columns the model expects.
    Handles XGBoost, LightGBM, Random Forest, etc.
    """
    feature_cols = None

    # sklearn models
    if hasattr(model, "feature_names_in_"):
        feature_cols = list(model.feature_names_in_)

    # xgboost models
    else:
        booster = getattr(model, "get_booster", lambda: None)()
        if booster is not None and getattr(booster, "feature_names", None):
            feature_cols = list(booster.feature_names)

    # fallback: everything except Date + label
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in ("Date", "return_1d")]

    # safety check
    missing = [m for m in feature_cols if m not in df.columns]
    if missing:
        raise ValueError(f"Model expects missing columns: {missing}")

    return feature_cols

def _signal_from_pred(pred, threshold):
    if pred > threshold:
        return 1
    if pred < -threshold:
        return -1
    return 0

def _max_drawdown(series):
    max_dd = 0
    peak = series[0]

    for x in series:
        if x > peak:
            peak = x
        dd = (peak - x) / peak
        if dd > max_dd:
            max_dd = dd

    return max_dd

def _annualized_return(total_return, n_days):
    if n_days <= 0:
        return 0
    years = n_days / 252
    return (1 + total_return) ** (1 / years) - 1

def run_backtest(
    ticker,
    model_path,
    threshold=0.002,
    initial_capital=10000,
    position_size=1.0,
    trading_cost=0.0005,
    retrain=False,
    retrain_model_callable=None
):
    logger.info(f"Running backtest for {ticker}...")

    # Load data
    df = _load_features(ticker)

    if "return_1d" not in df.columns:
        raise ValueError("Missing column 'return_1d' in features file.")

    # Load model
    model = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")

    feature_cols = _infer_feature_columns(model, df)
    X_full = df[feature_cols]

    preds, acts, signals, positions, equity = [], [], [], [], []

    portfolio = initial_capital
    position = 0

    warmup = max(50, int(len(df) * 0.05))

    # Walk-forward loop
    for i in range(warmup, len(df)):
        X_row = X_full.iloc[[i]]

        # optional walk-forward retraining
        if retrain:
            if retrain_model_callable is None:
                raise ValueError("retrain=True but no retrain_model_callable provided.")
            df_train = df.iloc[:i]
            model = retrain_model_callable(df_train)

        pred = float(model.predict(X_row)[0])
        actual = float(df["return_1d"].iloc[i])

        signal = _signal_from_pred(pred, threshold)

        prev_position = position

        # --- Trading logic ---
        if prev_position == 0 and signal != 0:  # entering
            cost = portfolio * position_size * trading_cost
            portfolio -= cost
            position = signal

        elif prev_position != 0 and signal == 0:  # exiting
            cost = portfolio * position_size * trading_cost
            portfolio -= cost
            position = 0

        elif prev_position != 0 and signal != prev_position:  # reverse
            cost = portfolio * position_size * trading_cost
            portfolio -= cost
            cost = portfolio * position_size * trading_cost
            portfolio -= cost
            position = signal

        # portfolio PnL
        if position != 0:
            portfolio *= (1 + actual * position_size)

        preds.append(pred)
        acts.append(actual)
        signals.append(signal)
        positions.append(position)
        equity.append(portfolio)

    # Convert arrays
    eq_series = pd.Series(equity, index=df["Date"].iloc[warmup:].reset_index(drop=True))
    pos_series = pd.Series(positions, index=eq_series.index)

    preds = np.array(preds)
    acts = np.array(acts)
    signals = np.array(signals)

    # Metrics
    mae = mean_absolute_error(acts, preds)
    rmse = math.sqrt(mean_squared_error(acts, preds))
    r2 = r2_score(acts, preds)

    total_return = (eq_series.iloc[-1] / initial_capital) - 1
    ann_return = _annualized_return(total_return, len(eq_series))

    daily_returns = eq_series.pct_change().dropna()
    sharpe = (
        (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        if daily_returns.std() != 0
        else np.nan
    )

    max_dd = _max_drawdown(eq_series.values)

    nonflat = signals != 0
    win_rate = (
        np.mean(np.sign(acts[nonflat]) == signals[nonflat]) if nonflat.any() else np.nan
    )

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "total_return": float(total_return),
        "annualized_return": float(ann_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
        "trades": int(nonflat.sum()),
        "start_date": str(df["Date"].iloc[warmup]),
        "end_date": str(df["Date"].iloc[-1]),
        "n_days": len(eq_series)
    }

    # Output DF
    out = df.iloc[warmup:].copy()
    out["predicted_return"] = preds
    out["signal"] = signals
    out["position"] = positions
    out["equity"] = eq_series.values
    out["actual_return"] = acts

    return {
        "equity": eq_series,
        "positions": pos_series,
        "preds": preds,
        "actuals": acts,
        "metrics": metrics,
        "df": out
    }

