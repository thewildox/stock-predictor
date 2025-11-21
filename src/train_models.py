import os
import joblib
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def find_project_root(marker_folder="data"):
    """Walk upward from cwd until a folder containing `marker_folder` is found."""
    folder = os.getcwd()
    while True:
        if os.path.isdir(os.path.join(folder, marker_folder)):
            return folder
        parent = os.path.dirname(folder)
        if parent == folder:
            raise FileNotFoundError(f"Could not find a parent folder containing '{marker_folder}'.")
        folder = parent


def load_features_for_ticker(ticker):
    """Load processed features CSV for ticker from project data/processed."""
    project_root = find_project_root("data")
    path = os.path.join(project_root, "data", "processed", f"{ticker}_features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed features not found: {path}")
    df = pd.read_csv(path, parse_dates=["Date"])
    logger.info(f"Loaded features: {path} | rows={len(df)}")
    return df, project_root


def _prepare_rf_data(df):
    """Prepare X,y for Random Forest using your RF feature set."""
    target = "return_1d"
    features = df.columns.drop(["Date", "return_1d"])
    X = df[features].copy()
    y = df[target].copy()
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    return X_train, X_test, y_train, y_test


def _prepare_xgb_data(df):
    """Prepare X,y for XGBoost using your XGB feature set."""
    X = df.drop(columns=[
        "Date", "Close", "Open", "High", "Low", "Volume",
        "return_1d", "return_3d", "return_7d"
    ]).copy()
    y = df["return_1d"].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    return X_train, X_test, y_train, y_test


def train_and_save_models(ticker: str):
    """
    Train Random Forest and XGBoost for given ticker, save models under <project_root>/models/,
    and save test-set predictions to <project_root>/data/processed/{ticker}_predictions.csv.

    Returns:
        results (dict): {
            "RandomForest": {"model": rf_model, "mae":..., "rmse":..., "r2":..., "model_path":...},
            "XGBoost": {...}
        }
    """
    df, project_root = load_features_for_ticker(ticker)

    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    results = {}

    # --- Random Forest ---
    X_train_rf, X_test_rf, y_train_rf, y_test_rf = _prepare_rf_data(df)

    rf_model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
    rf_model.fit(X_train_rf, y_train_rf)

    preds_rf = rf_model.predict(X_test_rf)
    mae_rf = mean_absolute_error(y_test_rf, preds_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test_rf, preds_rf))
    r2_rf = r2_score(y_test_rf, preds_rf)

    rf_path = os.path.join(models_dir, f"{ticker}_rf_model.pkl")
    joblib.dump(rf_model, rf_path)
    logger.info(f"Saved Random Forest model -> {rf_path}")

    results["RandomForest"] = {
        "model": rf_model,
        "mae": mae_rf,
        "rmse": rmse_rf,
        "r2": r2_rf,
        "model_path": rf_path
    }

    # Save RF predictions into processed CSV (aligned to test index)
    rf_pred_df = df.iloc[len(X_train_rf):].copy().reset_index(drop=True)
    rf_pred_df["predicted_return_rf"] = preds_rf
    rf_pred_path = os.path.join(project_root, "data", "processed", f"{ticker}_predictions_rf.csv")
    rf_pred_df.to_csv(rf_pred_path, index=False)
    logger.info(f"Saved RF predictions -> {rf_pred_path}")

    # --- XGBoost ---
    X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = _prepare_xgb_data(df)

    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_model.fit(X_train_xgb, y_train_xgb)

    preds_xgb = xgb_model.predict(X_test_xgb)
    mae_xgb = mean_absolute_error(y_test_xgb, preds_xgb)
    mse_xgb = mean_squared_error(y_test_xgb, preds_xgb)
    r2_xgb = r2_score(y_test_xgb, preds_xgb)

    xgb_path = os.path.join(models_dir, f"{ticker}_xgb_model.pkl")
    joblib.dump(xgb_model, xgb_path)
    logger.info(f"Saved XGBoost model -> {xgb_path}")

    results["XGBoost"] = {
        "model": xgb_model,
        "mae": mae_xgb,
        "mse": mse_xgb,
        "r2": r2_xgb,
        "model_path": xgb_path
    }

    # Save XGB predictions
    # Note: X_test_xgb has a different split method; we will align indices by using index from original df
    # Build a DataFrame aligned with the test portion:
    test_start_idx = len(df) - len(X_test_xgb)
    xgb_pred_df = df.iloc[test_start_idx:].copy().reset_index(drop=True)
    xgb_pred_df["predicted_return_xgb"] = preds_xgb
    xgb_pred_path = os.path.join(project_root, "data", "processed", f"{ticker}_predictions_xgb.csv")
    xgb_pred_df.to_csv(xgb_pred_path, index=False)
    logger.info(f"Saved XGB predictions -> {xgb_pred_path}")

    return results


if __name__ == "__main__":
    # quick CLI usage
    res = train_and_save_models("TSLA")
    logger.info("Training results:")
    for k, v in res.items():
        logger.info(f"{k}: MAE={v['mae']:.6f} RMSE={v.get('rmse', np.nan):.6f} R2={v['r2']:.4f} saved_at={v['model_path']}")
