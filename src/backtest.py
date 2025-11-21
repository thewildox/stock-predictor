import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def backtest(model_path, df):
    # Load trained model
    model = joblib.load(model_path)

    # Target and features
    y = df["return_1d"]
    X = df.drop(columns=["Date", "return_1d"])

    preds = []
    actuals = []

    # Walk-forward simulation (REAL backtest)
    for i in range(50, len(df)):  # start after 50 rows to avoid warm-up
        X_train = X.iloc[:i]
        y_train = y.iloc[:i]

        X_test = X.iloc[i:i+1]

        # Predict next day
        pred = model.predict(X_test)[0]

        preds.append(pred)
        actuals.append(y.iloc[i])

    preds = np.array(preds)
    actuals = np.array(actuals)

    # Metrics
    mae = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    r2 = r2_score(actuals, preds)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "preds": preds,
        "actuals": actuals
    }
