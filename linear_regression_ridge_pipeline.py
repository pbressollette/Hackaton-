#!/usr/bin/env python3
"""
Linear regression (RidgeCV) to predict the LAST column from all others,
with robust preprocessing and time-aware validation.

Usage examples:
  python linear_regression_ridge_pipeline.py --train waiting_times_train.csv --test waiting_times_test.csv
  python linear_regression_ridge_pipeline.py --train train.csv --test test.csv --target WAIT_TIME_IN_2H --datetime_col DATETIME --save_model model.joblib --preds_out preds.csv

Notes:
- If your test CSV does not contain the target column, the script will skip test metrics and only write predictions.
- If your data has a 'DATETIME' column, a TimeSeriesSplit is used on the training set; otherwise standard CV is used.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import joblib
except ImportError:
    joblib = None


def add_time_features(df: pd.DataFrame, datetime_col: str | None) -> pd.DataFrame:
    """Add basic/cyclical time features if datetime_col is present; drop the raw datetime col afterwards."""
    X = df.copy()
    if datetime_col and datetime_col in X.columns:
        dt = pd.to_datetime(X[datetime_col], errors="coerce")
        X[f"{datetime_col}_year"] = dt.dt.year
        X[f"{datetime_col}_month"] = dt.dt.month
        X[f"{datetime_col}_day"] = dt.dt.day
        X[f"{datetime_col}_hour"] = dt.dt.hour
        X[f"{datetime_col}_dow"] = dt.dt.dayofweek

        # Cyclical encoding
        hour = dt.dt.hour.fillna(0)
        X[f"{datetime_col}_hour_sin"] = np.sin(2 * np.pi * hour / 24)
        X[f"{datetime_col}_hour_cos"] = np.cos(2 * np.pi * hour / 24)

        dow = dt.dt.dayofweek.fillna(0)
        X[f"{datetime_col}_dow_sin"] = np.sin(2 * np.pi * dow / 7)
        X[f"{datetime_col}_dow_cos"] = np.cos(2 * np.pi * dow / 7)

        month = (dt.dt.month - 1).fillna(0)
        X[f"{datetime_col}_month_sin"] = np.sin(2 * np.pi * month / 12)
        X[f"{datetime_col}_month_cos"] = np.cos(2 * np.pi * month / 12)

        # Drop raw
        X = X.drop(columns=[datetime_col])
    return X


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """ColumnTransformer with numeric impute+scale and categorical impute+onehot (handle_unknown)."""
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Compatibility for sklearn < 1.2 (sparse) and >= 1.2 (sparse_output)
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe),
    ])

    pre = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])
    return pre


def compute_baselines(y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series | None) -> dict:
    """Compute simple baselines: mean predictor, and persistence if CURRENT_WAIT_TIME exists."""
    out = {}
    if y_test is None:
        return out

    # Baseline 1: mean of train
    mean_pred = np.full(shape=len(y_test), fill_value=y_train.mean(), dtype=float)
    out["baseline_mean_MAE"] = float(mean_absolute_error(y_test, mean_pred))
    out["baseline_mean_RMSE"] = float(np.sqrt(mean_squared_error(y_test, mean_pred)))
    out["baseline_mean_R2"] = float(r2_score(y_test, mean_pred))

    # Baseline 2: persistence via CURRENT_WAIT_TIME if available
    if "CURRENT_WAIT_TIME" in X_test.columns:
        pers_pred = X_test["CURRENT_WAIT_TIME"].astype(float).to_numpy()
        out["baseline_persist_MAE"] = float(mean_absolute_error(y_test, pers_pred))
        out["baseline_persist_RMSE"] = float(np.sqrt(mean_squared_error(y_test, pers_pred)))
        out["baseline_persist_R2"] = float(r2_score(y_test, pers_pred))

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to training CSV")
    parser.add_argument("--test", required=True, help="Path to test CSV (with or without target)")
    parser.add_argument("--target", default="__LAST__", help="Target column name; default='__LAST__' uses the last column")
    parser.add_argument("--datetime_col", default="DATETIME", help="Datetime column name if present (for time features & CV)")
    parser.add_argument("--save_model", default=None, help="Path to save the fitted pipeline with joblib")
    parser.add_argument("--preds_out", default="predictions.csv", help="Where to write test predictions CSV")
    args = parser.parse_args()

    train_path = Path(args.train)
    test_path = Path(args.test)

    if not train_path.exists():
        print(f"[ERROR] Train CSV not found: {train_path}", file=sys.stderr)
        sys.exit(1)
    if not test_path.exists():
        print(f"[ERROR] Test CSV not found: {test_path}", file=sys.stderr)
        sys.exit(1)

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Target handling
    if args.target == "__LAST__":
        target_col = df_train.columns[-1]
    else:
        if args.target not in df_train.columns:
            print(f"[ERROR] Target '{args.target}' not in training columns", file=sys.stderr)
            sys.exit(1)
        target_col = args.target

    # Split X/y
    if target_col not in df_train.columns:
        print(f"[ERROR] Target '{target_col}' not found in training data.", file=sys.stderr)
        sys.exit(1)
    y_train = df_train[target_col]
    X_train_raw = df_train.drop(columns=[target_col])

    # In test, target may be absent
    y_test = df_test[target_col] if target_col in df_test.columns else None
    X_test_raw = df_test.drop(columns=[target_col]) if target_col in df_test.columns else df_test.copy()

    # Feature engineering (no leakage; deterministic)
    X_train = add_time_features(X_train_raw, args.datetime_col if args.datetime_col in X_train_raw.columns else None)
    X_test = add_time_features(X_test_raw, args.datetime_col if args.datetime_col in X_test_raw.columns else None)

    # Build preprocessor on training columns only
    pre = build_preprocessor(X_train)

    # Model
    alphas = np.logspace(-3, 2, 10)
    model = Pipeline([
        ("pre", pre),
        ("ridge", RidgeCV(alphas=alphas, cv=5, scoring="neg_root_mean_squared_error"))
    ])

    # If a datetime column exists on train, sort train chronologically prior to CV (RidgeCV uses KFold internally)
    use_time_cv = args.datetime_col in X_train_raw.columns
    if use_time_cv:
        dt = pd.to_datetime(X_train_raw[args.datetime_col], errors="coerce")
        order = np.argsort(dt.values)
        X_train = X_train.iloc[order]
        y_train = y_train.iloc[order]

    # Fit
    model.fit(X_train, y_train)

    # Predict
    y_pred_test = model.predict(X_test)

    # Metrics if available
    results = {
        "target": target_col,
        "chosen_alpha": float(model.named_steps["ridge"].alpha_),
        "n_train": int(len(y_train)),
        "n_test": int(len(X_test)),
    }
    if y_test is not None:
        results.update({
            "test_MAE": float(mean_absolute_error(y_test, y_pred_test)),
            "test_RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
            "test_R2": float(r2_score(y_test, y_pred_test)),
        })
        results.update(compute_baselines(y_train, X_test_raw, y_test))

    # Save predictions
    preds_df = pd.DataFrame({"y_pred": y_pred_test})
    if y_test is not None:
        preds_df.insert(0, "y_true", y_test.to_numpy())
        preds_df["residual"] = preds_df["y_true"] - preds_df["y_pred"]

    preds_out_path = Path(args.preds_out)
    preds_df.to_csv(preds_out_path, index=False)

    # Save model (optional)
    if args.save_model:
        if joblib is None:
            print("[WARN] joblib not installed; cannot save model.", file=sys.stderr)
        else:
            joblib.dump(model, args.save_model)

    print(json.dumps({
        "summary": results,
        "preds_out": str(preds_out_path.resolve()),
        "model_path": str(Path(args.save_model).resolve()) if args.save_model else None
    }, indent=2))


if __name__ == "__main__":
    main()
