"""
Model training pipeline.
Trains XGBoost and Random Forest, evaluates both, saves the best model.
Run: python ml/train.py
"""
import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

MODEL_DIR = "ml/models"
DATA_PATH = "ml/data/salaries.csv"


def load_and_preprocess(path: str):
    df = pd.read_csv(path)

    le_title = LabelEncoder()
    le_location = LabelEncoder()
    le_education = LabelEncoder()

    df["job_title_enc"] = le_title.fit_transform(df["job_title"])
    df["location_enc"] = le_location.fit_transform(df["location"])
    df["education_enc"] = le_education.fit_transform(df["education"])

    encoders = {
        "job_title": le_title,
        "location": le_location,
        "education": le_education,
    }

    feature_cols = (
        ["job_title_enc", "location_enc", "education_enc", "years_experience", "num_skills"]
        + [c for c in df.columns if c.startswith("skill_")]
    )

    X = df[feature_cols]
    y = df["salary"]

    return X, y, encoders, feature_cols


def evaluate_model(model, X_test, y_test, name: str) -> dict:
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"\n{name} Results:")
    print(f"  MAE:  ${mae:,.0f}")
    print(f"  RMSE: ${rmse:,.0f}")
    print(f"  R²:   {r2:.4f}")
    return {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 4)}


def train():
    print("Loading data...")
    X, y, encoders, feature_cols = load_and_preprocess(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")

    # Train Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_metrics = evaluate_model(rf, X_test, y_test, "Random Forest")

    # Train XGBoost
    print("\nTraining XGBoost...")
    xgb = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    xgb_metrics = evaluate_model(xgb, X_test, y_test, "XGBoost")

    # Pick best model
    best_model = xgb if xgb_metrics["r2"] >= rf_metrics["r2"] else rf
    best_name = "XGBoost" if xgb_metrics["r2"] >= rf_metrics["r2"] else "RandomForest"
    best_metrics = xgb_metrics if best_name == "XGBoost" else rf_metrics

    print(f"\nBest model: {best_name} (R²={best_metrics['r2']})")

    # Save everything
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, f"{MODEL_DIR}/model.pkl")
    joblib.dump(encoders, f"{MODEL_DIR}/encoders.pkl")
    joblib.dump(feature_cols, f"{MODEL_DIR}/feature_cols.pkl")

    metadata = {
        "model_name": best_name,
        "metrics": best_metrics,
        "rf_metrics": rf_metrics,
        "xgb_metrics": xgb_metrics,
        "feature_cols": feature_cols,
        "training_samples": len(X_train),
    }
    with open(f"{MODEL_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel saved to {MODEL_DIR}/")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    train()
