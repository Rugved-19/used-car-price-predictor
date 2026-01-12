import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = "data/dataset.csv"
MODEL_DIR = "model"


def train_and_save_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    # =========================
    # BASIC CLEANING
    # =========================
    df = df.dropna(subset=["Price"])

    df["Mileage"] = df["Mileage"].astype(str).str.extract(r"(\d+\.?\d*)").astype(float)
    df["Engine"] = df["Engine"].astype(str).str.extract(r"(\d+)").astype(float)
    df["Power"] = df["Power"].astype(str).str.extract(r"(\d+\.?\d*)").astype(float)

    df.fillna(df.median(numeric_only=True), inplace=True)

    # =========================
    # FEATURE ENGINEERING
    # =========================
    df["Manufacturer"] = df["Name"].str.split().str[0]

    # Drop text columns (VERY IMPORTANT)
    df.drop(["Name", "Location"], axis=1, inplace=True, errors="ignore")

    # Convert year to car age
    CURRENT_YEAR = 2026
    df["Year"] = CURRENT_YEAR - df["Year"]

    # =========================
    # ONE-HOT ENCODING (KEY FIX)
    # =========================
    df = pd.get_dummies(
        df,
        columns=["Manufacturer", "Fuel_Type", "Transmission", "Owner_Type"],
        drop_first=True
    )

    # =========================
    # SPLIT FEATURES / TARGET
    # =========================
    X = df.drop("Price", axis=1)
    y = df["Price"]

    # 🔒 FINAL SAFETY CHECK (CRITICAL)
    X = X.apply(pd.to_numeric)

    # =========================
    # SCALE
    # =========================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # =========================
    # MODEL
    # =========================
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_scaled, y)

    # =========================
    # SAVE ARTIFACTS
    # =========================
    joblib.dump(model, f"{MODEL_DIR}/car_price_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(X.columns.tolist(), f"{MODEL_DIR}/features.pkl")

    return model, scaler, X.columns.tolist()


if __name__ == "__main__":
    train_and_save_model()
