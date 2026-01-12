import os
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = "data/dataset.csv"
MODEL_DIR = "model"


def train_and_save_model():
    # =========================
    # SETUP
    # =========================
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    # =========================
    # CLEAN PRICE (🔥 CRITICAL FIX)
    # =========================
    df["Price"] = (
        df["Price"]
        .astype(str)
        .str.replace("Lakhs", "", regex=False)
        .str.replace("Lakh", "", regex=False)
        .str.strip()
    )
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["Price"])

    # =========================
    # CLEAN NUMERIC FEATURES
    # =========================
    df["Mileage"] = (
        df["Mileage"].astype(str).str.extract(r"(\d+\.?\d*)").astype(float)
    )
    df["Engine"] = (
        df["Engine"].astype(str).str.extract(r"(\d+)").astype(float)
    )
    df["Power"] = (
        df["Power"].astype(str).str.extract(r"(\d+\.?\d*)").astype(float)
    )

    df.fillna(df.median(numeric_only=True), inplace=True)

    # =========================
    # FEATURE ENGINEERING
    # =========================
    df["Manufacturer"] = df["Name"].str.split().str[0]

    # Drop raw text columns
    df.drop(["Name", "Location"], axis=1, inplace=True, errors="ignore")

    # Convert Year → Car Age
    CURRENT_YEAR = 2026
    df["Year"] = CURRENT_YEAR - df["Year"]

    # =========================
    # ONE-HOT ENCODING
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

    # 🔒 FINAL SAFETY CONVERSION
    X = X.apply(pd.to_numeric, errors="coerce")
    X.fillna(0, inplace=True)

    # =========================
    # SCALE FEATURES
    # =========================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # =========================
    # TRAIN MODEL
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
    joblib.dump(model, os.path.join(MODEL_DIR, "car_price_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, "features.pkl"))

    return model, scaler, X.columns.tolist()
