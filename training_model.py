# train_model.py
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

    # ----- CLEAN DATA (KEEP SIMPLE & SAFE) -----
    df = df.dropna(subset=["Price"])

    df["Mileage"] = df["Mileage"].astype(str).str.extract(r"(\d+\.?\d*)").astype(float)
    df["Engine"] = df["Engine"].astype(str).str.extract(r"(\d+)").astype(float)
    df["Power"] = df["Power"].astype(str).str.extract(r"(\d+\.?\d*)").astype(float)

    df.fillna(df.median(numeric_only=True), inplace=True)

    # Manufacturer
    df["Manufacturer"] = df["Name"].str.split().str[0]

    df.drop(["Name", "Location"], axis=1, inplace=True)

    # Car age
    current_year = 2026
    df["Year"] = current_year - df["Year"]

    # One-hot encode
    df = pd.get_dummies(
        df,
        columns=["Manufacturer", "Fuel_Type", "Transmission", "Owner_Type"],
        drop_first=True
    )

    X = df.drop("Price", axis=1)
    y = df["Price"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled, y)

    joblib.dump(model, f"{MODEL_DIR}/car_price_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(X.columns.tolist(), f"{MODEL_DIR}/features.pkl")

    return model, scaler, X.columns.tolist()


if __name__ == "__main__":
    train_and_save_model()
