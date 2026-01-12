import pandas as pd
import numpy as np
import datetime
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# =========================
# CONFIG
# =========================
DATA_PATH = r"E:\used_car_app\data\dataset.csv"
MODEL_PATH = r"E:\used_car_app\model\car_price_model.pkl"
SCALER_PATH = r"E:\used_car_app\model\scaler.pkl"
FEATURES_PATH = r"E:\used_car_app\model\features.pkl"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)

# Drop index column if exists
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

# Drop missing target
df.dropna(subset=["Price"], inplace=True)

# =========================
# FEATURE ENGINEERING
# =========================

# Manufacturer from Name
df["Manufacturer"] = df["Name"].apply(lambda x: x.split()[0])

# Drop columns not used
df.drop(["Name", "Location"], axis=1, inplace=True)

# Convert Year → Car Age
current_year = datetime.datetime.now().year
df["Year"] = current_year - df["Year"]

# Clean Mileage
df["Mileage"] = df["Mileage"].str.extract(r"(\d+\.?\d*)").astype(float)

# Clean Engine
df["Engine"] = df["Engine"].str.extract(r"(\d+)").astype(float)

# Clean Power
df["Power"] = df["Power"].str.extract(r"(\d+\.?\d*)").astype(float)

# Fill numeric missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Drop New_Price (as per notebook logic)
df.drop("New_Price", axis=1, inplace=True)

# =========================
# ONE-HOT ENCODING
# =========================
df = pd.get_dummies(
    df,
    columns=["Manufacturer", "Fuel_Type", "Transmission", "Owner_Type"],
    drop_first=True
)

# =========================
# FEATURES & TARGET
# =========================
X = df.drop("Price", axis=1)
y = df["Price"]

# Save feature names (VERY IMPORTANT)
joblib.dump(X.columns.tolist(), FEATURES_PATH)

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# =========================
# SCALING
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, SCALER_PATH)

# =========================
# MODEL TRAINING
# =========================
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))

# =========================
# SAVE MODEL
# =========================
joblib.dump(model, MODEL_PATH)

print("✅ Model trained and saved successfully")
