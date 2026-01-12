import streamlit as st
import pandas as pd
import numpy as np
import datetime
import joblib
import time
import os
import plotly.express as px

# 🔥 IMPORT TRAINING FUNCTION (NO SUBPROCESS)
from training_model import train_and_save_model

# =========================
# PATHS (CLOUD SAFE)
# =========================
MODEL_PATH = "model/car_price_model.pkl"
SCALER_PATH = "model/scaler.pkl"
FEATURES_PATH = "model/features.pkl"
DATA_PATH = "data/dataset.csv"
SELLER_DB = "data/seller_listings.csv"

# =========================
# LOAD OR TRAIN MODEL (ONCE)
# =========================
@st.cache_resource
def load_or_train_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("🚀 Training model for first deployment..."):
            model, scaler, feature_columns = train_and_save_model()
    else:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_columns = joblib.load(FEATURES_PATH)

    return model, scaler, feature_columns


model, scaler, feature_columns = load_or_train_model()
df = pd.read_csv(DATA_PATH)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Smart Used Car Pricing",
    layout="centered"
)

# =========================
# THEME TOGGLE
# =========================
theme = st.sidebar.toggle("🌗 Dark / Light Mode", value=True)

bg = "#0f2027" if theme else "#f8fafc"
card = "#1e293b" if theme else "#ffffff"
text = "white" if theme else "#111827"

st.markdown(
    f"""
    <style>
    body {{ background-color: {bg}; }}
    .card {{
        background-color: {card};
        padding: 22px;
        border-radius: 16px;
        color: {text};
        box-shadow: 0px 12px 30px rgba(0,0,0,0.25);
    }}
    .price {{
        font-size: 28px;
        font-weight: bold;
        text-align: center;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# TITLE
# =========================
st.markdown("<h1 style='text-align:center'>🚗 Smart Used Car Pricing</h1>", unsafe_allow_html=True)
st.caption("AI-powered realistic car valuation")

# =========================
# INPUTS
# =========================
car_names = sorted(df["Name"].dropna().unique().tolist())

st.markdown("<div class='card'>", unsafe_allow_html=True)

name = st.selectbox("Car Name", car_names)
year = st.number_input("Manufacturing Year", 1995, datetime.datetime.now().year, 2016)
kms = st.number_input("Kilometers Driven", 0, 500000, 45000)

fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Electric"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Owner Type", ["First", "Second", "Third", "Fourth & Above"])

mileage = st.number_input("Mileage (kmpl)", 5.0, 40.0, 18.5)
engine = st.number_input("Engine (CC)", 500, 5000, 1200)
power = st.number_input("Power (bhp)", 40, 500, 90)
seats = st.selectbox("Seats", [4, 5, 6, 7, 8])

st.markdown("</div><br>", unsafe_allow_html=True)

# =========================
# FEATURE ENGINEERING
# =========================
current_year = datetime.datetime.now().year
car_age = current_year - year
manufacturer = name.split()[0] if name else "Unknown"

base = {
    "Year": car_age,
    "Kilometers_Driven": kms,
    "Mileage": mileage,
    "Engine": engine,
    "Power": power,
    "Seats": seats
}

input_df = pd.DataFrame([base])

for col in feature_columns:
    if col.startswith("Manufacturer_"):
        input_df[col] = 1 if col == f"Manufacturer_{manufacturer}" else 0
    elif col.startswith("Fuel_Type_"):
        input_df[col] = 1 if col == f"Fuel_Type_{fuel}" else 0
    elif col.startswith("Transmission_"):
        input_df[col] = 1 if col == f"Transmission_{transmission}" else 0
    elif col.startswith("Owner_Type_"):
        input_df[col] = 1 if col == f"Owner_Type_{owner}" else 0

input_df = input_df.reindex(columns=feature_columns, fill_value=0)
input_scaled = scaler.transform(input_df)

# =========================
# PREDICTION
# =========================
if st.button("💰 Predict Price"):

    st.image(
        "https://media.giphy.com/media/26ufdipQqU2lhNA4g/giphy.gif",
        width=280
    )

    with st.spinner("Calculating best market price..."):
        time.sleep(1.5)

    preds = np.array([tree.predict(input_scaled)[0] for tree in model.estimators_])
    mean_price = preds.mean()
    low, high = np.percentile(preds, [10, 90])

    st.markdown(
        f"""
        <div class='card'>
            <div class='price'>₹ {mean_price:.2f} Lakhs</div>
            <p style='text-align:center'>
            Expected Range: ₹ {low:.2f} – ₹ {high:.2f} Lakhs
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # =========================
    # SAVE SELLER DATA
    # =========================
    record = {
        "Car": name,
        "Year": year,
        "Kilometers": kms,
        "Predicted_Price": round(mean_price, 2),
        "Date": datetime.datetime.now()
    }

    os.makedirs("data", exist_ok=True)

    if os.path.exists(SELLER_DB):
        old = pd.read_csv(SELLER_DB)
        pd.concat([old, pd.DataFrame([record])]).to_csv(SELLER_DB, index=False)
    else:
        pd.DataFrame([record]).to_csv(SELLER_DB, index=False)

    # =========================
    # PRICE VS KM CHART
    # =========================
    st.subheader("📈 Market Price vs Kilometers")

    plot_df = df[["Kilometers_Driven", "Price"]].dropna()
    if len(plot_df) > 500:
        plot_df = plot_df.sample(500)

    fig = px.scatter(
        plot_df,
        x="Kilometers_Driven",
        y="Price",
        opacity=0.45,
        title="Used Car Market Trend"
    )

    fig.add_scatter(
        x=[kms],
        y=[mean_price],
        mode="markers",
        marker=dict(size=14, color="red"),
        name="Your Car"
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# SELLER DASHBOARD
# =========================
st.subheader("🗂 Seller Dashboard")

if os.path.exists(SELLER_DB):
    history = pd.read_csv(SELLER_DB)
    st.dataframe(history, use_container_width=True)
else:
    st.info("No seller listings yet.")
    st.write("Sell your car to see it listed here!")