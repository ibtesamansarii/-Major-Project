import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# Create dataset and model
# ---------------------------
@st.cache_data
def create_and_train_model():
    np.random.seed(42)
    data = {
        "N": np.random.randint(0, 140, 500),
        "P": np.random.randint(5, 145, 500),
        "K": np.random.randint(5, 205, 500),
        "pH": np.round(np.random.uniform(4.5, 8.5, 500), 2),
        "moisture": np.round(np.random.uniform(10, 90, 500), 2)
    }

    df = pd.DataFrame(data)

    def label_fertility(row):
        if row["N"] > 100 and row["P"] > 100 and row["K"] > 150:
            return "High"
        elif row["N"] > 50 and row["P"] > 50 and row["K"] > 70:
            return "Medium"
        else:
            return "Low"

    df["fertility"] = df.apply(label_fertility, axis=1)
    df["fertility_label"] = df["fertility"].map({"Low": 0, "Medium": 1, "High": 2})

    X = df[["N", "P", "K", "pH", "moisture"]]
    y = df["fertility_label"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model

model = create_and_train_model()

# ---------------------------
# Crop Suggestion Function
# ---------------------------
def suggest_crop(n, p, k, ph, moisture):
    input_data = pd.DataFrame([[n, p, k, ph, moisture]], columns=["N", "P", "K", "pH", "moisture"])
    pred = model.predict(input_data)[0]

    fertility = {0: "Low", 1: "Medium", 2: "High"}[pred]

    crop_suggestion = {
        "Low": ["Legumes", "Barley"],
        "Medium": ["Maize", "Soybean"],
        "High": ["Wheat", "Sugarcane"]
    }

    return fertility, crop_suggestion[fertility]

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🌾 Soil Fertility & Crop Recommendation")

st.sidebar.header("Input Soil Parameters")
n = st.sidebar.slider("Nitrogen (N)", 0, 140, 60)
p = st.sidebar.slider("Phosphorus (P)", 5, 145, 60)
k = st.sidebar.slider("Potassium (K)", 5, 205, 100)
ph = st.sidebar.slider("pH", 4.5, 8.5, 6.5)
moisture = st.sidebar.slider("Moisture (%)", 10.0, 90.0, 50.0)

if st.sidebar.button("Predict Fertility & Suggest Crops"):
    fertility_level, crops = suggest_crop(n, p, k, ph, moisture)
    st.success(f"🌱 Predicted Soil Fertility: **{fertility_level}**")
    st.info(f"🌾 Recommended Crops: {', '.join(crops)}")
