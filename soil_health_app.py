import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Set page config (must be the first Streamlit command)
st.set_page_config(page_title="Soil Health Analyzer", layout="wide")

# Create synthetic dataset and train model
def create_dataset():
    np.random.seed(42)
    data = {
        "N": np.random.randint(0, 140, 500),
        "P": np.random.randint(5, 145, 500),
        "K": np.random.randint(5, 205, 500),
        "pH": np.round(np.random.uniform(4.5, 8.5, 500), 2),
        "moisture": np.round(np.random.uniform(10, 90, 500), 2),
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
    return df

df = create_dataset()
df["fertility_label"] = df["fertility"].map({"Low": 0, "Medium": 1, "High": 2})

X = df[["N", "P", "K", "pH", "moisture"]]
y = df["fertility_label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

fertility_map = {0: "Low", 1: "Medium", 2: "High"}

# App Title
st.title("🌱 Soil Health and Crop Recommendation System")

# Navigation menu
menu = st.radio("Choose Action", ["🌾 Crop Prediction", "🧪 Fertilizer Tips"])

if menu == "🌾 Crop Prediction":
    st.header("🧪 Enter Soil Parameters")

    n = st.slider("Nitrogen (N)", 0, 150, 50)
    p = st.slider("Phosphorus (P)", 0, 150, 50)
    k = st.slider("Potassium (K)", 0, 200, 50)
    ph = st.slider("pH Level", 4.0, 9.0, 6.5)
    moisture = st.slider("Moisture (%)", 0, 100, 50)

    if st.button("🔍 Predict Crop"):
        input_df = pd.DataFrame([[n, p, k, ph, moisture]], columns=["N", "P", "K", "pH", "moisture"])
        prediction = model.predict(input_df)[0]
        fertility = fertility_map[prediction]

        crop_recommendation = {
            "Low": ["Legumes", "Barley"],
            "Medium": ["Maize", "Rice"],   # Rice instead of Soybean
            "High": ["Wheat", "Sugarcane"]
        }

        st.markdown(f"""
        <div style='padding: 20px; border: 2px solid #4caf50; border-radius: 10px; background-color: #e8f5e9;'>
            <h2 style='color:#2e7d32; font-size:36px;'><b>🌿 Predicted Fertility Level: {fertility}</b></h2>
            <h3 style='color:#1565c0; font-size:30px;'><b>🌾 Recommended Crops: {', '.join(crop_recommendation[fertility])}</b></h3>
        </div>
        """, unsafe_allow_html=True)

elif menu == "🧪 Fertilizer Tips":
    st.header("📋 Fertilizer Tips Based on Fertility Level")

    tips = {
        "Low": "Use organic compost, urea, and DAP to boost NPK levels.",
        "Medium": "Apply balanced NPK fertilizers and practice crop rotation.",
        "High": "Maintain current fertilization and monitor moisture & pH regularly."
    }

    for level, tip in tips.items():
        st.markdown(f"""
        <div style='border: 1px solid #ccc; padding: 15px; border-radius: 8px; background: #f0f4c3; margin-bottom: 15px;'>
            <h3 style='color: #827717;'><b>Fertility Level: {level}</b></h3>
            <p style='font-size: 18px;'>{tip}</p>
        </div>
        """, unsafe_allow_html=True)
