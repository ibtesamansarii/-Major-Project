import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Must be the first Streamlit command
st.set_page_config(page_title="Soil Health Analyzer", layout="wide")

# ------------------------------
# Model Creation
# ------------------------------
@st.cache_data
def train_model():
    np.random.seed(42)
    data = {
        "N": np.random.randint(0, 140, 500),
        "P": np.random.randint(5, 145, 500),
        "K": np.random.randint(5, 205, 500),
        "pH": np.round(np.random.uniform(4.5, 8.5, 500), 2),
        "moisture": np.round(np.random.uniform(10, 90, 500), 2)
    }
    df = pd.DataFrame(data)

    def label(row):
        if row["N"] > 100 and row["P"] > 100 and row["K"] > 150:
            return "High"
        elif row["N"] > 50 and row["P"] > 50 and row["K"] > 70:
            return "Medium"
        else:
            return "Low"

    df["fertility"] = df.apply(label, axis=1)
    df["label"] = df["fertility"].map({"Low": 0, "Medium": 1, "High": 2})

    X = df[["N", "P", "K", "pH", "moisture"]]
    y = df["label"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

fertility_map = {0: "Low", 1: "Medium", 2: "High"}
crop_recommendations = {
    "Low": ["Legumes", "Barley"],
    "Medium": ["Maize", "Soybean"],
    "High": ["Wheat", "Sugarcane"]
}

# ------------------------------
# UI Components
# ------------------------------
st.title("🌱 Interactive Soil Health Analyzer & Crop Advisor")

st.markdown("""
Welcome to the Soil Health Analyzer! Upload your soil data or enter values below to:
- 🔍 Predict soil fertility level
- 🌾 Get crop recommendations
- 📊 Visualize nutrient levels and distribution
""")

# ------------------------------
# Sidebar Inputs
# ------------------------------
st.sidebar.header("🔧 Manual Input")
n = st.sidebar.slider("Nitrogen (N)", 0, 140, 60)
p = st.sidebar.slider("Phosphorus (P)", 5, 145, 60)
k = st.sidebar.slider("Potassium (K)", 5, 205, 100)
ph = st.sidebar.slider("pH", 4.5, 8.5, 6.5)
moisture = st.sidebar.slider("Moisture (%)", 10.0, 90.0, 50.0)

if st.sidebar.button("🌾 Predict Crop Suggestion"):
    input_data = pd.DataFrame([[n, p, k, ph, moisture]], columns=["N", "P", "K", "pH", "moisture"])
    prediction = model.predict(input_data)[0]
    fertility = fertility_map[prediction]
    crops = crop_recommendations[fertility]
    st.success(f"Predicted Fertility Level: **{fertility}**")
    st.info(f"Recommended Crops: {', '.join(crops)}")

# ------------------------------
# File Upload
# ------------------------------
st.markdown("---")
st.subheader("📂 Upload CSV for Batch Prediction")
sample_csv = """N,P,K,pH,moisture
50,30,40,6.5,35
85,60,70,6.8,45
120,110,160,6.2,50
35,20,25,5.5,30
"""

st.download_button("📥 Download Sample CSV", sample_csv.encode('utf-8'), "sample_soil_data.csv")
file = st.file_uploader("Upload your CSV file here", type="csv")

if file:
    df = pd.read_csv(file)
    try:
        preds = model.predict(df[["N", "P", "K", "pH", "moisture"]])
        df["Fertility"] = [fertility_map[p] for p in preds]
        df["Recommended Crops"] = df["Fertility"].map(lambda x: ", ".join(crop_recommendations[x]))

        st.dataframe(df)
        st.download_button("📤 Download Results", df.to_csv(index=False).encode('utf-8'), "predicted_soil_data.csv")

        st.subheader("📊 Visual Analysis")
        fig, ax = plt.subplots(1, 3, figsize=(18, 4))
        sns.histplot(df["N"], kde=True, ax=ax[0], color='green').set(title="Nitrogen Distribution")
        sns.histplot(df["P"], kde=True, ax=ax[1], color='orange').set(title="Phosphorus Distribution")
        sns.histplot(df["K"], kde=True, ax=ax[2], color='purple').set(title="Potassium Distribution")
        st.pyplot(fig)

        pie_fig, pie_ax = plt.subplots()
        pie_data = df["Fertility"].value_counts()
        pie_ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
        pie_ax.axis('equal')
        st.pyplot(pie_fig)

    except Exception as e:
        st.error(f"Error in processing file: {e}")
