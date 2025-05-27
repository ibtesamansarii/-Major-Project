import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Page config must be first
st.set_page_config(page_title="Soil Health Analyzer", layout="wide")

# ------------------------------
# Model training (cached)
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
    "Medium": ["Maize", "Rice"],  # Rice instead of Soybean
    "High": ["Wheat", "Sugarcane"]
}

# ------------------------------
# Functions
# ------------------------------
def predict_fertility_and_crops(n, p, k, ph, moisture):
    input_df = pd.DataFrame([[n, p, k, ph, moisture]], columns=["N", "P", "K", "pH", "moisture"])
    pred = model.predict(input_df)[0]
    fertility = fertility_map[pred]
    crops = crop_recommendations[fertility]
    return fertility, crops

def style_crop_tags(crops):
    tags_html = ""
    colors = {"Legumes": "#6ab04c", "Barley": "#f39c12", "Maize": "#2980b9", "Rice": "#27ae60", "Wheat": "#d35400", "Sugarcane": "#8e44ad"}
    for crop in crops:
        color = colors.get(crop, "#34495e")
        tags_html += f"<span style='background-color:{color}; color:white; padding:6px 15px; margin:5px; font-weight:bold; border-radius:12px; font-size:20px; display:inline-block'>{crop}</span>"
    return tags_html

# ------------------------------
# UI Layout
# ------------------------------
st.title("🌱 Soil Health & Crop Suggestion Analyzer")

with st.sidebar:
    st.header("Enter Soil Parameters")
    n = st.slider("Nitrogen (N)", 0, 140, 60)
    p = st.slider("Phosphorus (P)", 5, 145, 60)
    k = st.slider("Potassium (K)", 5, 205, 100)
    ph = st.slider("pH", 4.5, 8.5, 6.5)
    moisture = st.slider("Moisture (%)", 10.0, 90.0, 50.0)

st.markdown("---")

# Prediction button & result
if st.button("Predict Fertility and Suggest Crops"):

    fertility, crops = predict_fertility_and_crops(n, p, k, ph, moisture)

    color_map = {"Low": "#FF6347", "Medium": "#FFA500", "High": "#32CD32"}
    st.markdown(f"### Fertility Level: <span style='color:{color_map[fertility]}; font-weight:bold; font-size:28px'>{fertility}</span>", unsafe_allow_html=True)

    st.markdown("**Recommended Crops:**")
    st.markdown(style_crop_tags(crops), unsafe_allow_html=True)

    # Nutrient bar chart for input values
    st.markdown("---")
    st.markdown("### Soil Nutrient Levels")
    thresholds = {"N": 100, "P": 100, "K": 150}
    nutrients = ["N", "P", "K"]
    values = [n, p, k]

    fig, ax = plt.subplots()
    bars = ax.bar(nutrients, values, color='cornflowerblue', label='Input Values')
    ax.axhline(thresholds["N"], color='r', linestyle='--', label='N Threshold')
    ax.axhline(thresholds["P"], color='g', linestyle='--', label='P Threshold')
    ax.axhline(thresholds["K"], color='purple', linestyle='--', label='K Threshold')
    ax.set_ylim(0, max(values + list(thresholds.values())) + 20)
    ax.set_ylabel('Amount (mg/kg)')
    ax.set_title("Nutrient Levels vs. Fertility Thresholds")
    ax.legend()

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    st.pyplot(fig)

    st.markdown("---")
    st.markdown(f"**Soil pH:** {ph} (ideal range: 6.0 - 7.5)")
    st.markdown(f"**Soil Moisture:** {moisture}% (optimal varies by crop)")

# ------------------------------
# Upload CSV for batch prediction
# ------------------------------
st.markdown("---")
st.header("📂 Upload CSV file for batch soil fertility prediction")

uploaded_file = st.file_uploader("Upload a CSV file with columns: N, P, K, pH, moisture", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        required_cols = {"N", "P", "K", "pH", "moisture"}
        if not required_cols.issubset(set(df.columns)):
            st.error(f"CSV must contain columns: {required_cols}")
        else:
            X = df[["N", "P", "K", "pH", "moisture"]]
            preds = model.predict(X)
            df["Fertility Level"] = [fertility_map[p] for p in preds]
            df["Recommended Crops"] = df["Fertility Level"].map(crop_recommendations)

            # Show dataframe with crop tags styled
            def format_crops(crops):
                return ", ".join(crops)

            df["Recommended Crops"] = df["Recommended Crops"].apply(format_crops)

            st.dataframe(df)

    except Exception as e:
        st.error(f"Error processing file: {e}")
