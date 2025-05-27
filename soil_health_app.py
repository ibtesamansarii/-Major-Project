import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Must be first Streamlit command
st.set_page_config(page_title="Soil Health Analyzer", layout="wide")

# ------------------------------
# Train or load model
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
    "Medium": ["Maize", "Rice"],       # Soybean replaced by Rice here
    "High": ["Wheat", "Sugarcane"]
}

# ------------------------------
# UI
# ------------------------------

st.title("🌱 Soil Health & Crop Suggestion Analyzer")

with st.sidebar:
    st.header("Enter Soil Parameters")
    n = st.slider("Nitrogen (N)", 0, 140, 60)
    p = st.slider("Phosphorus (P)", 5, 145, 60)
    k = st.slider("Potassium (K)", 5, 205, 100)
    ph = st.slider("pH", 4.5, 8.5, 6.5)
    moisture = st.slider("Moisture (%)", 10.0, 90.0, 50.0)

if st.button("Predict Fertility and Suggest Crops"):

    input_df = pd.DataFrame([[n, p, k, ph, moisture]], columns=["N", "P", "K", "pH", "moisture"])
    pred = model.predict(input_df)[0]
    fertility = fertility_map[pred]
    crops = crop_recommendations[fertility]

    # Colored badge for fertility level
    color_map = {"Low": "#FF6347", "Medium": "#FFA500", "High": "#32CD32"}  # tomato, orange, limegreen
    st.markdown(f"### Fertility Level: <span style='color:{color_map[fertility]}; font-weight:bold'>{fertility}</span>", unsafe_allow_html=True)

    # Show crop suggestions as tags/buttons
    st.markdown("**Recommended Crops:**")
    cols = st.columns(len(crops))
    for i, crop in enumerate(crops):
        cols[i].button(crop)

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

    # Label bars with values
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    st.pyplot(fig)

    # Show input pH and moisture info
    st.markdown("---")
    st.markdown(f"**Soil pH:** {ph} (ideal range: 6.0 - 7.5)")
    st.markdown(f"**Soil Moisture:** {moisture}% (optimal varies by crop)")
