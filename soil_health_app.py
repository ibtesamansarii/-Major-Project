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

# Train the model
df = create_dataset()
df["fertility_label"] = df["fertility"].map({"Low": 0, "Medium": 1, "High": 2})

X = df[["N", "P", "K", "pH", "moisture"]]
y = df["fertility_label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

fertility_map = {0: "Low", 1: "Medium", 2: "High"}

crop_recommendation = {
    "Low": ["Legumes", "Barley"],
    "Medium": ["Maize", "Rice"],   # Rice instead of Soybean
    "High": ["Wheat", "Sugarcane"]
}

# ---- UI ----
st.title("🌱 Soil Health and Crop Recommendation System")

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

        st.markdown(f"""
        <div style='padding: 20px; border: 3px solid #4caf50; border-radius: 15px; background-color: #e8f5e9; margin-top: 20px;'>
            <h1 style='color:#2e7d32; font-size:48px; font-weight:bold;'>🌿 Predicted Fertility Level: {fertility}</h1>
            <h2 style='color:#1565c0; font-size:36px; font-weight:bold;'>🌾 Recommended Crops: {', '.join(crop_recommendation[fertility])}</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.header("📂 Or Upload CSV for Batch Soil Health Analysis")

    uploaded_file = st.file_uploader("Upload CSV (columns: N, P, K, pH, moisture)", type=["csv"])

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)

            required_cols = {"N", "P", "K", "pH", "moisture"}
            if not required_cols.issubset(df_upload.columns):
                st.error(f"CSV missing required columns: {required_cols}")
            else:
                X_upload = df_upload[list(required_cols)]
                df_upload["fertility_label"] = model.predict(X_upload)
                df_upload["Fertility"] = df_upload["fertility_label"].map(fertility_map)
                df_upload["Recommended Crops"] = df_upload["Fertility"].map(crop_recommendation)

                # Show fertility distribution chart
                st.subheader("📊 Fertility Level Distribution")
                fertility_counts = df_upload["Fertility"].value_counts().rename_axis('Fertility').reset_index(name='Counts')
                st.bar_chart(fertility_counts.set_index("Fertility"))

                st.subheader("Sample Predictions")
                sample_df = df_upload.head(10).copy()
                sample_df["Recommended Crops"] = sample_df["Recommended Crops"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
                st.dataframe(sample_df[["N", "P", "K", "pH", "moisture", "Fertility", "Recommended Crops"]])

        except Exception as e:
            st.error(f"Error processing CSV file: {e}")

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
            <h3 style='color: #827717; font-weight: bold;'>Fertility Level: {level}</h3>
            <p style='font-size: 18px;'>{tip}</p>
        </div>
        """, unsafe_allow_html=True)

# --------

# Optional: Button to download sample CSV for testing
st.sidebar.header("Sample CSV Data")
if st.sidebar.button("Download Sample CSV"):
    sample_data = {
        "N": [45, 110, 60, 80],
        "P": [30, 120, 55, 90],
        "K": [40, 160, 80, 130],
        "pH": [6.2, 7.1, 6.5, 6.9],
        "moisture": [50, 70, 55, 65]
    }
    sample_df = pd.DataFrame(sample_data)
    csv = sample_df.to_csv(index=False)
    st.sidebar.download_button("Download CSV", csv, "sample_soil_data.csv", "text/csv")
