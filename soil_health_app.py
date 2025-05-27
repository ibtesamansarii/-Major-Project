import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Set Streamlit page config (must be first Streamlit command)
st.set_page_config(page_title="Soil Health Analyzer", layout="wide")

# Generate synthetic dataset for training (you can replace with real data)
def create_dataset():
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
    return df

# Load and train model
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

# Manual Input Prediction
st.header("🧪 Predict Soil Fertility and Get Crop Suggestions")
n = st.slider("Nitrogen (N)", 0, 150, 50)
p = st.slider("Phosphorus (P)", 0, 150, 50)
k = st.slider("Potassium (K)", 0, 200, 50)
ph = st.slider("pH Level", 4.0, 9.0, 6.5)
moisture = st.slider("Moisture (%)", 0, 100, 50)

if st.button("Predict Crop Suggestion"):
    input_df = pd.DataFrame([[n, p, k, ph, moisture]], columns=["N", "P", "K", "pH", "moisture"])
    prediction = model.predict(input_df)[0]
    fertility = fertility_map[prediction]

    crop_recommendation = {
        "Low": ["Legumes", "Barley"],
        "Medium": ["Maize", "Rice"],
        "High": ["Wheat", "Sugarcane"]
    }

    st.markdown(f"""
        <div style='font-size:22px; font-weight:bold;'>
            🧬 Predicted Fertility Level: <span style='color:blue'>{fertility}</span><br>
            🌾 Recommended Crops: <span style='color:green'>{', '.join(crop_recommendation[fertility])}</span>
        </div>
    """, unsafe_allow_html=True)

# Analysis from CSV
st.markdown("---")
st.header("📊 Soil Health Analysis from CSV")

uploaded_analysis_file = st.file_uploader("Upload CSV for Soil Health Analysis", type=["csv"], key="analyze_csv")

if uploaded_analysis_file is not None:
    try:
        df_upload = pd.read_csv(uploaded_analysis_file)
        required_columns = {"N", "P", "K", "pH", "moisture"}

        if not required_columns.issubset(df_upload.columns):
            st.error("❌ CSV must contain columns: N, P, K, pH, moisture")
        else:
            # Predict fertility levels
            X_input = df_upload[["N", "P", "K", "pH", "moisture"]]
            df_upload["Fertility"] = model.predict(X_input)
            df_upload["Fertility"] = df_upload["Fertility"].map(fertility_map)

            # 📈 Average Nutrient Levels
            st.subheader("📈 Average Nutrient Levels")
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg Nitrogen (N)", f"{df_upload['N'].mean():.1f} mg/kg")
            col2.metric("Avg Phosphorus (P)", f"{df_upload['P'].mean():.1f} mg/kg")
            col3.metric("Avg Potassium (K)", f"{df_upload['K'].mean():.1f} mg/kg")

            # 📊 Fertility Distribution
            st.subheader("🌾 Fertility Distribution")
            fertility_dist = df_upload["Fertility"].value_counts()
            st.bar_chart(fertility_dist)

            # 📝 Summary
            st.subheader("📝 Soil Health Summary")
            total = len(df_upload)
            summary = df_upload["Fertility"].value_counts(normalize=True).mul(100).round(1)
            high = summary.get("High", 0)
            medium = summary.get("Medium", 0)
            low = summary.get("Low", 0)

            st.markdown(f"""
            <div style='font-size:18px'>
            🟢 <b>{high}%</b> of samples have <b>High</b> fertility.<br>
            🟠 <b>{medium}%</b> of samples have <b>Medium</b> fertility.<br>
            🔴 <b>{low}%</b> of samples have <b>Low</b> fertility.
            </div>
            """, unsafe_allow_html=True)

            # 🌱 Recommendations
            st.subheader("🌱 Soil Fertility Recommendations")
            if low > 30:
                st.warning("🚨 Many samples are low in fertility. Consider organic compost, urea, or DAP fertilizers.")
            elif medium > 50:
                st.info("⚠️ Moderate fertility observed. Balanced fertilizers or crop rotation may help.")
            else:
                st.success("✅ Soil fertility is healthy overall. Maintain current practices and monitor regularly.")

            # Optional: Show table
            with st.expander("🔍 See Detailed Data"):
                st.dataframe(df_upload)

    except Exception as e:
        st.error(f"❌ Failed to process the file: {e}")
