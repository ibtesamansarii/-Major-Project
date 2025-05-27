import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page config at the top
st.set_page_config(page_title="Soil Health Analyzer", layout="wide")

st.title("🌱 Soil Health and Crop Recommendation System")

# Fertility mapping
fertility_map = {0: "Low", 1: "Medium", 2: "High"}

# Sample training data generator
@st.cache_data
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
    df["fertility_label"] = df["fertility"].map({"Low": 0, "Medium": 1, "High": 2})
    return df

# Train model
df = create_dataset()
X = df[["N", "P", "K", "pH", "moisture"]]
y = df["fertility_label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------- CSV Analysis Section ----------

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
            X_input = df_upload[["N", "P", "K", "pH", "moisture"]]
            df_upload["Fertility"] = model.predict(X_input)
            df_upload["Fertility"] = df_upload["Fertility"].map(fertility_map)

            # Display basic stats
            st.subheader("📈 Nutrient Averages")
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg Nitrogen (N)", f"{df_upload['N'].mean():.1f} mg/kg")
            col2.metric("Avg Phosphorus (P)", f"{df_upload['P'].mean():.1f} mg/kg")
            col3.metric("Avg Potassium (K)", f"{df_upload['K'].mean():.1f} mg/kg")

            # Fertility distribution
            st.subheader("🌾 Fertility Level Distribution")
            st.bar_chart(df_upload["Fertility"].value_counts())

            # Summary insights
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

            # Recommendations
            st.subheader("🌱 General Recommendations")
            if low > 30:
                st.warning("🚨 A large portion of samples are low in fertility. Consider adding organic compost or nitrogen-rich fertilizers.")
            elif medium > 50:
                st.info("⚠️ Most samples are in the medium range. You may need moderate soil enrichment.")
            else:
                st.success("✅ Soil health looks generally good. Maintain current practices and monitor pH/moisture.")

    except Exception as e:
        st.error(f"Failed to analyze file: {e}")
