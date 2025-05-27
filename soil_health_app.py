import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

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

# ---------------------------
# Batch Prediction via File Upload
# ---------------------------
st.markdown("---")
st.subheader("📁 Batch Prediction with CSV Upload")

# Built-in real-life sample CSV
sample_csv_data = """N,P,K,pH,moisture
50,30,40,6.5,35
85,60,70,6.8,45
120,110,160,6.2,50
35,20,25,5.5,30
60,45,80,7.0,55
100,90,130,6.4,48
140,130,180,6.6,60
25,10,15,5.0,28
110,80,120,7.1,52
75,65,85,6.9,49
"""
sample_bytes = sample_csv_data.encode("utf-8")
st.download_button("⬇️ Download Sample CSV", sample_bytes, file_name="real_soil_data.csv", mime="text/csv")

uploaded_file = st.file_uploader("Upload CSV file with columns: N, P, K, pH, moisture", type=["csv"])

if uploaded_file:
    df_input = pd.read_csv(uploaded_file)
    try:
        preds = model.predict(df_input[["N", "P", "K", "pH", "moisture"]])
        fertility_labels = {0: "Low", 1: "Medium", 2: "High"}
        df_input["Fertility"] = [fertility_labels[p] for p in preds]

        crop_map = {
            "Low": ["Legumes", "Barley"],
            "Medium": ["Maize", "Soybean"],
            "High": ["Wheat", "Sugarcane"]
        }
        df_input["Recommended Crops"] = df_input["Fertility"].map(lambda x: ", ".join(crop_map[x]))

        st.dataframe(df_input)
        csv = df_input.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results", csv, "fertility_predictions.csv", "text/csv")

        # -------------------------
        # Add Charts for Uploaded Data
        # -------------------------
        st.subheader("🔍 Data Visualization")

        with st.expander("Show Nutrient Histograms"):
            fig, axs = plt.subplots(1, 3, figsize=(15, 4))
            sns.histplot(df_input["N"], kde=True, ax=axs[0], color='green').set(title="Nitrogen (N)")
            sns.histplot(df_input["P"], kde=True, ax=axs[1], color='orange').set(title="Phosphorus (P)")
            sns.histplot(df_input["K"], kde=True, ax=axs[2], color='purple').set(title="Potassium (K)")
            st.pyplot(fig)

        with st.expander("Show Fertility Distribution"):
            st.write("Soil Fertility Level Breakdown")
            pie_data = df_input["Fertility"].value_counts()
            fig2, ax2 = plt.subplots()
            ax2.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90, colors=['#7fc97f','#beaed4','#fdc086'])
            ax2.axis('equal')
            st.pyplot(fig2)

    except Exception as e:
        st.error("❌ Error in processing file: " + str(e))
