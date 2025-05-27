import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ------------------ Model Preparation ------------------

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

# Crop recommendations (rice instead of soybean)
crop_recommendation = {
    "Low": ["Legumes", "Barley"],
    "Medium": ["Maize", "Rice"],
    "High": ["Wheat", "Sugarcane"]
}

# Fertilizer tips with colors
fertilizer_tips = {
    "Low": ("Use organic compost, urea, and DAP to boost NPK levels.", "#FF6347"),       # Tomato Red
    "Medium": ("Apply balanced NPK fertilizers and practice crop rotation.", "#FFA500"),  # Orange
    "High": ("Maintain current fertilization and monitor moisture & pH regularly.", "#32CD32")  # LimeGreen
}

# ------------------ Streamlit UI ------------------

st.set_page_config(page_title="Soil Health Analyzer", layout="wide")

st.title("🌱 Soil Health and Crop Recommendation System")

menu = st.radio("Choose an Option", ["🌾 Crop Prediction", "🧪 Fertilizer Tips", "📊 Soil Health CSV Analysis"], index=0)

if menu == "🌾 Crop Prediction":
    st.header("🧪 Enter Soil Parameters for Crop Prediction")

    n = st.slider("Nitrogen (N)", 0, 150, 50)
    p = st.slider("Phosphorus (P)", 0, 150, 50)
    k = st.slider("Potassium (K)", 0, 200, 50)
    ph = st.slider("pH Level", 4.0, 9.0, 6.5)
    moisture = st.slider("Moisture (%)", 0, 100, 50)

    if st.button("🔍 Predict Crop"):
        input_df = pd.DataFrame([[n, p, k, ph, moisture]], columns=["N", "P", "K", "pH", "moisture"])
        pred = model.predict(input_df)[0]
        fertility = fertility_map[pred]

        crops = ", ".join(crop_recommendation[fertility])
        st.markdown(f"""
            <div style='padding: 20px; border-radius: 15px; background-color: #e8f5e9; border: 3px solid #4caf50;'>
                <h2 style='color:#2e7d32; font-size:42px; font-weight:700;'>🌿 Fertility Level: {fertility}</h2>
                <h3 style='color:#1565c0; font-size:36px; font-weight:700;'>🌾 Recommended Crops: {crops}</h3>
            </div>
        """, unsafe_allow_html=True)

elif menu == "🧪 Fertilizer Tips":
    st.header("📋 Get Fertilizer Tips Based on Soil Parameters")

    n = st.number_input("Nitrogen (N)", min_value=0, max_value=150, value=50)
    p = st.number_input("Phosphorus (P)", min_value=0, max_value=150, value=50)
    k = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50)
    ph = st.number_input("pH Level", min_value=4.0, max_value=9.0, value=6.5, format="%.1f")
    moisture = st.number_input("Moisture (%)", min_value=0, max_value=100, value=50)

    if st.button("Get Fertilizer Tips"):
        input_df = pd.DataFrame([[n, p, k, ph, moisture]], columns=["N", "P", "K", "pH", "moisture"])
        pred = model.predict(input_df)[0]
        fertility = fertility_map[pred]

        tip, color = fertilizer_tips[fertility]

        st.markdown(f"""
            <div style="
                background-color: {color};
                color: white;
                padding: 30px;
                border-radius: 20px;
                font-size: 28px;
                font-weight: 900;
                text-align: center;
                margin-top: 20px;
            ">
                Fertility Level: {fertility} <br><br>
                Fertilizer Tip: {tip}
            </div>
        """, unsafe_allow_html=True)

elif menu == "📊 Soil Health CSV Analysis":
    st.header("📂 Upload CSV file with columns: N, P, K, pH, moisture")

    uploaded_file = st.file_uploader("Choose CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df_csv = pd.read_csv(uploaded_file)
            required_cols = {"N", "P", "K", "pH", "moisture"}
            if not required_cols.issubset(set(df_csv.columns)):
                st.error(f"CSV must contain these columns: {', '.join(required_cols)}")
            else:
                # Predict fertility for all rows
                X_csv = df_csv[["N", "P", "K", "pH", "moisture"]]
                preds = model.predict(X_csv)
                df_csv["Fertility"] = [fertility_map[p] for p in preds]

                st.subheader("🍃 Fertility Prediction Sample")
                st.dataframe(df_csv.head())

                # Show average nutrient values
                avg_n = df_csv["N"].mean()
                avg_p = df_csv["P"].mean()
                avg_k = df_csv["K"].mean()
                avg_ph = df_csv["pH"].mean()
                avg_moisture = df_csv["moisture"].mean()

                st.subheader("🌾 Average Nutrient Levels")
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Nitrogen (N)", f"{avg_n:.2f} mg/kg")
                col2.metric("Phosphorus (P)", f"{avg_p:.2f} mg/kg")
                col3.metric("Potassium (K)", f"{avg_k:.2f} mg/kg")
                col4.metric("pH Level", f"{avg_ph:.2f}")
                col5.metric("Moisture (%)", f"{avg_moisture:.2f}")

                # Fertility distribution bar chart
                st.subheader("📊 Fertility Distribution")
                fert_counts = df_csv["Fertility"].value_counts()
                st.bar_chart(fert_counts)

                # Fertilizer tip based on majority fertility
                majority_fertility = fert_counts.idxmax()
                tip, color = fertilizer_tips[majority_fertility]

                st.markdown(f"""
                    <div style="
                        background-color: {color};
                        color: white;
                        padding: 30px;
                        border-radius: 20px;
                        font-size: 28px;
                        font-weight: 900;
                        text-align: center;
                        margin-top: 20px;
                    ">
                        Majority Soil Fertility Level: {majority_fertility} <br><br>
                        Fertilizer Tip: {tip}
                    </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error processing file: {e}")
