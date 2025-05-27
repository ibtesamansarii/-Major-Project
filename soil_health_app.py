import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- Setup ---
st.set_page_config(page_title="Soil Health Analyzer", layout="wide")

# --- Create synthetic dataset and train model ---
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
reverse_fertility_map = {v: k for k, v in fertility_map.items()}

crop_recommendation = {
    "Low": ["Legumes", "Barley"],
    "Medium": ["Maize", "Rice"],
    "High": ["Wheat", "Sugarcane"]
}
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Assuming you have your trained model & fertility_map loaded as `model` and `fertility_map`

st.title("🧪 Fertilizer Tips Based on Soil Parameters")

n = st.number_input("Nitrogen (N)", min_value=0, max_value=150, value=50)
p = st.number_input("Phosphorus (P)", min_value=0, max_value=150, value=50)
k = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50)
ph = st.number_input("pH Level", min_value=4.0, max_value=9.0, value=6.5, format="%.1f")
moisture = st.number_input("Moisture (%)", min_value=0, max_value=100, value=50)

if st.button("Get Fertilizer Tips"):
    input_df = pd.DataFrame([[n, p, k, ph, moisture]], columns=["N", "P", "K", "pH", "moisture"])
    pred = model.predict(input_df)[0]
    fertility = fertility_map[pred]

    tips = {
        "Low": "Use organic compost, urea, and DAP to boost NPK levels.",
        "Medium": "Apply balanced NPK fertilizers and practice crop rotation.",
        "High": "Maintain current fertilization and monitor moisture & pH regularly."
    }

    colors = {
        "Low": "#FF6347",       # Tomato red (high contrast)
        "Medium": "#FFA500",    # Orange
        "High": "#32CD32",      # LimeGreen
    }

    st.markdown(f"""
        <div style="
            background-color: {colors[fertility]};
            color: white;
            padding: 25px;
            border-radius: 15px;
            font-size: 28px;
            font-weight: 900;
            text-align: center;
            margin-top: 20px;
        ">
            Fertility Level: {fertility} <br><br>
            Fertilizer Tip: {tips[fertility]}
        </div>
    """, unsafe_allow_html=True)

fertilizer_tips = {
    "Low": {
        "tip": "Use organic compost, urea, and DAP to boost NPK levels.",
        "color": "#ffcccc",
    },
    "Medium": {
        "tip": "Apply balanced NPK fertilizers and practice crop rotation.",
        "color": "#fff4cc",
    },
    "High": {
        "tip": "Maintain current fertilization and monitor moisture & pH regularly.",
        "color": "#ccffcc",
    },
}

# --- Sidebar navigation ---
st.sidebar.title("Navigation")
selected_tab = st.sidebar.radio("Go to", ["Crop Prediction", "CSV Upload & Analysis", "Fertilizer Tips"])

# --- Crop Prediction ---
if selected_tab == "Crop Prediction":
    st.title("🌾 Crop Prediction")

    st.markdown(
        "<style> \
        .big-button > button {font-size: 20px; padding: 15px 25px;} \
        </style>",
        unsafe_allow_html=True,
    )

    n = st.slider("Nitrogen (N)", 0, 150, 50)
    p = st.slider("Phosphorus (P)", 0, 150, 50)
    k = st.slider("Potassium (K)", 0, 200, 50)
    ph = st.slider("pH Level", 4.0, 9.0, 6.5, 0.1)
    moisture = st.slider("Moisture (%)", 0, 100, 50)

    if st.button("🔍 Predict Crop", key="predict_crop"):
        input_df = pd.DataFrame([[n, p, k, ph, moisture]], columns=["N", "P", "K", "pH", "moisture"])
        prediction = model.predict(input_df)[0]
        fertility = fertility_map[prediction]

        st.markdown(
            f"""
            <div style='
                background-color: #e0f7fa;
                border-radius: 15px;
                padding: 30px;
                margin-top: 20px;
                text-align: center;
                border: 3px solid #00796b;
            '>
                <h1 style='color: #004d40; font-weight: 900; font-size: 48px;'>🌿 Fertility Level: {fertility}</h1>
                <h2 style='color: #00796b; font-weight: 700; font-size: 36px;'>Recommended Crops: {", ".join(crop_recommendation[fertility])}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

# --- CSV Upload & Soil Health Analysis ---
elif selected_tab == "CSV Upload & Analysis":
    st.title("📊 Upload CSV for Soil Health Analysis")

    uploaded_file = st.file_uploader("Upload your soil data CSV file here", type=["csv"])

    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file)
            required_cols = ["N", "P", "K", "pH", "moisture"]

            if not set(required_cols).issubset(df_uploaded.columns):
                st.error(f"CSV file must contain columns: {required_cols}")
            else:
                X_uploaded = df_uploaded[required_cols]  # fixed order for model input
                preds = model.predict(X_uploaded)
                df_uploaded["Fertility"] = [fertility_map[p] for p in preds]

                st.subheader("Soil Fertility Results")
                st.dataframe(df_uploaded)

                # Show average nutrient values
                avg_n = df_uploaded["N"].mean()
                avg_p = df_uploaded["P"].mean()
                avg_k = df_uploaded["K"].mean()

                col1, col2, col3 = st.columns(3)
                col1.metric("Average Nitrogen (N)", f"{avg_n:.2f} mg/kg")
                col2.metric("Average Phosphorus (P)", f"{avg_p:.2f} mg/kg")
                col3.metric("Average Potassium (K)", f"{avg_k:.2f} mg/kg")

                # Fertility distribution bar chart
                st.subheader("Fertility Distribution")
                dist = df_uploaded["Fertility"].value_counts()
                st.bar_chart(dist)

                # Summary
                total = len(df_uploaded)
                count_high = (df_uploaded["Fertility"] == "High").sum()
                count_medium = (df_uploaded["Fertility"] == "Medium").sum()
                count_low = (df_uploaded["Fertility"] == "Low").sum()

                st.markdown(
                    f"""
                    - 🟢 **High Fertility:** {count_high} samples ({(count_high/total)*100:.1f}%)
                    - 🟠 **Medium Fertility:** {count_medium} samples ({(count_medium/total)*100:.1f}%)
                    - 🔴 **Low Fertility:** {count_low} samples ({(count_low/total)*100:.1f}%)
                    """
                )

                # Recommendations based on fertility levels
                st.subheader("Soil Health Recommendations")
                if count_low / total > 0.3:
                    st.warning("🚨 A large portion of samples have Low fertility. Consider adding organic compost or nitrogen-rich fertilizers.")
                elif count_medium / total > 0.5:
                    st.info("⚠️ Majority of samples have Medium fertility. Moderate soil enrichment recommended.")
                else:
                    st.success("✅ Soil health looks good overall. Keep monitoring pH and moisture regularly.")

        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.info("Please upload a CSV file to analyze soil health.")

# --- Fertilizer Tips ---
elif selected_tab == "Fertilizer Tips":
    st.title("🧪 Fertilizer Tips Based on Fertility Level")

    for level, data in fertilizer_tips.items():
        st.markdown(
            f"""
            <div style='
                background-color: {data['color']};
                padding: 20px;
                margin-bottom: 20px;
                border-radius: 15px;
                border: 2px solid #888;
            '>
                <h2 style='color: #333; font-weight: 900;'>{level} Fertility</h2>
                <p style='font-size: 18px;'>{data['tip']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
