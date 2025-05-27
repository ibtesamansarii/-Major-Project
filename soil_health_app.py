import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Set page config (must be first)
st.set_page_config(page_title="🌱 Soil Health & Crop Prediction", layout="wide")

# --- Data & Model Setup ---

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

df_train = create_dataset()
fertility_map = {"Low": 0, "Medium": 1, "High": 2}
reverse_fertility_map = {v: k for k, v in fertility_map.items()}
df_train["fertility_label"] = df_train["fertility"].map(fertility_map)

X_train = df_train[["N", "P", "K", "pH", "moisture"]]
y_train = df_train["fertility_label"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Crop suggestions based on fertility
crop_recommendation = {
    "Low": ["Legumes", "Barley"],
    "Medium": ["Maize", "Rice"],
    "High": ["Wheat", "Sugarcane"],
}

# Fertilizer tips by fertility level with bright colors
fertilizer_tips = {
    "Low": {
        "tip": "Use organic compost, urea, and DAP to boost NPK levels.",
        "color": "#ffcccc",  # light red
    },
    "Medium": {
        "tip": "Apply balanced NPK fertilizers and practice crop rotation.",
        "color": "#fff3b0",  # light yellow
    },
    "High": {
        "tip": "Maintain current fertilization and monitor moisture & pH regularly.",
        "color": "#ccffcc",  # light green
    },
}

# --- UI ---

# Custom CSS for capsule-shaped buttons
st.markdown(
    """
    <style>
    .capsule-btn {
        display: inline-block;
        padding: 12px 36px;
        margin: 5px;
        border-radius: 50px;
        font-size: 20px;
        font-weight: 600;
        cursor: pointer;
        user-select: none;
        border: 2px solid #4CAF50;
        color: #4CAF50;
        background-color: white;
        transition: all 0.3s ease;
    }
    .capsule-btn:hover {
        background-color: #4CAF50;
        color: white;
    }
    .capsule-btn-selected {
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Session state for tab selection
if "selected_tab" not in st.session_state:
    st.session_state.selected_tab = "Crop Prediction"

def select_tab(tab):
    st.session_state.selected_tab = tab

# Tabs
cols = st.columns([1, 1, 1])
with cols[0]:
    if st.button("🌾 Crop Prediction", key="tab_crop"):
        select_tab("Crop Prediction")
with cols[1]:
    if st.button("📂 CSV Upload & Soil Analysis", key="tab_csv"):
        select_tab("CSV Upload")
with cols[2]:
    if st.button("🧪 Fertilizer Tips", key="tab_fertilizer"):
        select_tab("Fertilizer Tips")

st.markdown("---")

# --- Crop Prediction Tab ---
if st.session_state.selected_tab == "Crop Prediction":
    st.header("🌾 Crop Prediction from Soil Parameters")

    n = st.slider("Nitrogen (N) [mg/kg]", 0, 150, 50)
    p = st.slider("Phosphorus (P) [mg/kg]", 0, 150, 50)
    k = st.slider("Potassium (K) [mg/kg]", 0, 200, 50)
    ph = st.slider("pH Level", 4.0, 9.0, 6.5, step=0.1)
    moisture = st.slider("Moisture (%)", 0, 100, 50)

    if st.button("🔍 Predict Crop"):
        input_df = pd.DataFrame([[n, p, k, ph, moisture]], columns=["N", "P", "K", "pH", "moisture"])
        pred = model.predict(input_df)[0]
        fertility = reverse_fertility_map[pred]
        crops = crop_recommendation[fertility]

        st.markdown(
            f"""
            <div style='
                background-color: #e8f5e9; 
                border: 3px solid #4caf50; 
                border-radius: 15px; 
                padding: 25px; 
                margin-top: 20px; 
                text-align: center;'>
                <h1 style='color:#2e7d32; font-weight: 900; font-size: 48px;'>🌿 Fertility Level: {fertility}</h1>
                <h2 style='color:#1565c0; font-weight: 800; font-size: 36px;'>🌾 Recommended Crops: {", ".join(crops)}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

# --- CSV Upload & Soil Analysis Tab ---
elif st.session_state.selected_tab == "CSV Upload":
    st.header("📂 Upload CSV & Analyze Soil Health")

    uploaded_file = st.file_uploader(
        "Upload CSV file with columns: N, P, K, pH, moisture", type=["csv"]
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            required_cols = {"N", "P", "K", "pH", "moisture"}
            if not required_cols.issubset(df.columns):
                st.error(f"CSV must contain these columns: {required_cols}")
            else:
                # Predict fertility for each row
                X = df[list(required_cols)]
                preds = model.predict(X)
                df["Fertility"] = [reverse_fertility_map[p] for p in preds]

                # Show sample data
                st.subheader("📋 Sample Data Preview")
                st.dataframe(df.head())

                # Show averages
                st.subheader("📊 Average Nutrient Levels")
                avg_n = df["N"].mean()
                avg_p = df["P"].mean()
                avg_k = df["K"].mean()
                avg_ph = df["pH"].mean()
                avg_moisture = df["moisture"].mean()

                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Avg Nitrogen (N)", f"{avg_n:.1f} mg/kg")
                col2.metric("Avg Phosphorus (P)", f"{avg_p:.1f} mg/kg")
                col3.metric("Avg Potassium (K)", f"{avg_k:.1f} mg/kg")
                col4.metric("Avg pH Level", f"{avg_ph:.2f}")
                col5.metric("Avg Moisture (%)", f"{avg_moisture:.1f}")

                # Fertility distribution chart
                st.subheader("🌾 Fertility Distribution")
                fert_counts = df["Fertility"].value_counts().rename_axis('Fertility').reset_index(name='Counts')
                st.bar_chart(fert_counts.set_index("Fertility"))

                # Summary & recommendations
                st.subheader("📝 Soil Health Summary")
                total = len(df)
                high = (df["Fertility"] == "High").sum()
                medium = (df["Fertility"] == "Medium").sum()
                low = (df["Fertility"] == "Low").sum()

                st.markdown(f"""
                - 🟢 **{high} samples ({(high/total)*100:.1f}%)** have **High** fertility.
                - 🟠 **{medium} samples ({(medium/total)*100:.1f}%)** have **Medium** fertility.
                - 🔴 **{low} samples ({(low/total)*100:.1f}%)** have **Low** fertility.
                """)

                st.subheader("🌱 Recommendations")
                if low / total > 0.3:
                    st.warning("🚨 High number of low fertility samples. Consider adding organic compost or nitrogen-rich fertilizers.")
                elif medium / total > 0.5:
                    st.info("⚠️ Majority samples are medium fertility. Moderate soil enrichment is recommended.")
                else:
                    st.success("✅ Soil health looks good overall. Keep monitoring pH and moisture regularly.")

        except Exception as e:
            st.error(f"Error reading file: {e}")

    else:
        st.info("Please upload a CSV file to analyze soil health.")

# --- Fertilizer Tips Tab ---
elif st.session_state.selected_tab == "Fertilizer Tips":
    st.header("🧪 Fertilizer Tips Based on Fertility Level")

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
