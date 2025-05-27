import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- Data & Model Preparation ---

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

crop_recommendation = {
    "Low": ["Legumes", "Barley"],
    "Medium": ["Maize", "Rice"],  # Rice instead of Soybean
    "High": ["Wheat", "Sugarcane"]
}

fertilizer_tips = {
    "Low": "Use organic compost, urea, and DAP to boost NPK levels.",
    "Medium": "Apply balanced NPK fertilizers and practice crop rotation.",
    "High": "Maintain current fertilization and monitor moisture & pH regularly."
}

# --- Streamlit UI ---

st.set_page_config(page_title="Soil Health Analyzer", layout="wide")

st.title("🌱 Soil Health and Crop Recommendation System")

# Create two capsule-shaped buttons horizontally

col1, col2 = st.columns(2)

with col1:
    crop_button = st.button("🌾 Crop Prediction", key="crop_btn", 
                           help="Input soil values or upload CSV to predict crop recommendations",
                           args=None,
                           kwargs=None)
with col2:
    fert_button = st.button("🧪 Fertilizer Tips", key="fert_btn", 
                           help="View fertilizer tips based on fertility levels")

# Use session state to track selection
if "selected_option" not in st.session_state:
    st.session_state.selected_option = None

if crop_button:
    st.session_state.selected_option = "crop"

if fert_button:
    st.session_state.selected_option = "fertilizer"

if st.session_state.selected_option == "crop":
    st.header("🌿 Crop Prediction")

    st.markdown(
        """
        Enter soil parameters manually or upload a CSV file for batch analysis.
        CSV must contain columns: N, P, K, pH, moisture
        """
    )

    # Manual input
    with st.form("manual_input_form"):
        n = st.slider("Nitrogen (N)", 0, 150, 50)
        p = st.slider("Phosphorus (P)", 0, 150, 50)
        k = st.slider("Potassium (K)", 0, 200, 50)
        ph = st.slider("pH Level", 4.0, 9.0, 6.5)
        moisture = st.slider("Moisture (%)", 0, 100, 50)

        submit_manual = st.form_submit_button("🔍 Predict Crop for Manual Input")

    if submit_manual:
        input_df = pd.DataFrame([[n, p, k, ph, moisture]], columns=["N", "P", "K", "pH", "moisture"])
        pred_label = model.predict(input_df)[0]
        fertility = fertility_map[pred_label]
        crops = crop_recommendation[fertility]

        st.markdown(
            f"""
            <div style='padding: 20px; border-radius: 12px; background-color: #d7f0d2; border: 2px solid #4caf50; margin-top:20px;'>
                <h2 style='color:#2e7d32; font-size:40px; font-weight:bold;'>🌿 Fertility Level: {fertility}</h2>
                <h3 style='color:#1565c0; font-size:34px; font-weight:bold;'>🌾 Recommended Crops: {', '.join(crops)}</h3>
            </div>
            """, unsafe_allow_html=True
        )

    st.markdown("---")

    # CSV Upload for batch prediction
    uploaded_file = st.file_uploader("Upload CSV for batch soil health analysis", type=["csv"])
    if uploaded_file:
        try:
            df_upload = pd.read_csv(uploaded_file)

            required_cols = {"N", "P", "K", "pH", "moisture"}
            if not required_cols.issubset(df_upload.columns):
                st.error(f"CSV missing columns. Required columns: {required_cols}")
            else:
                X_upload = df_upload[list(required_cols)]
                df_upload["fertility_label"] = model.predict(X_upload)
                df_upload["Fertility"] = df_upload["fertility_label"].map(fertility_map)
                df_upload["Recommended Crops"] = df_upload["Fertility"].map(crop_recommendation)

                # Show summary stats
                st.subheader("📊 Soil Fertility Summary from CSV")
                fertility_counts = df_upload["Fertility"].value_counts().rename_axis('Fertility').reset_index(name='Counts')
                st.bar_chart(fertility_counts.set_index("Fertility"))

                st.subheader("Sample Predictions")
                # Show first 10 samples with fertility and recommendations
                display_df = df_upload.head(10)[["N", "P", "K", "pH", "moisture", "Fertility", "Recommended Crops"]]
                # Format recommended crops as comma-separated string
                display_df["Recommended Crops"] = display_df["Recommended Crops"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
                st.dataframe(display_df)

        except Exception as e:
            st.error(f"Error processing file: {e}")

elif st.session_state.selected_option == "fertilizer":
    st.header("🧪 Fertilizer Tips Based on Soil Fertility Levels")

    for level in ["Low", "Medium", "High"]:
        tip = fertilizer_tips[level]
        st.markdown(
            f"""
            <div style='border-radius: 10px; background-color: #fff3e0; padding: 20px; margin-bottom: 15px; border-left: 8px solid #ffa726;'>
                <h2 style='font-weight:bold; font-size:32px; color:#ef6c00;'>Fertility Level: {level}</h2>
                <p style='font-size:20px;'>{tip}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

else:
    # Initial screen - show capsule style options bigger and centered

    st.markdown(
        """
        <style>
        .capsule-btn {
            display: inline-block;
            padding: 20px 60px;
            margin: 20px 30px;
            font-size: 28px;
            font-weight: 700;
            color: white;
            background-color: #4caf50;
            border-radius: 50px;
            cursor: pointer;
            user-select: none;
            text-align: center;
            width: 250px;
            transition: background-color 0.3s ease;
        }
        .capsule-btn:hover {
            background-color: #388e3c;
        }
        .capsule-container {
            text-align: center;
            margin-top: 150px;
        }
        </style>
        <div class="capsule-container">
            <div id="crop" class="capsule-btn" onclick="window.parent.postMessage({func:'selectOption', option:'crop'}, '*')">🌾 Crop Prediction</div>
            <div id="fert" class="capsule-btn" onclick="window.parent.postMessage({func:'selectOption', option:'fertilizer'}, '*')">🧪 Fertilizer Tips</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# JS to catch clicks from the capsules and update Streamlit session state
# Streamlit currently doesn't support onclick handlers directly,
# so we simulate by listening to postMessage events
st.markdown(
    """
    <script>
    window.addEventListener('message', event => {
        if (event.data.func === 'selectOption') {
            const option = event.data.option;
            window.parent.document.querySelector('iframe').contentWindow.postMessage({func:'setSessionState', option: option}, '*');
        }
    });
    </script>
    """,
    unsafe_allow_html=True,
)

# Listen to messages and update session state in Streamlit (hacky but works)
# This part may require additional integration or can be replaced by st.radio or st.selectbox as fallback.
