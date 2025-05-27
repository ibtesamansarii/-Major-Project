import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- Prepare and train model on synthetic data ---
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
df_train["fertility_label"] = df_train["fertility"].map(fertility_map)

X_train = df_train[["N", "P", "K", "pH", "moisture"]]
y_train = df_train["fertility_label"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

reverse_fertility_map = {v: k for k, v in fertility_map.items()}

# --- Streamlit UI ---
st.title("📂 Upload CSV & Analyze Soil Health")

uploaded_file = st.file_uploader("Upload CSV file with columns: N, P, K, pH, moisture", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        required_cols = {"N", "P", "K", "pH", "moisture"}
        if not required_cols.issubset(df.columns):
            st.error(f"CSV must contain these columns: {required_cols}")
        else:
            # Predict fertility
            X = df[list(required_cols)]
            preds = model.predict(X)
            df["Fertility"] = [reverse_fertility_map[p] for p in preds]

            # Show data preview
            st.subheader("📋 Sample Data")
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

            # Summary and recommendations
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
                st.success("✅ Soil health looks good overall. Keep monitoring pH and moisture levels regularly.")

    except Exception as e:
        st.error(f"Error reading file: {e}")

else:
    st.info("Please upload a CSV file to analyze soil health.")
