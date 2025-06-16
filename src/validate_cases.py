import streamlit as st
import pandas as pd

DATA_PATH = "../data/predictions/predictions.csv"
OUTPUT_FEEDBACK = "../data/new_feedback.csv"

st.title("🕵️ Validación de Casos Sospechosos - AML")

# Load model predictions
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()

# Show top N suspicious cases
st.subheader("🔍 Casos más relevantes para revisar")
threshold = st.slider("Filtrar por score mínimo (probabilidad ROS):", 0.0, 1.0, 0.8, 0.01)

filtered = df[df["prediction_score"] >= threshold].copy()
st.write(f"Mostrando {len(filtered)} casos sospechosos")

# Add selection for feedback
feedback_options = ["", "ROS Confirmado", "No ROS"]
filtered["feedback"] = ""

for i in range(len(filtered)):
    client_row = filtered.iloc[i]
    with st.expander(f"Cliente: {client_row['client_id']} | Score: {client_row['prediction_score']:.2f}"):
        st.dataframe(client_row.drop(["feedback"]).to_frame().T)
        feedback = st.selectbox(f"👉 Clasificación manual del cliente {client_row['client_id']}", feedback_options, key=f"fb_{i}")
        filtered.at[client_row.name, "feedback"] = feedback

# Download feedback
if st.button("💾 Exportar feedback CSV"):
    export_df = filtered[filtered["feedback"].isin(["ROS Confirmado", "No ROS"])][["client_id", "feedback"]].copy()
    export_df["label_ros"] = export_df["feedback"].map({"ROS Confirmado": 1, "No ROS": 0})
    export_df[["client_id", "label_ros"]].to_csv(OUTPUT_FEEDBACK, index=False)
    st.success(f"✅ Feedback exportado a: {OUTPUT_FEEDBACK}")
