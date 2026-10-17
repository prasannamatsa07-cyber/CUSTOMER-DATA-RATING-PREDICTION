"""
app.py
======
Streamlit web application for the E-Commerce Customer Behaviour
Customer Rating Predictor.

The trained pipeline (models/model.pkl) is loaded once at startup.
Users fill in the feature form and receive a predicted Customer Rating (1–5).

Run locally:
    streamlit run app.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Rating Predictor",
    page_icon="🛒",
    layout="centered",
)

# ─────────────────────────────────────────────
# Load model artefact (cached)
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")


@st.cache_resource(show_spinner="Loading model …")
def load_artefact(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def model_exists(path: str) -> bool:
    return os.path.isfile(path)


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("🛒 E-Commerce Customer Rating Predictor")
st.markdown(
    """
    Enter the details of a customer order below and click **Predict Rating**
    to see what rating (1 – 5 ⭐) the customer is likely to give.
    """
)

if not model_exists(MODEL_PATH):
    st.error(
        "⚠️ Model file not found.  "
        "Please run `python train_model.py` first to generate `models/model.pkl`."
    )
    st.stop()

artefact = load_artefact(MODEL_PATH)

pipeline      = artefact["pipeline"]
feature_cols  = artefact["feature_cols"]
num_cols      = artefact["num_cols"]
cat_cols      = artefact["cat_cols"]
cat_uniques   = artefact["cat_uniques"]
num_ranges    = artefact["num_ranges"]
best_model    = artefact["best_model"]
test_accuracy = artefact["test_accuracy"]
results       = artefact["results"]

# ─────────────────────────────────────────────
# Model info sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Model Information")
    st.metric("Best Model", best_model)
    st.metric("Test Accuracy", f"{test_accuracy * 100:.2f}%")

    st.subheader("All Models Comparison")
    rows = []
    for m, v in results.items():
        rows.append({
            "Model": m,
            "CV Accuracy": f"{v['cv_mean'] * 100:.2f}%",
            "Test Accuracy": f"{v['test_acc'] * 100:.2f}%",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    st.markdown("---")
    st.caption("Dataset: E-Commerce Customer Behaviour")
    st.caption("Target: Customer Rating (1–5)")

# ─────────────────────────────────────────────
# Helper: render a labelled input for each feature
# ─────────────────────────────────────────────
FRIENDLY_LABELS = {
    "Age":                       "Age (years)",
    "Gender":                    "Gender",
    "City":                      "City",
    "Product_Category":          "Product Category",
    "Unit_Price":                "Unit Price (₺)",
    "Quantity":                  "Quantity",
    "Discount_Amount":           "Discount Amount (₺)",
    "Total_Amount":              "Total Amount (₺)",
    "Payment_Method":            "Payment Method",
    "Device_Type":               "Device Type",
    "Session_Duration_Minutes":  "Session Duration (minutes)",
    "Pages_Viewed":              "Pages Viewed",
    "Is_Returning_Customer":     "Returning Customer?",
    "Delivery_Time_Days":        "Delivery Time (days)",
}

STAR_COLORS = {1: "🔴", 2: "🟠", 3: "🟡", 4: "🟢", 5: "🌟"}

# ─────────────────────────────────────────────
# Input form
# ─────────────────────────────────────────────
st.subheader("📝 Order & Customer Details")

with st.form("prediction_form"):
    user_input = {}

    # Lay out inputs in two columns
    col1, col2 = st.columns(2)
    left_features  = feature_cols[: len(feature_cols) // 2 + len(feature_cols) % 2]
    right_features = feature_cols[len(feature_cols) // 2 + len(feature_cols) % 2 :]

    def render_input(col_widget, feat):
        label = FRIENDLY_LABELS.get(feat, feat.replace("_", " "))

        if feat == "Is_Returning_Customer":
            val = col_widget.selectbox(label, options=["Yes", "No"], index=0)
            return 1 if val == "Yes" else 0

        if feat in cat_uniques:
            options = cat_uniques[feat]
            return col_widget.selectbox(label, options=options)

        # Numeric
        info = num_ranges.get(feat, {"min": 0.0, "max": 1000.0, "mean": 0.0})
        mn, mx, mu = info["min"], info["max"], info["mean"]

        if feat in ("Age", "Quantity", "Pages_Viewed",
                    "Session_Duration_Minutes", "Delivery_Time_Days"):
            return col_widget.number_input(
                label,
                min_value=int(mn),
                max_value=int(mx),
                value=int(mu),
                step=1,
            )
        else:
            return col_widget.number_input(
                label,
                min_value=float(mn),
                max_value=float(mx),
                value=round(float(mu), 2),
                step=0.01,
                format="%.2f",
            )

    for feat in left_features:
        user_input[feat] = render_input(col1, feat)

    for feat in right_features:
        user_input[feat] = render_input(col2, feat)

    submitted = st.form_submit_button("🔮 Predict Rating", use_container_width=True)

# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────
if submitted:
    # Build a DataFrame in the exact column order the pipeline expects
    input_df = pd.DataFrame([user_input])[feature_cols]

    prediction  = pipeline.predict(input_df)[0]
    probas      = pipeline.predict_proba(input_df)[0]
    class_order = pipeline.classes_

    icon = STAR_COLORS.get(int(prediction), "⭐")

    st.markdown("---")
    st.subheader("🎯 Prediction Result")

    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.metric(
            label="Predicted Customer Rating",
            value=f"{icon} {prediction} / 5",
        )

    with res_col2:
        st.write("**Probability per rating class:**")
        proba_df = pd.DataFrame({
            "Rating": [f"{STAR_COLORS.get(int(c), '⭐')} {c}" for c in class_order],
            "Probability": [f"{p * 100:.1f}%" for p in probas],
        }).set_index("Rating")
        st.dataframe(proba_df, use_container_width=True)

    # Interpretation banner
    if prediction >= 4:
        st.success(f"✅ Great news! The model predicts a **high rating of {prediction}** — "
                   "the customer is likely satisfied.")
    elif prediction == 3:
        st.warning(f"⚠️ The model predicts a **neutral rating of {prediction}** — "
                   "there's room for improvement.")
    else:
        st.error(f"❌ The model predicts a **low rating of {prediction}** — "
                 "action may be needed to improve the customer experience.")

    with st.expander("🔍 View raw input sent to model"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)
