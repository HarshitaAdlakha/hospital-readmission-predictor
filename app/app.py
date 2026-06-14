"""
Streamlit demo app — Hospital Readmission Predictor
Run: streamlit run app/app.py
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ------------------------------------------------------------------ #
# Page config
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


@st.cache_resource
def load_models():
    binary_path = os.path.join(MODELS_DIR, "binary_pipeline.pkl")
    multi_path = os.path.join(MODELS_DIR, "multiclass_pipeline.pkl")
    preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.pkl")
    engineer_path = os.path.join(MODELS_DIR, "feature_engineer.pkl")

    if not all(os.path.exists(p) for p in [binary_path, multi_path, preprocessor_path, engineer_path]):
        return None, None, None, None

    return (
        joblib.load(preprocessor_path),
        joblib.load(engineer_path),
        joblib.load(binary_path),
        joblib.load(multi_path),
    )


preprocessor, engineer, binary_model, multi_model = load_models()

# ------------------------------------------------------------------ #
# Sidebar — patient input form
# ------------------------------------------------------------------ #
st.sidebar.header("🩺 Patient Information")
st.sidebar.markdown("---")

with st.sidebar:
    age_raw = st.selectbox(
        "Age Group",
        ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
         "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"],
        index=6,
    )
    gender = st.selectbox("Gender", ["Male", "Female"])
    race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other"])
    time_in_hospital = st.slider("Days in Hospital", 1, 14, 4)
    num_lab_procedures = st.slider("# Lab Procedures", 1, 132, 43)
    num_procedures = st.slider("# Procedures", 0, 6, 1)
    num_medications = st.slider("# Medications", 1, 81, 16)
    number_diagnoses = st.slider("# Diagnoses", 1, 16, 7)
    number_outpatient = st.slider("# Outpatient Visits (prior year)", 0, 42, 0)
    number_emergency = st.slider("# Emergency Visits (prior year)", 0, 76, 0)
    number_inpatient = st.slider("# Inpatient Visits (prior year)", 0, 21, 0)
    insulin = st.selectbox("Insulin Dosage", ["No", "Steady", "Up", "Down"])
    diabetesMed = st.selectbox("On Diabetes Medication?", ["Yes", "No"])
    A1Cresult = st.selectbox("HbA1c Result", ["None", ">8", ">7", "Norm"])
    change = st.selectbox("Medication Changed?", ["No", "Ch"])

# ------------------------------------------------------------------ #
# Build input dataframe
# ------------------------------------------------------------------ #
input_dict = {
    "age": age_raw,
    "gender": gender,
    "race": race,
    "time_in_hospital": time_in_hospital,
    "num_lab_procedures": num_lab_procedures,
    "num_procedures": num_procedures,
    "num_medications": num_medications,
    "number_diagnoses": number_diagnoses,
    "number_outpatient": number_outpatient,
    "number_emergency": number_emergency,
    "number_inpatient": number_inpatient,
    "insulin": insulin,
    "diabetesMed": diabetesMed,
    "A1Cresult": A1Cresult,
    "change": change,
    # defaults for columns expected by preprocessor
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "medical_specialty": "InternalMedicine",
    "diag_1": "250",
    "diag_2": "250",
    "diag_3": "250",
    "max_glu_serum": "None",
    "metformin": "No",
    "repaglinide": "No",
    "nateglinide": "No",
    "chlorpropamide": "No",
    "glimepiride": "No",
    "acetohexamide": "No",
    "glipizide": "No",
    "glyburide": "No",
    "tolbutamide": "No",
    "pioglitazone": "No",
    "rosiglitazone": "No",
    "acarbose": "No",
    "miglitol": "No",
    "troglitazone": "No",
    "tolazamide": "No",
    "examide": "No",
    "citoglipton": "No",
    "glyburide-metformin": "No",
    "glipizide-metformin": "No",
    "glimepiride-pioglitazone": "No",
    "metformin-rosiglitazone": "No",
    "metformin-pioglitazone": "No",
}
input_df = pd.DataFrame([input_dict])

# ------------------------------------------------------------------ #
# Main page
# ------------------------------------------------------------------ #
st.title("🏥 Hospital Readmission Predictor")
st.markdown(
    """
    **Predicting 30-day readmission risk for diabetic patients using an ensemble ML model.**
    Adjust patient parameters in the sidebar and click **Predict** to see results.
    """
)
st.markdown("---")

col1, col2, col3 = st.columns(3)
col1.metric("Dataset", "130 US Hospitals")
col2.metric("Features", "29 Clinical")
col3.metric("Models", "Stacking + AdaBoost")

st.markdown("---")

# ------------------------------------------------------------------ #
# Predict button
# ------------------------------------------------------------------ #
if st.button("🔍 Predict Readmission", type="primary", use_container_width=True):
    if preprocessor is None:
        st.error(
            "Models not found. Please run `python train.py` first to train and save the models, "
            "then restart the app."
        )
    else:
        with st.spinner("Running prediction..."):
            try:
                X_proc = preprocessor.transform(input_df)
                X_eng = engineer.transform(X_proc)

                # align columns
                binary_cols = binary_model.feature_names_
                multi_cols = multi_model.feature_names_

                X_bin = X_eng.reindex(columns=binary_cols, fill_value=0)
                X_mul = X_eng.reindex(columns=multi_cols, fill_value=0)

                bin_proba = binary_model.predict_proba(X_bin)[0]
                mul_proba = multi_model.predict_proba(X_mul)[0]
                mul_classes = multi_model.classes_

                readmit_prob = bin_proba[1]
                mul_pred = mul_classes[np.argmax(mul_proba)]

            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.stop()

        st.markdown("### Results")
        r1, r2 = st.columns(2)

        # Binary result
        with r1:
            st.subheader("30-Day Readmission Risk")
            color = "🔴" if readmit_prob > 0.5 else "🟢"
            verdict = "HIGH RISK" if readmit_prob > 0.5 else "LOW RISK"
            st.metric(label=f"{color} {verdict}", value=f"{readmit_prob:.1%}")
            st.progress(float(readmit_prob))

        # Multiclass result
        with r2:
            st.subheader("Predicted Readmission Timing")
            label_map = {"NO": "No Readmission", "<30": "< 30 Days", ">30": "> 30 Days"}
            display_label = label_map.get(str(mul_pred), str(mul_pred))
            st.metric(label="Predicted Class", value=display_label)
            for cls, prob in zip(mul_classes, mul_proba):
                display = label_map.get(str(cls), str(cls))
                st.write(f"**{display}**: {prob:.1%}")
                st.progress(float(prob))

        # Clinical insights
        st.markdown("---")
        st.subheader("Key Risk Factors")
        factors = []
        if number_inpatient > 0:
            factors.append(f"Patient has **{number_inpatient} prior inpatient visit(s)** — recurrency is the strongest readmission predictor")
        if time_in_hospital > 7:
            factors.append(f"Long hospital stay (**{time_in_hospital} days**) indicates higher severity")
        if number_emergency > 0:
            factors.append(f"**{number_emergency} prior emergency visit(s)** suggest unstable condition")
        if num_medications > 20:
            factors.append(f"High medication count (**{num_medications}**) correlates with complex comorbidities")
        if A1Cresult == ">8":
            factors.append("HbA1c > 8 indicates **poorly controlled diabetes**")
        if change == "Ch":
            factors.append("Recent **medication change** may signal deterioration")

        if factors:
            for f in factors:
                st.markdown(f"- {f}")
        else:
            st.markdown("No major risk flags identified for this patient profile.")

# ------------------------------------------------------------------ #
# About section
# ------------------------------------------------------------------ #
with st.expander("ℹ️ About This Project"):
    st.markdown(
        """
        **Hospital Readmission Predictor** — Harshita Adlakha | Amazon ML Summer Program 2024

        This project tackles the critical healthcare problem of unnecessary hospital readmissions,
        which cost the US healthcare system over **$26 billion annually**. Using a dataset of
        **130 US hospitals (1999–2008)**, we built two ML models:

        | Task | Model | F1 Score |
        |------|-------|----------|
        | Binary (readmitted within 30 days?) | Stacking (MLP + ExtraTrees) | 0.64 |
        | Multiclass (No / <30 / >30 days) | AdaBoost | 0.60 |

        **Key engineered features:** recurrency, medication_change_ratio, lab_tests_per_med,
        patient_severity, age_times_medications

        **Source code:** [GitHub](https://github.com/HarshitaAdlakha/hospital-readmission-predictor)
        """
    )
