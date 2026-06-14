"""
Feature engineering for Hospital Readmission Prediction.
Creates domain-informed composite features from cleaned patient data.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Derives new clinical features that improve model signal:
      - recurrency            : whether patient has prior inpatient visits
      - medication_change_ratio: fraction of medications that changed
      - lab_tests_per_med     : lab workload relative to medications
      - patient_severity      : composite severity score
      - number_prior_visits   : total outpatient + emergency + inpatient
      - age_times_medications : interaction term (age × num_medications)
    """

    MEDICATION_COLS = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide",
        "glimepiride", "acetohexamide", "glipizide", "glyburide",
        "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
        "miglitol", "troglitazone", "tolazamide", "examide",
        "citoglipton", "insulin", "glyburide-metformin",
        "glipizide-metformin", "glimepiride-pioglitazone",
        "metformin-rosiglitazone", "metformin-pioglitazone",
    ]

    def fit(self, X: pd.DataFrame, y=None):
        self._med_cols_present = [c for c in self.MEDICATION_COLS if c in X.columns]
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        df = X.copy()

        # encode medication columns: "No"→0, "Steady"→1, "Up"/"Down"→2
        change_map = {"No": 0, "Steady": 1, "Up": 2, "Down": 2}
        for col in self._med_cols_present:
            df[col] = df[col].map(change_map).fillna(0)

        # feature: has the patient been admitted before?
        if "number_inpatient" in df.columns:
            df["recurrency"] = (df["number_inpatient"] > 0).astype(int)

        # feature: fraction of meds that were changed (Up or Down)
        if self._med_cols_present:
            changed = (df[self._med_cols_present] == 2).sum(axis=1)
            total = (df[self._med_cols_present] > 0).sum(axis=1).replace(0, np.nan)
            df["medication_change_ratio"] = (changed / total).fillna(0)

        # feature: prior visits aggregate
        prior_cols = [c for c in ("number_outpatient", "number_emergency", "number_inpatient") if c in df.columns]
        if prior_cols:
            df["number_prior_visits"] = df[prior_cols].sum(axis=1)

        # feature: lab intensity per medication
        if "num_lab_procedures" in df.columns and "num_medications" in df.columns:
            safe_meds = df["num_medications"].replace(0, np.nan)
            df["lab_tests_per_med"] = (df["num_lab_procedures"] / safe_meds).fillna(0)

        # feature: composite severity (procedures + diagnoses + emergency visits)
        sev_cols = [c for c in ("num_procedures", "number_diagnoses", "number_emergency") if c in df.columns]
        if sev_cols:
            df["patient_severity"] = df[sev_cols].sum(axis=1)

        # feature: age × medications interaction
        if "age" in df.columns and "num_medications" in df.columns:
            df["age_times_medications"] = df["age"] * df["num_medications"]

        return df
