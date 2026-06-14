"""
Data preprocessing pipeline for the Hospital Readmission Prediction project.
Handles missing values, outliers, encoding, and scaling.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, RobustScaler


METRIC_FEATURES = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]

NON_METRIC_FEATURES = [
    "race", "gender", "age", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "medical_specialty", "diag_1", "diag_2", "diag_3",
    "max_glu_serum", "A1Cresult", "change", "diabetesMed",
]

DROP_COLS = ["encounter_id", "patient_nbr", "patient_id", "weight", "payer_code", "country"]

AGE_MAP = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
    "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
    "[80-90)": 85, "[90-100)": 95,
}

DISCHARGE_MAP = {
    1: "home", 2: "facility", 3: "facility", 4: "home", 5: "facility",
    6: "home", 7: "AMA", 8: "home", 9: "facility", 10: "deceased",
    11: "facility", 12: "facility", 13: "hospice", 14: "hospice",
    15: "facility", 16: "facility", 17: "facility", 18: "facility",
    19: "deceased", 20: "facility", 21: "facility", 22: "facility",
    23: "facility", 24: "facility", 25: "facility", 26: "facility",
    27: "facility", 28: "facility", 29: "facility", 30: "facility",
}

ADMISSION_SOURCE_MAP = {
    1: "physician_referral", 2: "clinic_referral", 3: "HMO_referral",
    4: "transfer", 5: "transfer", 6: "transfer", 7: "emergency",
    8: "court", 9: "not_available", 10: "transfer", 11: "normal_delivery",
    12: "sick_baby", 13: "extramural_baby", 14: "transfer",
    15: "not_available", 17: "not_available", 18: "transfer",
    19: "transfer", 20: "transfer", 21: "transfer", 22: "transfer",
    23: "emergency", 24: "court", 25: "transfer", 26: "transfer",
}


def _map_diag(code):
    """Map ICD-9 diagnosis codes to disease categories."""
    if pd.isna(code) or code in ("?", "E", "V"):
        return "other"
    try:
        c = float(str(code).replace("E", "").replace("V", ""))
    except ValueError:
        return "other"
    if 390 <= c <= 459 or c == 785:
        return "circulatory"
    if 460 <= c <= 519 or c == 786:
        return "respiratory"
    if 520 <= c <= 579 or c == 787:
        return "digestive"
    if 250 <= c < 251:
        return "diabetes"
    if 800 <= c <= 999:
        return "injury"
    if 710 <= c <= 739:
        return "musculoskeletal"
    if 580 <= c <= 629 or c == 788:
        return "genitourinary"
    if 140 <= c <= 239:
        return "neoplasms"
    return "other"


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Full preprocessing pipeline:
      1. Drop irrelevant columns
      2. Handle missing values
      3. Map categorical codes to readable labels
      4. Encode categoricals with LabelEncoder
      5. Scale numerics with RobustScaler
    """

    def __init__(self, missing_threshold: float = 0.5):
        self.missing_threshold = missing_threshold
        self._label_encoders: dict[str, LabelEncoder] = {}
        self._scaler = RobustScaler()
        self._drop_high_missing: list[str] = []
        self._cat_cols: list[str] = []
        self._num_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        df = self._initial_clean(df)

        # drop columns with too many missing values
        missing_rate = df.isnull().mean()
        self._drop_high_missing = missing_rate[missing_rate > self.missing_threshold].index.tolist()
        df = df.drop(columns=self._drop_high_missing, errors="ignore")

        df = self._map_categoricals(df)
        df = self._impute(df)

        self._cat_cols = df.select_dtypes(include="object").columns.tolist()
        self._num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in self._cat_cols:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            self._label_encoders[col] = le

        self._scaler.fit(df[self._num_cols])
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        df = X.copy()
        df = self._initial_clean(df)
        df = df.drop(columns=self._drop_high_missing, errors="ignore")
        df = self._map_categoricals(df)
        df = self._impute(df)

        for col in self._cat_cols:
            if col in df.columns:
                le = self._label_encoders[col]
                df[col] = df[col].astype(str).map(
                    lambda x, le=le: x if x in le.classes_ else le.classes_[0]
                )
                df[col] = le.transform(df[col])

        if self._num_cols:
            present = [c for c in self._num_cols if c in df.columns]
            df[present] = self._scaler.transform(df[present])

        return df

    def _initial_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        target_cols = [c for c in ("readmitted", "readmitted_binary") if c in df.columns]
        drop = [c for c in DROP_COLS if c in df.columns]
        df = df.drop(columns=drop, errors="ignore")

        # replace sentinel missing values
        df.replace("?", np.nan, inplace=True)
        df.replace("Unknown/Invalid", np.nan, inplace=True)
        return df

    def _map_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        if "age" in df.columns:
            df["age"] = df["age"].map(AGE_MAP)

        if "discharge_disposition_id" in df.columns:
            df["discharge_disposition_id"] = (
                df["discharge_disposition_id"].map(DISCHARGE_MAP).fillna("facility")
            )

        if "admission_source_id" in df.columns:
            df["admission_source_id"] = (
                df["admission_source_id"].map(ADMISSION_SOURCE_MAP).fillna("not_available")
            )

        for diag_col in ("diag_1", "diag_2", "diag_3"):
            if diag_col in df.columns:
                df[diag_col] = df[diag_col].apply(_map_diag)

        return df

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "unknown")
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(df[col].median())
        return df
