"""Unit tests for DataPreprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import DataPreprocessor, _map_diag


def _make_sample_df(n=50):
    np.random.seed(42)
    return pd.DataFrame({
        "age": np.random.choice(
            ["[30-40)", "[50-60)", "[70-80)"], n
        ),
        "gender": np.random.choice(["Male", "Female"], n),
        "race": np.random.choice(["Caucasian", "AfricanAmerican"], n),
        "time_in_hospital": np.random.randint(1, 14, n),
        "num_lab_procedures": np.random.randint(1, 100, n),
        "num_procedures": np.random.randint(0, 6, n),
        "num_medications": np.random.randint(1, 50, n),
        "number_diagnoses": np.random.randint(1, 10, n),
        "number_outpatient": np.random.randint(0, 5, n),
        "number_emergency": np.random.randint(0, 3, n),
        "number_inpatient": np.random.randint(0, 5, n),
        "discharge_disposition_id": np.random.choice([1, 2, 3, 6], n),
        "admission_source_id": np.random.choice([1, 7], n),
        "diag_1": np.random.choice(["250", "401", "?", "E11"], n),
        "diag_2": np.random.choice(["250", "428"], n),
        "diag_3": np.random.choice(["V45", "401"], n),
        "max_glu_serum": np.random.choice(["None", ">200", "Norm"], n),
        "A1Cresult": np.random.choice(["None", ">8", "Norm"], n),
        "change": np.random.choice(["No", "Ch"], n),
        "diabetesMed": np.random.choice(["Yes", "No"], n),
        "insulin": np.random.choice(["No", "Steady", "Up"], n),
        "medical_specialty": np.random.choice(["InternalMedicine", np.nan], n),
        "encounter_id": np.arange(n),
        "patient_nbr": np.arange(n),
        "weight": [np.nan] * n,
        "payer_code": [np.nan] * n,
    })


class TestMapDiag:
    def test_diabetes(self):
        assert _map_diag("250") == "diabetes"

    def test_circulatory(self):
        assert _map_diag("410") == "circulatory"

    def test_nan(self):
        assert _map_diag(np.nan) == "other"

    def test_question_mark(self):
        assert _map_diag("?") == "other"


class TestDataPreprocessor:
    def test_fit_transform_shape(self):
        df = _make_sample_df()
        pp = DataPreprocessor()
        pp.fit(df)
        out = pp.transform(df)
        assert out.shape[0] == len(df)
        assert out.shape[1] > 0

    def test_no_missing_after_transform(self):
        df = _make_sample_df()
        pp = DataPreprocessor()
        pp.fit(df)
        out = pp.transform(df)
        assert out.isnull().sum().sum() == 0

    def test_drop_cols_removed(self):
        df = _make_sample_df()
        pp = DataPreprocessor()
        pp.fit(df)
        out = pp.transform(df)
        for col in ("encounter_id", "patient_nbr", "weight", "payer_code"):
            assert col not in out.columns

    def test_age_numeric(self):
        df = _make_sample_df()
        pp = DataPreprocessor()
        pp.fit(df)
        out = pp.transform(df)
        assert pd.api.types.is_numeric_dtype(out["age"])

    def test_transform_without_targets(self):
        df = _make_sample_df()
        df["readmitted"] = 0
        df["readmitted_binary"] = 0
        pp = DataPreprocessor()
        pp.fit(df.drop(columns=["readmitted", "readmitted_binary"]))
        out = pp.transform(df.drop(columns=["readmitted", "readmitted_binary"]))
        assert "readmitted" not in out.columns
