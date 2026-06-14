"""Unit tests for FeatureEngineer."""

import numpy as np
import pandas as pd

from src.feature_engineering import FeatureEngineer


def _base_df(n=20):
    np.random.seed(0)
    return pd.DataFrame({
        "age": np.random.uniform(30, 80, n),
        "num_medications": np.random.randint(5, 30, n),
        "num_lab_procedures": np.random.randint(10, 80, n),
        "num_procedures": np.random.randint(0, 5, n),
        "number_diagnoses": np.random.randint(1, 10, n),
        "number_outpatient": np.random.randint(0, 3, n),
        "number_emergency": np.random.randint(0, 3, n),
        "number_inpatient": np.random.randint(0, 3, n),
        "insulin": np.random.choice(["No", "Steady", "Up", "Down"], n),
        "metformin": np.random.choice(["No", "Steady", "Up"], n),
    })


class TestFeatureEngineer:
    def test_new_columns_created(self):
        df = _base_df()
        fe = FeatureEngineer()
        fe.fit(df)
        out = fe.transform(df)
        for col in ("recurrency", "number_prior_visits", "patient_severity", "age_times_medications"):
            assert col in out.columns, f"Missing column: {col}"

    def test_recurrency_binary(self):
        df = _base_df()
        fe = FeatureEngineer()
        fe.fit(df)
        out = fe.transform(df)
        assert out["recurrency"].isin([0, 1]).all()

    def test_no_nulls(self):
        df = _base_df()
        fe = FeatureEngineer()
        fe.fit(df)
        out = fe.transform(df)
        assert out.isnull().sum().sum() == 0

    def test_prior_visits_sum(self):
        df = _base_df()
        fe = FeatureEngineer()
        fe.fit(df)
        out = fe.transform(df)
        expected = df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
        pd.testing.assert_series_equal(
            out["number_prior_visits"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )
