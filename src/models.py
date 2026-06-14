"""
Model definitions for Hospital Readmission Prediction.

Binary task   : Stacking ensemble (Neural Network + Extra Trees)
Multiclass task: AdaBoost with calibrated probabilities

Both expose a standard sklearn-compatible fit/predict/predict_proba interface
and include built-in SMOTE-based class-imbalance handling.
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif


def _select_k_best(k: int):
    return SelectKBest(score_func=f_classif, k=k)


class BinaryClassifier:
    """
    Stacking ensemble for 30-day readmission (binary).
    Base learners: MLP + ExtraTrees
    Meta learner: LogisticRegression
    """

    def __init__(self, n_features: int = 14, random_state: int = 42):
        self.n_features = n_features
        self.random_state = random_state
        self.pipeline_: ImbPipeline | None = None
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BinaryClassifier":
        self.feature_names_ = list(X.columns)

        mlp = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=300,
            random_state=self.random_state,
        )
        et = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1,
        )
        stacking = StackingClassifier(
            estimators=[("mlp", mlp), ("et", et)],
            final_estimator=LogisticRegression(max_iter=1000, random_state=self.random_state),
            cv=5,
            n_jobs=-1,
        )

        self.pipeline_ = ImbPipeline([
            ("selector", _select_k_best(min(self.n_features, X.shape[1]))),
            ("smote", SMOTE(random_state=self.random_state)),
            ("clf", stacking),
        ])
        self.pipeline_.fit(X.values, y.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline_.predict(X.values)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline_.predict_proba(X.values)

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        return self.pipeline_.score(X.values, y.values)


class MulticlassClassifier:
    """
    AdaBoost classifier for readmission timing (No / <30 days / >30 days).
    """

    def __init__(self, n_features: int = 19, n_estimators: int = 100, random_state: int = 42):
        self.n_features = n_features
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.pipeline_: ImbPipeline | None = None
        self.feature_names_: list[str] = []
        self.classes_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MulticlassClassifier":
        self.feature_names_ = list(X.columns)
        self.classes_ = np.unique(y.values)

        ada = AdaBoostClassifier(
            n_estimators=self.n_estimators,
            learning_rate=0.5,
            random_state=self.random_state,
        )

        self.pipeline_ = ImbPipeline([
            ("selector", _select_k_best(min(self.n_features, X.shape[1]))),
            ("smote", SMOTE(random_state=self.random_state)),
            ("clf", CalibratedClassifierCV(ada, cv=3)),
        ])
        self.pipeline_.fit(X.values, y.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline_.predict(X.values)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline_.predict_proba(X.values)

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        return self.pipeline_.score(X.values, y.values)
