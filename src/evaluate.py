"""
Evaluation utilities: metrics, confusion matrices, SHAP explanations,
and learning-curve helpers.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    roc_auc_score,
    roc_curve,
)


class ModelEvaluator:
    """Wraps common evaluation routines for both binary and multiclass tasks."""

    def __init__(self, task: str = "binary"):
        if task not in ("binary", "multiclass"):
            raise ValueError("task must be 'binary' or 'multiclass'")
        self.task = task

    # ------------------------------------------------------------------ #
    # Core metrics
    # ------------------------------------------------------------------ #

    def report(self, y_true, y_pred) -> str:
        avg = "binary" if self.task == "binary" else "macro"
        f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)
        cr = classification_report(y_true, y_pred, zero_division=0)
        header = f"F1 ({avg}): {f1:.4f}\n\n"
        return header + cr

    # ------------------------------------------------------------------ #
    # Plots
    # ------------------------------------------------------------------ #

    def plot_confusion_matrix(self, y_true, y_pred, labels=None, title="Confusion Matrix"):
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, display_labels=labels, ax=ax, colorbar=False
        )
        ax.set_title(title)
        plt.tight_layout()
        return fig

    def plot_roc_curve(self, y_true, y_proba, pos_label=1):
        if self.task != "binary":
            raise ValueError("ROC curve is only for binary classification")
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1], pos_label=pos_label)
        auc = roc_auc_score(y_true, y_proba[:, 1])
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", lw=2)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        plt.tight_layout()
        return fig

    def plot_feature_importance(self, feature_names: list[str], importances: np.ndarray, top_n: int = 15):
        idx = np.argsort(importances)[-top_n:]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh([feature_names[i] for i in idx], importances[idx], color="steelblue")
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {top_n} Feature Importances")
        plt.tight_layout()
        return fig

    def plot_class_distribution(self, y: pd.Series, title: str = "Class Distribution"):
        counts = y.value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(counts.index.astype(str), counts.values, color=["#4C72B0", "#DD8452", "#55A868"])
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_title(title)
        for i, v in enumerate(counts.values):
            ax.text(i, v + counts.max() * 0.01, str(v), ha="center", fontsize=10)
        plt.tight_layout()
        return fig
