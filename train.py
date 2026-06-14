"""
train.py — one-command model training script.

Usage:
    python train.py --train data/train.csv --test data/test.csv

Trains both the binary and multiclass pipelines, saves them to models/.
"""

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import joblib
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.models import BinaryClassifier, MulticlassClassifier

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def load_data(train_path: str, test_path: str):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"Train shape: {train_df.shape}  |  Test shape: {test_df.shape}")
    return train_df, test_df


def split_features_targets(df: pd.DataFrame):
    binary_target = "readmitted_binary" if "readmitted_binary" in df.columns else None
    multi_target = "readmitted" if "readmitted" in df.columns else None

    y_bin = df[binary_target] if binary_target else None
    y_mul = df[multi_target] if multi_target else None

    drop_targets = [c for c in (binary_target, multi_target) if c is not None]
    X = df.drop(columns=drop_targets, errors="ignore")
    return X, y_bin, y_mul


def train_pipeline(train_path: str, test_path: str):
    print("=" * 60)
    print("Hospital Readmission Predictor — Training Pipeline")
    print("=" * 60)

    train_df, test_df = load_data(train_path, test_path)

    # ----------------------------------------------------------------
    # 1. Preprocessing
    # ----------------------------------------------------------------
    print("\n[1/5] Fitting preprocessor on training data...")
    X_train_raw, y_bin_train, y_mul_train = split_features_targets(train_df)
    X_test_raw, y_bin_test, y_mul_test = split_features_targets(test_df)

    preprocessor = DataPreprocessor(missing_threshold=0.5)
    preprocessor.fit(X_train_raw)

    X_train_proc = preprocessor.transform(X_train_raw)
    X_test_proc = preprocessor.transform(X_test_raw)
    print(f"   After preprocessing: {X_train_proc.shape[1]} features")

    # ----------------------------------------------------------------
    # 2. Feature Engineering
    # ----------------------------------------------------------------
    print("[2/5] Engineering features...")
    engineer = FeatureEngineer()
    engineer.fit(X_train_proc)

    X_train = engineer.transform(X_train_proc)
    X_test = engineer.transform(X_test_proc)
    print(f"   After engineering: {X_train.shape[1]} features")

    # ----------------------------------------------------------------
    # 3. Binary classification
    # ----------------------------------------------------------------
    if y_bin_train is not None:
        print("\n[3/5] Training Binary Classifier (Stacking: MLP + ExtraTrees)...")
        binary_clf = BinaryClassifier(n_features=14, random_state=42)
        binary_clf.fit(X_train, y_bin_train)

        preds_bin = binary_clf.predict(X_test)
        f1_bin = f1_score(y_bin_test, preds_bin, average="binary", zero_division=0)
        print(f"   Binary F1 on test set: {f1_bin:.4f}")

        joblib.dump(binary_clf, os.path.join(MODELS_DIR, "binary_pipeline.pkl"))
        print(f"   Saved → {MODELS_DIR}/binary_pipeline.pkl")
    else:
        print("[3/5] Skipping binary classifier (target column not found)")

    # ----------------------------------------------------------------
    # 4. Multiclass classification
    # ----------------------------------------------------------------
    if y_mul_train is not None:
        print("\n[4/5] Training Multiclass Classifier (AdaBoost)...")
        multi_clf = MulticlassClassifier(n_features=19, n_estimators=100, random_state=42)
        multi_clf.fit(X_train, y_mul_train)

        preds_mul = multi_clf.predict(X_test)
        f1_mul = f1_score(y_mul_test, preds_mul, average="macro", zero_division=0)
        print(f"   Multiclass F1 (macro) on test set: {f1_mul:.4f}")

        joblib.dump(multi_clf, os.path.join(MODELS_DIR, "multiclass_pipeline.pkl"))
        print(f"   Saved → {MODELS_DIR}/multiclass_pipeline.pkl")
    else:
        print("[4/5] Skipping multiclass classifier (target column not found)")

    # ----------------------------------------------------------------
    # 5. Save preprocessing objects
    # ----------------------------------------------------------------
    print("\n[5/5] Saving preprocessor and feature engineer...")
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.pkl"))
    joblib.dump(engineer, os.path.join(MODELS_DIR, "feature_engineer.pkl"))

    print("\n" + "=" * 60)
    print("Training complete. All artifacts saved to models/")
    print("Run the app: streamlit run app/app.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hospital Readmission models")
    parser.add_argument("--train", default="data/train.csv", help="Path to training CSV")
    parser.add_argument("--test", default="data/test.csv", help="Path to test CSV")
    args = parser.parse_args()
    train_pipeline(args.train, args.test)
