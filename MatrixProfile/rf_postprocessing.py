import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def train_rf_from_xlsx(xlsx_path: str, test_size: float = 0.3, random_state: int = 42) -> None:
    """
    Trains and evaluates a Random Forest classifier on matrix profile RF samples from an Excel file.

    Args:
        xlsx_path (str): Path to the Excel file with RF samples.
        test_size (float): Proportion of data to use for test split (default: 0.3).
        random_state (int): Random state for reproducibility (default: 42).

    Returns:
        None
    """
    # Load samples
    df = pd.read_excel(xlsx_path)

    # Feature columns: all columns starting with "mp_"
    feature_cols = [col for col in df.columns if col.startswith("mp_")]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int32)

    # Optional: Print class distribution
    print("Class balance:", np.bincount(y))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train RF
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        class_weight="balanced",  # good for unbalanced datasets
        random_state=random_state,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=3))

    # Feature importance (top 10)
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("Top 10 Features by Importance:")
    for i in range(min(10, len(importances))):
        print(f"{feature_cols[sorted_idx[i]]}: {importances[sorted_idx[i]]:.4f}")

if __name__ == "__main__":
    train_rf_from_xlsx("/home/jhagenbe_sw/ASIM/ecg-seizure-detection/MatrixProfile/rf_train_data/rf_train_samples.xlsx")
    # train_rf_from_xlsx("rf_train_samples_detection_window.xlsx")
