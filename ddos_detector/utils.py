"""
Utility functions for the DrDoS DNS detection pipeline.
Aggregates data loading/preprocessing, balancing, splitting, training,
comparison, evaluation, and persistence helpers into a single module.
"""

import os
import time
import warnings
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# =============================================================================
# Data loading and preprocessing
# =============================================================================


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load the DrDoS dataset from CSV file.
    """
    print("\n[1/7] Loading dataset...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"   Total samples: {len(df):,}")
    print(f"   Total features: {len(df.columns)}")
    return df


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, str]:
    """
    Clean and preprocess the dataset.
    """
    print("\n[2/7] Cleaning data...")

    columns_to_drop = ["Unnamed: 0", "Flow ID", " Timestamp"]
    existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=existing_cols_to_drop)

    if " Label" in df.columns:
        label_col = " Label"
    elif "Label" in df.columns:
        label_col = "Label"
    else:
        raise ValueError("Label column not found!")

    print(f"   Label column: '{label_col}'")
    print("   Replacing infinity values...")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    print(f"   Null values before: {df.isnull().sum().sum()}")
    df = df.dropna()
    print(f"   Null values after: {df.isnull().sum().sum()}")

    X = df.drop(columns=[label_col])
    y = df[label_col]

    print("   Encoding categorical features...")
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    print("\n   Class distribution:")
    for label, count in zip(*np.unique(y, return_counts=True)):
        percentage = (count / len(y)) * 100
        print(f"   - {label}: {count:,} samples ({percentage:.2f}%)")

    print(f"\n   Total features to be used: {len(X.columns)}")
    return X, y, label_col


def encode_labels(y: pd.Series) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Encode string labels to numeric values.
    """
    le_label = LabelEncoder()
    y_encoded = le_label.fit_transform(y)
    return y_encoded, le_label


# =============================================================================
# Balancing
# =============================================================================


def apply_smote_to_benign(
    X_benign: pd.DataFrame,
    y_benign: np.ndarray,
    target_samples: int,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Apply SMOTE to BENIGN (minority) class BEFORE splitting.
    """
    print("\n[3/7] Applying SMOTE to BENIGN class...")
    print(f"   Original BENIGN samples: {len(X_benign):,}")
    print(f"   Target BENIGN samples after SMOTE: {target_samples:,}")

    y_dummy_attack = np.ones(target_samples, dtype=int) * (1 - y_benign[0])
    X_dummy_attack = pd.DataFrame(
        np.zeros((target_samples, X_benign.shape[1])), columns=X_benign.columns
    )

    X_combined = pd.concat([X_benign, X_dummy_attack], ignore_index=True)
    y_combined = np.concatenate([y_benign, y_dummy_attack])

    smote = SMOTE(
        sampling_strategy={y_benign[0]: target_samples},
        random_state=random_state,
        k_neighbors=min(5, len(X_benign) - 1),
    )

    X_resampled, y_resampled = smote.fit_resample(X_combined, y_combined)

    benign_mask = y_resampled == y_benign[0]
    X_benign_smote = pd.DataFrame(X_resampled[benign_mask], columns=X_benign.columns)
    y_benign_smote = y_resampled[benign_mask]

    print(f"   BENIGN samples after SMOTE: {len(X_benign_smote):,}")
    print(f"   SMOTE generated {len(X_benign_smote) - len(X_benign):,} synthetic samples")
    return X_benign_smote, y_benign_smote


# =============================================================================
# Splitting
# =============================================================================


def split_data_after_smote(
    X_benign_original: pd.DataFrame,
    y_benign_original: np.ndarray,
    X_benign_smote: pd.DataFrame,
    y_benign_smote: np.ndarray,
    X_attack: pd.DataFrame,
    y_attack: np.ndarray,
    le_label,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets AFTER SMOTE.
    """
    print(f"\n[4/7] Splitting data with corrected strategy (test_size={test_size})...")
    print("   Strategy:")
    print("   - Test: ALL original BENIGN + equal DDoS samples")
    print("   - Train: SMOTE BENIGN + remaining DDoS")

    np.random.seed(random_state)

    num_benign_test = len(X_benign_original)
    num_attack_test = num_benign_test

    print("\n   Test set composition:")
    print(f"   - BENIGN (all original): {num_benign_test:,}")
    print(f"   - DDoS (random selection): {num_attack_test:,}")

    attack_test_indices = np.random.choice(len(X_attack), size=num_attack_test, replace=False)

    X_test_benign = X_benign_original.copy()
    y_test_benign = y_benign_original.copy()

    X_test_attack = X_attack.iloc[attack_test_indices].copy()
    y_test_attack = y_attack[attack_test_indices].copy()

    X_test = pd.concat([X_test_benign, X_test_attack], ignore_index=True)
    y_test = np.concatenate([y_test_benign, y_test_attack])

    shuffle_indices = np.random.permutation(len(X_test))
    X_test = X_test.iloc[shuffle_indices].copy()
    y_test = y_test[shuffle_indices].copy()

    print(f"   - Total test samples: {len(X_test):,}")

    num_train_total = int(len(X_test) * (1 - test_size) / test_size)

    print(f"\n   Calculating train set size for {test_size*100}% test ratio:")
    print(f"   - Total train samples needed: {num_train_total:,}")

    num_benign_train_available = len(X_benign_smote)

    if num_benign_train_available > num_train_total:
        print(
            f"   - SMOTE BENIGN ({num_benign_train_available:,}) > needed total ({num_train_total:,})"
        )
        print("   - Subsampling SMOTE BENIGN to fit ratio...")
        num_benign_train = min(num_benign_train_available, num_train_total // 2)
        num_attack_train = num_train_total - num_benign_train
    else:
        num_benign_train = num_benign_train_available
        num_attack_train = num_train_total - num_benign_train

    if num_attack_train < 0:
        print("   ⚠ Adjusting: using all SMOTE BENIGN and no DDoS in train")
        num_benign_train = num_benign_train_available
        num_attack_train = 0

    attack_train_mask = np.ones(len(X_attack), dtype=bool)
    attack_train_mask[attack_test_indices] = False
    attack_available_indices = np.where(attack_train_mask)[0]

    print("\n   Train set composition:")
    print(f"   - BENIGN (SMOTE) to use: {num_benign_train:,}")
    print(f"   - DDoS available: {len(attack_available_indices):,}")
    print(f"   - DDoS needed: {num_attack_train:,}")

    if num_benign_train < len(X_benign_smote):
        benign_train_indices = np.random.choice(
            len(X_benign_smote), size=num_benign_train, replace=False
        )
        X_train_benign = X_benign_smote.iloc[benign_train_indices].copy()
        y_train_benign = y_benign_smote[benign_train_indices].copy()
    else:
        X_train_benign = X_benign_smote.copy()
        y_train_benign = y_benign_smote.copy()

    if num_attack_train == 0:
        X_train_attack = pd.DataFrame(columns=X_attack.columns)
        y_train_attack = np.array([], dtype=y_attack.dtype)
    elif num_attack_train > len(attack_available_indices):
        print(
            f"   ⚠ Not enough DDoS samples! Using all available: {len(attack_available_indices):,}"
        )
        X_train_attack = X_attack.iloc[attack_available_indices].copy()
        y_train_attack = y_attack[attack_available_indices].copy()
    else:
        attack_train_indices = np.random.choice(
            attack_available_indices, size=num_attack_train, replace=False
        )
        X_train_attack = X_attack.iloc[attack_train_indices].copy()
        y_train_attack = y_attack[attack_train_indices].copy()

    X_train = pd.concat([X_train_benign, X_train_attack], ignore_index=True)
    y_train = np.concatenate([y_train_benign, y_train_attack])

    shuffle_indices = np.random.permutation(len(X_train))
    X_train = X_train.iloc[shuffle_indices].copy()
    y_train = y_train[shuffle_indices].copy()

    print(f"   - Total train samples: {len(X_train):,}")

    actual_test_size = len(X_test) / (len(X_train) + len(X_test))
    print("\n   Verification:")
    print(f"   - Train size: {len(X_train):,}")
    print(f"   - Test size: {len(X_test):,}")
    print(f"   - Actual test ratio: {actual_test_size:.2%}")
    print(f"   - BENIGN in test: {sum(y_test == le_label.transform(['BENIGN'])[0]):,}")
    print(f"   - DDoS in test: {sum(y_test == le_label.transform(['DrDoS_DNS'])[0]):,}")
    print(f"   - BENIGN in train: {sum(y_train == le_label.transform(['BENIGN'])[0]):,}")
    print(f"   - DDoS in train: {sum(y_train == le_label.transform(['DrDoS_DNS'])[0]):,}")

    return X_train, X_test, y_train, y_test


# =============================================================================
# Scaling and training
# =============================================================================


def scale_features(X_train: np.ndarray, X_test: np.ndarray) -> Tuple:
    """Scale features using StandardScaler."""
    print("\n[5/7] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("   Features scaled using StandardScaler")
    print(f"   Training shape: {X_train_scaled.shape}")
    print(f"   Test shape: {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled, scaler


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_iter: int = 1000,
    random_state: int = 42,
) -> LogisticRegression:
    """Train Logistic Regression classifier."""
    print("\n[6/7] Training Logistic Regression classifier...")
    print("   Χρησιμοποιούνται ΟΛΕΣ οι στήλες του dataset")
    print(f"   Total features used: {X_train.shape[1]}")

    clf = LogisticRegression(
        max_iter=max_iter, random_state=random_state, n_jobs=-1, verbose=1
    )
    clf.fit(X_train, y_train)
    print("\n   Training completed!")
    return clf


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 30,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train Random Forest classifier."""
    print("\n[6/7] Training Random Forest classifier...")
    print("   Χρησιμοποιούνται ΟΛΕΣ οι στήλες του dataset")
    print(f"   Total features used: {X_train.shape[1]}")

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )
    clf.fit(X_train, y_train)
    print("\n   Training completed!")
    return clf


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    kernel: str = "rbf",
    C: float = 1.0,
    random_state: int = 42,
) -> SVC:
    """Train Support Vector Machine classifier."""
    print("\n[6/7] Training SVM classifier...")
    print("   Χρησιμοποιούνται ΟΛΕΣ οι στήλες του dataset")
    print(f"   Total features used: {X_train.shape[1]}")

    clf = SVC(kernel=kernel, C=C, random_state=random_state, verbose=True)
    clf.fit(X_train, y_train)
    print("\n   Training completed!")
    return clf


def train_decision_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_depth: int = 30,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    random_state: int = 42,
) -> DecisionTreeClassifier:
    """Train Decision Tree classifier."""
    print("\n[6/7] Training Decision Tree classifier...")
    print("   Χρησιμοποιούνται ΟΛΕΣ οι στήλες του dataset")
    print(f"   Total features used: {X_train.shape[1]}")

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    print("\n   Training completed!")
    return clf


def train_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_neighbors: int = 5,
    weights: str = "uniform",
) -> KNeighborsClassifier:
    """Train K-Nearest Neighbors classifier."""
    print("\n[6/7] Training KNN classifier...")
    print("   Χρησιμοποιούνται ΟΛΕΣ οι στήλες του dataset")
    print(f"   Total features used: {X_train.shape[1]}")

    clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, n_jobs=-1)
    clf.fit(X_train, y_train)
    print("\n   Training completed!")
    return clf


# =============================================================================
# Evaluation
# =============================================================================


def evaluate_model(
    clf,
    X_test: np.ndarray,
    y_test: np.ndarray,
    le_label,
    feature_names: List[str],
) -> Dict:
    """Evaluate model and display comprehensive metrics."""
    print("\n[7/7] Evaluating model...")

    y_pred = clf.predict(X_test)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le_label.classes_))

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("\nSummary Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    print("\n" + "=" * 80)
    print("TOP 20 MOST IMPORTANT FEATURES")
    print("=" * 80)

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        importances = np.zeros(len(feature_names))

    feature_importance = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    print(feature_importance.head(20).to_string(index=False))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "feature_importance": feature_importance,
    }


def get_next_results_filename(base_name: str = "training_results", extension: str = "txt", output_dir: str | None = None) -> str:
    """Generate next available filename with incremental numbering."""
    base_path = Path(output_dir) if output_dir else Path.cwd()
    base_path.mkdir(parents=True, exist_ok=True)
    counter = 1
    while True:
        filename = base_path / f"{base_name}_{counter}.{extension}"
        if not filename.exists():
            return str(filename)
        counter += 1


def save_results_to_file(
    metrics: Dict,
    config: Dict,
    train_test_info: Dict,
    le_label,
    filename: str | None = None,
    output_dir: str | None = None,
) -> str:
    """Save training results to a text file."""
    if filename is None:
        filename = get_next_results_filename(output_dir=output_dir)
    else:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

    with open(filename, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("DrDoS DNS Attack Detection - Training Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Results saved to: {filename}\n")
        f.write("=" * 80 + "\n\n")

        f.write("CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        f.write("DATASET INFORMATION\n")
        f.write("-" * 80 + "\n")
        for key, value in train_test_info.items():
            if isinstance(value, (int, float)):
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value:,}\n")
            else:
                f.write(f"{key}: {value}\n")
        f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("RESULTS\n")
        f.write("=" * 80 + "\n\n")

        f.write("Confusion Matrix:\n")
        cm = metrics["confusion_matrix"]
        f.write(str(cm) + "\n\n")

        f.write("Classification Report:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
        f.write("-" * 80 + "\n")
        for class_name in le_label.classes_:
            f.write(
                f"{class_name:<15} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1_score']:<12.4f}\n"
            )
        f.write("\n")

        f.write("Summary Metrics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1-Score:  {metrics['f1_score']:.4f}\n")
        f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("TOP 20 MOST IMPORTANT FEATURES\n")
        f.write("=" * 80 + "\n")
        feature_importance = metrics["feature_importance"].head(20)
        f.write(f"{'Feature':<30} {'Importance':<15}\n")
        f.write("-" * 80 + "\n")
        for _, row in feature_importance.iterrows():
            f.write(f"{row['feature']:<30} {row['importance']:<15.6f}\n")
        f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("Pipeline completed successfully!\n")
        f.write("=" * 80 + "\n")

    return filename


# =============================================================================
# Comparison
# =============================================================================


def compare_models(results_dict: Dict, label_encoder) -> pd.DataFrame:
    """Compare multiple models and create a comparison table."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    comparison_data = []
    for model_name, metrics in results_dict.items():
        comparison_data.append(
            {
                "Model": model_name,
                "Accuracy": f"{metrics['accuracy']:.4f}",
                "Precision": f"{metrics['precision']:.4f}",
                "Recall": f"{metrics['recall']:.4f}",
                "F1-Score": f"{metrics['f1_score']:.4f}",
                "Training Time (s)": f"{metrics['training_time']:.2f}",
                "Total Time (s)": f"{metrics['total_time']:.2f}",
            }
        )

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values("Accuracy", ascending=False)

    print(comparison_df.to_string(index=False))
    print("\n" + "=" * 80)

    best_model = comparison_df.iloc[0]["Model"]
    print(f"🏆 Best Model: {best_model}")
    print("=" * 80)
    return comparison_df


def save_comparison_to_file(
    comparison_df: pd.DataFrame,
    results_dict: Dict,
    config: Dict,
    train_test_info: Dict,
    label_encoder,
    output_dir: str | None = None,
) -> str:
    """Save comparison results to a text file with auto-incrementing filename."""
    base_path = Path(output_dir) if output_dir else Path.cwd()
    base_path.mkdir(parents=True, exist_ok=True)
    counter = 1
    while True:
        candidate = base_path / f"comparison_results_{counter}.txt"
        if not candidate.exists():
            filename = candidate
            break
        counter += 1

    with open(filename, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("DrDoS DNS Attack Detection - Model Comparison Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Results saved to: {filename}\n")
        f.write("=" * 80 + "\n\n")

        f.write("CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

        f.write("\nDATASET INFORMATION\n")
        f.write("-" * 80 + "\n")
        for key, value in train_test_info.items():
            f.write(f"{key}: {value:,}" if isinstance(value, int) else f"{key}: {value}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("MODEL COMPARISON\n")
        f.write("=" * 80 + "\n\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")

        best_model = comparison_df.iloc[0]["Model"]
        f.write("=" * 80 + "\n")
        f.write(f"🏆 Best Model: {best_model}\n")
        f.write("=" * 80 + "\n\n")

        f.write("=" * 80 + "\n")
        f.write("DETAILED RESULTS FOR EACH MODEL\n")
        f.write("=" * 80 + "\n\n")

        for model_name, metrics in results_dict.items():
            f.write("-" * 80 + "\n")
            f.write(f"Model: {model_name}\n")
            f.write("-" * 80 + "\n\n")

            f.write("⏱️  Timing Information:\n")
            f.write(f"Training Time: {metrics['training_time']:.2f} seconds\n")
            f.write(f"Evaluation Time: {metrics['evaluation_time']:.2f} seconds\n")
            f.write(f"Total Time: {metrics['total_time']:.2f} seconds\n\n")

            f.write("Confusion Matrix:\n")
            cm = metrics["confusion_matrix"]
            f.write(str(cm) + "\n\n")

            f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall:    {metrics['recall']:.4f}\n")
            f.write(f"F1-Score:  {metrics['f1_score']:.4f}\n\n")

            f.write("Top 10 Features:\n")
            f.write("-" * 80 + "\n")
            top_features = metrics["feature_importance"].head(10)
            f.write(f"{'Feature':<30} {'Importance':<15}\n")
            f.write("-" * 80 + "\n")
            for _, row in top_features.iterrows():
                f.write(f"{row['feature']:<30} {row['importance']:<15.6f}\n")
            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("Comparison completed successfully!\n")
        f.write("=" * 80 + "\n")

    return str(filename)


# =============================================================================
# Persistence
# =============================================================================


import pickle
from pathlib import Path


def save_model(
    model,
    scaler,
    label_encoder,
    feature_names: List[str],
    filepath: str = "drdos_detector_model.pkl",
) -> None:
    """Save trained model and preprocessing objects."""
    print("\n" + "=" * 80)
    print("Saving model and scaler...")

    model_data = {
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_names": feature_names,
    }

    with open(filepath, "wb") as f:
        pickle.dump(model_data, f)

    print(f"Model saved as '{filepath}'")
    print("=" * 80)
    print("Pipeline completed successfully!")
    print("=" * 80)


def load_model(filepath: str = "drdos_detector_model.pkl") -> Dict:
    """Load trained model and preprocessing objects."""
    with open(filepath, "rb") as f:
        model_data = pickle.load(f)
    return model_data
