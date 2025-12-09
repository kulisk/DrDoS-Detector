"""
Main training pipeline for DrDoS DNS Attack Detection.
Uses the consolidated utilities in utils.py for all processing steps.
"""

import time
import warnings
import numpy as np
from pathlib import Path

from utils import (
    apply_smote_to_benign,
    clean_data,
    compare_models,
    encode_labels,
    evaluate_model,
    load_dataset,
    save_comparison_to_file,
    save_model,
    save_results_to_file,
    scale_features,
    split_data_after_smote,
    train_decision_tree,
    train_knn,
    train_logistic_regression,
    train_random_forest,
    train_svm,
)

warnings.filterwarnings("ignore")

# ============================================================================
# MODEL SELECTION - Enable/Disable models here
# ============================================================================
ENABLE_MODELS = {
    "Logistic Regression": True,
    "Random Forest": True,
    "SVM": False,
    "Decision Tree": True,
    "KNN": False,
}

# Configuration
RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
CSV_PATH = str(BASE_DIR / "datasets" / "DrDoS_DNS.csv")
MODEL_PATH = str(BASE_DIR / "drdos_detector_model.pkl")
TEST_SIZE = 0.20  # Test set ratio (configurable)

# SMOTE configuration
SMOTE_TARGET_RATIO = 10  # SMOTE BENIGN to be X times the original BENIGN count

# Model parameters
MODEL_PARAMS = {
    "Logistic Regression": {
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
    },
    "Random Forest": {
        "n_estimators": 100,
        "max_depth": 30,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": RANDOM_STATE,
    },
    "SVM": {
        "kernel": "rbf",
        "C": 1.0,
        "random_state": RANDOM_STATE,
    },
    "Decision Tree": {
        "max_depth": 30,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": RANDOM_STATE,
    },
    "KNN": {
        "n_neighbors": 5,
        "weights": "uniform",
    },
}


def main():
    """Main training pipeline with corrected SMOTE-first strategy."""
    np.random.seed(RANDOM_STATE)

    enabled_models = [name for name, enabled in ENABLE_MODELS.items() if enabled]
    num_models = len(enabled_models)

    if num_models == 0:
        print("ERROR: No models enabled! Please enable at least one model in ENABLE_MODELS.")
        return

    print("=" * 80)
    print("DrDoS DNS Attack Detection - Training Pipeline")
    print("=" * 80)
    print(f"Configuration: TEST_SIZE={TEST_SIZE}, SMOTE_RATIO={SMOTE_TARGET_RATIO}x")
    print(f"Enabled Models: {', '.join(enabled_models)}")
    print("=" * 80)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(CSV_PATH)
    X, y, _ = clean_data(df)
    y_encoded, le_label = encode_labels(y)

    print("\n[3/7] Separating classes...")
    benign_mask = y == "BENIGN"
    attack_mask = y == "DrDoS_DNS"

    X_benign_original = X[benign_mask].reset_index(drop=True)
    y_benign_original = y_encoded[benign_mask]
    X_attack = X[attack_mask].reset_index(drop=True)
    y_attack = y_encoded[attack_mask]

    print(f"   Original BENIGN samples: {len(X_benign_original):,}")
    print(f"   Original DDoS samples: {len(X_attack):,}")

    target_benign_samples = len(X_benign_original) * SMOTE_TARGET_RATIO
    X_benign_smote, y_benign_smote = apply_smote_to_benign(
        X_benign_original,
        y_benign_original,
        target_benign_samples,
        RANDOM_STATE,
    )

    X_train, X_test, y_train, y_test = split_data_after_smote(
        X_benign_original,
        y_benign_original,
        X_benign_smote,
        y_benign_smote,
        X_attack,
        y_attack,
        le_label,
        TEST_SIZE,
        RANDOM_STATE,
    )

    train_test_info = {
        "Total Dataset Samples": len(df),
        "Original BENIGN Samples": len(X_benign_original),
        "Original DDoS Samples": len(X_attack),
        "SMOTE BENIGN Generated": len(X_benign_smote),
        "Train Set Size": len(X_train),
        "Test Set Size": len(X_test),
        "Test Ratio": f"{TEST_SIZE * 100:.1f}%",
        "BENIGN in Test": sum(y_test == 0),
        "DDoS in Test": sum(y_test == 1),
        "BENIGN in Train": sum(y_train == 0),
        "DDoS in Train": sum(y_train == 1),
        "Total Features": len(X.columns),
    }

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    results_dict = {}
    models_dict = {}

    for model_name, enabled in ENABLE_MODELS.items():
        if not enabled:
            continue

        print("\n" + "=" * 80)
        print(f"Training: {model_name}")
        print("=" * 80)

        start_time = time.time()

        if model_name == "Logistic Regression":
            clf = train_logistic_regression(
                X_train_scaled, y_train, **MODEL_PARAMS[model_name]
            )
        elif model_name == "Random Forest":
            clf = train_random_forest(X_train_scaled, y_train, **MODEL_PARAMS[model_name])
        elif model_name == "SVM":
            clf = train_svm(X_train_scaled, y_train, **MODEL_PARAMS[model_name])
        elif model_name == "Decision Tree":
            clf = train_decision_tree(
                X_train_scaled, y_train, **MODEL_PARAMS[model_name]
            )
        elif model_name == "KNN":
            clf = train_knn(X_train_scaled, y_train, **MODEL_PARAMS[model_name])
        else:
            continue

        training_time = time.time() - start_time

        eval_start_time = time.time()
        metrics = evaluate_model(clf, X_test_scaled, y_test, le_label, X.columns.tolist())
        evaluation_time = time.time() - eval_start_time
        total_time = time.time() - start_time

        metrics["training_time"] = training_time
        metrics["evaluation_time"] = evaluation_time
        metrics["total_time"] = total_time

        print("\n⏱️  Timing Information:")
        print(f"   Training Time: {training_time:.2f} seconds")
        print(f"   Evaluation Time: {evaluation_time:.2f} seconds")
        print(f"   Total Time: {total_time:.2f} seconds")

        results_dict[model_name] = metrics
        models_dict[model_name] = clf

    if num_models > 1:
        print("\n" + "=" * 80)
        print("COMPARING ALL MODELS")
        print("=" * 80)
        comparison_df = compare_models(results_dict, le_label)

        config = {
            "CSV_PATH": CSV_PATH,
            "TEST_SIZE": TEST_SIZE,
            "SMOTE_TARGET_RATIO": SMOTE_TARGET_RATIO,
            "RANDOM_STATE": RANDOM_STATE,
            "Enabled Models": ", ".join(enabled_models),
        }

        comparison_file = save_comparison_to_file(
            comparison_df, results_dict, config, train_test_info, le_label, output_dir=REPORTS_DIR
        )
        print(f"\nComparison results saved to: {comparison_file}")

        best_model_name = comparison_df.iloc[0]["Model"]
        best_model = models_dict[best_model_name]
        model_path = f"best_model_{best_model_name.replace(' ', '_').lower()}.pkl"
        save_model(best_model, scaler, le_label, X.columns.tolist(), model_path)
        print(f"Best model ({best_model_name}) saved as: {model_path}")

    else:
        model_name = enabled_models[0]
        clf = models_dict[model_name]
        metrics = results_dict[model_name]

        save_model(clf, scaler, le_label, X.columns.tolist(), MODEL_PATH)

        config = {
            "CSV_PATH": CSV_PATH,
            "MODEL_PATH": MODEL_PATH,
            "TEST_SIZE": TEST_SIZE,
            "SMOTE_TARGET_RATIO": SMOTE_TARGET_RATIO,
            "RANDOM_STATE": RANDOM_STATE,
        }
        config.update(MODEL_PARAMS[model_name])

        results_file = save_results_to_file(metrics, config, train_test_info, le_label, output_dir=str(REPORTS_DIR))
        print(f"\nResults saved to: {results_file}")

    print("\n" + "=" * 80)
    print("Pipeline completed successfully!")
    print("=" * 80)
    return models_dict, scaler, le_label, results_dict


if __name__ == "__main__":
    main()
