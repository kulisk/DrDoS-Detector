# -*- coding: utf-8 -*-
"""
Malicious DoH Detection - Main Script
=====================================
Detects DNS-over-HTTPS traffic (Stage 1: DoH vs non-DoH).

Author: DrDoS-Detector Team
"""

import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

from utils import (
    data_loading_stage,
    preprocessing_stage,
    training_stage,
    evaluation_stage,
    save_report,
    save_model,
)

print("="*80)
print("MALICIOUS DNS-over-HTTPS (DoH) DETECTION")
print("="*80)
print("\nStage 1: DoH vs non-DoH traffic classification")
print()

# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
SAMPLE_LIMIT = 50000
BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "doh_detection_best_model.pkl"

# ============================================================================
# DATASET PATHS
# ============================================================================

STAGE1_DOH = BASE_DIR / 'datasets' / 'cira-cic-dohbrw-2020' / 'l1-doh.csv'
STAGE1_NONDOH = BASE_DIR / 'datasets' / 'cira-cic-dohbrw-2020' / 'l1-nondoh.csv'

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

MODEL_PARAMS = {
    'Random Forest': {
        'n_estimators': 100,
        'max_depth': 20,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    },
    'Decision Tree': {
        'max_depth': 20,
        'random_state': RANDOM_STATE
    }
}


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline with stage calls"""
    
    # Stage 1: Data Loading
    df = data_loading_stage(STAGE1_DOH, STAGE1_NONDOH, SAMPLE_LIMIT)
    if df is None:
        print("\n[ERROR] Failed to load data")
        return
    
    # Stage 2: Preprocessing
    X_train, X_test, y_train, y_test, le_label = preprocessing_stage(df, TEST_SIZE, RANDOM_STATE)
    
    # Stage 3: Training
    results, scaler, X_test_scaled = training_stage(X_train, X_test, y_train, MODEL_PARAMS, RANDOM_STATE)
    
    # Stage 4: Evaluation
    best_model_name, best_accuracy = evaluation_stage(results, y_test, le_label)

    # Save report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / 'doh_detection_report.txt'
    save_report(results, y_test, le_label, report_path)

    # Persist best model
    best_model = results[best_model_name]['model']
    save_model(
        model=best_model,
        scaler=scaler,
        label_encoder=le_label,
        feature_names=X_train.columns.tolist(),
        filepath=str(MODEL_PATH),
        model_name=best_model_name,
    )
    
    # Summary
    print("\n" + "="*80)
    print("Best Model: {} ({:.4f})".format(best_model_name, best_accuracy))
    print("="*80)
    
    print("\n" + "="*80)
    print("MALICIOUS DoH DETECTION - COMPLETED")
    print("="*80)
    
    print("\nBest model: {best_model_name}")
    print("Can detect:")
    print("  - DNS-over-HTTPS (DoH) traffic")
    print("  - Encrypted DNS communication")
    print("  - Malware using DoH for C&C")
    
    print("\n" + "="*80)
    print(f"Report saved: {report_path}")
    print(f"Best model saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
