# -*- coding: utf-8 -*-
"""
DNS Exfiltration Detection - Main Script
========================================
Detects data exfiltration attempts through DNS queries.

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
)

print("="*80)
print("DNS EXFILTRATION DETECTION")
print("="*80)
print("\nDetecting data exfiltration through DNS queries.")
print("Using features: domain length, entropy, subdomain patterns\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"

# ============================================================================
# DATASET PATHS
# ============================================================================

STATELESS_FILES = [
    BASE_DIR / 'datasets' / 'stateless_features-benign_1.pcap.csv',
    BASE_DIR / 'datasets' / 'stateless_features-benign_2.pcap.csv',
]

STATEFUL_FILES = [
    BASE_DIR / 'datasets' / 'stateful_features-benign_1.pcap.csv',
    BASE_DIR / 'datasets' / 'stateful_features-benign_2.pcap.csv',
]

ATTACK_FOLDERS = [
    BASE_DIR / 'CIC-Bell-DNS-EXF-2021 dataset' / 'Attack_heavy_Benign' / 'Attacks',
    BASE_DIR / 'CIC-Bell-DNS-EXF-2021 dataset' / 'Attack_Light_Benign' / 'Attacks',
]

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
    },
    'Logistic Regression': {
        'max_iter': 1000,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
}


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline with stage calls"""
    
    # Stage 1: Data Loading
    df = data_loading_stage(STATELESS_FILES, STATEFUL_FILES, ATTACK_FOLDERS)
    if df is None:
        return
    
    # Stage 2: Preprocessing
    X_train, X_test, y_train, y_test, le_label = preprocessing_stage(df, TEST_SIZE, RANDOM_STATE)
    
    # Stage 3: Training
    results, scaler, X_test_scaled = training_stage(X_train, X_test, y_train, MODEL_PARAMS, RANDOM_STATE)
    
    # Stage 4: Evaluation
    best_model_name, best_accuracy = evaluation_stage(results, y_test, le_label)

    # Save report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / 'dns_exfiltration_report.txt'
    save_report(results, y_test, le_label, report_path)
    
    # Summary
    print("\n" + "="*80)
    print("Best Model: {} ({:.4f})".format(best_model_name, best_accuracy))
    print("="*80)
    
    print("\n[6/6] Complete!")
    
    print("\n" + "="*80)
    print("DNS EXFILTRATION DETECTION - COMPLETED")
    print("="*80)
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
