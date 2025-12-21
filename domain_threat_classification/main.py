# -*- coding: utf-8 -*-
"""DNS Domain Threat Classification (ML) - Main Script
================================
Trains a machine learning model to classify domains into 4 categories
using the CIC-Bell-DNS 2021 dataset CSVs.

Uses:
- CIC-Bell-DNS 2021: CSV_benign.csv, CSV_malware.csv, CSV_phishing.csv, CSV_spam.csv

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
print("DNS DOMAIN THREAT CLASSIFICATION (ML)")
print("="*80)
print("\nClassifying domains into Benign/Malware/Phishing/Spam using ML\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / 'reports'
OUTPUT_REPORT = REPORTS_DIR / 'dns_reflector_analysis_report.txt'

RANDOM_STATE = 42
TEST_SIZE = 0.20

# ============================================================================
# DATASET PATHS (ONLY CSVs)
# ============================================================================

LABELED_DOMAIN_CSVS = {
    'Benign': BASE_DIR / 'datasets' / 'CIC-Bell-DNS 2021' / 'CSV_benign.csv',
    'Malware': BASE_DIR / 'datasets' / 'CIC-Bell-DNS 2021' / 'CSV_malware.csv',
    'Phishing': BASE_DIR / 'datasets' / 'CIC-Bell-DNS 2021' / 'CSV_phishing.csv',
    'Spam': BASE_DIR / 'datasets' / 'CIC-Bell-DNS 2021' / 'CSV_spam.csv',
}

# To keep training time reasonable with very large benign CSVs.
MAX_SAMPLES_PER_CLASS = 50_000


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline"""

    df = data_loading_stage(
        LABELED_DOMAIN_CSVS,
        max_samples_per_class=MAX_SAMPLES_PER_CLASS,
        random_state=RANDOM_STATE,
    )
    if df is None or df.empty:
        print("\n[ERROR] No training data found!")
        return

    X_train, X_test, y_train, y_test, le_label = preprocessing_stage(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    model_params = {
        'Random Forest': {
            'n_estimators': 200,
            'max_depth': 25,
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'class_weight': 'balanced_subsample',
        },
        'Decision Tree': {
            'max_depth': 25,
            'random_state': RANDOM_STATE,
            'class_weight': 'balanced',
        },
        'Logistic Regression': {
            'max_iter': 2000,
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'solver': 'lbfgs',
            'class_weight': 'balanced',
        },
    }

    results, scaler, X_test_scaled = training_stage(
        X_train,
        X_test,
        y_train,
        model_params,
        random_state=RANDOM_STATE,
    )

    best_model_name, best_accuracy = evaluation_stage(results, y_test, le_label)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    save_report(results, y_test, le_label, OUTPUT_REPORT)

    print("\n" + "="*80)
    print(f"Best Model: {best_model_name} ({best_accuracy:.4f})")
    print("DNS DOMAIN THREAT CLASSIFICATION (ML) - COMPLETED")
    print("="*80)
    print(f"Report saved: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
