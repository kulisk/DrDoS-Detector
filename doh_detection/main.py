# -*- coding: utf-8 -*-
"""
Malicious DoH Detection - Main Script
=====================================
Detects DNS-over-HTTPS traffic (Stage 1: DoH vs non-DoH).

Author: DrDoS-Detector Team
"""

import warnings
warnings.filterwarnings('ignore')

from utils import data_loading_stage, preprocessing_stage, training_stage, evaluation_stage

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

# ============================================================================
# DATASET PATHS
# ============================================================================

STAGE1_DOH = '../datasets/l1-doh.csv'
STAGE1_NONDOH = '../datasets/l1-nondoh.csv'

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


if __name__ == "__main__":
    main()
