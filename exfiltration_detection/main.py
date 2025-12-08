# -*- coding: utf-8 -*-
"""
DNS Exfiltration Detection - Main Script
========================================
Detects data exfiltration attempts through DNS queries.

Author: DrDoS-Detector Team
"""

import warnings
warnings.filterwarnings('ignore')

from utils import data_loading_stage, preprocessing_stage, training_stage, evaluation_stage

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

# ============================================================================
# DATASET PATHS
# ============================================================================

STATELESS_FILES = [
    '../datasets/stateless_features-benign_1.pcap.csv',
    '../datasets/stateless_features-benign_2.pcap.csv',
]

STATEFUL_FILES = [
    '../datasets/stateful_features-benign_1.pcap.csv',
    '../datasets/stateful_features-benign_2.pcap.csv',
]

ATTACK_FOLDERS = [
    '../CIC-Bell-DNS-EXF-2021 dataset/Attack_heavy_Benign/Attacks',
    '../CIC-Bell-DNS-EXF-2021 dataset/Attack_Light_Benign/Attacks',
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
    
    # Summary
    print(f"\n{'='*80}")
    print(f"Best Model: {best_model_name} ({best_accuracy:.4f})")
    print("="*80)
    
    print("\n[6/6] Complete!")
    
    print("\n" + "="*80)
    print("DNS EXFILTRATION DETECTION - COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()
