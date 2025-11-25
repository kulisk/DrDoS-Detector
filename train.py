"""
Main training pipeline for DrDoS DNS Attack Detection
Orchestrates the complete training workflow

CORRECTED STRATEGY:
1. Load and clean data
2. Separate BENIGN and DDoS classes
3. Apply SMOTE to BENIGN (minority class) FIRST
4. Split: Test = ALL original BENIGN + equal DDoS, Train = SMOTE BENIGN + remaining DDoS
5. Train model
"""

import warnings
import numpy as np

from data_preprocessing import load_dataset, clean_data, encode_labels
from data_balancing import apply_smote_to_benign
from data_splitting import split_data_after_smote
from model_training import scale_features, train_random_forest
from model_evaluation import evaluate_model
from model_persistence import save_model

warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
CSV_PATH = r'c:\Users\Kulis\Documents\Πτυχιακή\DrDoS-Detector\DrDoS_DNS.csv'
MODEL_PATH = 'drdos_detector_model.pkl'
TEST_SIZE = 0.20  # Test set ratio (configurable)

# SMOTE configuration
SMOTE_TARGET_RATIO = 10  # SMOTE BENIGN to be X times the original BENIGN count

# Model parameters
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 30,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE
}


def main():
    """Main training pipeline with corrected SMOTE-first strategy"""
    
    # Set random seed
    np.random.seed(RANDOM_STATE)
    
    print("="*80)
    print("DrDoS DNS Attack Detection - Training Pipeline (CORRECTED)")
    print("="*80)
    print(f"Configuration: TEST_SIZE={TEST_SIZE}, SMOTE_RATIO={SMOTE_TARGET_RATIO}x")
    
    # Step 1: Load dataset
    df = load_dataset(CSV_PATH)
    
    # Step 2: Clean and preprocess data
    X, y, label_col = clean_data(df)
    
    # Encode labels
    y_encoded, le_label = encode_labels(y)
    
    # Step 3: Separate classes BEFORE any splitting
    print("\n[3/7] Separating classes...")
    benign_mask = y == 'BENIGN'
    attack_mask = y == 'DrDoS_DNS'
    
    X_benign_original = X[benign_mask].reset_index(drop=True)
    y_benign_original = y_encoded[benign_mask]
    X_attack = X[attack_mask].reset_index(drop=True)
    y_attack = y_encoded[attack_mask]
    
    print(f"   Original BENIGN samples: {len(X_benign_original):,}")
    print(f"   Original DDoS samples: {len(X_attack):,}")
    
    # Step 4: Apply SMOTE to BENIGN class FIRST
    target_benign_samples = len(X_benign_original) * SMOTE_TARGET_RATIO
    X_benign_smote, y_benign_smote = apply_smote_to_benign(
        X_benign_original,
        y_benign_original,
        target_benign_samples,
        RANDOM_STATE
    )
    
    # Step 5: Split data (test = all original BENIGN + equal DDoS, train = SMOTE + remaining DDoS)
    X_train, X_test, y_train, y_test = split_data_after_smote(
        X_benign_original,
        y_benign_original,
        X_benign_smote,
        y_benign_smote,
        X_attack,
        y_attack,
        le_label,
        TEST_SIZE,
        RANDOM_STATE
    )
    
    # Step 6: Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Step 7: Train model
    clf = train_random_forest(X_train_scaled, y_train, **RF_PARAMS)
    
    # Step 8: Evaluate model
    metrics = evaluate_model(clf, X_test_scaled, y_test, le_label, X.columns.tolist())
    
    # Save model
    save_model(clf, scaler, le_label, X.columns.tolist(), MODEL_PATH)
    
    return clf, scaler, le_label, metrics


if __name__ == "__main__":
    main()
