"""
Main training pipeline for DrDoS DNS Attack Detection
Orchestrates the complete training workflow
"""

import warnings
import numpy as np

from data_preprocessing import load_dataset, clean_data, encode_labels
from data_splitting import split_balanced_data
from data_balancing import balance_with_smote
from model_training import scale_features, train_random_forest
from model_evaluation import evaluate_model
from model_persistence import save_model

warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
CSV_PATH = r'c:\Users\Kulis\Documents\Πτυχιακή\DrDoS-Detector\DrDoS_DNS.csv'
MODEL_PATH = 'drdos_detector_model.pkl'

# Model parameters
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 30,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE
}


def main():
    """Main training pipeline"""
    
    # Set random seed
    np.random.seed(RANDOM_STATE)
    
    print("="*80)
    print("DrDoS DNS Attack Detection - Training Pipeline")
    print("="*80)
    
    # Step 1: Load dataset
    df = load_dataset(CSV_PATH)
    
    # Step 2: Clean and preprocess data
    X, y, label_col = clean_data(df)
    
    # Encode labels
    y_encoded, le_label = encode_labels(y)
    
    # Step 3: Split data with balanced test set
    (X_train_original, X_test, y_train_original, y_test,
     X_train_benign, y_train_benign, X_train_attack, y_train_attack) = split_balanced_data(
        X, y, y_encoded, le_label, RANDOM_STATE
    )
    
    # Step 4: Balance training data with SMOTE
    X_train_balanced, y_train_balanced = balance_with_smote(
        X_train_benign, y_train_benign,
        X_train_attack, y_train_attack,
        le_label, RANDOM_STATE
    )
    
    # Step 5: Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train_balanced, X_test
    )
    
    # Step 6: Train model
    clf = train_random_forest(X_train_scaled, y_train_balanced, **RF_PARAMS)
    
    # Step 7: Evaluate model
    metrics = evaluate_model(clf, X_test_scaled, y_test, le_label, X.columns.tolist())
    
    # Save model
    save_model(clf, scaler, le_label, X.columns.tolist(), MODEL_PATH)
    
    return clf, scaler, le_label, metrics


if __name__ == "__main__":
    main()
