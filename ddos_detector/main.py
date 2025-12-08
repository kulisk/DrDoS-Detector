# -*- coding: utf-8 -*-
"""
DrDoS Amplification Detection - Main Script
===========================================
Main execution script for DrDoS DNS Attack Detection.

Author: DrDoS-Detector Team
"""

import warnings
warnings.filterwarnings('ignore')

from utils import preprocessing_stage, training_stage, evaluation_stage, persistence_stage

print("="*80)
print("DrDoS DNS ATTACK DETECTION")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_STATE = 42
CSV_PATH = '../datasets/DrDoS_DNS.csv'
TEST_SIZE = 0.20
SMOTE_TARGET_RATIO = 10

# ============================================================================
# MODEL SELECTION
# ============================================================================

ENABLE_MODELS = {
    'Logistic Regression': True,
    'Random Forest': True,
    'Decision Tree': True,
    'SVM': False,  # Slow
    'KNN': False   # Very slow
}

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

MODEL_PARAMS = {
    'Logistic Regression': {
        'max_iter': 1000,
        'random_state': RANDOM_STATE
    },
    'Random Forest': {
        'n_estimators': 100,
        'max_depth': 30,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    },
    'Decision Tree': {
        'max_depth': 30,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': RANDOM_STATE
    },
    'SVM': {
        'kernel': 'rbf',
        'C': 1.0,
        'random_state': RANDOM_STATE,
        'verbose': True
    },
    'KNN': {
        'n_neighbors': 5,
        'weights': 'uniform',
        'n_jobs': -1
    }
}


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline with stage calls"""
    
    # Stage 1: Preprocessing
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, le = preprocessing_stage(
        CSV_PATH, SMOTE_TARGET_RATIO, TEST_SIZE, RANDOM_STATE
    )
    
    # Stage 2: Training and Evaluation
    print("\n[6/7] Training models...")
    results = {}
    
    for model_name, enabled in ENABLE_MODELS.items():
        if not enabled:
            continue
        
        print(f"\n   Training {model_name}...")
        clf, train_time = training_stage(model_name, X_train_scaled, y_train, MODEL_PARAMS[model_name])
        
        eval_results = evaluation_stage(clf, X_test_scaled, y_test, le)
        
        results[model_name] = {
            'model': clf,
            'accuracy': eval_results['accuracy'],
            'train_time': train_time,
            'eval_time': eval_results['eval_time'],
            'confusion_matrix': eval_results['confusion_matrix'],
            'report': eval_results['report']
        }
        
        print(f"      Accuracy: {eval_results['accuracy']:.4f}")
        print(f"      Train time: {train_time:.2f}s")
    
    # Stage 3: Comparison and Persistence
    print("\n[7/7] Model comparison...")
    
    best_model_name = None
    best_accuracy = 0
    
    for name, result in results.items():
        print(f"\n{'='*80}")
        print(f"{name}")
        print("="*80)
        print(f"\nAccuracy: {result['accuracy']:.4f}")
        print(f"Training time: {result['train_time']:.2f}s")
        print(f"\nConfusion Matrix:")
        print(result['confusion_matrix'])
        print(f"\nClassification Report:")
        print(result['report'])
        
        if result['accuracy'] > best_accuracy:
            best_accuracy = result['accuracy']
            best_model_name = name
    
    if best_model_name:
        print(f"\n{'='*80}")
        print(f"BEST MODEL: {best_model_name} ({best_accuracy:.4f})")
        print("="*80)
        
        best_clf = results[best_model_name]['model']
        persistence_stage(best_clf, scaler, le, f'best_model_{best_model_name.lower().replace(" ", "_")}.pkl')
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
