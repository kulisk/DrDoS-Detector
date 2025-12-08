# -*- coding: utf-8 -*-
"""
DrDoS Amplification Detection - Utility Functions
=================================================
All helper functions organized by pipeline stages.

Author: DrDoS-Detector Team
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import pickle
import time


# ============================================================================
# STAGE FUNCTIONS (Top-level pipeline stages)
# ============================================================================

def preprocessing_stage(csv_path, smote_ratio, test_size, random_state):
    """
    Complete preprocessing pipeline: Load → Clean → Encode → Balance → Split → Scale
    
    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder)
    """
    # Load and clean
    df = _load_dataset(csv_path)
    X, y, label_col = _clean_data(df)
    y_encoded, le = _encode_labels(y)
    
    # Balance with SMOTE
    X_smote, y_smote = _apply_smote(X, y_encoded, smote_ratio, random_state)
    
    # Split
    X_train, X_test, y_train, y_test = _split_data(X_smote, y_smote, test_size, random_state)
    
    # Scale
    X_train_scaled, X_test_scaled, scaler = _scale_features(X_train, X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le


def training_stage(model_name, X_train, y_train, params):
    """
    Train a model with given parameters
    
    Returns:
        tuple: (trained_model, training_time)
    """
    return _train_model(model_name, X_train, y_train, params)


def evaluation_stage(model, X_test, y_test, label_encoder):
    """
    Evaluate model performance
    
    Returns:
        dict: {'accuracy', 'predictions', 'confusion_matrix', 'report', 'eval_time'}
    """
    return _evaluate_model(model, X_test, y_test, label_encoder)


def persistence_stage(model, scaler, label_encoder, filename):
    """
    Save trained model with scaler and label encoder
    """
    _save_model(model, scaler, label_encoder, filename)


# ============================================================================
# PREPROCESSING HELPERS
# ============================================================================

def _load_dataset(csv_path):
    """Load the DrDoS dataset from CSV file"""
    print("\n[1/7] Loading dataset...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"   Total samples: {len(df):,}")
    print(f"   Total features: {len(df.columns)}")
    return df


def _clean_data(df):
    """Clean and preprocess the dataset"""
    print("\n[2/7] Cleaning data...")
    
    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Identify label column
    label_col = ' Label' if ' Label' in df.columns else 'Label'
    
    # Separate features and labels
    X = df.drop(columns=[label_col])
    y = df[label_col]
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN with 0
    X = X.fillna(0)
    
    print(f"   Features: {X.shape[1]}")
    print(f"   Samples: {X.shape[0]:,}")
    
    return X, y, label_col


def _encode_labels(y):
    """Encode string labels to numeric"""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print("\n   Label distribution:")
    for label, count in zip(*np.unique(y_encoded, return_counts=True)):
        label_name = le.inverse_transform([label])[0]
        pct = 100 * count / len(y_encoded)
        print(f"     {label_name}: {count:,} ({pct:.2f}%)")
    
    return y_encoded, le


# ============================================================================
# BALANCING HELPERS
# ============================================================================

def _apply_smote(X, y, target_ratio, random_state):
    """Apply SMOTE to balance BENIGN class"""
    print("\n[3/7] Applying SMOTE to BENIGN class...")
    
    benign_count = np.sum(y == 0)
    ddos_count = np.sum(y == 1)
    
    print(f"   Original BENIGN: {benign_count:,}")
    print(f"   Original DDoS: {ddos_count:,}")
    
    target_benign = benign_count * target_ratio
    
    smote = SMOTE(
        sampling_strategy={0: target_benign},
        random_state=random_state,
        n_jobs=-1
    )
    
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    new_benign = np.sum(y_resampled == 0)
    print(f"   After SMOTE BENIGN: {new_benign:,} (multiplied by {target_ratio}x)")
    print(f"   Total samples after SMOTE: {len(X_resampled):,}")
    
    return X_resampled, y_resampled


# ============================================================================
# SPLITTING HELPERS
# ============================================================================

def _split_data(X, y, test_size, random_state):
    """Split data into train/test sets"""
    print("\n[4/7] Splitting data...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"   Train samples: {len(X_train):,}")
    print(f"   Test samples: {len(X_test):,}")
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# SCALING HELPERS
# ============================================================================

def _scale_features(X_train, X_test):
    """Scale features using StandardScaler"""
    print("\n[5/7] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# ============================================================================
# TRAINING HELPERS
# ============================================================================

def _train_model(model_name, X_train, y_train, params):
    """Train a single model"""
    
    models = {
        'Logistic Regression': LogisticRegression(**params),
        'Random Forest': RandomForestClassifier(**params),
        'Decision Tree': DecisionTreeClassifier(**params),
        'SVM': SVC(**params),
        'KNN': KNeighborsClassifier(**params)
    }
    
    clf = models[model_name]
    
    start_time = time.time()
    clf.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    return clf, training_time


# ============================================================================
# EVALUATION HELPERS
# ============================================================================

def _evaluate_model(clf, X_test, y_test, le):
    """Evaluate model performance"""
    
    start_time = time.time()
    y_pred = clf.predict(X_test)
    eval_time = time.time() - start_time
    
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'confusion_matrix': cm,
        'report': report,
        'eval_time': eval_time
    }


# ============================================================================
# PERSISTENCE HELPERS
# ============================================================================

def _save_model(clf, scaler, le, filename):
    """Save trained model"""
    model_data = {
        'model': clf,
        'scaler': scaler,
        'label_encoder': le
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n   Model saved: {filename}")


def load_model(filename='model.pkl'):
    """Load trained model"""
    with open(filename, 'rb') as f:
        model_data = pickle.load(f)
    return model_data
