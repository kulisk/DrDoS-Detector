# -*- coding: utf-8 -*-
"""
Malicious DoH Detection - Utility Functions
===========================================
Helper functions organized by pipeline stages for DoH traffic detection.

Author: DrDoS-Detector Team
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ============================================================================
# STAGE FUNCTIONS (Top-level pipeline stages)
# ============================================================================

def data_loading_stage(doh_file, nondoh_file, sample_limit):
    """
    Load DoH and non-DoH datasets
    
    Returns:
        DataFrame: Combined dataset with labels
    """
    return _load_doh_data(doh_file, nondoh_file, sample_limit)


def preprocessing_stage(df, test_size, random_state):
    """
    Preprocess data: clean → encode → split
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, label_encoder)
    """
    X, y, le_label = _preprocess_data(df)
    X_train, X_test, y_train, y_test = _split_data(X, y, test_size, random_state)
    return X_train, X_test, y_train, y_test, le_label


def training_stage(X_train, X_test, y_train, model_params, random_state):
    """
    Scale features and train models
    
    Returns:
        tuple: (results_dict, scaler, X_test_scaled)
    """
    X_train_scaled, X_test_scaled, scaler = _scale_features(X_train, X_test)
    results = _train_models(X_train_scaled, X_test_scaled, y_train, model_params, random_state)
    return results, scaler, X_test_scaled


def evaluation_stage(results, y_test, label_encoder):
    """
    Evaluate all models and find best one
    
    Returns:
        tuple: (best_model_name, best_accuracy)
    """
    return _evaluate_models(results, y_test, label_encoder)


# ============================================================================
# DATA LOADING HELPERS
# ============================================================================

def _load_doh_data(doh_file, nondoh_file, sample_limit):
    """Load DoH vs non-DoH data"""
    print("[1/5] Loading dataset...")
    
    data = []
    
    # Load DoH traffic
    if os.path.exists(doh_file):
        df_doh = pd.read_csv(doh_file, nrows=sample_limit)
        df_doh['Label'] = 'DoH'
        data.append(df_doh)
        print("  [OK] DoH samples: {len(df_doh):,}")
    else:
        print("  [ERROR] File not found: {doh_file}")
    
    # Load non-DoH traffic
    if os.path.exists(nondoh_file):
        df_nondoh = pd.read_csv(nondoh_file, nrows=sample_limit)
        df_nondoh['Label'] = 'non-DoH'
        data.append(df_nondoh)
        print("  [OK] non-DoH samples: {len(df_nondoh):,}")
    else:
        print("  [ERROR] File not found: {nondoh_file}")
    
    if not data:
        return None
    
    df_combined = pd.concat(data, ignore_index=True)
    print("\n  Total samples: {len(df_combined):,}")
    
    return df_combined


# ============================================================================
# PREPROCESSING HELPERS
# ============================================================================

def _preprocess_data(df):
    """Preprocess DoH detection data"""
    print("\n[2/5] Preprocessing data...")
    
    # Separate features and labels
    X = df.drop(columns=['Label'])
    y = df['Label']
    
    # Handle missing values
    X = X.fillna(0)
    print("  Null values: {X.isnull().sum().sum()}")
    
    # Encode categorical columns
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Encode labels
    le_label = LabelEncoder()
    y_encoded = le_label.fit_transform(y)
    
    print("  Features: {X.shape[1]}")
    print("  Classes: {le_label.classes_}")
    
    return X, y_encoded, le_label


def _split_data(X, y, test_size, random_state):
    """Split data into train/test sets"""
    print("\n[3/5] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print("  Train: {len(X_train):,}, Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


# ============================================================================
# SCALING HELPERS
# ============================================================================

def _scale_features(X_train, X_test):
    """Scale features using StandardScaler"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# ============================================================================
# TRAINING HELPERS
# ============================================================================

def _train_models(X_train_scaled, X_test_scaled, y_train, model_params, random_state):
    """Train DoH detection models"""
    print("\n[4/5] Training models...")
    
    models = {
        'Random Forest': RandomForestClassifier(**model_params['Random Forest']),
        'Decision Tree': DecisionTreeClassifier(**model_params['Decision Tree'])
    }
    
    results = {}
    
    for name, clf in models.items():
        print("\n  Training {name}...")
        clf.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = clf.predict(X_test_scaled)
        
        # Evaluate
        accuracy = accuracy_score(y_train, clf.predict(X_train_scaled))
        
        results[name] = {
            'model': clf,
            'accuracy': accuracy,
            'predictions': y_pred
        }
        
        print("    Train Accuracy: {accuracy:.4f}")
    
    return results


# ============================================================================
# EVALUATION HELPERS
# ============================================================================

def _evaluate_models(results, y_test, le_label):
    """Evaluate DoH detection models"""
    print("\n[5/5] Evaluation...")
    
    best_model_name = None
    best_accuracy = 0
    
    for name, result in results.items():
        print("\n{'='*80}")
        print("{name}")
        print("="*80)
        
        y_pred = result['predictions']
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\nTest Accuracy: {accuracy:.4f}")
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=le_label.classes_))
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
    
    return best_model_name, best_accuracy
