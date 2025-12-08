# -*- coding: utf-8 -*-
"""
DNS Exfiltration Detection - Utility Functions
==============================================
Helper functions organized by pipeline stages.

Author: DrDoS-Detector Team
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ============================================================================
# STAGE FUNCTIONS (Top-level pipeline stages)
# ============================================================================

def data_loading_stage(stateless_files, stateful_files, attack_folders):
    """
    Load all datasets from multiple sources
    
    Returns:
        DataFrame: Combined dataset with labels
    """
    return _load_exfiltration_data(stateless_files, stateful_files, attack_folders)


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
    Scale features and train all models
    
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

def _load_exfiltration_data(stateless_files, stateful_files, attack_folders):
    """Load and merge stateless and stateful features"""
    print("[1/6] Loading dataset...")
    
    all_data = []
    
    # Load attack data
    for folder in attack_folders:
        if os.path.exists(folder):
            print("\n  Checking {folder}...")
            for file in os.listdir(folder):
                if file.endswith('.csv'):
                    filepath = os.path.join(folder, file)
                    try:
                        df = pd.read_csv(filepath)
                        # Add label based on filename
                        if 'phishing' in file.lower():
                            df['Label'] = 'Phishing'
                        elif 'malware' in file.lower():
                            df['Label'] = 'Malware'
                        elif 'spam' in file.lower():
                            df['Label'] = 'Spam'
                        else:
                            df['Label'] = 'Malicious'
                        
                        all_data.append(df)
                        print("    [OK] {file}: {len(df):,} samples")
                    except Exception as e:
                        print("    [ERROR] {file}: {str(e)}")
    
    # Load benign data
    for filepath in stateless_files + stateful_files:
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                df['Label'] = 'Benign'
                all_data.append(df)
                print("  [OK] {os.path.basename(filepath)}: {len(df):,} samples")
            except Exception as e:
                print("  [ERROR] {os.path.basename(filepath)}: {str(e)}")
    
    if not all_data:
        print("\n[ERROR] No data found!")
        return None
    
    # Combine all data
    df_combined = pd.concat(all_data, ignore_index=True)
    
    print("\n  Total samples: {len(df_combined):,}")
    print("  Total features: {len(df_combined.columns)-1}")
    
    # Display label distribution
    print("\n  Label distribution:")
    for label, count in df_combined['Label'].value_counts().items():
        pct = 100 * count / len(df_combined)
        print("    - {label}: {count:,} ({pct:.2f}%)")
    
    return df_combined


# ============================================================================
# PREPROCESSING HELPERS
# ============================================================================

def _preprocess_data(df):
    """Clean and prepare data for training"""
    print("\n[2/6] Preprocessing data...")
    
    # Separate features and labels
    X = df.drop(columns=['Label'])
    y = df['Label']
    
    # Handle missing values
    X = X.fillna(0)
    print("  Null values after: {X.isnull().sum().sum()}")
    
    # Encode categorical columns
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Encode labels
    le_label = LabelEncoder()
    y_encoded = le_label.fit_transform(y)
    
    print("  Features: {X.shape[1]}")
    print("  Classes: {len(le_label.classes_)}")
    
    return X, y_encoded, le_label


def _split_data(X, y, test_size, random_state):
    """Split data into train/test sets"""
    print("\n[3/6] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print("  Train samples: {len(X_train):,}")
    print("  Test samples: {len(X_test):,}")
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
    """Train multiple classifiers"""
    print("\n[4/6] Training models...")
    
    models = {
        'Random Forest': RandomForestClassifier(**model_params['Random Forest']),
        'Decision Tree': DecisionTreeClassifier(**model_params['Decision Tree']),
        'Logistic Regression': LogisticRegression(**model_params['Logistic Regression'])
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
    """Evaluate and compare all models"""
    print("\n[5/6] Evaluation results...")
    
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
