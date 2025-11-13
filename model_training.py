"""
Model training utilities
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def scale_features(X_train: np.ndarray, X_test: np.ndarray) -> tuple:
    """
    Scale features using StandardScaler.
    
    Args:
        X_train: Training features
        X_test: Test features
        
    Returns:
        Tuple of (scaled X_train, scaled X_test, scaler)
    """
    print("\n[5/7] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"   Features scaled using StandardScaler")
    print(f"   Training shape: {X_train_scaled.shape}")
    print(f"   Test shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, scaler


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 30,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    random_state: int = 42
) -> RandomForestClassifier:
    """
    Train Random Forest classifier.
    
    Args:
        X_train: Scaled training features
        y_train: Training labels
        n_estimators: Number of trees
        max_depth: Maximum depth of trees
        min_samples_split: Minimum samples to split
        min_samples_leaf: Minimum samples in leaf
        random_state: Random seed
        
    Returns:
        Trained RandomForestClassifier
    """
    print("\n[6/7] Training Random Forest classifier...")
    print("   Χρησιμοποιούνται ΟΛΕΣ οι στήλες του dataset")
    print(f"   Total features used: {X_train.shape[1]}")
    
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    clf.fit(X_train, y_train)
    
    print("\n   Training completed!")
    
    return clf
