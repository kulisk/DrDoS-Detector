"""
Model training utilities
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


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


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_iter: int = 1000,
    random_state: int = 42
) -> LogisticRegression:
    """
    Train Logistic Regression classifier.
    
    Args:
        X_train: Scaled training features
        y_train: Training labels
        max_iter: Maximum iterations for convergence
        random_state: Random seed
        
    Returns:
        Trained LogisticRegression
    """
    print("\n[6/7] Training Logistic Regression classifier...")
    print("   Χρησιμοποιούνται ΟΛΕΣ οι στήλες του dataset")
    print(f"   Total features used: {X_train.shape[1]}")
    
    clf = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    clf.fit(X_train, y_train)
    
    print("\n   Training completed!")
    
    return clf


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


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    kernel: str = 'rbf',
    C: float = 1.0,
    random_state: int = 42
) -> SVC:
    """
    Train Support Vector Machine classifier.
    
    Args:
        X_train: Scaled training features
        y_train: Training labels
        kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
        C: Regularization parameter
        random_state: Random seed
        
    Returns:
        Trained SVC
    """
    print("\n[6/7] Training SVM classifier...")
    print("   Χρησιμοποιούνται ΟΛΕΣ οι στήλες του dataset")
    print(f"   Total features used: {X_train.shape[1]}")
    
    clf = SVC(
        kernel=kernel,
        C=C,
        random_state=random_state,
        verbose=True
    )
    
    clf.fit(X_train, y_train)
    
    print("\n   Training completed!")
    
    return clf


def train_decision_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_depth: int = 30,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    random_state: int = 42
) -> DecisionTreeClassifier:
    """
    Train Decision Tree classifier.
    
    Args:
        X_train: Scaled training features
        y_train: Training labels
        max_depth: Maximum depth of tree
        min_samples_split: Minimum samples to split
        min_samples_leaf: Minimum samples in leaf
        random_state: Random seed
        
    Returns:
        Trained DecisionTreeClassifier
    """
    print("\n[6/7] Training Decision Tree classifier...")
    print("   Χρησιμοποιούνται ΟΛΕΣ οι στήλες του dataset")
    print(f"   Total features used: {X_train.shape[1]}")
    
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    
    clf.fit(X_train, y_train)
    
    print("\n   Training completed!")
    
    return clf


def train_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_neighbors: int = 5,
    weights: str = 'uniform'
) -> KNeighborsClassifier:
    """
    Train K-Nearest Neighbors classifier.
    
    Args:
        X_train: Scaled training features
        y_train: Training labels
        n_neighbors: Number of neighbors
        weights: Weight function ('uniform' or 'distance')
        
    Returns:
        Trained KNeighborsClassifier
    """
    print("\n[6/7] Training KNN classifier...")
    print("   Χρησιμοποιούνται ΟΛΕΣ οι στήλες του dataset")
    print(f"   Total features used: {X_train.shape[1]}")
    
    clf = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        n_jobs=-1
    )
    
    clf.fit(X_train, y_train)
    
    print("\n   Training completed!")
    
    return clf
