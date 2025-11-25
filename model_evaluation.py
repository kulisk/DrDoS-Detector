"""
Model evaluation and metrics utilities
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)


def evaluate_model(
    clf,
    X_test: np.ndarray,
    y_test: np.ndarray,
    le_label,
    feature_names: list
) -> dict:
    """
    Evaluate model and display comprehensive metrics.
    
    Args:
        clf: Trained classifier
        X_test: Scaled test features
        y_test: Test labels
        le_label: Label encoder
        feature_names: List of feature names
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n[7/7] Evaluating model...")
    
    y_pred = clf.predict(X_test)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le_label.classes_))
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\nSummary Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Feature Importance
    print("\n" + "="*80)
    print("TOP 20 MOST IMPORTANT FEATURES")
    print("="*80)
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(20).to_string(index=False))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'feature_importance': feature_importance
    }
