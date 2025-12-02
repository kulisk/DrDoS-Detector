"""
Model evaluation and metrics utilities
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)


def get_next_results_filename(base_name='training_results', extension='txt'):
    """
    Generate next available filename with incremental numbering.
    
    Args:
        base_name: Base name for the file
        extension: File extension (without dot)
        
    Returns:
        Next available filename (e.g., 'training_results_1.txt')
    """
    counter = 1
    while True:
        filename = f"{base_name}_{counter}.{extension}"
        if not os.path.exists(filename):
            return filename
        counter += 1


def save_results_to_file(
    metrics: dict,
    config: dict,
    train_test_info: dict,
    le_label,
    filename: str = None
) -> str:
    """
    Save training results to a text file.
    
    Args:
        metrics: Dictionary with evaluation metrics
        config: Configuration parameters
        train_test_info: Information about train/test split
        le_label: Label encoder
        filename: Optional filename (auto-generated if None)
        
    Returns:
        Path to the saved file
    """
    if filename is None:
        filename = get_next_results_filename()
    
    with open(filename, 'w', encoding='utf-8') as f:
        # Header
        f.write("="*80 + "\n")
        f.write("DrDoS DNS Attack Detection - Training Results\n")
        f.write("="*80 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Results saved to: {filename}\n")
        f.write("="*80 + "\n\n")
        
        # Configuration
        f.write("CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Dataset Information
        f.write("DATASET INFORMATION\n")
        f.write("-" * 80 + "\n")
        for key, value in train_test_info.items():
            if isinstance(value, (int, float)):
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value:,}\n")
            else:
                f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Results
        f.write("="*80 + "\n")
        f.write("RESULTS\n")
        f.write("="*80 + "\n\n")
        
        # Confusion Matrix
        f.write("Confusion Matrix:\n")
        cm = metrics['confusion_matrix']
        f.write(str(cm) + "\n\n")
        
        # Classification Report (reconstructed)
        f.write("Classification Report:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
        f.write("-" * 80 + "\n")
        for class_name in le_label.classes_:
            f.write(f"{class_name:<15} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1_score']:<12.4f}\n")
        f.write("\n")
        
        # Summary Metrics
        f.write("Summary Metrics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1-Score:  {metrics['f1_score']:.4f}\n")
        f.write("\n")
        
        # Feature Importance
        f.write("="*80 + "\n")
        f.write("TOP 20 MOST IMPORTANT FEATURES\n")
        f.write("="*80 + "\n")
        feature_importance = metrics['feature_importance'].head(20)
        f.write(f"{'Feature':<30} {'Importance':<15}\n")
        f.write("-" * 80 + "\n")
        for idx, row in feature_importance.iterrows():
            f.write(f"{row['feature']:<30} {row['importance']:<15.6f}\n")
        f.write("\n")
        
        # Footer
        f.write("="*80 + "\n")
        f.write("Pipeline completed successfully!\n")
        f.write("="*80 + "\n")
    
    return filename


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
    
    # Feature Importance (handle different model types)
    print("\n" + "="*80)
    print("TOP 20 MOST IMPORTANT FEATURES")
    print("="*80)
    
    # Check if model has feature_importances_ (RandomForest, DecisionTree)
    # or coef_ (LogisticRegression, SVM)
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
    elif hasattr(clf, 'coef_'):
        # For Logistic Regression, use absolute values of coefficients
        importances = np.abs(clf.coef_[0])
    else:
        importances = np.zeros(len(feature_names))
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
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
