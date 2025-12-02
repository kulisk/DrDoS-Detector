"""
Model comparison utilities
"""

import pandas as pd
from datetime import datetime


def compare_models(results_dict: dict, label_encoder) -> pd.DataFrame:
    """
    Compare multiple models and create a comparison table.
    
    Args:
        results_dict: Dictionary with model names as keys and metrics as values
        label_encoder: Label encoder for class names
        
    Returns:
        DataFrame with comparison results
    """
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    comparison_data = []
    
    for model_name, metrics in results_dict.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1_score']:.4f}",
            'Training Time (s)': f"{metrics['training_time']:.2f}",
            'Total Time (s)': f"{metrics['total_time']:.2f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by accuracy descending
    comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
    
    print(comparison_df.to_string(index=False))
    print("\n" + "="*80)
    
    # Find best model
    best_idx = comparison_df.index[0]
    best_model = comparison_df.iloc[0]['Model']
    print(f"🏆 Best Model: {best_model}")
    print("="*80)
    
    return comparison_df


def save_comparison_to_file(comparison_df: pd.DataFrame, results_dict: dict, 
                            config: dict, train_test_info: dict, label_encoder):
    """
    Save comparison results to a text file with auto-incrementing filename.
    
    Args:
        comparison_df: DataFrame with comparison results
        results_dict: Dictionary with all model results
        config: Configuration dictionary
        train_test_info: Train/test split information
        label_encoder: Label encoder
        
    Returns:
        str: Filename where results were saved
    """
    # Find next available filename
    counter = 1
    while True:
        filename = f"comparison_results_{counter}.txt"
        try:
            with open(filename, 'r'):
                counter += 1
        except FileNotFoundError:
            break
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("DrDoS DNS Attack Detection - Model Comparison Results\n")
        f.write("="*80 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Results saved to: {filename}\n")
        f.write("="*80 + "\n\n")
        
        # Configuration
        f.write("CONFIGURATION\n")
        f.write("-"*80 + "\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        
        # Dataset info
        f.write("\nDATASET INFORMATION\n")
        f.write("-"*80 + "\n")
        for key, value in train_test_info.items():
            f.write(f"{key}: {value:,}" if isinstance(value, int) else f"{key}: {value}\n")
        
        # Model Comparison
        f.write("\n" + "="*80 + "\n")
        f.write("MODEL COMPARISON\n")
        f.write("="*80 + "\n\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        # Best model
        best_model = comparison_df.iloc[0]['Model']
        f.write("="*80 + "\n")
        f.write(f"🏆 Best Model: {best_model}\n")
        f.write("="*80 + "\n\n")
        
        # Detailed results for each model
        f.write("="*80 + "\n")
        f.write("DETAILED RESULTS FOR EACH MODEL\n")
        f.write("="*80 + "\n\n")
        
        for model_name, metrics in results_dict.items():
            f.write("-"*80 + "\n")
            f.write(f"Model: {model_name}\n")
            f.write("-"*80 + "\n\n")
            
            # Timing Information
            f.write("⏱️  Timing Information:\n")
            f.write(f"Training Time: {metrics['training_time']:.2f} seconds\n")
            f.write(f"Evaluation Time: {metrics['evaluation_time']:.2f} seconds\n")
            f.write(f"Total Time: {metrics['total_time']:.2f} seconds\n\n")
            
            # Confusion Matrix
            f.write("Confusion Matrix:\n")
            cm = metrics['confusion_matrix']
            f.write(str(cm) + "\n\n")
            
            # Metrics
            f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall:    {metrics['recall']:.4f}\n")
            f.write(f"F1-Score:  {metrics['f1_score']:.4f}\n\n")
            
            # Top 10 features
            f.write("Top 10 Features:\n")
            f.write("-"*80 + "\n")
            top_features = metrics['feature_importance'].head(10)
            f.write(f"{'Feature':<30} {'Importance':<15}\n")
            f.write("-"*80 + "\n")
            for _, row in top_features.iterrows():
                f.write(f"{row['feature']:<30} {row['importance']:<15.6f}\n")
            f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("Comparison completed successfully!\n")
        f.write("="*80 + "\n")
    
    return filename
