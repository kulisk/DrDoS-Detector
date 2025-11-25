"""
Model persistence utilities
"""

import pickle


def save_model(
    model,
    scaler,
    label_encoder,
    feature_names: list,
    filepath: str = 'drdos_detector_model.pkl'
) -> None:
    """
    Save trained model and preprocessing objects.
    
    Args:
        model: Trained classifier
        scaler: Fitted StandardScaler
        label_encoder: Fitted LabelEncoder
        feature_names: List of feature names
        filepath: Path to save the model
    """
    print("\n" + "="*80)
    print("Saving model and scaler...")
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_names': feature_names
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved as '{filepath}'")
    print("="*80)
    print("Pipeline completed successfully!")
    print("="*80)


def load_model(filepath: str = 'drdos_detector_model.pkl') -> dict:
    """
    Load trained model and preprocessing objects.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Dictionary containing model, scaler, label_encoder, and feature_names
    """
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data
