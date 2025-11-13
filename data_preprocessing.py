"""
Data loading and preprocessing utilities for DrDoS Detection
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load the DrDoS dataset from CSV file.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Loaded DataFrame
    """
    print("\n[1/7] Loading dataset...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"   Total samples: {len(df):,}")
    print(f"   Total features: {len(df.columns)}")
    return df


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, str]:
    """
    Clean and preprocess the dataset.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Tuple of (features DataFrame, labels Series, label column name)
    """
    print("\n[2/7] Cleaning data...")
    
    # Remove useless columns
    columns_to_drop = ['Unnamed: 0', 'Flow ID', ' Timestamp']
    existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=existing_cols_to_drop)
    
    # Find label column
    if ' Label' in df.columns:
        label_col = ' Label'
    elif 'Label' in df.columns:
        label_col = 'Label'
    else:
        raise ValueError("Label column not found!")
    
    print(f"   Label column: '{label_col}'")
    
    # Handle null values
    print(f"   Null values before: {df.isnull().sum().sum()}")
    df = df.dropna()
    print(f"   Null values after: {df.isnull().sum().sum()}")
    
    # Handle infinity values
    print("   Replacing infinity values...")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    # Separate features and labels
    X = df.drop(columns=[label_col])
    y = df[label_col]
    
    # Encode non-numeric columns
    print("   Encoding categorical features...")
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    print(f"\n   Class distribution:")
    for label, count in zip(*np.unique(y, return_counts=True)):
        percentage = (count / len(y)) * 100
        print(f"   - {label}: {count:,} samples ({percentage:.2f}%)")
    
    print(f"\n   Total features to be used: {len(X.columns)}")
    
    return X, y, label_col


def encode_labels(y: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    """
    Encode string labels to numeric values.
    
    Args:
        y: Series of string labels
        
    Returns:
        Tuple of (encoded labels array, label encoder)
    """
    le_label = LabelEncoder()
    y_encoded = le_label.fit_transform(y)
    return y_encoded, le_label
