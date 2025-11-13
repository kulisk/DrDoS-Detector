"""
SMOTE and data balancing utilities
"""

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE


def balance_with_smote(
    X_train_benign: pd.DataFrame,
    y_train_benign: np.ndarray,
    X_train_attack: pd.DataFrame,
    y_train_attack: np.ndarray,
    le_label,
    random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Balance training data using undersampling + SMOTE.
    
    Strategy:
    1. Undersample majority class (DrDoS_DNS) to 10x minority class
    2. Apply SMOTE to minority class to balance
    
    Args:
        X_train_benign: BENIGN training features
        y_train_benign: BENIGN training labels
        X_train_attack: DrDoS_DNS training features
        y_train_attack: DrDoS_DNS training labels
        le_label: Label encoder
        random_state: Random seed
        
    Returns:
        Tuple of (balanced X, balanced y)
    """
    print("\n[4/7] Applying SMOTE with undersampling to balance training set...")
    print("   Strategy: Undersample majority class first to manage memory")
    
    # Undersample majority class to 10x minority class
    target_majority_samples = len(X_train_benign) * 10
    
    print(f"   Undersampling DrDoS_DNS from {len(X_train_attack):,} to {target_majority_samples:,}")
    
    # Random selection from majority class
    np.random.seed(random_state)
    attack_downsample_indices = np.random.choice(
        len(X_train_attack),
        size=target_majority_samples,
        replace=False
    )
    
    X_train_attack_downsampled = X_train_attack.iloc[attack_downsample_indices].copy()
    y_train_attack_downsampled = y_train_attack[attack_downsample_indices].copy()
    
    # Combine for new training set
    X_train_downsampled = pd.concat([X_train_benign, X_train_attack_downsampled], ignore_index=True)
    y_train_downsampled = np.concatenate([y_train_benign, y_train_attack_downsampled])
    
    print(f"\n   Training set after undersampling:")
    print(f"   - Total: {len(X_train_downsampled):,}")
    print(f"   - BENIGN: {sum(y_train_downsampled == le_label.transform(['BENIGN'])[0]):,}")
    print(f"   - DrDoS_DNS: {sum(y_train_downsampled == le_label.transform(['DrDoS_DNS'])[0]):,}")
    
    # Apply SMOTE
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_downsampled, y_train_downsampled)
    
    print(f"\n   Training set after SMOTE:")
    print(f"   - Total samples: {len(X_train_balanced):,}")
    for label_idx, label_name in enumerate(le_label.classes_):
        count = sum(y_train_balanced == label_idx)
        print(f"   - {label_name}: {count:,}")
    
    return X_train_balanced, y_train_balanced
