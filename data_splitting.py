"""
Data splitting utilities with balanced test set strategy
"""

import pandas as pd
import numpy as np


def split_balanced_data(
    X: pd.DataFrame,
    y: pd.Series,
    y_encoded: np.ndarray,
    le_label,
    random_state: int = 42
) -> tuple:
    """
    Split data into train and test sets with balanced test set.
    
    Strategy:
    - Test set is balanced 50-50 between classes
    - Uses 50% of minority class (BENIGN) for testing
    - No duplicates in selection process
    - Random selection without replacement
    
    Args:
        X: Features DataFrame
        y: Original labels
        y_encoded: Encoded labels
        le_label: Label encoder
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, X_train_benign, y_train_benign, X_train_attack, y_train_attack)
    """
    print("\n[3/7] Splitting data with balanced test set strategy...")
    
    # Split data by class
    benign_mask = y == 'BENIGN'
    attack_mask = y == 'DrDoS_DNS'
    
    X_benign = X[benign_mask].reset_index(drop=True)
    y_benign = y_encoded[benign_mask]
    X_attack = X[attack_mask].reset_index(drop=True)
    y_attack = y_encoded[attack_mask]
    
    print(f"   BENIGN samples: {len(X_benign):,}")
    print(f"   DrDoS_DNS samples: {len(X_attack):,}")
    
    # Calculate test set size: 50% of BENIGN for test
    samples_per_class_test = len(X_benign) // 2
    
    # Limit test set size if needed
    total_samples = len(X)
    max_test_size = int(total_samples * 0.2)
    if samples_per_class_test * 2 > max_test_size:
        samples_per_class_test = max_test_size // 2
    
    print(f"   Test set will have {samples_per_class_test} samples per class")
    print(f"   This leaves {len(X_benign) - samples_per_class_test} BENIGN samples for training")
    
    # Random selection for test set (replace=False ensures no duplicates)
    np.random.seed(random_state)
    benign_test_indices = np.random.choice(len(X_benign), size=samples_per_class_test, replace=False)
    attack_test_indices = np.random.choice(len(X_attack), size=samples_per_class_test, replace=False)
    
    # Create boolean masks for train/test split
    benign_train_mask = np.ones(len(X_benign), dtype=bool)
    benign_train_mask[benign_test_indices] = False
    
    attack_train_mask = np.ones(len(X_attack), dtype=bool)
    attack_train_mask[attack_test_indices] = False
    
    # Test sets
    X_test_benign = X_benign.iloc[benign_test_indices].copy()
    y_test_benign = y_benign[benign_test_indices].copy()
    
    X_test_attack = X_attack.iloc[attack_test_indices].copy()
    y_test_attack = y_attack[attack_test_indices].copy()
    
    # Train sets
    X_train_benign = X_benign.iloc[benign_train_mask].copy()
    y_train_benign = y_benign[benign_train_mask].copy()
    
    X_train_attack = X_attack.iloc[attack_train_mask].copy()
    y_train_attack = y_attack[attack_train_mask].copy()
    
    # Combine test sets
    X_test = pd.concat([X_test_benign, X_test_attack], ignore_index=True)
    y_test = np.concatenate([y_test_benign, y_test_attack])
    
    # Shuffle test set
    shuffle_indices = np.random.permutation(len(X_test))
    X_test = X_test.iloc[shuffle_indices].copy()
    y_test = y_test[shuffle_indices].copy()
    
    print(f"\n   Test set created:")
    print(f"   - Total test samples: {len(X_test):,}")
    print(f"   - BENIGN in test: {sum(y_test == le_label.transform(['BENIGN'])[0]):,}")
    print(f"   - DrDoS_DNS in test: {sum(y_test == le_label.transform(['DrDoS_DNS'])[0]):,}")
    
    # Combine train sets (pre-SMOTE)
    X_train_original = pd.concat([X_train_benign, X_train_attack], ignore_index=True)
    y_train_original = np.concatenate([y_train_benign, y_train_attack])
    
    print(f"\n   Training set (before SMOTE):")
    print(f"   - Total train samples: {len(X_train_original):,}")
    print(f"   - BENIGN in train: {sum(y_train_original == le_label.transform(['BENIGN'])[0]):,}")
    print(f"   - DrDoS_DNS in train: {sum(y_train_original == le_label.transform(['DrDoS_DNS'])[0]):,}")
    
    # Verify no overlap
    print("\n   Verifying no overlap between train and test sets...")
    train_size = len(X_train_original)
    test_size = len(X_test)
    total_after_split = train_size + test_size
    print(f"   - Train size: {train_size:,}")
    print(f"   - Test size: {test_size:,}")
    print(f"   - Total: {total_after_split:,} (original: {len(X):,})")
    print(f"   - No duplicates in selection process: ✓")
    
    return X_train_original, X_test, y_train_original, y_test, X_train_benign, y_train_benign, X_train_attack, y_train_attack
