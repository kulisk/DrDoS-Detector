"""
Data splitting utilities with balanced test set strategy
Splits AFTER SMOTE has been applied to BENIGN class
"""

import pandas as pd
import numpy as np


def split_data_after_smote(
    X_benign_original: pd.DataFrame,
    y_benign_original: np.ndarray,
    X_benign_smote: pd.DataFrame,
    y_benign_smote: np.ndarray,
    X_attack: pd.DataFrame,
    y_attack: np.ndarray,
    le_label,
    test_size: float = 0.20,
    random_state: int = 42
) -> tuple:
    """
    Split data into train and test sets AFTER SMOTE.
    
    Strategy (CORRECTED):
    1. Test set = ALL original BENIGN + equal number of DDoS (random)
    2. Train set = SMOTE BENIGN + remaining DDoS (random selection to achieve test_size ratio)
    3. Test set should be test_size (default 20%) of total data used
    
    Args:
        X_benign_original: Original BENIGN features (will ALL go to test)
        y_benign_original: Original BENIGN labels
        X_benign_smote: SMOTE-augmented BENIGN features (will go to train)
        y_benign_smote: SMOTE-augmented BENIGN labels
        X_attack: All DDoS features
        y_attack: All DDoS labels
        le_label: Label encoder
        test_size: Proportion of test set (default 0.20)
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    print(f"\n[4/7] Splitting data with corrected strategy (test_size={test_size})...")
    print("   Strategy:")
    print("   - Test: ALL original BENIGN + equal DDoS samples")
    print("   - Train: SMOTE BENIGN + remaining DDoS")
    
    np.random.seed(random_state)
    
    # Test set: ALL original BENIGN + same number of DDoS (random selection)
    num_benign_test = len(X_benign_original)
    num_attack_test = num_benign_test  # Equal for balanced test
    
    print(f"\n   Test set composition:")
    print(f"   - BENIGN (all original): {num_benign_test:,}")
    print(f"   - DDoS (random selection): {num_attack_test:,}")
    
    # Random selection of DDoS for test
    attack_test_indices = np.random.choice(
        len(X_attack),
        size=num_attack_test,
        replace=False
    )
    
    X_test_benign = X_benign_original.copy()
    y_test_benign = y_benign_original.copy()
    
    X_test_attack = X_attack.iloc[attack_test_indices].copy()
    y_test_attack = y_attack[attack_test_indices].copy()
    
    # Combine and shuffle test set
    X_test = pd.concat([X_test_benign, X_test_attack], ignore_index=True)
    y_test = np.concatenate([y_test_benign, y_test_attack])
    
    shuffle_indices = np.random.permutation(len(X_test))
    X_test = X_test.iloc[shuffle_indices].copy()
    y_test = y_test[shuffle_indices].copy()
    
    print(f"   - Total test samples: {len(X_test):,}")
    
    # Calculate how many train samples we need for the test_size ratio
    # test_size = test / (train + test)
    # train = test * (1 - test_size) / test_size
    num_train_total = int(len(X_test) * (1 - test_size) / test_size)
    
    print(f"\n   Calculating train set size for {test_size*100}% test ratio:")
    print(f"   - Total train samples needed: {num_train_total:,}")
    
    # Train set: Use available SMOTE BENIGN + calculated DDoS samples
    num_benign_train_available = len(X_benign_smote)
    
    # If SMOTE generated too many BENIGN, subsample them
    if num_benign_train_available > num_train_total:
        print(f"   - SMOTE BENIGN ({num_benign_train_available:,}) > needed total ({num_train_total:,})")
        print(f"   - Subsampling SMOTE BENIGN to fit ratio...")
        # Use all available, adjust with DDoS
        num_benign_train = min(num_benign_train_available, num_train_total // 2)
        num_attack_train = num_train_total - num_benign_train
    else:
        num_benign_train = num_benign_train_available
        num_attack_train = num_train_total - num_benign_train
    
    # Ensure positive values
    if num_attack_train < 0:
        print(f"   ⚠ Adjusting: using all SMOTE BENIGN and no DDoS in train")
        num_benign_train = num_benign_train_available
        num_attack_train = 0
    
    # Remaining DDoS after test selection
    attack_train_mask = np.ones(len(X_attack), dtype=bool)
    attack_train_mask[attack_test_indices] = False
    attack_available_indices = np.where(attack_train_mask)[0]
    
    print(f"\n   Train set composition:")
    print(f"   - BENIGN (SMOTE) to use: {num_benign_train:,}")
    print(f"   - DDoS available: {len(attack_available_indices):,}")
    print(f"   - DDoS needed: {num_attack_train:,}")
    
    # Select BENIGN for training (subsample if needed)
    if num_benign_train < len(X_benign_smote):
        benign_train_indices = np.random.choice(
            len(X_benign_smote),
            size=num_benign_train,
            replace=False
        )
        X_train_benign = X_benign_smote.iloc[benign_train_indices].copy()
        y_train_benign = y_benign_smote[benign_train_indices].copy()
    else:
        X_train_benign = X_benign_smote.copy()
        y_train_benign = y_benign_smote.copy()
    
    # Select DDoS for training
    if num_attack_train == 0:
        X_train_attack = pd.DataFrame(columns=X_attack.columns)
        y_train_attack = np.array([], dtype=y_attack.dtype)
    elif num_attack_train > len(attack_available_indices):
        print(f"   ⚠ Not enough DDoS samples! Using all available: {len(attack_available_indices):,}")
        X_train_attack = X_attack.iloc[attack_available_indices].copy()
        y_train_attack = y_attack[attack_available_indices].copy()
    else:
        # Random selection from available DDoS
        attack_train_indices = np.random.choice(
            attack_available_indices,
            size=num_attack_train,
            replace=False
        )
        X_train_attack = X_attack.iloc[attack_train_indices].copy()
        y_train_attack = y_attack[attack_train_indices].copy()
    
    # Combine and shuffle train set
    X_train = pd.concat([X_train_benign, X_train_attack], ignore_index=True)
    y_train = np.concatenate([y_train_benign, y_train_attack])
    
    shuffle_indices = np.random.permutation(len(X_train))
    X_train = X_train.iloc[shuffle_indices].copy()
    y_train = y_train[shuffle_indices].copy()
    
    print(f"   - Total train samples: {len(X_train):,}")
    
    # Verify the split
    actual_test_size = len(X_test) / (len(X_train) + len(X_test))
    print(f"\n   Verification:")
    print(f"   - Train size: {len(X_train):,}")
    print(f"   - Test size: {len(X_test):,}")
    print(f"   - Actual test ratio: {actual_test_size:.2%}")
    print(f"   - BENIGN in test: {sum(y_test == le_label.transform(['BENIGN'])[0]):,}")
    print(f"   - DDoS in test: {sum(y_test == le_label.transform(['DrDoS_DNS'])[0]):,}")
    print(f"   - BENIGN in train: {sum(y_train == le_label.transform(['BENIGN'])[0]):,}")
    print(f"   - DDoS in train: {sum(y_train == le_label.transform(['DrDoS_DNS'])[0]):,}")
    
    return X_train, X_test, y_train, y_test
