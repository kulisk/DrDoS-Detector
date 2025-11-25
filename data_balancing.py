"""
SMOTE and data balancing utilities
"""

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE


def apply_smote_to_benign(
    X_benign: pd.DataFrame,
    y_benign: np.ndarray,
    target_samples: int,
    random_state: int = 42
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Apply SMOTE to BENIGN (minority) class BEFORE splitting.
    
    Args:
        X_benign: BENIGN features
        y_benign: BENIGN labels
        target_samples: Target number of samples after SMOTE
        random_state: Random seed
        
    Returns:
        Tuple of (SMOTE augmented X, SMOTE augmented y)
    """
    print("\n[3/7] Applying SMOTE to BENIGN class...")
    print(f"   Original BENIGN samples: {len(X_benign):,}")
    print(f"   Target BENIGN samples after SMOTE: {target_samples:,}")
    
    # Create a temporary dataset with enough DDoS samples for SMOTE to work
    # SMOTE needs at least 2 classes, so we add dummy DDoS samples
    y_dummy_attack = np.ones(target_samples, dtype=int) * (1 - y_benign[0])
    X_dummy_attack = pd.DataFrame(
        np.zeros((target_samples, X_benign.shape[1])),
        columns=X_benign.columns
    )
    
    # Combine
    X_combined = pd.concat([X_benign, X_dummy_attack], ignore_index=True)
    y_combined = np.concatenate([y_benign, y_dummy_attack])
    
    # Apply SMOTE with specific sampling strategy
    smote = SMOTE(
        sampling_strategy={y_benign[0]: target_samples},
        random_state=random_state,
        k_neighbors=min(5, len(X_benign) - 1)
    )
    
    X_resampled, y_resampled = smote.fit_resample(X_combined, y_combined)
    
    # Extract only BENIGN samples (SMOTE-generated)
    benign_mask = y_resampled == y_benign[0]
    X_benign_smote = pd.DataFrame(X_resampled[benign_mask], columns=X_benign.columns)
    y_benign_smote = y_resampled[benign_mask]
    
    print(f"   BENIGN samples after SMOTE: {len(X_benign_smote):,}")
    print(f"   SMOTE generated {len(X_benign_smote) - len(X_benign):,} synthetic samples")
    
    return X_benign_smote, y_benign_smote
