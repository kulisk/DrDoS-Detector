"""
Test script για να ελέγξουμε τη φόρτωση και επεξεργασία δεδομένων
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("TEST: Φόρτωση και Επεξεργασία Δεδομένων")
print("=" * 70)

# 1. Φόρτωση
print("\n1️⃣ Φόρτωση δεδομένων...")
data = pd.read_csv('DrDoS_DNS.csv', low_memory=False)
print(f"✓ Φορτώθηκαν {len(data):,} εγγραφές με {data.shape[1]} στήλες")

# 2. Έλεγχος τύπων
print("\n2️⃣ Έλεγχος τύπων δεδομένων...")
print(f"Τύποι στηλών:\n{data.dtypes.value_counts()}")

# 3. Επιλογή αριθμητικών στηλών
print("\n3️⃣ Επιλογή αριθμητικών στηλών...")
label_col = data[' Label'] if ' Label' in data.columns else None
numeric_data = data.select_dtypes(include=[np.number])
print(f"✓ Αριθμητικές στήλες: {numeric_data.shape[1]}")

if label_col is not None:
    data = numeric_data.copy()
    data[' Label'] = label_col
else:
    data = numeric_data

print(f"✓ Νέο shape: {data.shape}")

# 4. Καθαρισμός inf/nan
print("\n4️⃣ Καθαρισμός inf/nan...")
initial_len = len(data)

# Αντικατάσταση inf με NaN
data = data.replace([np.inf, -np.inf], np.nan)

# Γέμισμα με median
for col in data.columns:
    if col != ' Label':
        if data[col].isnull().any():
            median_val = data[col].median()
            if pd.notna(median_val):
                data[col].fillna(median_val, inplace=True)
            else:
                data[col].fillna(0, inplace=True)

# Διαγραφή γραμμών με NaN στο Label
data = data.dropna(subset=[' Label'])

print(f"✓ Από {initial_len:,} → {len(data):,} εγγραφές")

# 5. Έλεγχος κλάσεων
print("\n5️⃣ Έλεγχος κλάσεων...")
if ' Label' in data.columns:
    label_counts = data[' Label'].value_counts()
    print(f"Κατανομή κλάσεων:")
    for label, count in label_counts.items():
        pct = (count / len(data)) * 100
        print(f"   {label}: {count:,} ({pct:.2f}%)")
    
    if len(label_counts) < 2:
        print("❌ ΠΡΟΒΛΗΜΑ: Μόνο μία κλάση!")
        print("\nΔοκιμάζω εναλλακτική μέθοδο...")
        
        # Επαναφόρτωση χωρίς επιλογή numeric
        print("\n📂 Επαναφόρτωση δεδομένων...")
        data = pd.read_csv('DrDoS_DNS.csv', low_memory=False)
        
        # Κρατάμε Label ξεχωριστά
        label_col = data[' Label']
        
        # Διαγράφουμε προβληματικές στήλες (string IPs κτλ)
        columns_to_drop = []
        for col in data.columns:
            if col != ' Label':
                # Αν η στήλη έχει strings που δεν μπορούν να γίνουν αριθμοί
                try:
                    pd.to_numeric(data[col], errors='raise')
                except:
                    columns_to_drop.append(col)
        
        print(f"Διαγραφή {len(columns_to_drop)} μη-αριθμητικών στηλών")
        data = data.drop(columns=columns_to_drop)
        
        # Μετατροπή σε numeric
        for col in data.columns:
            if col != ' Label':
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Αντικατάσταση inf/nan
        data = data.replace([np.inf, -np.inf], np.nan)
        for col in data.columns:
            if col != ' Label':
                median_val = data[col].median()
                if pd.notna(median_val):
                    data[col].fillna(median_val, inplace=True)
                else:
                    data[col].fillna(0, inplace=True)
        
        print(f"✓ Τελικό shape: {data.shape}")
        
        # Έλεγχος ξανά
        label_counts = data[' Label'].value_counts()
        print(f"Νέα κατανομή:")
        for label, count in label_counts.items():
            pct = (count / len(data)) * 100
            print(f"   {label}: {count:,} ({pct:.2f}%)")
else:
    print("❌ Δεν βρέθηκε στήλη ' Label'")
    exit(1)

# 6. Διαχωρισμός X, y
print("\n6️⃣ Διαχωρισμός features και target...")
X = data.drop(' Label', axis=1)

# Σωστό mapping: BENIGN -> 0, DrDoS_DNS -> 1
print(f"Unique labels: {data[' Label'].unique()}")
y = data[' Label'].apply(lambda x: 0 if x in ['Normal', 'BENIGN'] else 1)

print(f"X shape: {X.shape}")
print(f"y unique values: {np.unique(y)}")
print(f"y counts: {pd.Series(y).value_counts()}")

if len(np.unique(y)) < 2:
    print("❌ ΤΕΛΙΚΟ ΠΡΟΒΛΗΜΑ: Μόνο μία κλάση υπάρχει!")
    exit(1)

# 7. Κανονικοποίηση
print("\n7️⃣ Κανονικοποίηση...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
print("✓ Κανονικοποίηση ολοκληρώθηκε")

# 8. SMOTE
print("\n8️⃣ Εφαρμογή SMOTE...")
class_counts = pd.Series(y).value_counts()
min_samples = class_counts.min()
k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1

print(f"Min samples: {min_samples}, k_neighbors: {k_neighbors}")

if k_neighbors >= 1:
    smote = SMOTE(sampling_strategy=1.0, random_state=42, k_neighbors=k_neighbors)
    X_balanced, y_balanced = smote.fit_resample(X_scaled, y)
    
    print(f"✓ Balanced dataset: {len(X_balanced):,}")
    print(f"   Normal: {sum(y_balanced == 0):,}")
    print(f"   Attack: {sum(y_balanced == 1):,}")
else:
    print("❌ Δεν μπορεί να εφαρμοστεί SMOTE (πολύ λίγα samples)")
    X_balanced = X_scaled
    y_balanced = y.values

# 9. Split
print("\n9️⃣ Train/Test Split...")
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.3, random_state=42, stratify=y_balanced
)

print(f"Train: {len(X_train):,}")
print(f"  Normal: {sum(y_train == 0):,} ({sum(y_train == 0)/len(y_train)*100:.1f}%)")
print(f"  Attack: {sum(y_train == 1):,} ({sum(y_train == 1)/len(y_train)*100:.1f}%)")

print(f"Test: {len(X_test):,}")
print(f"  Normal: {sum(y_test == 0):,} ({sum(y_test == 0)/len(y_test)*100:.1f}%)")
print(f"  Attack: {sum(y_test == 1):,} ({sum(y_test == 1)/len(y_test)*100:.1f}%)")

print("\n" + "=" * 70)
print("✅ ΟΛΟΚΛΗΡΩΘΗΚΕ ΕΠΙΤΥΧΩΣ!")
print("=" * 70)
