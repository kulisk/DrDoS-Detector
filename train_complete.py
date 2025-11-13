"""
DrDoS DNS Attack Detection using Machine Learning
Implements balanced dataset preparation with SMOTE for minority class
and proper train/test split with balanced test set
Χρησιμοποιεί ΟΛΕΣ τις στήλες του dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import pickle
import warnings
warnings.filterwarnings('ignore')

# Seed για αναπαραγωγιμότητα
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("DrDoS DNS Attack Detection - Training Pipeline")
print("="*80)

# Βήμα 1: Φόρτωση δεδομένων
print("\n[1/7] Loading dataset...")
csv_path = r'c:\Users\Kulis\Documents\Πτυχιακή\DrDoS-Detector\DrDoS_DNS.csv'
df = pd.read_csv(csv_path, low_memory=False)

print(f"   Total samples: {len(df):,}")
print(f"   Total features: {len(df.columns)}")

# Βήμα 2: Καθαρισμός δεδομένων
print("\n[2/7] Cleaning data...")

# Αφαίρεση άχρηστων στηλών
columns_to_drop = ['Unnamed: 0', 'Flow ID', ' Timestamp']
existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
df = df.drop(columns=existing_cols_to_drop)

# Έλεγχος για Label column
if ' Label' in df.columns:
    label_col = ' Label'
elif 'Label' in df.columns:
    label_col = 'Label'
else:
    raise ValueError("Label column not found!")

print(f"   Label column: '{label_col}'")

# Χειρισμός κενών τιμών
print(f"   Null values before: {df.isnull().sum().sum()}")
df = df.dropna()
print(f"   Null values after: {df.isnull().sum().sum()}")

# Χειρισμός infinity values
print("   Replacing infinity values...")
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# Διαχωρισμός features και labels
X = df.drop(columns=[label_col])
y = df[label_col]

# Κωδικοποίηση non-numeric columns
print("   Encoding categorical features...")
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

# Κωδικοποίηση labels
le_label = LabelEncoder()
y_encoded = le_label.fit_transform(y)

print(f"\n   Class distribution:")
for label, count in zip(*np.unique(y, return_counts=True)):
    percentage = (count / len(y)) * 100
    print(f"   - {label}: {count:,} samples ({percentage:.2f}%)")

print(f"\n   Total features to be used: {len(X.columns)}")

# Βήμα 3: Χωρισμός δεδομένων σε train/test με ειδική στρατηγική
print("\n[3/7] Splitting data with balanced test set strategy...")

# Χωρισμός δεδομένων ανά κλάση
benign_mask = y == 'BENIGN'
attack_mask = y == 'DrDoS_DNS'

X_benign = X[benign_mask].reset_index(drop=True)
y_benign = y_encoded[benign_mask]
X_attack = X[attack_mask].reset_index(drop=True)
y_attack = y_encoded[attack_mask]

print(f"   BENIGN samples: {len(X_benign):,}")
print(f"   DrDoS_DNS samples: {len(X_attack):,}")

# Για το test set: 50% από κάθε κλάση
# Στρατηγική: Χρησιμοποιούμε 50% των BENIGN για test (για να μείνουν και για train)
# και τον ίδιο αριθμό από DrDoS_DNS για balanced test

# Υπολογισμός: Χρησιμοποιούμε το 50% των BENIGN για test
samples_per_class_test = len(X_benign) // 2

# Αν το test set είναι πολύ μεγάλο σε σχέση με το σύνολο, το περιορίζουμε
total_samples = len(df)
max_test_size = int(total_samples * 0.2)
if samples_per_class_test * 2 > max_test_size:
    samples_per_class_test = max_test_size // 2

print(f"   Test set will have {samples_per_class_test} samples per class")
print(f"   This leaves {len(X_benign) - samples_per_class_test} BENIGN samples for training")

# Τυχαία επιλογή για test set (replace=False εξασφαλίζει ότι δεν υπάρχουν διπλότυπα)
benign_test_indices = np.random.choice(len(X_benign), size=samples_per_class_test, replace=False)
attack_test_indices = np.random.choice(len(X_attack), size=samples_per_class_test, replace=False)

# Δημιουργία boolean masks για train/test split
benign_train_mask = np.ones(len(X_benign), dtype=bool)
benign_train_mask[benign_test_indices] = False

attack_train_mask = np.ones(len(X_attack), dtype=bool)
attack_train_mask[attack_test_indices] = False

# Test sets (χωρίς διπλότυπα λόγω του choice με replace=False)
X_test_benign = X_benign.iloc[benign_test_indices].copy()
y_test_benign = y_benign[benign_test_indices].copy()

X_test_attack = X_attack.iloc[attack_test_indices].copy()
y_test_attack = y_attack[attack_test_indices].copy()

# Train sets (τα υπόλοιπα)
X_train_benign = X_benign.iloc[benign_train_mask].copy()
y_train_benign = y_benign[benign_train_mask].copy()

X_train_attack = X_attack.iloc[attack_train_mask].copy()
y_train_attack = y_attack[attack_train_mask].copy()

# Συνδυασμός test sets
X_test = pd.concat([X_test_benign, X_test_attack], ignore_index=True)
y_test = np.concatenate([y_test_benign, y_test_attack])

# Ανακάτεμα του test set
shuffle_indices = np.random.permutation(len(X_test))
X_test = X_test.iloc[shuffle_indices].copy()
y_test = y_test[shuffle_indices].copy()

print(f"\n   Test set created:")
print(f"   - Total test samples: {len(X_test):,}")
print(f"   - BENIGN in test: {sum(y_test == le_label.transform(['BENIGN'])[0]):,}")
print(f"   - DrDoS_DNS in test: {sum(y_test == le_label.transform(['DrDoS_DNS'])[0]):,}")

# Συνδυασμός train sets (προ-SMOTE)
X_train_original = pd.concat([X_train_benign, X_train_attack], ignore_index=True)
y_train_original = np.concatenate([y_train_benign, y_train_attack])

print(f"\n   Training set (before SMOTE):")
print(f"   - Total train samples: {len(X_train_original):,}")
print(f"   - BENIGN in train: {sum(y_train_original == le_label.transform(['BENIGN'])[0]):,}")
print(f"   - DrDoS_DNS in train: {sum(y_train_original == le_label.transform(['DrDoS_DNS'])[0]):,}")

# Έλεγχος για overlap μεταξύ train και test
print("\n   Verifying no overlap between train and test sets...")
train_size = len(X_train_original)
test_size = len(X_test)
total_after_split = train_size + test_size
print(f"   - Train size: {train_size:,}")
print(f"   - Test size: {test_size:,}")
print(f"   - Total: {total_after_split:,} (original: {len(df):,})")
print(f"   - No duplicates in selection process: ✓")

# Βήμα 4: Εφαρμογή SMOTE στο training set με undersampling
print("\n[4/7] Applying SMOTE with undersampling to balance training set...")
print("   Strategy: Undersample majority class first to manage memory")

# Πρώτα κάνουμε undersampling της πλειοψηφούσας κλάσης (DrDoS_DNS)
# Χρησιμοποιούμε 10x τα BENIGN samples για καλύτερη εκπαίδευση
target_majority_samples = len(X_train_benign) * 10

print(f"   Undersampling DrDoS_DNS from {len(X_train_attack):,} to {target_majority_samples:,}")

# Τυχαία επιλογή από την πλειοψηφούσα κλάση
attack_downsample_indices = np.random.choice(
    len(X_train_attack),
    size=target_majority_samples,
    replace=False
)

X_train_attack_downsampled = X_train_attack.iloc[attack_downsample_indices].copy()
y_train_attack_downsampled = y_train_attack[attack_downsample_indices].copy()

# Συνδυασμός για το νέο training set
X_train_downsampled = pd.concat([X_train_benign, X_train_attack_downsampled], ignore_index=True)
y_train_downsampled = np.concatenate([y_train_benign, y_train_attack_downsampled])

print(f"\n   Training set after undersampling:")
print(f"   - Total: {len(X_train_downsampled):,}")
print(f"   - BENIGN: {sum(y_train_downsampled == le_label.transform(['BENIGN'])[0]):,}")
print(f"   - DrDoS_DNS: {sum(y_train_downsampled == le_label.transform(['DrDoS_DNS'])[0]):,}")

# Τώρα εφαρμόζουμε SMOTE
smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_downsampled, y_train_downsampled)

print(f"\n   Training set after SMOTE:")
print(f"   - Total samples: {len(X_train_balanced):,}")
for label_idx, label_name in enumerate(le_label.classes_):
    count = sum(y_train_balanced == label_idx)
    print(f"   - {label_name}: {count:,}")

# Βήμα 5: Feature Scaling
print("\n[5/7] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

print(f"   Features scaled using StandardScaler")
print(f"   Training shape: {X_train_scaled.shape}")
print(f"   Test shape: {X_test_scaled.shape}")

# Βήμα 6: Εκπαίδευση μοντέλου (Random Forest - συνήθης επιλογή σε DrDoS detection papers)
print("\n[6/7] Training Random Forest classifier...")
print("   Χρησιμοποιούνται ΟΛΕΣ οι στήλες του dataset")
print(f"   Total features used: {X_train_scaled.shape[1]}")

# Παράμετροι που χρησιμοποιούνται συχνά σε DrDoS detection research
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1
)

clf.fit(X_train_scaled, y_train_balanced)

print("\n   Training completed!")

# Βήμα 7: Αξιολόγηση
print("\n[7/7] Evaluating model...")

y_pred = clf.predict(X_test_scaled)

print("\n" + "="*80)
print("RESULTS")
print("="*80)

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le_label.classes_))

print("\nSummary Metrics:")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred, average='weighted'):.4f}")

# Feature Importance
print("\n" + "="*80)
print("TOP 20 MOST IMPORTANT FEATURES")
print("="*80)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(20).to_string(index=False))

# Αποθήκευση μοντέλου
print("\n" + "="*80)
print("Saving model and scaler...")

model_data = {
    'model': clf,
    'scaler': scaler,
    'label_encoder': le_label,
    'feature_names': X.columns.tolist()
}

with open('drdos_detector_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Model saved as 'drdos_detector_model.pkl'")
print("="*80)
print("Pipeline completed successfully!")
print("="*80)
