# DrDoS DNS Attack Detection - Project Documentation

## Επισκόπηση
Σύστημα ανίχνευσης επιθέσεων DrDoS (Distributed Reflection Denial of Service) με χρήση Machine Learning (Random Forest). Το σύστημα εφαρμόζει **SMOTE ΠΡΙΝ το splitting** για σωστή αντιμετώπιση ανισορροπημένων δεδομένων και εξασφαλίζει ότι το test set περιέχει **ΜΟΝΟ πραγματικά δεδομένα** (όχι SMOTE).

---

## Δομή Αρχείων και Σειρά Εκτέλεσης

### 🚀 Κύριο Script Εκτέλεσης

#### **`train.py`**
Το κύριο script που ορχηστρώνει ολόκληρη τη διαδικασία εκπαίδευσης.

**Εκτελεί με τη σειρά:**
1. Φόρτωση δεδομένων
2. Καθαρισμό και προεπεξεργασία
3. Διαχωρισμό σε BENIGN και DDoS classes
4. **Εφαρμογή SMOTE στα BENIGN (ΠΡΙΝ το splitting)**
5. Χωρισμό σε train/test sets (test = ΟΛΑ τα original BENIGN + ίσα DDoS)
6. Κανονικοποίηση features
7. Εκπαίδευση Random Forest
8. Αξιολόγηση μοντέλου
9. Αποθήκευση μοντέλου

**Εκτέλεση:**
```bash
python train.py
```

---

## Modules (με σειρά κλήσης)

### 1️⃣ **`data_preprocessing.py`**
Φόρτωση και καθαρισμός του dataset.

**Συναρτήσεις:**
- `load_dataset(csv_path)` - Φορτώνει το CSV αρχείο
- `clean_data(df)` - Καθαρίζει τα δεδομένα:
  - Αφαιρεί άχρηστες στήλες (Unnamed: 0, Flow ID, Timestamp)
  - Χειρίζεται null και infinity τιμές
  - Κωδικοποιεί categorical features
  - Διαχωρίζει features από labels
- `encode_labels(y)` - Μετατρέπει string labels σε αριθμούς

**Έξοδος:**
- Features DataFrame (X)
- Labels Series (y)
- Label Encoder

---

### 2️⃣ **`data_balancing.py`**
Εφαρμογή SMOTE στα BENIGN δεδομένα **ΠΡΙΝ το splitting**.

**Συναρτήσεις:**
- `apply_smote_to_benign(X_benign, y_benign, target_samples, random_state)` - Εφαρμόζει SMOTE:

**Στρατηγική:**
- **SMOTE εφαρμόζεται ΠΡΩΤΑ** στην minority class (BENIGN)
- Αυξάνει τα BENIGN samples από ~3.4K → ~33.5K (10x)
- Δημιουργεί συνθετικά δείγματα για εξισορρόπηση
- Ρυθμιζόμενο target (SMOTE_TARGET_RATIO στο train.py)

**Έξοδος:**
- SMOTE-augmented BENIGN features και labels
- Τα original BENIGN διατηρούνται ξεχωριστά για το test set

---

### 3️⃣ **`data_splitting.py`**
Χωρισμός δεδομένων σε training και test sets **ΜΕΤΑ το SMOTE**.

**Συναρτήσεις:**
- `split_data_after_smote(X_benign_original, y_benign_original, X_benign_smote, y_benign_smote, X_attack, y_attack, le_label, test_size, random_state)` - Χωρίζει τα δεδομένα:

**Στρατηγική (ΔΙΟΡΘΩΜΕΝΗ):**
1. **Test Set:**
   - **ΟΛΑ** τα original BENIGN samples (3,354)
   - Ίσος αριθμός DDoS samples (3,354) - τυχαία επιλογή
   - Σύνολο: 6,708 samples (50-50 balanced)
   - **ΚΑΝΕΝΑ SMOTE δεδομένο**

2. **Train Set:**
   - SMOTE BENIGN samples (subsample για να ταιριάξει το test_size ratio)
   - Ίσος αριθμός DDoS samples - τυχαία επιλογή από τα υπόλοιπα
   - Εξισορροπημένο 50-50

3. **Test Size Ratio:**
   - Ρυθμιζόμενο (default 20%)
   - Υπολογίζεται αυτόματα: test / (train + test)

**Έξοδος:**
- X_train, y_train (balanced, περιέχει SMOTE)
- X_test, y_test (balanced, ΜΟΝΟ original data)

---

### 4️⃣ **`model_training.py`**
Κανονικοποίηση features και εκπαίδευση μοντέλου.

**Συναρτήσεις:**
- `scale_features(X_train, X_test)` - StandardScaler για κανονικοποίηση
  - Fit στο training set
  - Transform σε train και test
  
- `train_random_forest(X_train, y_train, ...)` - Εκπαίδευση Random Forest:
  - 100 trees
  - max_depth = 30
  - Παράλληλη επεξεργασία (n_jobs=-1)
  - Χρησιμοποιεί **ΟΛΕΣ** τις 84 στήλες

**Έξοδος:**
- Scaler (fitted)
- Trained Random Forest Classifier

---

### 5️⃣ **`model_evaluation.py`**
Αξιολόγηση του εκπαιδευμένου μοντέλου.

**Συναρτήσεις:**
- `evaluate_model(clf, X_test, y_test, le_label, feature_names)` - Υπολογίζει:

**Μετρικές:**
- Confusion Matrix
- Classification Report
- Accuracy, Precision, Recall, F1-Score
- Feature Importance (Top 20)

**Σημείωση:** Το test set περιέχει **ΜΟΝΟ πραγματικά δεδομένα**, όχι SMOTE!

**Έξοδος:**
- Dictionary με όλες τις μετρικές
- Εμφάνιση αποτελεσμάτων στην κονσόλα

---

### 6️⃣ **`model_persistence.py`**
Αποθήκευση και φόρτωση του μοντέλου.

**Συναρτήσεις:**
- `save_model(model, scaler, label_encoder, feature_names, filepath)` - Αποθηκεύει:
  - Trained model
  - Scaler
  - Label encoder
  - Feature names
  
- `load_model(filepath)` - Φορτώνει αποθηκευμένο μοντέλο

**Έξοδος:**
- `drdos_detector_model.pkl` - Pickle file με όλα τα απαραίτητα objects

---

## Ροή Δεδομένων (ΔΙΟΡΘΩΜΕΝΗ)

```
DrDoS_DNS.csv (5M+ samples, 99.93% DDoS, 0.07% BENIGN)
    ↓
[1] data_preprocessing.py
    ├─ Καθαρισμός (null, inf)
    ├─ Encoding (categorical → numeric)
    └─ Διαχωρισμός X, y
    ↓
[2] Διαχωρισμός Classes
    ├─ BENIGN: 3,354 samples (original)
    └─ DDoS: 4,908,665 samples
    ↓
[3] data_balancing.py - SMOTE ΠΡΙΝ ΤΟ SPLITTING
    ├─ Input: BENIGN (3,354)
    ├─ SMOTE: 3,354 → 33,540 (10x)
    └─ Output: SMOTE BENIGN (33,540)
    ↓
[4] data_splitting.py - Splitting ΜΕΤΑ ΤΟ SMOTE
    ├─ Test Set (6,708):
    │   ├─ ALL original BENIGN: 3,354
    │   └─ DDoS (random): 3,354
    │   └─ Ratio: 50-50, ΚΑΝΕΝΑ SMOTE!
    │
    └─ Train Set (26,832):
        ├─ SMOTE BENIGN (subsample): 13,416
        └─ DDoS (random): 13,416
        └─ Ratio: 50-50, balanced
    ↓
    └─ Test ratio: 20% (configurable)
    ↓
[5] model_training.py
    ├─ StandardScaler (normalization)
    └─ Random Forest Training
    ↓
[6] model_evaluation.py
    ├─ Predictions on PURE original data
    ├─ Metrics Calculation
    └─ Results: 99.94% Accuracy
    ↓
[7] model_persistence.py
    └─ Save → drdos_detector_model.pkl
```

---

## Αποτελέσματα

### 📊 Performance Metrics
- **Accuracy:** 99.94%
- **Precision:** 99.94%
- **Recall:** 99.94%
- **F1-Score:** 99.94%
- **Errors:** 4/6,708 predictions
- **Test Set:** 6,708 samples (100% original data, 0% SMOTE)

### 🎯 Top Features
1. Source IP (13.3%)
2. Min Packet Length (8.2%)
3. Avg Fwd Segment Size (7.2%)
4. Average Packet Size (7.2%)
5. Fwd Packet Length Min (7.2%)

---

## Χαρακτηριστικά Υλοποίησης

### ✅ Σωστή Διαχείριση SMOTE
- **SMOTE εφαρμόζεται ΠΡΙΝ το splitting** (όχι μετά!)
- Test set περιέχει **ΜΟΝΟ original BENIGN** data
- Train set περιέχει **SMOTE-augmented** data
- Αποφυγή data leakage

### ✅ Test Set Strategy
- **ΟΛΑ τα original BENIGN** για realistic evaluation
- **Εξισορροπημένο 50-50** με ίσα DDoS samples
- **Κανένα συνθετικό δεδομένο** (SMOTE-free)
- **Τυχαία επιλογή DDoS** χωρίς διπλότυπα

### ✅ Ρυθμιζόμενες Παράμετροι
- `TEST_SIZE` - Test set ratio (default 0.20 = 20%)
- `SMOTE_TARGET_RATIO` - SMOTE multiplier (default 10x)
- Εύκολη προσαρμογή στο `train.py`

### ✅ Τεχνικά Χαρακτηριστικά
- **Χρήση ΟΛΩΝ των στηλών** (84 features)
- **Random Forest** με 100 trees
- **StandardScaler** normalization
- **Modular design** για συντήρηση
- **Reproducible** (random_state=42)

---

## Χρήση

### Εκπαίδευση
```bash
python train.py
```

### Ρύθμιση Παραμέτρων
Επεξεργασία του `train.py`:
```python
TEST_SIZE = 0.20              # Test set ratio (20%)
SMOTE_TARGET_RATIO = 10       # SMOTE multiplier (10x original BENIGN)
```

### Χρήση Αποθηκευμένου Μοντέλου
```python
from model_persistence import load_model

# Φόρτωση
model_data = load_model('drdos_detector_model.pkl')
clf = model_data['model']
scaler = model_data['scaler']
label_encoder = model_data['label_encoder']

# Πρόβλεψη
X_new_scaled = scaler.transform(X_new)
predictions = clf.predict(X_new_scaled)
labels = label_encoder.inverse_transform(predictions)
```

---

## Απαιτήσεις

```
pandas
numpy
scikit-learn
imbalanced-learn
```

## Dataset

- **Αρχείο:** `DrDoS_DNS.csv`
- **Samples:** 5,074,413
- **Features:** 88 (χρησιμοποιούνται 84)
- **Classes:** BENIGN (0.07%), DrDoS_DNS (99.93%)
- **Ανισορροπία:** ~1:1,464 ratio

---

## Βασικές Διαφορές από Λάθος Υλοποιήσεις

### ❌ ΛΑΘΟΣ Approach:
1. Split data → Train/Test
2. Apply SMOTE → Training set
3. **Πρόβλημα:** SMOTE δεδομένα leak στο test set ή test με ανισορροπημένα δεδομένα

### ✅ ΣΩΣΤΟ Approach (αυτό το project):
1. **Apply SMOTE FIRST** → BENIGN augmentation
2. **Split AFTER** → Test = ALL original BENIGN + equal DDoS, Train = SMOTE + DDoS
3. **Αποτέλεσμα:** Test set καθαρό, αξιόπιστη αξιολόγηση

---

## Τεχνικές Λεπτομέρειες

### SMOTE Implementation
- Χρήση `imblearn.over_sampling.SMOTE`
- k_neighbors = min(5, len(BENIGN) - 1)
- Δημιουργία συνθετικών samples με interpolation

### Data Splitting Logic
- Test ratio calculation: `train = test * (1 - test_size) / test_size`
- Subsampling SMOTE αν χρειαστεί για να ταιριάξει το ratio
- Balanced train set για καλύτερη εκπαίδευση

### Random Forest Parameters
- n_estimators: 100
- max_depth: 30
- min_samples_split: 5
- min_samples_leaf: 2
- n_jobs: -1 (παράλληλη επεξεργασία)
