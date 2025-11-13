# DrDoS DNS Attack Detection - Project Documentation

## Επισκόπηση
Σύστημα ανίχνευσης επιθέσεων DrDoS (Distributed Reflection Denial of Service) με χρήση Machine Learning (Random Forest). Το σύστημα χρησιμοποιεί SMOTE για την αντιμετώπιση ανισορροπημένων δεδομένων και εξασφαλίζει εξισορροπημένο test set για αξιόπιση αξιολόγηση.

---

## Δομή Αρχείων και Σειρά Εκτέλεσης

### 🚀 Κύριο Script Εκτέλεσης

#### **`train.py`**
Το κύριο script που ορχηστρώνει ολόκληρη τη διαδικασία εκπαίδευσης.

**Εκτελεί με τη σειρά:**
1. Φόρτωση δεδομένων
2. Καθαρισμό και προεπεξεργασία
3. Χωρισμό σε train/test sets
4. Εφαρμογή SMOTE για εξισορρόπηση
5. Κανονικοποίηση features
6. Εκπαίδευση Random Forest
7. Αξιολόγηση μοντέλου
8. Αποθήκευση μοντέλου

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

### 2️⃣ **`data_splitting.py`**
Χωρισμός δεδομένων σε training και test sets με ειδική στρατηγική.

**Συναρτήσεις:**
- `split_balanced_data(X, y, y_encoded, le_label, random_state)` - Χωρίζει τα δεδομένα:
  
**Στρατηγική:**
- Test set: **Απόλυτα εξισορροπημένο** (50% BENIGN, 50% DrDoS_DNS)
- Χρησιμοποιεί 50% των BENIGN για test
- Τυχαία επιλογή **χωρίς διπλότυπα** (replace=False)
- Test set δεν περιλαμβάνει **κανένα SMOTE δεδομένο**

**Έξοδος:**
- X_train_original, y_train_original (pre-SMOTE)
- X_test, y_test (balanced 50-50)
- Χωριστά train sets για κάθε κλάση

---

### 3️⃣ **`data_balancing.py`**
Εξισορρόπηση training set με undersampling και SMOTE.

**Συναρτήσεις:**
- `balance_with_smote(X_train_benign, y_train_benign, X_train_attack, y_train_attack, le_label, random_state)` - Εξισορροπεί τα δεδομένα:

**Στρατηγική:**
1. **Undersampling:** Μειώνει την πλειοψηφούσα κλάση (DrDoS_DNS) στο 10x των BENIGN
   - Από ~4.9M → ~17K samples
   - Για διαχείριση μνήμης
2. **SMOTE:** Αυξάνει την μειοψηφούσα κλάση (BENIGN) για να ταιριάξει με DrDoS_DNS
   - Από ~1.7K → ~17K samples
   - Δημιουργεί συνθετικά δείγματα

**Έξοδος:**
- Balanced training set (50-50, ~33K samples)

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

## Ροή Δεδομένων

```
DrDoS_DNS.csv (5M+ samples)
    ↓
[1] data_preprocessing.py
    ├─ Καθαρισμός (null, inf)
    ├─ Encoding (categorical → numeric)
    └─ Διαχωρισμός X, y
    ↓
[2] data_splitting.py
    ├─ Balanced Test Set: 3,354 samples (50-50)
    └─ Train Set: 4.9M samples (ανισορροπημένο)
    ↓
[3] data_balancing.py
    ├─ Undersampling: 4.9M → 17K (DrDoS_DNS)
    ├─ SMOTE: 1.7K → 17K (BENIGN)
    └─ Balanced Train: 33K samples (50-50)
    ↓
[4] model_training.py
    ├─ StandardScaler (normalization)
    └─ Random Forest Training
    ↓
[5] model_evaluation.py
    ├─ Predictions
    ├─ Metrics Calculation
    └─ Results: 99.97% Accuracy
    ↓
[6] model_persistence.py
    └─ Save → drdos_detector_model.pkl
```

---

## Αποτελέσματα

### 📊 Performance Metrics
- **Accuracy:** 99.97%
- **Precision:** 99.97%
- **Recall:** 99.97%
- **F1-Score:** 99.97%
- **Errors:** 1/3,354 predictions

### 🎯 Top Features
1. Source IP (13.2%)
2. Fwd Packet Length Min (8.6%)
3. Average Packet Size (7.3%)
4. Avg Fwd Segment Size (7.1%)
5. Fwd Packet Length Mean (7.1%)

---

## Χαρακτηριστικά Υλοποίησης

✅ **Χρήση SMOTE** για την υπολυπόμενη κλάση (BENIGN)  
✅ **Test set εξισορροπημένο** 50-50 για αξιόπιστη αξιολόγηση  
✅ **Κανένα SMOTE δεδομένο** στο test set  
✅ **Τυχαία επιλογή χωρίς διπλότυπα** (random_state + replace=False)  
✅ **Χρήση ΟΛΩΝ των στηλών** (84 features)  
✅ **Undersampling** για διαχείριση μνήμης  
✅ **Modular design** για εύκολη συντήρηση  

---

## Χρήση

### Εκπαίδευση
```bash
python train.py
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
