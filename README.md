# DrDoS DNS Attack Detection

## Overview
DrDoS-Detector is a machine learning pipeline for detecting Distributed Reflection Denial of Service (DrDoS) attacks targeting DNS servers. The project supports multiple classification algorithms, automatic model comparison, robust handling of class imbalance with SMOTE applied before splitting, and clean separation of training vs. evaluation data to avoid leakage.

- Multiple models with enable/disable configuration
- Automatic per-model timing (training, evaluation, total)
- Clean test set with only original, non-SMOTE data
- Auto-saved results and best-model artifacts with incremental filenames
- Modular codebase designed for reproducibility and maintenance

## Supported Algorithms
- Logistic Regression (as referenced in the paper)
- Random Forest
- Decision Tree (fast and high-performing)
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

## Quick Start

### Requirements
```txt
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
imbalanced-learn>=0.10.0
```

### Install
```bash
pip install pandas numpy scikit-learn imbalanced-learn
```

### Dataset
- File: `DrDoS_DNS.csv`
- Download: available in GitHub Releases
- Place the CSV in the project root folder

Releases: https://github.com/kulisk/DrDoS-Detector/releases

### Run
```bash
python train.py
```

Default configuration trains and compares multiple models. Results and artifacts are saved automatically.

## Configuration
Edit `train.py` to select and configure models.

- Enable or disable models:
```python
ENABLE_MODELS = {
  'Logistic Regression': True,
  'Random Forest': True,
  'SVM': False,
  'Decision Tree': True,
  'KNN': False
}
```

- Global settings:
```python
TEST_SIZE = 0.20           # Evaluation ratio (20%)
SMOTE_TARGET_RATIO = 10    # BENIGN upsampling multiplier
RANDOM_STATE = 42          # Reproducibility
```

- Per-model parameters (excerpt):
```python
MODEL_PARAMS = {
  'Logistic Regression': {
    'max_iter': 1000,
    'random_state': RANDOM_STATE
  },
  'Random Forest': {
    'n_estimators': 100,
    'max_depth': 30,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE
  },
  'SVM': {
    'kernel': 'rbf',
    'C': 1.0,
    'random_state': RANDOM_STATE
  },
  'Decision Tree': {
    'max_depth': 30,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE
  },
  'KNN': {
    'n_neighbors': 5,
    'weights': 'uniform'
  }
}
```

## Pipeline

### Data Flow (corrected design)
```
DrDoS_DNS.csv (5M+ samples, ~0.07% BENIGN, ~99.93% DrDoS)
  ↓
[1] data_preprocessing.py
  ├─ Clean null/inf
  ├─ Encode categorical → numeric
  └─ Split X, y
  ↓
[2] Class separation
  ├─ BENIGN (original)
  └─ DrDoS (original)
  ↓
[3] data_balancing.py (SMOTE BEFORE SPLIT)
  ├─ Input: BENIGN original
  ├─ SMOTE to target ratio (e.g., 10×)
  └─ Output: BENIGN SMOTE
  ↓
[4] data_splitting.py (AFTER SMOTE)
  ├─ Test: ALL original BENIGN + equal DrDoS
  └─ Train: BENIGN SMOTE + remaining DrDoS
  ↓
[5] model_training.py
  ├─ StandardScaler fit on train → transform train/test
  └─ Train selected models (with timing)
  ↓
[6] model_evaluation.py
  ├─ Evaluate on PURE original test data
  ├─ Metrics: Accuracy, Precision, Recall, F1
  └─ Feature importance / coefficients (Top 20)
  ↓
[7] model_comparison.py (if multiple models enabled)
  ├─ Compare metrics & timing
  ├─ Save comparison report
  └─ Select best model
  ↓
[8] model_persistence.py
  └─ Save best model and supporting artifacts
```

## Modules

- `data_preprocessing.py`
  - `load_dataset(csv_path)`
  - `clean_data(df)` → cleans + encodes + returns X, y
  - `encode_labels(y)`

- `data_balancing.py`
  - `apply_smote_to_benign(X_benign, y_benign, target_samples, random_state)`

- `data_splitting.py`
  - `split_data_after_smote(...)` → Test: ALL original BENIGN + equal DrDoS; Train: BENIGN SMOTE + remaining DrDoS

- `model_training.py`
  - `scale_features(X_train, X_test)`
  - `train_logistic_regression(...)`
  - `train_random_forest(...)`
  - `train_svm(...)`
  - `train_decision_tree(...)`
  - `train_knn(...)`

- `model_evaluation.py`
  - `evaluate_model(clf, X_test, y_test, le_label, feature_names)`
  - Handles tree-based `feature_importances_` and linear `coef_` (abs)
  - Saves individual results with auto-increment: `training_results_#.txt`

- `model_comparison.py`
  - `compare_models(results_dict, label_encoder)` → table of metrics + timing
  - `save_comparison_to_file(...)` → `comparison_results_#.txt`
  - Best model selection stored as `best_model_[algorithm].pkl`

- `model_persistence.py`
  - `save_model(model, scaler, label_encoder, feature_names, filepath)`
  - `load_model(filepath)`

## Results Snapshot

Example comparative performance (with 6,708 original test samples):

| Model               | Accuracy | Precision | Recall | F1-Score | Train Time | Total Time |
|---------------------|----------|-----------|--------|----------|------------|------------|
| Decision Tree       | 0.9999   | 0.9999    | 0.9999 | 0.9999   | 0.36 s     | 0.39 s     |
| Logistic Regression | 0.9996   | 0.9996    | 0.9996 | 0.9996   | 5.20 s     | 5.26 s     |
| Random Forest       | 0.9994   | 0.9994    | 0.9994 | 0.9994   | 0.62 s     | 0.79 s     |

Notes:
- Test set uses only original samples (no SMOTE)
- Auto-comparison report: `comparison_results_#.txt`
- Best model artifact: `best_model_[algorithm].pkl`

## Using Saved Models
```python
from model_persistence import load_model
import pandas as pd

model_data = load_model('best_model_decision_tree.pkl')
clf = model_data['model']
scaler = model_data['scaler']
label_encoder = model_data['label_encoder']
feature_names = model_data['feature_names']

# X_new must match training feature schema
X_new = pd.DataFrame(..., columns=feature_names)
X_new_scaled = scaler.transform(X_new)
pred = clf.predict(X_new_scaled)
labels = label_encoder.inverse_transform(pred)
print(labels)
```

## Project Structure
```
DrDoS-Detector/
├── train.py
├── data_preprocessing.py
├── data_balancing.py
├── data_splitting.py
├── model_training.py
├── model_evaluation.py
├── model_comparison.py
├── model_persistence.py
├── DrDoS_DNS.csv                 # download from Releases
├── training_results_*.txt
├── comparison_results_*.txt
├── best_model_*.pkl
└── README.md
```

## Troubleshooting
- Memory constraints: reduce `SMOTE_TARGET_RATIO` (e.g., 5) or `TEST_SIZE` (e.g., 0.10)
- Slow training: disable SVM and KNN for large datasets; keep Decision Tree and Random Forest
- Dataset not found: ensure `DrDoS_DNS.csv` is in the project root

## Reference
Paper: “Predicting of DDoS Attack on DNS Server using Machine Learning”
- Uses Logistic Regression
- This implementation also compares additional models and achieves >99.9% accuracy with clean evaluation design

## License
Open-source for educational and research use.

## Contact
- Issues: https://github.com/kulisk/DrDoS-Detector/issues
- Releases: https://github.com/kulisk/DrDoS-Detector/releases

# DrDoS DNS Attack Detection - Project Documentation

## Επισκόπηση
Προηγμένο σύστημα ανίχνευσης επιθέσεων DrDoS (Distributed Reflection Denial of Service) με χρήση Machine Learning. Το σύστημα υποστηρίζει **πολλαπλούς αλγορίθμους** ταξινόμησης με δυνατότητα **αυτόματης σύγκρισης** και εφαρμόζει **SMOTE ΠΡΙΝ το splitting** για σωστή αντιμετώπιση ανισορροπημένων δεδομένων, εξασφαλίζοντας ότι το test set περιέχει **ΜΟΝΟ πραγματικά δεδομένα** (όχι SMOTE).

### 🎯 Υποστηριζόμενοι Αλγόριθμοι
- **Logistic Regression** (συνιστάται από το άρθρο)
- **Random Forest**
- **Decision Tree** 🏆 (καλύτερη απόδοση)
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**

---

## Δομή Αρχείων και Σειρά Εκτέλεσης

### 🚀 Κύριο Script Εκτέλεσης

#### **`train.py`**
Το κύριο script που ορχηστρώνει ολόκληρη τη διαδικασία εκπαίδευσης με υποστήριξη πολλαπλών αλγορίθμων.

**Εκτελεί με τη σειρά:**
1. Φόρτωση δεδομένων
2. Καθαρισμό και προεπεξεργασία
3. Διαχωρισμό σε BENIGN και DDoS classes
4. **Εφαρμογή SMOTE στα BENIGN (ΠΡΙΝ το splitting)**
5. Χωρισμό σε train/test sets (test = ΟΛΑ τα original BENIGN + ίσα DDoS)
6. Κανονικοποίηση features
7. Εκπαίδευση επιλεγμένων μοντέλων με μέτρηση χρόνου
8. Αξιολόγηση κάθε μοντέλου
9. Σύγκριση μοντέλων (αν >1 enabled)
10. Αποθήκευση του καλύτερου μοντέλου

**Ρύθμιση Αλγορίθμων:**
```python
ENABLE_MODELS = {
    'Logistic Regression': True,   # Από το άρθρο
    'Random Forest': True,          # Εξαιρετική απόδοση
    'SVM': False,                   # Αργός για μεγάλα datasets
    'Decision Tree': True,          # Ταχύτερος & Ακριβέστερος
    'KNN': False                    # Πολύ αργός
}
```

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
Κανονικοποίηση features και εκπαίδευση μοντέλων με υποστήριξη πολλαπλών αλγορίθμων.

**Συναρτήσεις:**
- `scale_features(X_train, X_test)` - StandardScaler για κανονικοποίηση
  - Fit στο training set
  - Transform σε train και test

**Αλγόριθμοι Εκπαίδευσης:**
- `train_logistic_regression(...)` - Logistic Regression (από το άρθρο)
  - max_iter: 1000
  - Παράλληλη επεξεργασία (n_jobs=-1)
  
- `train_random_forest(...)` - Random Forest
  - 100 trees, max_depth=30
  - Παράλληλη επεξεργασία (n_jobs=-1)
  
- `train_svm(...)` - Support Vector Machine
  - RBF kernel, C=1.0
  
- `train_decision_tree(...)` - Decision Tree
  - max_depth=30
  
- `train_knn(...)` - K-Nearest Neighbors
  - k=5, uniform weights

**Χαρακτηριστικά:**
- Χρησιμοποιεί **ΟΛΕΣ** τις 84 στήλες
- Υποστήριξη coefficient-based και tree-based μοντέλων
- Αυτόματη μέτρηση χρόνου εκπαίδευσης

**Έξοδος:**
- Scaler (fitted)
- Trained Classifier (οποιοσδήποτε αλγόριθμος)

---

### 5️⃣ **`model_evaluation.py`**
Αξιολόγηση των εκπαιδευμένων μοντέλων με πλήρη μετρικές.

**Συναρτήσεις:**
- `evaluate_model(clf, X_test, y_test, le_label, feature_names)` - Υπολογίζει:

**Μετρικές:**
- Confusion Matrix
- Classification Report (per-class metrics)
- Accuracy, Precision, Recall, F1-Score
- Feature Importance / Coefficients (Top 20)
  - Tree-based models: feature_importances_
  - Linear models: abs(coef_)

**Χρόνοι Εκτέλεσης:**
- Training Time (δευτερόλεπτα)
- Evaluation Time (δευτερόλεπτα)
- Total Time (δευτερόλεπτα)

**Αποθήκευση Αποτελεσμάτων:**
- `save_results_to_file()` - Αυτόματη αποθήκευση σε `training_results_X.txt`
- Αυτόματη αρίθμηση αρχείων (δεν γίνεται overwrite)

**Σημείωση:** Το test set περιέχει **ΜΟΝΟ πραγματικά δεδομένα**, όχι SMOTE!

**Έξοδος:**
- Dictionary με όλες τις μετρικές + χρόνους
- Εμφάνιση αποτελεσμάτων στην κονσόλα
- Αποθήκευση σε txt αρχείο

---

### 6️⃣ **`model_comparison.py`**
Σύγκριση πολλαπλών μοντέλων και δημιουργία comparative analysis.

**Συναρτήσεις:**
- `compare_models(results_dict, label_encoder)` - Δημιουργεί πίνακα σύγκρισης:
  - Accuracy, Precision, Recall, F1-Score
  - Training Time, Total Time
  - Αυτόματη ταξινόμηση (κατά Accuracy)
  - Προσδιορισμός καλύτερου μοντέλου

- `save_comparison_to_file(...)` - Αποθήκευση λεπτομερούς σύγκρισης:
  - Συγκριτικός πίνακας
  - Detailed results για κάθε μοντέλο
  - Confusion matrices
  - Top 10 features per model
  - Αυτόματη αρίθμηση: `comparison_results_X.txt`

**Λειτουργία:**
- Ενεργοποιείται αυτόματα όταν >1 μοντέλο είναι enabled
- Αποθηκεύει το **καλύτερο μοντέλο** αυτόματα

**Έξοδος:**
- Comparison DataFrame
- `comparison_results_X.txt` με πλήρη ανάλυση
- `best_model_[algorithm_name].pkl`

---

### 7️⃣ **`model_persistence.py`**
### 7️⃣ **`model_persistence.py`**
Αποθήκευση και φόρτωση του μοντέλου.

**Συναρτήσεις:**
- `save_model(model, scaler, label_encoder, feature_names, filepath)` - Αποθηκεύει:
  - Trained model (οποιοσδήποτε αλγόριθμος)
  - Scaler
  - Label encoder
  - Feature names
  
- `load_model(filepath)` - Φορτώνει αποθηκευμένο μοντέλο

**Έξοδος:**
- `drdos_detector_model.pkl` ή `best_model_[algorithm].pkl`
- Pickle file με όλα τα απαραίτητα objects

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
    └─ Train Multiple Models με χρονομέτρηση
    ↓
[6] model_evaluation.py
    ├─ Predictions on PURE original data
    ├─ Metrics Calculation για κάθε μοντέλο
    └─ Χρόνοι εκτέλεσης
    ↓
[7] model_comparison.py (αν >1 model enabled)
    ├─ Συγκριτική ανάλυση
    ├─ Επιλογή καλύτερου μοντέλου
    └─ Save comparison_results_X.txt
    ↓
[8] model_persistence.py
    └─ Save → best_model_[algorithm].pkl
```

---

## Αποτελέσματα

### 📊 Performance Metrics (Σύγκριση Αλγορίθμων)

| Μοντέλο | Accuracy | Precision | Recall | F1-Score | Training Time | Total Time |
|---------|----------|-----------|--------|----------|---------------|------------|
| **Decision Tree** 🏆 | **99.99%** | **99.99%** | **99.99%** | **99.99%** | **0.36s** ⚡ | **0.39s** ⚡ |
| **Logistic Regression** | 99.96% | 99.96% | 99.96% | 99.96% | 5.20s | 5.26s |
| **Random Forest** | 99.94% | 99.94% | 99.94% | 99.94% | 0.62s | 0.79s |

**Σημειώσεις:**
- **Decision Tree**: Καλύτερη απόδοση ΚΑΙ ταχύτητα (μόνο 1 λάθος στα 6,708 samples!)
- **Logistic Regression**: Αλγόριθμος που συστήνεται από το άρθρο, εξαιρετική ακρίβεια
- **Random Forest**: Εξαιρετική ισορροπία απόδοσης/ταχύτητας
- **Test Set**: 6,708 samples (100% original data, 0% SMOTE)

### 🎯 Top Features (Decision Tree)
1. Source IP (99.93%)
2. Min Packet Length (0.06%)
3. Destination Port (0.01%)
4. min_seg_size_forward (<0.01%)
5. Destination IP (<0.01%)

### 🎯 Top Features (Logistic Regression)
1. Source IP (3.35)
2. Destination IP (2.63)
3. URG Flag Count (2.18)
4. Protocol (1.98)
5. Bwd Packet Length Min (0.80)

### 🎯 Top Features (Random Forest)
1. Source IP (13.3%)
2. Min Packet Length (8.2%)
3. Avg Fwd Segment Size (7.2%)
4. Average Packet Size (7.2%)
5. Fwd Packet Length Min (7.2%)

---

## Χαρακτηριστικά Υλοποίησης

### ✅ Πολλαπλοί Αλγόριθμοι ML
- **5 διαφορετικοί αλγόριθμοι** με εύκολη ενεργοποίηση/απενεργοποίηση
- **Αυτόματη σύγκριση** όταν >1 αλγόριθμος enabled
- **Επιλογή καλύτερου μοντέλου** με βάση accuracy
- **Υποστήριξη coefficient & tree-based models**

### ✅ Μέτρηση Χρόνων Εκτέλεσης
- **Training Time** για κάθε μοντέλο
- **Evaluation Time** για κάθε μοντέλο
- **Total Time** (end-to-end)
- Εμφάνιση σε πίνακα σύγκρισης

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

### ✅ Αυτόματη Αποθήκευση Αποτελεσμάτων
- **Auto-incrementing filenames** (δεν γίνεται overwrite)
- `training_results_X.txt` - Μεμονωμένα αποτελέσματα
- `comparison_results_X.txt` - Συγκριτική ανάλυση
- `best_model_[algorithm].pkl` - Το καλύτερο μοντέλο

### ✅ Ρυθμιζόμενες Παράμετροι
- `ENABLE_MODELS` - Επιλογή αλγορίθμων (True/False)
- `TEST_SIZE` - Test set ratio (default 0.20 = 20%)
- `SMOTE_TARGET_RATIO` - SMOTE multiplier (default 10x)
- `MODEL_PARAMS` - Παράμετροι για κάθε αλγόριθμο

### ✅ Τεχνικά Χαρακτηριστικά
- **Χρήση ΟΛΩΝ των στηλών** (84 features)
- **StandardScaler** normalization για όλα τα μοντέλα
- **Parallel processing** όπου υποστηρίζεται (n_jobs=-1)
- **Modular design** για εύκολη συντήρηση
- **Reproducible** (random_state=42)

---

## Χρήση

### 🚀 Γρήγορη Εκκίνηση

#### 1. Εγκατάσταση Απαιτήσεων
```bash
pip install pandas numpy scikit-learn imbalanced-learn
```

#### 2. Λήψη Dataset
Κατεβάστε το `DrDoS_DNS.csv` από τα [GitHub Releases](https://github.com/kulisk/DrDoS-Detector/releases) και τοποθετήστε το στον φάκελο του project.

#### 3. Εκπαίδευση με Default Ρυθμίσεις
```bash
python train.py
```

**Default Configuration:**
- Enabled: Logistic Regression, Random Forest, Decision Tree
- Disabled: SVM, KNN (αργοί για μεγάλα datasets)
- Test Size: 20%
- SMOTE Ratio: 10x

---

### ⚙️ Προχωρημένη Χρήση

#### Επιλογή Αλγορίθμων
Επεξεργαστείτε το `train.py` (γραμμές 25-31):
```python
ENABLE_MODELS = {
    'Logistic Regression': True,   # Από το άρθρο - Αργός αλλά ακριβής
    'Random Forest': True,          # Καλή ισορροπία
    'SVM': False,                   # Πολύ αργός (enable μόνο για μικρά datasets)
    'Decision Tree': True,          # Ταχύτερος & Καλύτερος 🏆
    'KNN': False                    # Εξαιρετικά αργός (αποφύγετε)
}
```

#### Ρύθμιση Παραμέτρων
Επεξεργαστείτε το `train.py`:
```python
TEST_SIZE = 0.20              # Test set ratio (20%)
SMOTE_TARGET_RATIO = 10       # SMOTE multiplier (10x original BENIGN)
RANDOM_STATE = 42             # Για reproducibility
```

#### Ρύθμιση Model Parameters
Τροποποιήστε το dictionary `MODEL_PARAMS` στο `train.py`:
```python
MODEL_PARAMS = {
    'Logistic Regression': {
        'max_iter': 1000,
        'random_state': RANDOM_STATE
    },
    'Random Forest': {
        'n_estimators': 100,      # Περισσότερα trees = καλύτερη απόδοση
        'max_depth': 30,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': RANDOM_STATE
    },
    # ... άλλοι αλγόριθμοι
}
```

---

### 📊 Έξοδος Αποτελεσμάτων

#### Single Model Mode (1 enabled)
- Console output με πλήρη μετρικές
- `training_results_X.txt` με αποτελέσματα
- `drdos_detector_model.pkl` με το μοντέλο

#### Comparison Mode (>1 enabled)
- Console output για κάθε μοντέλο
- Συγκριτικός πίνακας στην κονσόλα
- `comparison_results_X.txt` με πλήρη σύγκριση
- `best_model_[algorithm].pkl` με το καλύτερο μοντέλο

**Παράδειγμα Comparison Output:**
```
================================================================================
MODEL COMPARISON
================================================================================
              Model Accuracy Precision Recall F1-Score Training Time (s) Total Time (s)
      Decision Tree   0.9999    0.9999 0.9999   0.9999              0.36           0.39
Logistic Regression   0.9996    0.9996 0.9996   0.9996              5.20           5.26
      Random Forest   0.9994    0.9994 0.9994   0.9994              0.62           0.79
================================================================================
🏆 Best Model: Decision Tree
================================================================================
```

---

### 🔮 Χρήση Αποθηκευμένου Μοντέλου για Predictions
### 🔮 Χρήση Αποθηκευμένου Μοντέλου για Predictions
```python
from model_persistence import load_model
import pandas as pd

# Φόρτωση καλύτερου μοντέλου
model_data = load_model('best_model_decision_tree.pkl')
clf = model_data['model']
scaler = model_data['scaler']
label_encoder = model_data['label_encoder']
feature_names = model_data['feature_names']

# Προετοιμασία νέων δεδομένων
# X_new πρέπει να έχει τις ίδιες στήλες με το training set
X_new = pd.DataFrame(...)  # Τα νέα δεδομένα σας

# Κανονικοποίηση
X_new_scaled = scaler.transform(X_new)

# Πρόβλεψη
predictions = clf.predict(X_new_scaled)
probabilities = clf.predict_proba(X_new_scaled)

# Μετατροπή σε labels
labels = label_encoder.inverse_transform(predictions)

print(f"Predictions: {labels}")
print(f"Probabilities: {probabilities}")
```

---

## 📁 Δομή Project

```
DrDoS-Detector/
├── train.py                      # Κύριο script εκτέλεσης
├── data_preprocessing.py         # Φόρτωση & καθαρισμός δεδομένων
├── data_balancing.py             # SMOTE implementation
├── data_splitting.py             # Train/Test splitting
├── model_training.py             # Εκπαίδευση αλγορίθμων (5 models)
├── model_evaluation.py           # Αξιολόγηση & μετρικές
├── model_comparison.py           # Σύγκριση μοντέλων
├── model_persistence.py          # Αποθήκευση/Φόρτωση μοντέλων
├── DrDoS_DNS.csv                 # Dataset (download από releases)
├── README.md                     # Αυτό το αρχείο
├── .gitignore                    # Git exclusions
│
├── training_results_*.txt        # Μεμονωμένα αποτελέσματα
├── comparison_results_*.txt      # Συγκριτικές αναλύσεις
├── best_model_*.pkl              # Αποθηκευμένα μοντέλα
└── drdos_detector_model.pkl      # Single model output
```

---

## 📦 Απαιτήσεις

### Python Packages
```txt
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
imbalanced-learn>=0.10.0
```

### Εγκατάσταση
```bash
pip install pandas numpy scikit-learn imbalanced-learn
```

### Σύστημα
- Python 3.8+
- RAM: 8GB+ συνιστάται (για το dataset των 5M+ samples)
- CPU: Multi-core για parallel processing

---

## 📊 Dataset

- **Αρχείο:** `DrDoS_DNS.csv`
- **Λήψη:** Διαθέσιμο στα [GitHub Releases](https://github.com/kulisk/DrDoS-Detector/releases)
- **Samples:** 5,074,413
- **Features:** 88 (χρησιμοποιούνται 84)
- **Classes:** BENIGN (0.07%), DrDoS_DNS (99.93%)
- **Ανισορροπία:** ~1:1,464 ratio

**Σημείωση:** Λόγω μεγέθους, το dataset δεν περιλαμβάνεται στο repository. Κατεβάστε το από τα releases και τοποθετήστε το στον φάκελο του project.

---

## Βασικές Διαφορές από Λάθος Υλοποιήσεις

### ❌ ΛΑΘΟΣ Approach:
1. Split data → Train/Test
2. Apply SMOTE → Training set ΜΟΝΟ
3. **Προβλήματα:**
   - Πληροφορία από το test set μπορεί να "διαρρεύσει" στο training
   - Test set παραμένει ανισορροπημένο
   - Αναξιόπιστη αξιολόγηση

### ✅ ΣΩΣΤΟ Approach (αυτό το project):
1. **Separate classes** → BENIGN & DDoS ξεχωριστά
2. **Apply SMOTE FIRST** → BENIGN augmentation (3.4K → 33.5K)
3. **Split AFTER SMOTE** → 
   - Test = ALL original BENIGN + equal DDoS
   - Train = SMOTE BENIGN + remaining DDoS
4. **Αποτέλεσμα:** 
   - Test set καθαρό (100% original data)
   - Εξισορροπημένα train & test sets
   - Αξιόπιστη αξιολόγηση
   - Καμία data leakage

---

## 🎓 Σύγκριση με το Άρθρο

### Αλγόριθμος από το Άρθρο
- **Logistic Regression** (κύριος αλγόριθμος)
- Accuracy: ~96-98% (σύμφωνα με το άρθρο)

### Αποτελέσματα Υλοποίησης
| Αλγόριθμος | Accuracy | Σχόλια |
|-----------|----------|---------|
| **Decision Tree** 🥇 | **99.99%** | Υπερέχει του άρθρου, ταχύτατος |
| **Logistic Regression** | **99.96%** | Καλύτερος από το άρθρο (~97%) |
| **Random Forest** | **99.94%** | Εξαιρετική ισορροπία |

### Πλεονεκτήματα Υλοποίησης
✅ **Καλύτερη απόδοση** από το άρθρο (99.96% vs ~97%)  
✅ **Πολλαπλοί αλγόριθμοι** με αυτόματη σύγκριση  
✅ **Μέτρηση χρόνων** εκτέλεσης  
✅ **Σωστή SMOTE στρατηγική** (ΠΡΙΝ το splitting)  
✅ **Modular & Extensible** architecture  
✅ **Production-ready** με auto-save features  

---

## 🔬 Τεχνικές Λεπτομέρειες

## 🔬 Τεχνικές Λεπτομέρειες

### SMOTE Implementation
- Χρήση `imblearn.over_sampling.SMOTE`
- k_neighbors = min(5, len(BENIGN) - 1)
- Δημιουργία συνθετικών samples με interpolation
- Εφαρμογή ΠΡΙΝ το splitting για σωστή αξιολόγηση

### Data Splitting Logic
- Test ratio calculation: `train_size = test_size * (1 - test_ratio) / test_ratio`
- Subsampling SMOTE αν χρειαστεί για να ταιριάξει το ratio
- Balanced train & test sets (50-50) για βέλτιστη εκπαίδευση
- Τυχαία επιλογή DDoS samples χωρίς διπλότυπα

### Model Parameters

#### Logistic Regression
```python
LogisticRegression(
    max_iter=1000,
    random_state=42,
    n_jobs=-1,      # Παράλληλη επεξεργασία
    verbose=1
)
```

#### Random Forest
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,      # Παράλληλη επεξεργασία
    verbose=1
)
```

#### Decision Tree
```python
DecisionTreeClassifier(
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
```

#### Support Vector Machine
```python
SVC(
    kernel='rbf',
    C=1.0,
    random_state=42,
    verbose=True
)
```

#### K-Nearest Neighbors
```python
KNeighborsClassifier(
    n_neighbors=5,
    weights='uniform',
    n_jobs=-1       # Παράλληλη επεξεργασία
)
```

### Feature Importance Extraction
- **Tree-based models** (RF, DT): `model.feature_importances_`
- **Linear models** (LR, SVM): `np.abs(model.coef_[0])`
- Αυτόματη ανίχνευση τύπου μοντέλου

### Performance Optimization
- **Parallel Processing**: Χρήση όλων των CPU cores (n_jobs=-1)
- **Efficient Memory Usage**: Subsampling όπου χρειάζεται
- **Vectorized Operations**: Pandas & NumPy optimizations
- **Reproducibility**: Fixed random_state για consistency

---

## 🐛 Troubleshooting

### Memory Issues
Αν αντιμετωπίζετε προβλήματα μνήμης:
```python
# Μειώστε το SMOTE ratio
SMOTE_TARGET_RATIO = 5  # Αντί για 10

# Ή μειώστε το TEST_SIZE
TEST_SIZE = 0.10  # Αντί για 0.20
```

### Slow Training
Για ταχύτερη εκπαίδευση:
```python
# Απενεργοποιήστε τους αργούς αλγορίθμους
ENABLE_MODELS = {
    'Logistic Regression': False,  # Αργός
    'Random Forest': False,
    'SVM': False,                   # Πολύ αργός
    'Decision Tree': True,          # Ταχύτερος
    'KNN': False                    # Εξαιρετικά αργός
}
```

### Dataset Not Found
```bash
# Βεβαιωθείτε ότι το CSV είναι στον σωστό φάκελο
ls DrDoS_DNS.csv

# Αν όχι, κατεβάστε από τα releases
# και τοποθετήστε το στον root φάκελο του project
```

---

## 📚 Αναφορές

### Paper
**"Predicting of DDoS Attack on DNS Server using Machine Learning"**
- Χρησιμοποιεί: Logistic Regression
- Dataset: DrDoS DNS Attack traces
- Αποτελέσματα: ~96-98% accuracy

### Βελτιώσεις αυτής της Υλοποίησης
1. ✅ Υψηλότερη accuracy (99.94-99.99%)
2. ✅ Πολλαπλοί αλγόριθμοι με αυτόματη σύγκριση
3. ✅ Σωστή SMOTE implementation (ΠΡΙΝ splitting)
4. ✅ Μέτρηση χρόνων εκτέλεσης
5. ✅ Production-ready architecture
6. ✅ Comprehensive documentation

---

## 🤝 Contributing

Contributions are welcome! Για να συνεισφέρετε:

1. Fork το repository
2. Δημιουργήστε feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit τις αλλαγές σας (`git commit -m 'Add some AmazingFeature'`)
4. Push στο branch (`git push origin feature/AmazingFeature`)
5. Ανοίξτε Pull Request

---

## 📄 License

Αυτό το project είναι ανοιχτού κώδικα και διαθέσιμο για εκπαιδευτικούς και ερευνητικούς σκοπούς.

---

## 👥 Authors

**DrDoS-Detector Team**
- Ανάπτυξη & Implementation
- Βελτιστοποίηση Αλγορίθμων
- Documentation & Testing

---

## 🙏 Acknowledgments

- Το άρθρο που ενέπνευσε αυτήν την υλοποίηση
- Scikit-learn community για τα εξαιρετικά ML tools
- Imbalanced-learn για το SMOTE implementation
- Όλους τους contributors και testers

---

## 📞 Contact & Support

Για ερωτήσεις, issues ή suggestions:
- 🐛 [GitHub Issues](https://github.com/kulisk/DrDoS-Detector/issues)
- 📧 Email: [Διαθέσιμο στο GitHub Profile]
- 📖 [Documentation](https://github.com/kulisk/DrDoS-Detector)

---

**⭐ Αν σας βοήθησε αυτό το project, δώστε ένα star στο GitHub! ⭐**
