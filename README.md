# DDoS Detection με Logistic Regression και SMOTE

Αυτό το project περιέχει την υλοποίηση συστήματος ανίχνευσης DDoS επιθέσεων σε DNS servers χρησιμοποιώντας Logistic Regression με SMOTE-balanced δεδομένα.

## Δομή Project

### Κύριο Script

1. **ddos_detection_analysis.py** - Ολοκληρωμένη ανάλυση DDoS Detection
   - Φόρτωση και προεπεξεργασία δεδομένων
   - Οπτικοποίηση δεδομένων
   - Εφαρμογή SMOTE για ισορροπία κλάσεων (50-50)
   - Εκπαίδευση μοντέλου Logistic Regression
   - Αξιολόγηση και οπτικοποίηση αποτελεσμάτων
   - Ανάλυση σημαντικότητας χαρακτηριστικών

### Utility Scripts

2. **check_files.py** - Έλεγχος ύπαρξης αρχείων και μοντέλων
3. **00_project_summary.py** - Περίληψη όλων των διαθέσιμων scripts

## Εγκατάσταση

### 1. Δημιουργία Virtual Environment

```powershell
python -m venv .venv
```

### 2. Ενεργοποίηση Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Εγκατάσταση Dependencies

```powershell
pip install pandas numpy scikit-learn matplotlib seaborn joblib scapy pyshark imbalanced-learn scipy
```

## Χρήση

### Πλήρης Ανάλυση

Εκτελέστε το κύριο script που κάνει όλη την ανάλυση:

```powershell
# Ενεργοποίηση Virtual Environment
.\.venv\Scripts\Activate.ps1

# Εκτέλεση ανάλυσης
python ddos_detection_analysis.py
```

Αυτό το script:
- Φορτώνει το dataset `DrDoS_DNS.csv`
- Δημιουργεί όλες τις οπτικοποιήσεις
- Εφαρμόζει SMOTE για ισορροπία 50-50
- Εκπαιδεύει το μοντέλο με SMOTE-balanced δεδομένα
- Αποθηκεύει το μοντέλο στον φάκελο `trained_model/`
- Αποθηκεύει όλα τα αποτελέσματα και γραφήματα στο `results/`

### Έλεγχος Αρχείων

```powershell
python check_files.py
```

## Δομή Φακέλων

```
.
├── .venv/                          # Virtual environment
├── ddos_detection_analysis.py      # Κύριο script
├── check_files.py                  # Utility script
├── 00_project_summary.py           # Project summary
├── DrDoS_DNS.csv                   # Dataset
├── results/                        # Αποτελέσματα ανάλυσης
│   ├── correlation_matrix.png
│   ├── class_distribution_before_smote.png
│   ├── smote_balance_comparison.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   ├── model_metrics.csv
│   └── feature_importance.csv
├── trained_model/                  # Εκπαιδευμένο μοντέλο (SMOTE)
│   ├── logistic_regression_model.joblib
│   ├── scaler.joblib
│   └── feature_names.csv
└── pcap-01-12/                     # PCAP files (προαιρετικά)
```

## Αποτελέσματα

Μετά την εκτέλεση του `ddos_detection_analysis.py`, θα δημιουργηθούν:

### Μοντέλο

**SMOTE-Balanced Model** (50-50 ισορροπία)
- Εκπαιδευμένο με SMOTE για ισορροπία κλάσεων
- Καλύτερη ανίχνευση τόσο για normal όσο και για attack traffic
- Συνιστάται για παραγωγική χρήση

### Γραφήματα

- `correlation_matrix.png` - Συσχετίσεις χαρακτηριστικών
- `class_distribution_before_smote.png` - Αρχική κατανομή κλάσεων
- `smote_balance_comparison.png` - Πριν/μετά το SMOTE
- `confusion_matrix.png` - Confusion matrix του μοντέλου
- `roc_curve.png` - ROC curve και AUC score
- `feature_importance.png` - Σημαντικότητα χαρακτηριστικών

### CSV Files

- `model_metrics.csv` - Μετρικές απόδοσης μοντέλου
- `feature_importance.csv` - Λίστα σημαντικότητας features

## Βιβλιογραφία

Abdusalam Yahya & Ahmed Mohammed Omar (2025)  
"Predicting of DDoS Attack on DNS Server using Logistic Regression"  
HNSJ, 6(2)

## Σημειώσεις

- Το dataset πρέπει να είναι στο ίδιο directory με τα scripts
- Το μοντέλο εκπαιδεύεται ΜΟΝΟ με SMOTE-balanced δεδομένα (50-50)
- Το SMOTE χρειάζεται αρκετό RAM για μεγάλα datasets
- Το μοντέλο αποθηκεύεται αυτόματα μετά την εκπαίδευση στο `trained_model/`
