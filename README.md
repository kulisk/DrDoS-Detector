# DDoS Detection με Logistic Regression και SMOTE

Αυτό το project περιέχει την υλοποίηση συστήματος ανίχνευσης DDoS επιθέσεων σε DNS servers χρησιμοποιώντας Logistic Regression με SMOTE-balanced δεδομένα.

## Δομή Project

### Κύρια Scripts

1. **ddos_detection_analysis.py** - Ολοκληρωμένη ανάλυση DDoS Detection
   - Φόρτωση και προεπεξεργασία δεδομένων
   - Οπτικοποίηση δεδομένων
   - Εφαρμογή SMOTE για ισορροπία κλάσεων (50-50)
   - Εκπαίδευση μοντέλου Logistic Regression
   - Αξιολόγηση και οπτικοποίηση αποτελεσμάτων
   - Ανάλυση σημαντικότητας χαρακτηριστικών

### Γραφικά Περιβάλλοντα (GUI)

2. **ddos_detection_gui.py** - Γραφικό Περιβάλλον DDoS Detection
   - Πλήρες GUI για εκπαίδευση και αξιολόγηση μοντέλου
   - Οπτικοποίηση αποτελεσμάτων σε πραγματικό χρόνο
   - Εύκολη διαχείριση όλων των φάσεων της ανάλυσης
   - Καταγραφή λεπτομερών logs

3. **ddos_detection_gui_wizard.py** - Wizard-Style GUI
   - Βήμα προς βήμα καθοδήγηση
   - Φιλική εμπειρία χρήστη
   - Επιλογή μοντέλου και παραμέτρων
   - Ολοκληρωμένη παρουσίαση αποτελεσμάτων

### Utility Scripts

4. **test_data_loading.py** - Δοκιμαστικό script για φόρτωση δεδομένων
   - Έλεγχος φόρτωσης και επεξεργασίας δεδομένων
   - Επικύρωση ακεραιότητας δεδομένων
   - Διαχείριση missing values

5. **check_files.py** - Έλεγχος ύπαρξης αρχείων και μοντέλων

6. **00_project_summary.py** - Περίληψη όλων των διαθέσιμων scripts

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
pip install pandas numpy scikit-learn matplotlib seaborn joblib scapy pyshark imbalanced-learn scipy tk
```

**Σημείωση:** Για τα GUI applications (ddos_detection_gui.py, ddos_detection_gui_wizard.py), χρειάζεται και το tkinter που συνήθως είναι προεγκατεστημένο με την Python.

## Χρήση

### Πλήρης Ανάλυση (Command Line)

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

### Γραφικό Περιβάλλον (GUI Applications)

#### Standard GUI

Εκτελέστε το πλήρες γραφικό περιβάλλον:

```powershell
# Ενεργοποίηση Virtual Environment
.\.venv\Scripts\Activate.ps1

# Εκκίνηση GUI
python ddos_detection_gui.py
```

Το GUI προσφέρει:
- Φόρτωση δεδομένων με ένα κλικ
- Οπτικοποίηση σε πραγματικό χρόνο
- Εκπαίδευση μοντέλου με progress tracking
- Λεπτομερή αποτελέσματα και μετρικές
- Εξαγωγή αποτελεσμάτων

#### Wizard GUI (Προτεινόμενο για νέους χρήστες)

Εκτελέστε το wizard-style περιβάλλον για βήμα προς βήμα καθοδήγηση:

```powershell
# Ενεργοποίηση Virtual Environment
.\.venv\Scripts\Activate.ps1

# Εκκίνηση Wizard
python ddos_detection_gui_wizard.py
```

Το Wizard GUI σας καθοδηγεί μέσω:
1. Επιλογή μοντέλου και παραμέτρων
2. Ρύθμιση επιλογών ανάλυσης
3. Επεξεργασία δεδομένων
4. Παρουσίαση αποτελεσμάτων

### Δοκιμή Φόρτωσης Δεδομένων

Για να ελέγξετε ότι τα δεδομένα φορτώνονται σωστά:

```powershell
python test_data_loading.py
```

### Έλεγχος Αρχείων

```powershell
python check_files.py
```

## Δομή Φακέλων

```
.
├── .venv/                          # Virtual environment
├── ddos_detection_analysis.py      # Κύριο script (command line)
├── ddos_detection_gui.py           # Γραφικό περιβάλλον (GUI)
├── ddos_detection_gui_wizard.py    # Wizard GUI (βήμα προς βήμα)
├── test_data_loading.py            # Test script για δεδομένα
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
