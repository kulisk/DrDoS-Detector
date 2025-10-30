"""
Summary του DDoS Detection Project

Τρέξτε αυτό το script για να δείτε την περίληψη του project.
"""

import os
from datetime import datetime

def print_header(title):
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}\n")

def get_file_size(filepath):
    """Επιστρέφει το μέγεθος αρχείου σε human-readable format"""
    if not os.path.exists(filepath):
        return "N/A"
    size = os.path.getsize(filepath)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"

def main():
    print_header("DDOS DETECTION PROJECT - ΠΕΡΙΛΗΨΗ")
    
    print("📌 Αυτό το project υλοποιεί σύστημα ανίχνευσης DDoS επιθέσεων")
    print("   χρησιμοποιώντας Logistic Regression με SMOTE-balanced δεδομένα.\n")
    
    # Κύριο Script
    print_header("ΚΥΡΙΟ SCRIPT")
    
    script_info = {
        'name': 'ddos_detection_analysis.py',
        'description': 'Ολοκληρωμένη ανάλυση DDoS Detection',
        'features': [
            '✓ Φόρτωση και προεπεξεργασία δεδομένων',
            '✓ Οπτικοποίηση δεδομένων (correlation matrix, distributions)',
            '✓ Εφαρμογή SMOTE για ισορροπία κλάσεων (50-50)',
            '✓ Εκπαίδευση μοντέλου Logistic Regression',
            '✓ Αξιολόγηση μοντέλου (accuracy, precision, recall, F1, ROC)',
            '✓ Ανάλυση σημαντικότητας χαρακτηριστικών',
        ],
        'usage': 'python ddos_detection_analysis.py',
        'output': [
            'Μοντέλο: trained_model/',
            'Αποτελέσματα: results/',
            'Γραφήματα: PNG files στο results/',
        ]
    }
    
    exists = os.path.exists(script_info['name'])
    status = "✓" if exists else "✗"
    
    print(f"{status} {script_info['name']}")
    print(f"   {script_info['description']}")
    
    if exists:
        size = get_file_size(script_info['name'])
        print(f"   Μέγεθος: {size}")
    
    print(f"\n   Χαρακτηριστικά:")
    for feature in script_info['features']:
        print(f"   {feature}")
    
    print(f"\n   Χρήση:")
    print(f"   {script_info['usage']}")
    
    print(f"\n   Output:")
    for output in script_info['output']:
        print(f"   - {output}")
    
    # Utility Scripts
    print_header("UTILITY SCRIPTS")
    
    utils = [
        ('check_files.py', 'Έλεγχος ύπαρξης αρχείων και μοντέλων'),
        ('00_project_summary.py', 'Αυτό το script - περίληψη project'),
    ]
    
    for filename, desc in utils:
        exists = os.path.exists(filename)
        status = "✓" if exists else "✗"
        print(f"{status} {filename:30} - {desc}")
    
    # Directories
    print_header("ΒΑΣΙΚΟΙ ΦΑΚΕΛΟΙ")
    
    dirs = [
        ('DrDoS_DNS.csv', 'Dataset με δεδομένα DDoS επιθέσεων'),
        ('trained_model/', 'Εκπαιδευμένο μοντέλο (SMOTE-balanced)'),
        ('results/', 'Αποτελέσματα ανάλυσης και γραφήματα'),
        ('.venv/', 'Python virtual environment'),
        ('pcap-01-12/', 'PCAP files (προαιρετικά)'),
    ]
    
    print("📁 Κύρια Directories:\n")
    for path, desc in dirs:
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        size = ""
        if exists and os.path.isfile(path):
            size = f" ({get_file_size(path)})"
        print(f"{status} {path:30} - {desc}{size}")
    
    # Εκτέλεση
    print_header("ΟΔΗΓΙΕΣ ΕΚΤΕΛΕΣΗΣ")
    
    steps = [
        ("1", "Ενεργοποίηση Virtual Environment", r".\.venv\Scripts\Activate.ps1"),
        ("2", "Έλεγχος αρχείων (προαιρετικά)", "python check_files.py"),
        ("3", "Εκτέλεση πλήρους ανάλυσης", "python ddos_detection_analysis.py"),
    ]
    
    for num, desc, cmd in steps:
        print(f"{num}. {desc}")
        print(f"   → {cmd}\n")
    
    # Αποτελέσματα
    print_header("ΑΝΑΜΕΝΟΜΕΝΑ ΑΠΟΤΕΛΕΣΜΑΤΑ")
    
    print("📊 Μετά την εκτέλεση θα δημιουργηθούν:\n")
    
    print("1. ΜΟΝΤΕΛΟ (trained_model/):")
    print("   ✓ logistic_regression_model.joblib - Το εκπαιδευμένο μοντέλο")
    print("   ✓ scaler.joblib - StandardScaler για normalization")
    print("   ✓ feature_names.csv - Ονόματα χαρακτηριστικών")
    
    print("\n2. ΓΡΑΦΗΜΑΤΑ (results/):")
    print("   ✓ correlation_matrix.png - Συσχετίσεις features")
    print("   ✓ class_distribution_before_smote.png - Αρχική κατανομή")
    print("   ✓ smote_balance_comparison.png - Πριν/μετά SMOTE")
    print("   ✓ confusion_matrix.png - Confusion matrix")
    print("   ✓ roc_curve.png - ROC curve και AUC")
    print("   ✓ feature_importance.png - Σημαντικότητα features")
    
    print("\n3. CSV FILES (results/):")
    print("   ✓ model_metrics.csv - Μετρικές απόδοσης")
    print("   ✓ feature_importance.csv - Feature importance ranking")
    
    # Χαρακτηριστικά
    print_header("ΧΑΡΑΚΤΗΡΙΣΤΙΚΑ ΜΟΝΤΕΛΟΥ")
    
    print("🔧 Τεχνικά Χαρακτηριστικά:\n")
    print("   ✓ Αλγόριθμος: Logistic Regression")
    print("   ✓ Balancing: SMOTE (50-50 ισορροπία)")
    print("   ✓ Features: Όλα τα διαθέσιμα χαρακτηριστικά του dataset")
    print("   ✓ Normalization: StandardScaler")
    print("   ✓ Train/Test Split: 70% / 30%")
    print("   ✓ Random State: 42 (για reproducibility)")
    
    print("\n📈 Πλεονεκτήματα:")
    print("   ✓ Ισορροπημένο μοντέλο (καλή απόδοση σε Normal & Attack)")
    print("   ✓ Υψηλή ακρίβεια ανίχνευσης")
    print("   ✓ Κατάλληλο για παραγωγική χρήση")
    print("   ✓ Ερμηνεύσιμο (feature importance analysis)")
    
    # Σημειώσεις
    print_header("ΣΗΜΕΙΩΣΕΙΣ")
    
    print("⚠️ Απαιτήσεις:")
    print("   - Python 3.7+")
    print("   - Virtual environment (.venv/)")
    print("   - Dataset: DrDoS_DNS.csv")
    print("   - Επαρκής RAM για SMOTE (>4GB συνιστάται)")
    
    print("\n📖 Περισσότερες πληροφορίες:")
    print("   - Δες το README.md για λεπτομερή documentation")
    print("   - Τρέξε check_files.py για έλεγχο αρχείων")
    
    print("\n" + "="*70)
    print(" Για βοήθεια: python ddos_detection_analysis.py --help")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
