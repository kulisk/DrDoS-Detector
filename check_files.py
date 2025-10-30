"""
Έλεγχος αρχείων και φακέλων.
"""

import os
import sys

def check_files():
    """Έλεγχος ύπαρξης απαραίτητων αρχείων."""
    
    print("\nΈλεγχος αρχείων PCAP:")
    pcap_dir = "pcap-01-12"
    if os.path.exists(pcap_dir):
        print(f"✓ Ο φάκελος {pcap_dir} υπάρχει")
        # Έλεγχος των πρώτων 5 αρχείων
        files = sorted(os.listdir(pcap_dir))[:5]
        for file in files:
            full_path = os.path.join(pcap_dir, file)
            print(f"  - {file}: {'✓' if os.path.isfile(full_path) else '✗'}")
    else:
        print(f"✗ Ο φάκελος {pcap_dir} δεν βρέθηκε!")
        
    print("\nΈλεγχος μοντέλων paper:")
    paper_dir = "trained_models_paper"
    if os.path.exists(paper_dir):
        print(f"✓ Ο φάκελος {paper_dir} υπάρχει")
        files = ['logistic_regression_paper.joblib', 'scaler_paper.joblib', 'feature_names_paper.csv']
        for file in files:
            full_path = os.path.join(paper_dir, file)
            print(f"  - {file}: {'✓' if os.path.isfile(full_path) else '✗'}")
    else:
        print(f"✗ Ο φάκελος {paper_dir} δεν βρέθηκε!")
        
    print("\nΈλεγχος μοντέλων full:")
    full_dir = "trained_models_full"
    if os.path.exists(full_dir):
        print(f"✓ Ο φάκελος {full_dir} υπάρχει")
        files = ['logistic_regression_full.joblib', 'scaler_full.joblib', 'feature_names_full.csv']
        for file in files:
            full_path = os.path.join(full_dir, file)
            print(f"  - {file}: {'✓' if os.path.isfile(full_path) else '✗'}")
    else:
        print(f"✗ Ο φάκελος {full_dir} δεν βρέθηκε!")
        
    print("\nΈλεγχος φακέλων αποτελεσμάτων:")
    results_dirs = ['pcap_analysis_paper', 'pcap_analysis_full']
    for dir in results_dirs:
        if os.path.exists(dir):
            print(f"✓ Ο φάκελος {dir} υπάρχει")
        else:
            print(f"✗ Ο φάκελος {dir} δεν βρέθηκε!")

if __name__ == "__main__":
    print("="*70)
    print(" ΕΛΕΓΧΟΣ ΑΡΧΕΙΩΝ ΚΑΙ ΦΑΚΕΛΩΝ")
    print("="*70)
    
    try:
        check_files()
        print("\n✓ Ο έλεγχος ολοκληρώθηκε!")
        print("="*70)
    except Exception as e:
        print(f"\n❌ Σφάλμα: {str(e)}")
        sys.exit(1)