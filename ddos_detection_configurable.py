"""
DDoS Detection System - Διαμορφώσιμο Script
=============================================

Αυτό το script παρέχει πλήρη έλεγχο της ανάλυσης DDoS μέσω μεταβλητών διαμόρφωσης.
Ρυθμίστε τις παραμέτρους στην ενότητα CONFIGURATION και εκτελέστε το script.

Δυνατότητες:
- Φόρτωση και προεπεξεργασία δεδομένων
- Οπτικοποίηση δεδομένων (on/off)
- Εφαρμογή SMOTE για ισορροπία κλάσεων (on/off)
- Εκπαίδευση μοντέλου Logistic Regression
- Αξιολόγηση και οπτικοποίηση αποτελεσμάτων
- Ανάλυση σημαντικότητας χαρακτηριστικών
- Αποθήκευση μοντέλου και αποτελεσμάτων
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score, 
    recall_score, 
    f1_score,
    roc_curve,
    auc
)

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - ΜΕΤΑΒΛΗΤΕΣ ΔΙΑΜΟΡΦΩΣΗΣ
# ============================================================================

class Config:
    """Κλάση διαμόρφωσης για το DDoS Detection System"""
    
    # --- ΑΡΧΕΙΑ ΔΕΔΟΜΕΝΩΝ ---
    CSV_FILE = 'DrDoS_DNS.csv'                    # Αρχείο εισόδου CSV
    
    # --- ΦΑΚΕΛΟΙ ΕΞΟΔΟΥ ---
    RESULTS_DIR = 'results'                       # Φάκελος για γραφήματα και αποτελέσματα
    MODEL_DIR = 'trained_model'                   # Φάκελος για αποθήκευση μοντέλου
    
    # --- ΧΩΡΙΣΜΟΣ ΔΕΔΟΜΕΝΩΝ ---
    TEST_SIZE = 0.3                               # Ποσοστό δεδομένων για test (0.3 = 30%)
    RANDOM_STATE = 42                             # Seed για reproducibility
    
    # --- SMOTE ΡΥΘΜΙΣΕΙΣ ---
    USE_SMOTE = True                              # True: Εφαρμογή SMOTE, False: Χωρίς SMOTE
    SMOTE_SAMPLING_STRATEGY = 1.0                 # 1.0 = 50-50 ισορροπία, 0.5 = 1:2, κλπ
    SMOTE_K_NEIGHBORS = 5                         # Αριθμός γειτόνων για SMOTE
    
    # --- ΜΟΝΤΕΛΟ ---
    MODEL_MAX_ITER = 1000                         # Μέγιστες επαναλήψεις για Logistic Regression
    MODEL_VERBOSE = 0                             # 0: Χωρίς output, 1: Με progress
    
    # --- ΟΠΤΙΚΟΠΟΙΗΣΕΙΣ ---
    ENABLE_VISUALIZATIONS = True                  # True: Δημιουργία γραφημάτων, False: Όχι
    PLOT_CORRELATION_MATRIX = True                # Χάρτης συσχέτισης
    PLOT_CLASS_DISTRIBUTION = True                # Pie chart κατανομής κλάσεων
    PLOT_FEATURE_DISTRIBUTIONS = True             # Box plots χαρακτηριστικών
    PLOT_SMOTE_COMPARISON = True                  # Σύγκριση πριν/μετά SMOTE
    PLOT_CONFUSION_MATRIX = True                  # Confusion Matrix
    PLOT_ROC_CURVE = True                         # ROC Curve
    PLOT_FEATURE_IMPORTANCE = True                # Σημαντικότητα χαρακτηριστικών
    
    # --- ΑΠΟΘΗΚΕΥΣΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ ---
    SAVE_MODEL = True                             # Αποθήκευση εκπαιδευμένου μοντέλου
    SAVE_METRICS_CSV = True                       # Αποθήκευση μετρικών σε CSV
    SAVE_FEATURE_IMPORTANCE_CSV = True            # Αποθήκευση σημαντικότητας χαρακτηριστικών
    
    # --- ΑΝΑΦΟΡΕΣ ---
    PRINT_DETAILED_REPORT = True                  # Λεπτομερής αναφορά στην κονσόλα
    PRINT_CLASSIFICATION_REPORT = True            # Classification report
    PRINT_FEATURE_IMPORTANCE_TOP_N = 15           # Πόσα top features να εμφανιστούν
    
    # --- ΓΡΑΦΗΜΑΤΑ ---
    FIGURE_DPI = 150                              # DPI για αποθήκευση εικόνων
    FIGURE_FORMAT = 'png'                         # Μορφή αποθήκευσης (png, jpg, pdf)
    
    # --- ΠΡΟΧΩΡΗΜΕΝΕΣ ΡΥΘΜΙΣΕΙΣ ---
    HANDLE_INFINITY = True                        # Διαχείριση inf τιμών
    HANDLE_NAN = True                             # Διαχείριση NaN τιμών
    STRATIFY_SPLIT = True                         # Stratified split για train/test
    
    # --- TOP FEATURES ΓΙΑ BOX PLOTS ---
    TOP_FEATURES_FOR_PLOTS = [
        ' Source Port', 
        ' Protocol', 
        ' Flow Duration',
        ' Total Fwd Packets', 
        ' Total Length of Bwd Packets'
    ]

# ============================================================================
# ΒΟΗΘΗΤΙΚΕΣ ΣΥΝΑΡΤΗΣΕΙΣ
# ============================================================================

def print_header(title, char="="):
    """Εκτύπωση όμορφου header"""
    print(f"\n{char*70}")
    print(f" {title}")
    print(f"{char*70}\n")

def print_section(title):
    """Εκτύπωση τίτλου section"""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}\n")

def save_figure(fig, filename, dpi=None):
    """Αποθήκευση γραφήματος"""
    if not Config.ENABLE_VISUALIZATIONS:
        plt.close(fig)
        return
        
    if not os.path.exists(Config.RESULTS_DIR):
        os.makedirs(Config.RESULTS_DIR)
    
    dpi = dpi or Config.FIGURE_DPI
    filepath = os.path.join(Config.RESULTS_DIR, f"{filename}.{Config.FIGURE_FORMAT}")
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    if Config.PRINT_DETAILED_REPORT:
        print(f"✓ Αποθηκεύτηκε: {filepath}")

def ensure_dir(directory):
    """Δημιουργία φακέλου αν δεν υπάρχει"""
    if not os.path.exists(directory):
        os.makedirs(directory)

# ============================================================================
# 1. ΦΟΡΤΩΣΗ ΚΑΙ ΠΡΟΕΠΕΞΕΡΓΑΣΙΑ ΔΕΔΟΜΕΝΩΝ
# ============================================================================

def load_and_preprocess_data():
    """
    Φορτώνει και προεπεξεργάζεται τα δεδομένα
    
    Returns:
        X: Χαρακτηριστικά
        y: Labels
        feature_names: Ονόματα χαρακτηριστικών
        df: Original dataframe
    """
    print_section("ΦΟΡΤΩΣΗ ΚΑΙ ΠΡΟΕΠΕΞΕΡΓΑΣΙΑ ΔΕΔΟΜΕΝΩΝ")
    
    # Φόρτωση δεδομένων
    print(f"📂 Φόρτωση δεδομένων από '{Config.CSV_FILE}'...")
    if not os.path.exists(Config.CSV_FILE):
        raise FileNotFoundError(f"Το αρχείο {Config.CSV_FILE} δεν βρέθηκε!")
    
    df = pd.read_csv(Config.CSV_FILE)
    print(f"✓ Φορτώθηκαν {len(df):,} εγγραφές")
    
    # Έλεγχος αν υπάρχει η στήλη Label
    if ' Label' not in df.columns:
        raise ValueError("Η στήλη ' Label' δεν βρέθηκε στο dataset!")
    
    # Επιλογή αριθμητικών χαρακτηριστικών
    print("\n🔍 Επιλογή χαρακτηριστικών...")
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    features = [col for col in numeric_features if col != ' Label']
    X = df[features]
    y = df[' Label']
    
    print(f"✓ Χρησιμοποιούνται {len(features)} αριθμητικά χαρακτηριστικά")
    
    if Config.PRINT_DETAILED_REPORT:
        print(f"\n📊 Διαστάσεις Dataset:")
        print(f"   - Εγγραφές: {X.shape[0]:,}")
        print(f"   - Χαρακτηριστικά: {X.shape[1]}")
    
    # Καθαρισμός δεδομένων
    if Config.HANDLE_INFINITY or Config.HANDLE_NAN:
        print("\n🔄 Καθαρισμός δεδομένων...")
        original_len = len(X)
        
        if Config.HANDLE_INFINITY:
            X = X.replace([np.inf, -np.inf], np.nan)
        
        if Config.HANDLE_NAN:
            mask = X.notna().all(axis=1)
            X = X[mask]
            y = y[mask]
            df = df[mask].reset_index(drop=True)
            removed_rows = original_len - len(X)
            print(f"✓ Αφαιρέθηκαν {removed_rows:,} γραμμές με ελλιπείς τιμές")
    
    print(f"✓ Τελικό μέγεθος: {len(X):,} εγγραφές")
    
    # Μετατροπή labels σε binary
    y = (y == 'DrDoS_DNS').astype(int)
    
    # Κατανομή κλάσεων
    if Config.PRINT_DETAILED_REPORT:
        value_counts = y.value_counts()
        print("\n📈 Κατανομή κλάσεων (Αρχικά δεδομένα):")
        for label, count in value_counts.items():
            percentage = count / len(y) * 100
            class_name = 'Attack (DrDoS_DNS)' if label == 1 else 'Normal Traffic'
            print(f"   - {class_name}: {count:,} ({percentage:.2f}%)")
    
    return X, y, X.columns.tolist(), df

# ============================================================================
# 2. ΟΠΤΙΚΟΠΟΙΗΣΗ ΔΕΔΟΜΕΝΩΝ
# ============================================================================

def visualize_data(X, y, df):
    """Δημιουργεί οπτικοποιήσεις των δεδομένων"""
    if not Config.ENABLE_VISUALIZATIONS:
        return
    
    print_section("ΟΠΤΙΚΟΠΟΙΗΣΗ ΔΕΔΟΜΕΝΩΝ")
    ensure_dir(Config.RESULTS_DIR)
    
    # 1. Χάρτης συσχέτισης
    if Config.PLOT_CORRELATION_MATRIX:
        print("📊 Δημιουργία χάρτη συσχέτισης...")
        fig, ax = plt.subplots(figsize=(20, 16))
        correlation_matrix = X.corr()
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Χάρτης Συσχέτισης Χαρακτηριστικών', fontsize=16, fontweight='bold')
        save_figure(fig, 'correlation_matrix')
    
    # 2. Κατανομή κλάσεων
    if Config.PLOT_CLASS_DISTRIBUTION:
        print("📊 Δημιουργία pie chart κλάσεων...")
        fig, ax = plt.subplots(figsize=(10, 8))
        value_counts = y.value_counts()
        attack_count = value_counts.get(1, 0)
        normal_count = value_counts.get(0, 0)
        
        sizes = [attack_count, normal_count]
        labels_pie = ['Attack (DrDoS_DNS)', 'Normal Traffic']
        colors = ['#ff9999', '#66b3ff']
        
        ax.pie(sizes, labels=labels_pie, autopct='%1.1f%%', colors=colors, 
               startangle=90, textprops={'fontsize': 14})
        ax.set_title('Κατανομή Κλάσεων (Πριν SMOTE)', fontsize=16, fontweight='bold')
        save_figure(fig, 'class_distribution_before_smote')
    
    # 3. Box plots για top χαρακτηριστικά
    if Config.PLOT_FEATURE_DISTRIBUTIONS:
        print("📊 Δημιουργία box plots χαρακτηριστικών...")
        available_features = [f for f in Config.TOP_FEATURES_FOR_PLOTS if f in df.columns]
        
        if available_features:
            n_features = len(available_features)
            n_cols = min(3, n_features)
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_features == 1:
                axes = [axes]
            else:
                axes = axes.ravel() if n_features > 1 else [axes]
            
            for i, feature in enumerate(available_features):
                if i < len(axes):
                    sns.boxplot(data=df, x=' Label', y=feature, ax=axes[i])
                    axes[i].set_title(f'Distribution of {feature}', fontsize=12)
            
            # Απόκρυψη κενών subplots
            for i in range(n_features, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            save_figure(fig, 'top_features_distribution')

# ============================================================================
# 3. ΧΩΡΙΣΜΟΣ ΔΕΔΟΜΕΝΩΝ ΚΑΙ ΚΑΝΟΝΙΚΟΠΟΙΗΣΗ
# ============================================================================

def prepare_train_test_data(X, y):
    """
    Χωρίζει τα δεδομένα και τα κανονικοποιεί
    
    Returns:
        scaler, X_train, X_test, y_train, y_test
    """
    print_section("ΧΩΡΙΣΜΟΣ ΚΑΙ ΚΑΝΟΝΙΚΟΠΟΙΗΣΗ ΔΕΔΟΜΕΝΩΝ")
    
    # Κανονικοποίηση
    print("🔄 Κανονικοποίηση δεδομένων (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Χωρισμός σε train/test
    print(f"🔄 Χωρισμός σε train/test sets (test_size={Config.TEST_SIZE})...")
    
    split_params = {
        'test_size': Config.TEST_SIZE,
        'random_state': Config.RANDOM_STATE
    }
    
    if Config.STRATIFY_SPLIT:
        split_params['stratify'] = y
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, **split_params)
    
    if Config.PRINT_DETAILED_REPORT:
        print(f"✓ Train set: {X_train.shape[0]:,} εγγραφές ({X_train.shape[0]/len(X_scaled)*100:.1f}%)")
        print(f"✓ Test set: {X_test.shape[0]:,} εγγραφές ({X_test.shape[0]/len(X_scaled)*100:.1f}%)")
    
    return scaler, X_train, X_test, y_train, y_test

# ============================================================================
# 4. ΕΦΑΡΜΟΓΗ SMOTE
# ============================================================================

def apply_smote(X_train, y_train):
    """
    Εφαρμόζει SMOTE για ισορροπία δεδομένων (αν ενεργοποιημένο)
    
    Returns:
        X_train_balanced, y_train_balanced
    """
    if not Config.USE_SMOTE:
        print_section("SMOTE - ΑΠΕΝΕΡΓΟΠΟΙΗΜΕΝΟ")
        print("⚠️ Το SMOTE είναι απενεργοποιημένο. Εκπαίδευση με αρχικά δεδομένα...")
        return X_train, y_train
    
    print_section("ΕΦΑΡΜΟΓΗ SMOTE ΓΙΑ ΙΣΟΡΡΟΠΙΑ ΔΕΔΟΜΕΝΩΝ")
    
    # Import SMOTE
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        print("⚠️ Το imbalanced-learn δεν είναι εγκατεστημένο!")
        print("   Εγκαταστήστε το με: pip install imbalanced-learn")
        print("   Συνέχεια χωρίς SMOTE...")
        return X_train, y_train
    
    # Αρχική κατανομή
    if Config.PRINT_DETAILED_REPORT:
        print("📊 Αρχική Κατανομή (Training Set):")
        print(f"   Normal (0): {sum(y_train == 0):,} ({sum(y_train == 0)/len(y_train)*100:.2f}%)")
        print(f"   Attack (1): {sum(y_train == 1):,} ({sum(y_train == 1)/len(y_train)*100:.2f}%)")
        print(f"   Συνολικά: {len(y_train):,}")
    
    # Εφαρμογή SMOTE
    print(f"\n⚙️ Εφαρμογή SMOTE (sampling_strategy={Config.SMOTE_SAMPLING_STRATEGY})...")
    smote = SMOTE(
        sampling_strategy=Config.SMOTE_SAMPLING_STRATEGY,
        random_state=Config.RANDOM_STATE,
        k_neighbors=Config.SMOTE_K_NEIGHBORS
    )
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Νέα κατανομή
    if Config.PRINT_DETAILED_REPORT:
        print("\n✅ Νέα Κατανομή (μετά το SMOTE):")
        print(f"   Normal (0): {sum(y_train_balanced == 0):,} ({sum(y_train_balanced == 0)/len(y_train_balanced)*100:.2f}%)")
        print(f"   Attack (1): {sum(y_train_balanced == 1):,} ({sum(y_train_balanced == 1)/len(y_train_balanced)*100:.2f}%)")
        print(f"   Συνολικά: {len(y_train_balanced):,}")
        
        # Στατιστικά
        class_0_increase = sum(y_train_balanced == 0) - sum(y_train == 0)
        class_1_increase = sum(y_train_balanced == 1) - sum(y_train == 1)
        
        if class_0_increase > 0:
            print(f"\n📈 Στατιστικά SMOTE:")
            print(f"   Συνθετικά Normal samples: {class_0_increase:,}")
        if class_1_increase > 0:
            print(f"   Συνθετικά Attack samples: {class_1_increase:,}")
    
    # Οπτικοποίηση
    if Config.ENABLE_VISUALIZATIONS and Config.PLOT_SMOTE_COMPARISON:
        print("\n📊 Οπτικοποίηση ισορροπίας...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Πριν SMOTE
        sizes_before = [sum(y_train == 0), sum(y_train == 1)]
        labels_pie = ['Normal', 'Attack']
        colors = ['#66b3ff', '#ff9999']
        
        axes[0].pie(sizes_before, labels=labels_pie, autopct='%1.1f%%',
                    colors=colors, startangle=90, textprops={'fontsize': 12})
        axes[0].set_title('Πριν το SMOTE\n(Ανισορροπημένο)', 
                         fontsize=14, fontweight='bold')
        
        # Μετά SMOTE
        sizes_after = [sum(y_train_balanced == 0), sum(y_train_balanced == 1)]
        axes[1].pie(sizes_after, labels=labels_pie, autopct='%1.1f%%',
                    colors=colors, startangle=90, textprops={'fontsize': 12})
        axes[1].set_title(f'Μετά το SMOTE\n(Ισορροπημένο)', 
                         fontsize=14, fontweight='bold')
        
        save_figure(fig, 'smote_balance_comparison')
    
    return X_train_balanced, y_train_balanced

# ============================================================================
# 5. ΕΚΠΑΙΔΕΥΣΗ ΜΟΝΤΕΛΟΥ
# ============================================================================

def train_model(X_train, y_train, scaler, feature_names):
    """
    Εκπαιδεύει το μοντέλο Logistic Regression
    
    Returns:
        model: Το εκπαιδευμένο μοντέλο
    """
    print_section("ΕΚΠΑΙΔΕΥΣΗ ΜΟΝΤΕΛΟΥ LOGISTIC REGRESSION")
    
    smote_status = "με SMOTE-balanced δεδομένα" if Config.USE_SMOTE else "χωρίς SMOTE"
    print(f"🔧 Εκπαίδευση μοντέλου {smote_status}...")
    print(f"   - Training samples: {X_train.shape[0]:,}")
    print(f"   - Features: {X_train.shape[1]}")
    print(f"   - Max iterations: {Config.MODEL_MAX_ITER}")
    
    model = LogisticRegression(
        random_state=Config.RANDOM_STATE,
        max_iter=Config.MODEL_MAX_ITER,
        verbose=Config.MODEL_VERBOSE
    )
    
    model.fit(X_train, y_train)
    print("✓ Η εκπαίδευση ολοκληρώθηκε επιτυχώς!")
    
    # Αποθήκευση μοντέλου
    if Config.SAVE_MODEL:
        print(f"\n💾 Αποθήκευση μοντέλου στον φάκελο '{Config.MODEL_DIR}/'...")
        ensure_dir(Config.MODEL_DIR)
        
        joblib.dump(model, f'{Config.MODEL_DIR}/logistic_regression_model.joblib')
        joblib.dump(scaler, f'{Config.MODEL_DIR}/scaler.joblib')
        pd.Series(feature_names).to_csv(f'{Config.MODEL_DIR}/feature_names.csv', index=False)
        
        print(f"✓ Αποθηκεύτηκαν τα αρχεία:")
        print(f"  - logistic_regression_model.joblib")
        print(f"  - scaler.joblib")
        print(f"  - feature_names.csv")
    
    return model

# ============================================================================
# 6. ΑΞΙΟΛΟΓΗΣΗ ΜΟΝΤΕΛΟΥ
# ============================================================================

def evaluate_model(model, X_test, y_test):
    """Αξιολογεί το μοντέλο"""
    print_section("ΑΞΙΟΛΟΓΗΣΗ ΜΟΝΤΕΛΟΥ")
    
    # Προβλέψεις
    print("🔍 Υπολογισμός προβλέψεων...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Classification Report
    if Config.PRINT_CLASSIFICATION_REPORT:
        print("\n📊 Classification Report:")
        print("-" * 70)
        print(classification_report(y_test, y_pred,
                                  target_names=['Normal Traffic', 'Attack Traffic']))
    
    # Υπολογισμός μετρικών
    accuracy = accuracy_score(y_test, y_pred)
    precision_normal = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
    recall_normal = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    f1_normal = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
    precision_attack = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall_attack = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1_attack = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    
    # Σύνοψη μετρικών
    if Config.PRINT_DETAILED_REPORT:
        print("\n📊 Σύνοψη Μετρικών:")
        print(f"   Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\n   Normal Traffic:")
        print(f"     - Precision: {precision_normal:.4f}")
        print(f"     - Recall: {recall_normal:.4f}")
        print(f"     - F1-Score: {f1_normal:.4f}")
        print(f"\n   Attack Traffic:")
        print(f"     - Precision: {precision_attack:.4f}")
        print(f"     - Recall: {recall_attack:.4f}")
        print(f"     - F1-Score: {f1_attack:.4f}")
    
    # Αποθήκευση μετρικών
    if Config.SAVE_METRICS_CSV:
        ensure_dir(Config.RESULTS_DIR)
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Normal Precision', 'Normal Recall', 'Normal F1',
                       'Attack Precision', 'Attack Recall', 'Attack F1'],
            'Score': [accuracy, precision_normal, recall_normal, f1_normal,
                      precision_attack, recall_attack, f1_attack]
        })
        metrics_path = f'{Config.RESULTS_DIR}/model_metrics.csv'
        metrics_df.to_csv(metrics_path, index=False)
        if Config.PRINT_DETAILED_REPORT:
            print(f"\n✓ Μετρικές αποθηκεύτηκαν: {metrics_path}")
    
    # Confusion Matrix
    if Config.ENABLE_VISUALIZATIONS and Config.PLOT_CONFUSION_MATRIX:
        print("\n📊 Δημιουργία Confusion Matrix...")
        fig, ax = plt.subplots(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    xticklabels=['Normal', 'Attack'],
                    yticklabels=['Normal', 'Attack'])
        ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        save_figure(fig, 'confusion_matrix')
    
    # ROC Curve
    if Config.ENABLE_VISUALIZATIONS and Config.PLOT_ROC_CURVE:
        print("📊 Δημιουργία ROC Curve...")
        fig, ax = plt.subplots(figsize=(10, 8))
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve', fontsize=16, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        save_figure(fig, 'roc_curve')
    else:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
    
    return {
        'accuracy': accuracy,
        'precision_normal': precision_normal,
        'recall_normal': recall_normal,
        'f1_normal': f1_normal,
        'precision_attack': precision_attack,
        'recall_attack': recall_attack,
        'f1_attack': f1_attack,
        'roc_auc': roc_auc
    }

# ============================================================================
# 7. ΑΝΑΛΥΣΗ ΣΗΜΑΝΤΙΚΟΤΗΤΑΣ ΧΑΡΑΚΤΗΡΙΣΤΙΚΩΝ
# ============================================================================

def analyze_feature_importance(model, feature_names):
    """Αναλύει τη σημαντικότητα των χαρακτηριστικών"""
    print_section("ΑΝΑΛΥΣΗ ΣΗΜΑΝΤΙΚΟΤΗΤΑΣ ΧΑΡΑΚΤΗΡΙΣΤΙΚΩΝ")
    
    # Υπολογισμός σημαντικότητας
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': abs(model.coef_[0]),
        'coefficient': model.coef_[0]
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # Top N
    if Config.PRINT_DETAILED_REPORT:
        top_n = Config.PRINT_FEATURE_IMPORTANCE_TOP_N
        print(f"📊 Top {top_n} Σημαντικότερα Χαρακτηριστικά:")
        print("-" * 70)
        for idx, row in feature_importance.head(top_n).iterrows():
            sign = '+' if row['coefficient'] > 0 else '-'
            print(f"{row['feature']:50} {row['importance']:.6f} ({sign})")
    
    # Οπτικοποίηση
    if Config.ENABLE_VISUALIZATIONS and Config.PLOT_FEATURE_IMPORTANCE:
        print("\n📊 Δημιουργία γραφήματος σημαντικότητας...")
        fig, ax = plt.subplots(figsize=(12, 10))
        top_features = feature_importance.head(20)
        
        colors = ['#ff9999' if c < 0 else '#99ff99' for c in top_features['coefficient']]
        sns.barplot(data=top_features, x='importance', y='feature', palette=colors, ax=ax)
        
        ax.set_title('Top 20 Most Important Features', fontsize=16, fontweight='bold')
        ax.set_xlabel('Absolute Coefficient Value', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.grid(axis='x', alpha=0.3)
        
        save_figure(fig, 'feature_importance')
    
    # Αποθήκευση
    if Config.SAVE_FEATURE_IMPORTANCE_CSV:
        ensure_dir(Config.RESULTS_DIR)
        importance_path = f'{Config.RESULTS_DIR}/feature_importance.csv'
        feature_importance.to_csv(importance_path, index=False)
        if Config.PRINT_DETAILED_REPORT:
            print(f"✓ Αποθηκεύτηκε: {importance_path}")
    
    return feature_importance

# ============================================================================
# 8. ΤΕΛΙΚΗ ΣΥΝΟΨΗ
# ============================================================================

def print_summary(metrics):
    """Εκτύπωση τελικής σύνοψης"""
    print_section("ΤΕΛΙΚΗ ΣΥΝΟΨΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ")
    
    print("📌 Βασικά Αποτελέσματα:\n")
    
    print(f"1. ΣΥΝΟΛΙΚΗ ΑΠΟΔΟΣΗ:")
    print(f"   ✓ Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   ✓ ROC AUC: {metrics['roc_auc']:.4f}")
    
    print(f"\n2. NORMAL TRAFFIC DETECTION:")
    print(f"   ✓ Precision: {metrics['precision_normal']:.4f}")
    print(f"   ✓ Recall: {metrics['recall_normal']:.4f}")
    print(f"   ✓ F1-Score: {metrics['f1_normal']:.4f}")
    
    print(f"\n3. ATTACK TRAFFIC DETECTION:")
    print(f"   ✓ Precision: {metrics['precision_attack']:.4f}")
    print(f"   ✓ Recall: {metrics['recall_attack']:.4f}")
    print(f"   ✓ F1-Score: {metrics['f1_attack']:.4f}")
    
    print("\n4. ΧΑΡΑΚΤΗΡΙΣΤΙΚΑ ΜΟΝΤΕΛΟΥ:")
    if Config.USE_SMOTE:
        print(f"   ✓ Εκπαιδευμένο με SMOTE-balanced δεδομένα (sampling_strategy={Config.SMOTE_SAMPLING_STRATEGY})")
    else:
        print("   ✓ Εκπαιδευμένο με αρχικά (ανισορροπημένα) δεδομένα")
    print("   ✓ Logistic Regression")
    print(f"   ✓ Test set: {Config.TEST_SIZE*100:.0f}% των δεδομένων")
    
    print("\n5. ΑΠΟΘΗΚΕΥΜΕΝΑ ΑΡΧΕΙΑ:")
    if Config.SAVE_MODEL:
        print(f"   ✓ Μοντέλο: {Config.MODEL_DIR}/")
    if Config.ENABLE_VISUALIZATIONS:
        print(f"   ✓ Γραφήματα: {Config.RESULTS_DIR}/")
    if Config.SAVE_METRICS_CSV or Config.SAVE_FEATURE_IMPORTANCE_CSV:
        print(f"   ✓ CSV Αποτελέσματα: {Config.RESULTS_DIR}/")

# ============================================================================
# ΚΥΡΙΑ ΣΥΝΑΡΤΗΣΗ
# ============================================================================

def main():
    """Κύρια συνάρτηση που εκτελεί όλη την ανάλυση"""
    print_header("DDOS DETECTION - ΔΙΑΜΟΡΦΩΣΙΜΟ ΣΥΣΤΗΜΑ", "=")
    print("Ρυθμίσεις που χρησιμοποιούνται:")
    print(f"  - CSV File: {Config.CSV_FILE}")
    print(f"  - Test Size: {Config.TEST_SIZE*100:.0f}%")
    print(f"  - SMOTE: {'Ενεργοποιημένο' if Config.USE_SMOTE else 'Απενεργοποιημένο'}")
    print(f"  - Visualizations: {'Ενεργοποιημένες' if Config.ENABLE_VISUALIZATIONS else 'Απενεργοποιημένες'}")
    print(f"  - Save Model: {'Ναι' if Config.SAVE_MODEL else 'Όχι'}")
    print()
    
    try:
        # 1. Φόρτωση και προεπεξεργασία
        X, y, feature_names, df = load_and_preprocess_data()
        
        # 2. Οπτικοποίηση
        visualize_data(X, y, df)
        
        # 3. Χωρισμός και κανονικοποίηση
        scaler, X_train, X_test, y_train, y_test = prepare_train_test_data(X, y)
        
        # 4. SMOTE (αν ενεργοποιημένο)
        X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)
        
        # 5. Εκπαίδευση
        model = train_model(X_train_balanced, y_train_balanced, scaler, feature_names)
        
        # 6. Αξιολόγηση
        metrics = evaluate_model(model, X_test, y_test)
        
        # 7. Ανάλυση χαρακτηριστικών
        feature_importance = analyze_feature_importance(model, feature_names)
        
        # 8. Σύνοψη
        print_summary(metrics)
        
        print_header("ΟΛΟΚΛΗΡΩΘΗΚΕ ΕΠΙΤΥΧΩΣ!", "=")
        print("✅ Το μοντέλο εκπαιδεύτηκε και αξιολογήθηκε επιτυχώς")
        
        if Config.SAVE_MODEL:
            print(f"✅ Το μοντέλο αποθηκεύτηκε στο '{Config.MODEL_DIR}/'")
        
        if Config.ENABLE_VISUALIZATIONS:
            print(f"✅ Τα γραφήματα αποθηκεύτηκαν στο '{Config.RESULTS_DIR}/'")
        
        print("\n💡 Tip: Μπορείς να αλλάξεις τις ρυθμίσεις στην κλάση Config")
        print("   στην αρχή του αρχείου για να προσαρμόσεις τη λειτουργία!")
        
    except FileNotFoundError as e:
        print(f"\n❌ Σφάλμα: {str(e)}")
        return 1
    except Exception as e:
        print(f"\n❌ Σφάλμα κατά την εκτέλεση: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

# ============================================================================
# ΕΚΤΕΛΕΣΗ
# ============================================================================

if __name__ == "__main__":
    exit(main())
