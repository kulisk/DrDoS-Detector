"""
Ολοκληρωμένη Ανάλυση DDoS Detection με Logistic Regression και SMOTE

Αυτό το script εκτελεί την πλήρη ανάλυση για την ανίχνευση DDoS επιθέσεων:
1. Φόρτωση και προεπεξεργασία δεδομένων
2. Οπτικοποίηση δεδομένων
3. Εφαρμογή SMOTE για ισορροπία δεδομένων
4. Εκπαίδευση μοντέλου Logistic Regression
5. Αξιολόγηση και οπτικοποίηση αποτελεσμάτων
6. Ανάλυση σημαντικότητας χαρακτηριστικών

Το μοντέλο εκπαιδεύεται ΜΟΝΟ με SMOTE-balanced δεδομένα.
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

# Απόκρυψη warnings
warnings.filterwarnings('ignore')

# Ρυθμίσεις οπτικοποίησης
sns.set_palette("husl")
plt.style.use('seaborn-v0_8-whitegrid')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

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

def save_figure(fig, filename, results_dir='results', dpi=150):
    """Αποθήκευση γραφήματος"""
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    filepath = os.path.join(results_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Αποθηκεύτηκε: {filepath}")

# ============================================================================
# 1. ΦΟΡΤΩΣΗ ΚΑΙ ΠΡΟΕΠΕΞΕΡΓΑΣΙΑ ΔΕΔΟΜΕΝΩΝ
# ============================================================================

def load_and_preprocess_data(csv_file='DrDoS_DNS.csv'):
    """
    Φορτώνει και προεπεξεργάζεται τα δεδομένα
    
    Args:
        csv_file: Το αρχείο CSV με τα δεδομένα
        
    Returns:
        X: Χαρακτηριστικά
        y: Labels
        feature_names: Ονόματα χαρακτηριστικών
        df: Original dataframe για visualizations
    """
    print_section("ΦΟΡΤΩΣΗ ΚΑΙ ΠΡΟΕΠΕΞΕΡΓΑΣΙΑ ΔΕΔΟΜΕΝΩΝ")
    
    # Φόρτωση δεδομένων
    print("📂 Φόρτωση δεδομένων...")
    df = pd.read_csv(csv_file)
    print(f"✓ Φορτώθηκαν {len(df):,} εγγραφές")
    
    # Επιλογή αριθμητικών χαρακτηριστικών
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    features = [col for col in numeric_features if col != ' Label']
    X = df[features]
    y = df[' Label']
    
    print(f"\n📊 Διαστάσεις Dataset:")
    print(f"   - Εγγραφές: {X.shape[0]:,}")
    print(f"   - Χαρακτηριστικά: {X.shape[1]}")
    
    # Καθαρισμός δεδομένων
    print("\n🔄 Καθαρισμός δεδομένων...")
    X = X.replace([np.inf, -np.inf], np.nan)
    original_len = len(X)
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
    value_counts = y.value_counts()
    print("\n📈 Κατανομή κλάσεων (ΠΡΙΝ το SMOTE):")
    for label, count in value_counts.items():
        percentage = count / len(y) * 100
        class_name = 'Attack' if label == 1 else 'Normal'
        print(f"   - {class_name}: {count:,} ({percentage:.2f}%)")
    
    return X, y, X.columns.tolist(), df

# ============================================================================
# 2. ΟΠΤΙΚΟΠΟΙΗΣΗ ΔΕΔΟΜΕΝΩΝ
# ============================================================================

def visualize_data(X, y, df, results_dir='results'):
    """
    Δημιουργεί οπτικοποιήσεις των δεδομένων
    
    Args:
        X: Χαρακτηριστικά
        y: Labels
        df: Original dataframe
        results_dir: Φάκελος αποθήκευσης
    """
    print_section("ΟΠΤΙΚΟΠΟΙΗΣΗ ΔΕΔΟΜΕΝΩΝ")
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 1. Χάρτης συσχέτισης
    print("📊 Δημιουργία χάρτη συσχέτισης...")
    fig, ax = plt.subplots(figsize=(20, 16))
    correlation_matrix = X.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Χάρτης Συσχέτισης Χαρακτηριστικών', fontsize=16, fontweight='bold')
    save_figure(fig, 'correlation_matrix.png', results_dir)
    
    # 2. Κατανομή κλάσεων
    print("📊 Δημιουργία pie chart κλάσεων...")
    fig, ax = plt.subplots(figsize=(10, 8))
    value_counts = y.value_counts()
    attack_count = value_counts.get(1, 0)
    normal_count = value_counts.get(0, 0)
    
    sizes = [attack_count, normal_count]
    labels_pie = ['Attack', 'Normal']
    colors = ['#ff9999', '#66b3ff']
    
    ax.pie(sizes, labels=labels_pie, autopct='%1.1f%%', colors=colors, 
           startangle=90, textprops={'fontsize': 14})
    ax.set_title('Κατανομή Κλάσεων (Πριν SMOTE)', fontsize=16, fontweight='bold')
    save_figure(fig, 'class_distribution_before_smote.png', results_dir)
    
    # 3. Box plots για top χαρακτηριστικά
    print("📊 Δημιουργία box plots...")
    top_features = [' Source Port', ' Protocol', ' Flow Duration', 
                   ' Total Fwd Packets', ' Total Length of Bwd Packets']
    
    available_features = [f for f in top_features if f in df.columns]
    
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
        save_figure(fig, 'top_features_distribution.png', results_dir)

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
    print("🔄 Κανονικοποίηση δεδομένων...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Χωρισμός σε train/test
    print("🔄 Χωρισμός σε train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"   Train set: {X_train.shape[0]:,} εγγραφές")
    print(f"   Test set: {X_test.shape[0]:,} εγγραφές")
    
    return scaler, X_train, X_test, y_train, y_test

# ============================================================================
# 4. ΕΦΑΡΜΟΓΗ SMOTE
# ============================================================================

def apply_smote(X_train, y_train, results_dir='results'):
    """
    Εφαρμόζει SMOTE για ισορροπία δεδομένων
    
    Returns:
        X_train_balanced, y_train_balanced
    """
    print_section("ΕΦΑΡΜΟΓΗ SMOTE ΓΙΑ ΙΣΟΡΡΟΠΙΑ ΔΕΔΟΜΕΝΩΝ")
    
    # Import SMOTE
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        print("⚠️ Εγκατάσταση imbalanced-learn...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'imbalanced-learn'])
        from imblearn.over_sampling import SMOTE
    
    # Αρχική κατανομή
    print("📊 Αρχική Κατανομή (Training Set):")
    print(f"   Normal (0): {sum(y_train == 0):,} ({sum(y_train == 0)/len(y_train)*100:.2f}%)")
    print(f"   Attack (1): {sum(y_train == 1):,} ({sum(y_train == 1)/len(y_train)*100:.2f}%)")
    print(f"   Συνολικά: {len(y_train):,}")
    
    # Εφαρμογή SMOTE
    print("\n⚙️ Εφαρμογή SMOTE (στόχος: 50-50 ισορροπία)...")
    smote = SMOTE(sampling_strategy=1.0, random_state=42, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Νέα κατανομή
    print("\n✅ Νέα Κατανομή (μετά το SMOTE):")
    print(f"   Normal (0): {sum(y_train_balanced == 0):,} ({sum(y_train_balanced == 0)/len(y_train_balanced)*100:.2f}%)")
    print(f"   Attack (1): {sum(y_train_balanced == 1):,} ({sum(y_train_balanced == 1)/len(y_train_balanced)*100:.2f}%)")
    print(f"   Συνολικά: {len(y_train_balanced):,}")
    
    # Στατιστικά
    normal_increase = sum(y_train_balanced == 0) - sum(y_train == 0)
    print(f"\n📈 Στατιστικά SMOTE:")
    print(f"   Συνθετικά Normal samples: {normal_increase:,}")
    print(f"   Αύξηση: {(normal_increase / sum(y_train == 0)):.1f}x")
    
    # Οπτικοποίηση
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
    axes[1].set_title('Μετά το SMOTE\n(Ισορροπημένο 50-50)', 
                     fontsize=14, fontweight='bold')
    
    save_figure(fig, 'smote_balance_comparison.png', results_dir)
    
    return X_train_balanced, y_train_balanced

# ============================================================================
# 5. ΕΚΠΑΙΔΕΥΣΗ ΜΟΝΤΕΛΟΥ
# ============================================================================

def train_model(X_train_balanced, y_train_balanced, scaler, feature_names):
    """
    Εκπαιδεύει το μοντέλο με balanced δεδομένα
    
    Returns:
        model: Το εκπαιδευμένο μοντέλο
    """
    print_section("ΕΚΠΑΙΔΕΥΣΗ ΜΟΝΤΕΛΟΥ LOGISTIC REGRESSION")
    
    print("🔧 Εκπαίδευση μοντέλου με SMOTE-balanced δεδομένα...")
    model = LogisticRegression(random_state=42, max_iter=1000, verbose=0)
    model.fit(X_train_balanced, y_train_balanced)
    print("✓ Η εκπαίδευση ολοκληρώθηκε!")
    
    # Αποθήκευση μοντέλου
    print("\n💾 Αποθήκευση μοντέλου...")
    model_dir = 'trained_model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    joblib.dump(model, f'{model_dir}/logistic_regression_model.joblib')
    joblib.dump(scaler, f'{model_dir}/scaler.joblib')
    pd.Series(feature_names).to_csv(f'{model_dir}/feature_names.csv', index=False)
    
    print(f"✓ Μοντέλο αποθηκεύτηκε στο '{model_dir}/'")
    print(f"  - logistic_regression_model.joblib")
    print(f"  - scaler.joblib")
    print(f"  - feature_names.csv")
    
    return model

# ============================================================================
# 6. ΑΞΙΟΛΟΓΗΣΗ ΜΟΝΤΕΛΟΥ
# ============================================================================

def evaluate_model(model, X_test, y_test, results_dir='results'):
    """
    Αξιολογεί το μοντέλο
    """
    print_section("ΑΞΙΟΛΟΓΗΣΗ ΜΟΝΤΕΛΟΥ")
    
    # Προβλέψεις
    print("🔍 Υπολογισμός προβλέψεων...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Classification Report
    print("\n📊 Classification Report:")
    print("-" * 70)
    print(classification_report(y_test, y_pred,
                              target_names=['Normal Traffic', 'Attack Traffic']))
    
    # Μετρικές
    accuracy = accuracy_score(y_test, y_pred)
    precision_normal = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
    recall_normal = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    f1_normal = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
    precision_attack = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall_attack = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1_attack = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    
    # Σύνοψη μετρικών
    print("\n📊 Σύνοψη Μετρικών:")
    print(f"   Overall Accuracy: {accuracy:.4f}")
    print(f"\n   Normal Traffic:")
    print(f"     - Precision: {precision_normal:.4f}")
    print(f"     - Recall: {recall_normal:.4f}")
    print(f"     - F1-Score: {f1_normal:.4f}")
    print(f"\n   Attack Traffic:")
    print(f"     - Precision: {precision_attack:.4f}")
    print(f"     - Recall: {recall_attack:.4f}")
    print(f"     - F1-Score: {f1_attack:.4f}")
    
    # Αποθήκευση μετρικών
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Normal Precision', 'Normal Recall', 'Normal F1',
                   'Attack Precision', 'Attack Recall', 'Attack F1'],
        'Score': [accuracy, precision_normal, recall_normal, f1_normal,
                  precision_attack, recall_attack, f1_attack]
    })
    metrics_df.to_csv(f'{results_dir}/model_metrics.csv', index=False)
    print(f"\n✓ Μετρικές αποθηκεύτηκαν στο '{results_dir}/model_metrics.csv'")
    
    # Confusion Matrix
    print("\n📊 Δημιουργία Confusion Matrix...")
    fig, ax = plt.subplots(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    save_figure(fig, 'confusion_matrix.png', results_dir)
    
    # ROC Curve
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
    save_figure(fig, 'roc_curve.png', results_dir)
    
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

def analyze_feature_importance(model, feature_names, results_dir='results'):
    """
    Αναλύει τη σημαντικότητα των χαρακτηριστικών
    """
    print_section("ΑΝΑΛΥΣΗ ΣΗΜΑΝΤΙΚΟΤΗΤΑΣ ΧΑΡΑΚΤΗΡΙΣΤΙΚΩΝ")
    
    # Υπολογισμός σημαντικότητας
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': abs(model.coef_[0]),
        'coefficient': model.coef_[0]
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # Top 15
    print("📊 Top 15 Σημαντικότερα Χαρακτηριστικά:")
    print("-" * 70)
    for idx, row in feature_importance.head(15).iterrows():
        sign = '+' if row['coefficient'] > 0 else '-'
        print(f"{row['feature']:50} {row['importance']:.6f} ({sign})")
    
    # Οπτικοποίηση
    print("\n📊 Δημιουργία γραφήματος σημαντικότητας...")
    fig, ax = plt.subplots(figsize=(12, 10))
    top_features = feature_importance.head(20)
    
    colors = ['#ff9999' if c < 0 else '#99ff99' for c in top_features['coefficient']]
    sns.barplot(data=top_features, x='importance', y='feature', palette=colors, ax=ax)
    
    ax.set_title('Top 20 Most Important Features', fontsize=16, fontweight='bold')
    ax.set_xlabel('Absolute Coefficient Value', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    
    save_figure(fig, 'feature_importance.png', results_dir)
    
    # Αποθήκευση
    feature_importance.to_csv(f'{results_dir}/feature_importance.csv', index=False)
    print(f"✓ Αποθηκεύτηκε: {results_dir}/feature_importance.csv")
    
    return feature_importance

# ============================================================================
# 8. ΤΕΛΙΚΗ ΣΥΝΟΨΗ
# ============================================================================

def print_summary(metrics):
    """
    Εκτύπωση τελικής σύνοψης
    """
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
    print("   ✓ Εκπαιδευμένο με SMOTE-balanced δεδομένα (50-50)")
    print("   ✓ Logistic Regression με όλα τα διαθέσιμα features")
    print("   ✓ Κατάλληλο για ανίχνευση DDoS επιθέσεων σε πραγματικό χρόνο")
    
    print("\n5. ΑΠΟΘΗΚΕΥΜΕΝΑ ΑΡΧΕΙΑ:")
    print("   ✓ Μοντέλο: trained_model/")
    print("   ✓ Αποτελέσματα: results/")
    print("   ✓ Γραφήματα: PNG files στο results/")

# ============================================================================
# ΚΥΡΙΑ ΣΥΝΑΡΤΗΣΗ
# ============================================================================

def main():
    """
    Κύρια συνάρτηση που εκτελεί όλη την ανάλυση
    """
    print_header("DDOS DETECTION - LOGISTIC REGRESSION ΜΕ SMOTE", "=")
    print("Αυτό το script εκτελεί την πλήρη ανάλυση για την ανίχνευση DDoS επιθέσεων")
    print("χρησιμοποιώντας Logistic Regression με SMOTE-balanced δεδομένα.\n")
    
    try:
        # 1. Φόρτωση και προεπεξεργασία
        X, y, feature_names, df = load_and_preprocess_data()
        
        # 2. Οπτικοποίηση
        visualize_data(X, y, df)
        
        # 3. Χωρισμός και κανονικοποίηση
        scaler, X_train, X_test, y_train, y_test = prepare_train_test_data(X, y)
        
        # 4. SMOTE
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
        print("✅ Το μοντέλο εκπαιδεύτηκε και αποθηκεύτηκε επιτυχώς")
        print("✅ Όλα τα γραφήματα δημιουργήθηκαν και αποθηκεύτηκαν")
        print("✅ Τα αποτελέσματα είναι διαθέσιμα στον φάκελο 'results/'")
        print("✅ Το μοντέλο είναι διαθέσιμο στον φάκελο 'trained_model/'")
        
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
