"""
DDoS Detection Wizard GUI - Βήμα προς Βήμα Περιβάλλον
=====================================================
Wizard-style interface για φιλική εμπειρία χρήστη
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import threading
import queue
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Χρώματα θέματος
COLORS = {
    'primary': '#1e3a8a',      # Σκούρο μπλε
    'secondary': '#3b82f6',    # Μπλε
    'success': '#10b981',      # Πράσινο
    'danger': '#ef4444',       # Κόκκινο
    'warning': '#f59e0b',      # Πορτοκαλί
    'light': '#f3f4f6',        # Ανοιχτό γκρι
    'dark': '#111827',         # Σκούρο
    'white': '#ffffff'
}

class WizardPage(tk.Frame):
    """Βασική κλάση για wizard pages"""
    def __init__(self, parent, wizard):
        super().__init__(parent, bg=COLORS['white'])
        self.wizard = wizard
        self.pack(fill=tk.BOTH, expand=True)
        
    def on_show(self):
        """Καλείται όταν εμφανίζεται η σελίδα"""
        pass
    
    def on_hide(self):
        """Καλείται όταν κρύβεται η σελίδα"""
        pass
    
    def validate(self):
        """Επιστρέφει True αν μπορούμε να προχωρήσουμε"""
        return True


class WelcomePage(WizardPage):
    """Σελίδα Καλωσορίσματος"""
    def __init__(self, parent, wizard):
        super().__init__(parent, wizard)
        
        # Logo/Title
        title_frame = tk.Frame(self, bg=COLORS['primary'], height=200)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        tk.Label(
            title_frame,
            text="🛡️",
            font=("Arial", 72),
            bg=COLORS['primary'],
            fg=COLORS['white']
        ).pack(pady=20)
        
        tk.Label(
            title_frame,
            text="DDoS Detection System",
            font=("Arial", 32, "bold"),
            bg=COLORS['primary'],
            fg=COLORS['white']
        ).pack()
        
        tk.Label(
            title_frame,
            text="Ανίχνευση DDoS Επιθέσεων με Machine Learning",
            font=("Arial", 14),
            bg=COLORS['primary'],
            fg=COLORS['light']
        ).pack(pady=5)
        
        # Content
        content = tk.Frame(self, bg=COLORS['white'])
        content.pack(fill=tk.BOTH, expand=True, padx=50, pady=40)
        
        tk.Label(
            content,
            text="Καλώς ήρθατε!",
            font=("Arial", 24, "bold"),
            bg=COLORS['white'],
            fg=COLORS['dark']
        ).pack(pady=10)
        
        tk.Label(
            content,
            text="Αυτός ο οδηγός θα σας βοηθήσει να:",
            font=("Arial", 12),
            bg=COLORS['white'],
            fg=COLORS['dark'],
            justify=tk.LEFT
        ).pack(pady=10)
        
        features = [
            "✓ Επιλέξετε ή εκπαιδεύσετε ένα μοντέλο ανίχνευσης",
            "✓ Ρυθμίσετε τις παραμέτρους ανάλυσης",
            "✓ Δείτε την πρόοδο σε πραγματικό χρόνο",
            "✓ Αναλύσετε τα αποτελέσματα",
            "✓ Αποθηκεύσετε και εξάγετε αναφορές"
        ]
        
        for feature in features:
            tk.Label(
                content,
                text=feature,
                font=("Arial", 11),
                bg=COLORS['white'],
                fg=COLORS['dark'],
                anchor=tk.W
            ).pack(pady=3, anchor=tk.W, padx=50)
        
        tk.Label(
            content,
            text="\nΠατήστε 'Επόμενο' για να ξεκινήσετε",
            font=("Arial", 10, "italic"),
            bg=COLORS['white'],
            fg='gray'
        ).pack(pady=20)


class ModelSelectionPage(WizardPage):
    """Σελίδα Επιλογής Μοντέλου"""
    def __init__(self, parent, wizard):
        super().__init__(parent, wizard)
        
        # Header
        header = tk.Frame(self, bg=COLORS['secondary'], height=100)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text="Βήμα 1: Επιλογή Μοντέλου",
            font=("Arial", 24, "bold"),
            bg=COLORS['secondary'],
            fg=COLORS['white']
        ).pack(pady=30)
        
        # Content
        content = tk.Frame(self, bg=COLORS['white'])
        content.pack(fill=tk.BOTH, expand=True, padx=60, pady=40)
        
        tk.Label(
            content,
            text="Επιλέξτε πώς θέλετε να προχωρήσετε:",
            font=("Arial", 14),
            bg=COLORS['white']
        ).pack(pady=20)
        
        # Radio buttons
        self.wizard.model_choice = tk.StringVar(value="new")
        
        # Νέο μοντέλο
        new_frame = tk.Frame(content, bg=COLORS['light'], relief=tk.RAISED, borderwidth=2)
        new_frame.pack(fill=tk.X, pady=10, padx=20)
        
        rb_new = tk.Radiobutton(
            new_frame,
            text="🆕 Εκπαίδευση Νέου Μοντέλου",
            variable=self.wizard.model_choice,
            value="new",
            font=("Arial", 14, "bold"),
            bg=COLORS['light'],
            activebackground=COLORS['light'],
            command=self.update_description
        )
        rb_new.pack(anchor=tk.W, padx=20, pady=15)
        
        self.new_desc = tk.Label(
            new_frame,
            text="Θα εκπαιδεύσουμε ένα νέο μοντέλο από την αρχή χρησιμοποιώντας\n"
                 "το dataset DrDoS_DNS.csv με Logistic Regression και SMOTE.",
            font=("Arial", 10),
            bg=COLORS['light'],
            fg='gray',
            justify=tk.LEFT
        )
        self.new_desc.pack(anchor=tk.W, padx=40, pady=(0, 15))
        
        # Υπάρχον μοντέλο
        existing_frame = tk.Frame(content, bg=COLORS['light'], relief=tk.RAISED, borderwidth=2)
        existing_frame.pack(fill=tk.X, pady=10, padx=20)
        
        rb_existing = tk.Radiobutton(
            existing_frame,
            text="📂 Χρήση Εκπαιδευμένου Μοντέλου",
            variable=self.wizard.model_choice,
            value="existing",
            font=("Arial", 14, "bold"),
            bg=COLORS['light'],
            activebackground=COLORS['light'],
            command=self.update_description
        )
        rb_existing.pack(anchor=tk.W, padx=20, pady=15)
        
        self.existing_desc = tk.Label(
            existing_frame,
            text="",
            font=("Arial", 10),
            bg=COLORS['light'],
            fg='gray',
            justify=tk.LEFT
        )
        self.existing_desc.pack(anchor=tk.W, padx=40, pady=(0, 15))
        
        self.update_description()
    
    def update_description(self):
        """Ενημέρωση περιγραφής"""
        model_exists = os.path.exists('trained_model/logistic_regression_model.joblib')
        
        if model_exists:
            self.existing_desc.config(
                text="✅ Βρέθηκε εκπαιδευμένο μοντέλο. Θα το χρησιμοποιήσουμε\n"
                     "για άμεση αξιολόγηση χωρίς νέα εκπαίδευση.",
                fg=COLORS['success']
            )
        else:
            self.existing_desc.config(
                text="❌ Δεν βρέθηκε εκπαιδευμένο μοντέλο. Επιλέξτε εκπαίδευση νέου.",
                fg=COLORS['danger']
            )
    
    def on_show(self):
        self.update_description()
    
    def validate(self):
        if self.wizard.model_choice.get() == "existing":
            if not os.path.exists('trained_model/logistic_regression_model.joblib'):
                messagebox.showerror(
                    "Σφάλμα",
                    "Δεν βρέθηκε εκπαιδευμένο μοντέλο!\n"
                    "Παρακαλώ επιλέξτε 'Εκπαίδευση Νέου Μοντέλου'."
                )
                return False
        return True


class OptionsPage(WizardPage):
    """Σελίδα Επιλογών Ανάλυσης"""
    def __init__(self, parent, wizard):
        super().__init__(parent, wizard)
        
        # Header
        header = tk.Frame(self, bg=COLORS['secondary'], height=100)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text="Βήμα 2: Επιλογές Ανάλυσης",
            font=("Arial", 24, "bold"),
            bg=COLORS['secondary'],
            fg=COLORS['white']
        ).pack(pady=30)
        
        # Content
        content = tk.Frame(self, bg=COLORS['white'])
        content.pack(fill=tk.BOTH, expand=True, padx=60, pady=30)
        
        # Αριστερή στήλη - Διαδικασίες
        left_frame = tk.LabelFrame(
            content,
            text="🔧 Επιλογή Διαδικασιών",
            font=("Arial", 13, "bold"),
            bg=COLORS['white'],
            fg=COLORS['dark']
        )
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.wizard.visualize_var = tk.BooleanVar(value=True)
        self.wizard.evaluate_var = tk.BooleanVar(value=True)
        self.wizard.feature_importance_var = tk.BooleanVar(value=True)
        self.wizard.save_results_var = tk.BooleanVar(value=True)
        
        options = [
            ("📊 Οπτικοποίηση Δεδομένων", self.wizard.visualize_var),
            ("📈 Αξιολόγηση Μοντέλου", self.wizard.evaluate_var),
            ("🔍 Ανάλυση Σημαντικότητας Χαρακτηριστικών", self.wizard.feature_importance_var),
            ("💾 Αποθήκευση Αποτελεσμάτων", self.wizard.save_results_var)
        ]
        
        for text, var in options:
            cb = tk.Checkbutton(
                left_frame,
                text=text,
                variable=var,
                font=("Arial", 11),
                bg=COLORS['white']
            )
            cb.pack(anchor=tk.W, padx=20, pady=8)
        
        # Δεξιά στήλη - Παράμετροι
        right_frame = tk.LabelFrame(
            content,
            text="⚙️ Παράμετροι",
            font=("Arial", 13, "bold"),
            bg=COLORS['white'],
            fg=COLORS['dark']
        )
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Test Size
        tk.Label(
            right_frame,
            text="Μέγεθος Test Set:",
            font=("Arial", 11),
            bg=COLORS['white']
        ).pack(pady=(15, 5))
        
        self.wizard.test_size_var = tk.DoubleVar(value=0.30)
        
        self.split_label = tk.Label(
            right_frame,
            text="Train: 70% | Test: 30%",
            font=("Arial", 12, "bold"),
            bg=COLORS['white'],
            fg=COLORS['primary']
        )
        self.split_label.pack(pady=5)
        
        slider = tk.Scale(
            right_frame,
            from_=10,
            to=40,
            orient=tk.HORIZONTAL,
            resolution=5,
            length=250,
            command=self.update_split_label,
            bg=COLORS['white']
        )
        slider.set(30)
        slider.pack(pady=10)
        
        tk.Label(
            right_frame,
            text="ℹ️ Όσο μεγαλύτερο το test set,\nτόσο πιο αξιόπιστη η αξιολόγηση.",
            font=("Arial", 9, "italic"),
            bg=COLORS['white'],
            fg='gray',
            justify=tk.CENTER
        ).pack(pady=10)
    
    def update_split_label(self, value):
        test_pct = int(float(value))
        train_pct = 100 - test_pct
        self.wizard.test_size_var.set(test_pct / 100.0)
        self.split_label.config(text=f"Train: {train_pct}% | Test: {test_pct}%")


class ProcessingPage(WizardPage):
    """Σελίδα Επεξεργασίας"""
    def __init__(self, parent, wizard):
        super().__init__(parent, wizard)
        
        # Header
        header = tk.Frame(self, bg=COLORS['secondary'], height=100)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        self.header_label = tk.Label(
            header,
            text="Βήμα 3: Επεξεργασία...",
            font=("Arial", 24, "bold"),
            bg=COLORS['secondary'],
            fg=COLORS['white']
        )
        self.header_label.pack(pady=30)
        
        # Content
        content = tk.Frame(self, bg=COLORS['white'])
        content.pack(fill=tk.BOTH, expand=True, padx=40, pady=30)
        
        # Progress info
        progress_frame = tk.Frame(content, bg=COLORS['light'], relief=tk.SUNKEN, borderwidth=2)
        progress_frame.pack(fill=tk.X, pady=10)
        
        self.status_label = tk.Label(
            progress_frame,
            text="Προετοιμασία...",
            font=("Arial", 14, "bold"),
            bg=COLORS['light'],
            fg=COLORS['primary']
        )
        self.status_label.pack(pady=15)
        
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            mode='determinate',
            length=600
        )
        self.progress_bar.pack(pady=10, padx=20)
        
        self.progress_label = tk.Label(
            progress_frame,
            text="0%",
            font=("Arial", 11),
            bg=COLORS['light'],
            fg='gray'
        )
        self.progress_label.pack(pady=(0, 15))
        
        # Log area
        log_frame = tk.LabelFrame(
            content,
            text="📋 Λεπτομέρειες Διεργασίας",
            font=("Arial", 12, "bold"),
            bg=COLORS['white']
        )
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            wrap=tk.WORD,
            width=80,
            height=20,
            font=("Courier", 9)
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.processing_complete = False
        self.message_queue = queue.Queue()
    
    def enable_next_button(self):
        """Ενεργοποίηση next button (καλείται από main thread)"""
        self.processing_complete = True
        self.wizard.next_btn.config(state=tk.NORMAL)
        messagebox.showinfo("Επιτυχία", "Η ανάλυση ολοκληρώθηκε επιτυχώς!\n\nΠατήστε 'Επόμενο' για να δείτε τα αποτελέσματα.")
    
    def check_queue(self):
        """Έλεγχος για μηνύματα από το thread"""
        try:
            while True:
                msg = self.message_queue.get_nowait()
                if msg == "COMPLETE":
                    self.enable_next_button()
                    return
        except queue.Empty:
            pass
        # Συνέχιση ελέγχου
        self.after(100, self.check_queue)
    
    def on_show(self):
        """Ξεκινά την επεξεργασία"""
        self.processing_complete = False
        self.wizard.next_btn.config(state=tk.DISABLED)
        
        # Ξεκινάμε έλεγχο queue
        self.check_queue()
        
        # Ξεκινάμε σε thread
        thread = threading.Thread(target=self.run_processing)
        thread.daemon = True
        thread.start()
    
    def log_message(self, msg):
        """Προσθήκη μηνύματος στο log"""
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.update()
    
    def update_progress(self, percent, status=""):
        """Ενημέρωση progress"""
        self.progress_bar['value'] = percent
        self.progress_label.config(text=f"{int(percent)}%")
        if status:
            self.status_label.config(text=status)
        self.update()
    
    def run_processing(self):
        """Εκτέλεση όλης της ανάλυσης"""
        try:
            steps = []
            if self.wizard.model_choice.get() == "new":
                steps = ["load", "train", "evaluate"]
            else:
                steps = ["load", "load_model", "evaluate"]
            
            if self.wizard.visualize_var.get():
                steps.insert(1, "visualize")
            if self.wizard.feature_importance_var.get():
                steps.append("feature_importance")
            
            total_steps = len(steps)
            current_step = 0
            
            # Εκτέλεση κάθε βήματος
            for step in steps:
                current_step += 1
                progress = (current_step / total_steps) * 100
                
                if step == "load":
                    self.after(0, lambda p=progress: self.update_progress(p, "Φόρτωση δεδομένων..."))
                    self.load_data()
                elif step == "visualize":
                    self.after(0, lambda p=progress: self.update_progress(p, "Οπτικοποίηση..."))
                    self.visualize_data()
                elif step == "train":
                    self.after(0, lambda p=progress: self.update_progress(p, "Εκπαίδευση μοντέλου..."))
                    self.train_model()
                elif step == "load_model":
                    self.after(0, lambda p=progress: self.update_progress(p, "Φόρτωση μοντέλου..."))
                    self.load_existing_model()
                elif step == "evaluate":
                    self.after(0, lambda p=progress: self.update_progress(p, "Αξιολόγηση..."))
                    self.evaluate_model()
                elif step == "feature_importance":
                    self.after(0, lambda p=progress: self.update_progress(p, "Ανάλυση χαρακτηριστικών..."))
                    self.analyze_features()
            
            # Τελική ενημέρωση
            self.after(0, lambda: self.update_progress(100, "✅ Ολοκληρώθηκε!"))
            self.after(0, lambda: self.log_message("\n" + "=" * 70))
            self.after(0, lambda: self.log_message("✅ Η ΕΠΕΞΕΡΓΑΣΙΑ ΟΛΟΚΛΗΡΩΘΗΚΕ ΕΠΙΤΥΧΩΣ!"))
            self.after(0, lambda: self.log_message("=" * 70))
            
            # Στέλνουμε μήνυμα ολοκλήρωσης μέσω queue
            self.message_queue.put("COMPLETE")
            
        except Exception as e:
            error_msg = str(e)
            self.after(0, lambda: self.log_message(f"\n❌ ΣΦΑΛΜΑ: {error_msg}"))
            self.after(0, lambda: self.update_progress(0, "❌ Σφάλμα!"))
            self.after(0, lambda: messagebox.showerror("Σφάλμα", f"Προέκυψε σφάλμα:\n{error_msg}"))
    
    def load_data(self):
        """Φόρτωση δεδομένων"""
        self.log_message("📂 Φόρτωση DrDoS_DNS.csv...")
        self.log_message("   (Αυτό μπορεί να πάρει λίγο χρόνο για μεγάλο dataset...)")
        self.wizard.data = pd.read_csv('DrDoS_DNS.csv', low_memory=False)  # Όλα τα δεδομένα
        self.log_message(f"✓ Φορτώθηκαν {len(self.wizard.data):,} εγγραφές")
        
        # Καθαρισμός
        self.log_message("🧹 Καθαρισμός δεδομένων...")
        label_col = self.wizard.data[' Label']
        numeric_data = self.wizard.data.select_dtypes(include=[np.number])
        self.wizard.data = numeric_data.copy()
        self.wizard.data[' Label'] = label_col
        
        # Αντικατάσταση inf/nan
        self.wizard.data = self.wizard.data.replace([np.inf, -np.inf], np.nan)
        for col in self.wizard.data.columns:
            if col != ' Label':
                median_val = self.wizard.data[col].median()
                if pd.notna(median_val):
                    self.wizard.data[col].fillna(median_val, inplace=True)
        
        self.log_message(f"✓ Καθαρισμός ολοκληρώθηκε")
    
    def visualize_data(self):
        """Οπτικοποίηση"""
        self.log_message("📊 Δημιουργία οπτικοποιήσεων...")
        # Placeholder - θα εμφανιστεί στην επόμενη σελίδα
        self.log_message("✓ Οπτικοποιήσεις έτοιμες")
    
    def train_model(self):
        """Εκπαίδευση μοντέλου"""
        self.log_message("🔧 Προετοιμασία για εκπαίδευση...")
        
        X = self.wizard.data.drop(' Label', axis=1)
        y = self.wizard.data[' Label'].apply(lambda x: 0 if x in ['Normal', 'BENIGN'] else 1)
        
        self.log_message(f"📊 Κατανομή: Normal={sum(y==0):,}, Attack={sum(y==1):,}")
        
        # Κανονικοποίηση
        self.log_message("🔄 Κανονικοποίηση δεδομένων...")
        self.wizard.scaler = StandardScaler()
        X_scaled = self.wizard.scaler.fit_transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        self.log_message("✓ Κανονικοποίηση ολοκληρώθηκε")
        
        # SMOTE
        self.log_message("⚙️ Εφαρμογή SMOTE για ισορροπία κλάσεων...")
        self.log_message("   (Αυτό μπορεί να πάρει αρκετό χρόνο για μεγάλο dataset...)")
        
        min_samples = min(sum(y==0), sum(y==1))
        k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
        
        smote = SMOTE(sampling_strategy=1.0, random_state=42, k_neighbors=k_neighbors)
        X_balanced, y_balanced = smote.fit_resample(X_scaled, y)
        self.log_message(f"✓ Balanced: {len(X_balanced):,} samples")
        self.log_message(f"   Normal: {sum(y_balanced==0):,} | Attack: {sum(y_balanced==1):,}")
        
        # Split
        test_size = self.wizard.test_size_var.get()
        self.log_message(f"\n📊 Χωρισμός σε Train ({int((1-test_size)*100)}%) / Test ({int(test_size*100)}%)...")
        
        self.wizard.X_train, self.wizard.X_test, self.wizard.y_train, self.wizard.y_test = train_test_split(
            X_balanced, y_balanced, test_size=test_size, random_state=42, stratify=y_balanced
        )
        
        self.log_message(f"   Train: {len(self.wizard.X_train):,} samples")
        self.log_message(f"   Test: {len(self.wizard.X_test):,} samples")
        
        # Train
        self.log_message("\n🔧 Εκπαίδευση Logistic Regression...")
        self.log_message("   (Παρακαλώ περιμένετε...)")
        self.wizard.model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1, verbose=0)
        self.wizard.model.fit(self.wizard.X_train, self.wizard.y_train)
        
        # Save
        self.log_message("\n💾 Αποθήκευση μοντέλου...")
        os.makedirs('trained_model', exist_ok=True)
        joblib.dump(self.wizard.model, 'trained_model/logistic_regression_model.joblib')
        joblib.dump(self.wizard.scaler, 'trained_model/scaler.joblib')
        
        self.log_message("✓ Εκπαίδευση ολοκληρώθηκε και αποθηκεύτηκε!")
    
    def load_existing_model(self):
        """Φόρτωση μοντέλου"""
        self.log_message("📥 Φόρτωση εκπαιδευμένου μοντέλου...")
        self.wizard.model = joblib.load('trained_model/logistic_regression_model.joblib')
        self.wizard.scaler = joblib.load('trained_model/scaler.joblib')
        
        # Prep test data
        X = self.wizard.data.drop(' Label', axis=1)
        y = self.wizard.data[' Label'].apply(lambda x: 0 if x in ['Normal', 'BENIGN'] else 1)
        
        X_scaled = self.wizard.scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Balanced test sample
        test_size = int(len(X_scaled) * self.wizard.test_size_var.get())
        samples_per_class = min(sum(y==0), sum(y==1), test_size // 2)
        
        benign_idx = y[y == 0].index[:samples_per_class]
        attack_idx = y[y == 1].index[:samples_per_class]
        test_idx = np.concatenate([benign_idx, attack_idx])
        
        self.wizard.X_test = X_scaled[test_idx]
        self.wizard.y_test = y.iloc[test_idx].values
        
        self.log_message("✓ Μοντέλο φορτώθηκε!")
    
    def evaluate_model(self):
        """Αξιολόγηση"""
        self.log_message("📈 Αξιολόγηση μοντέλου...")
        
        y_pred = self.wizard.model.predict(self.wizard.X_test)
        y_pred_proba = self.wizard.model.predict_proba(self.wizard.X_test)[:, 1]
        
        # Μετρικές
        self.wizard.accuracy = accuracy_score(self.wizard.y_test, y_pred)
        self.wizard.cm = confusion_matrix(self.wizard.y_test, y_pred)
        self.wizard.roc_auc = roc_auc_score(self.wizard.y_test, y_pred_proba)
        
        self.log_message(f"✓ Accuracy: {self.wizard.accuracy:.4f}")
        self.log_message(f"✓ ROC AUC: {self.wizard.roc_auc:.4f}")
        
        # Save για results page
        self.wizard.y_pred = y_pred
        self.wizard.y_pred_proba = y_pred_proba
    
    def analyze_features(self):
        """Feature importance"""
        self.log_message("🔍 Ανάλυση σημαντικότητας...")
        try:
            coefficients = self.wizard.model.coef_[0]
            self.wizard.feature_importance = sorted(
                zip(range(len(coefficients)), np.abs(coefficients)),
                key=lambda x: x[1],
                reverse=True
            )[:15]
            self.log_message("✓ Ανάλυση ολοκληρώθηκε")
        except Exception as e:
            self.log_message(f"⚠️ Προειδοποίηση: {str(e)}")
            self.wizard.feature_importance = []
    
    def validate(self):
        return self.processing_complete


class ResultsPage(WizardPage):
    """Σελίδα Αποτελεσμάτων"""
    def __init__(self, parent, wizard):
        super().__init__(parent, wizard)
        
        # Header
        header = tk.Frame(self, bg=COLORS['success'], height=100)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text="✅ Αποτελέσματα Ανάλυσης",
            font=("Arial", 24, "bold"),
            bg=COLORS['success'],
            fg=COLORS['white']
        ).pack(pady=30)
        
        # Content
        content = tk.Frame(self, bg=COLORS['white'])
        content.pack(fill=tk.BOTH, expand=True, padx=40, pady=20)
        
        # Metrics
        metrics_frame = tk.LabelFrame(
            content,
            text="📊 Μετρικές Απόδοσης",
            font=("Arial", 13, "bold"),
            bg=COLORS['white']
        )
        metrics_frame.pack(fill=tk.X, pady=10)
        
        self.metrics_text = tk.Text(
            metrics_frame,
            height=10,
            font=("Courier", 11),
            bg=COLORS['light']
        )
        self.metrics_text.pack(fill=tk.X, padx=20, pady=15)
        
        # Visualization
        viz_frame = tk.LabelFrame(
            content,
            text="📈 Οπτικοποιήσεις",
            font=("Arial", 13, "bold"),
            bg=COLORS['white']
        )
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.viz_canvas_frame = tk.Frame(viz_frame, bg=COLORS['white'])
        self.viz_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Save button
        save_btn = tk.Button(
            content,
            text="💾 Αποθήκευση Αναφοράς",
            command=self.save_report,
            bg=COLORS['primary'],
            fg=COLORS['white'],
            font=("Arial", 12, "bold"),
            cursor="hand2",
            padx=20,
            pady=10
        )
        save_btn.pack(pady=15)
    
    def on_show(self):
        """Εμφάνιση αποτελεσμάτων"""
        # Metrics
        metrics = f"""
═══════════════════════════════════════════════════════════════════
                        ΑΠΟΤΕΛΕΣΜΑΤΑ ΑΝΑΛΥΣΗΣ
═══════════════════════════════════════════════════════════════════

📊 ΣΥΝΟΛΙΚΗ ΑΠΟΔΟΣΗ:
   • Accuracy:  {self.wizard.accuracy:.4f} ({self.wizard.accuracy*100:.2f}%)
   • ROC AUC:   {self.wizard.roc_auc:.4f}

📈 CONFUSION MATRIX:
                    Predicted
                  Normal    Attack
   Actual Normal   {self.wizard.cm[0,0]:6d}    {self.wizard.cm[0,1]:6d}
          Attack   {self.wizard.cm[1,0]:6d}    {self.wizard.cm[1,1]:6d}

✅ Το μοντέλο είναι έτοιμο για χρήση!
"""
        
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(1.0, metrics)
        
        # Visualizations
        self.create_visualizations()
    
    def create_visualizations(self):
        """Δημιουργία γραφημάτων"""
        for widget in self.viz_canvas_frame.winfo_children():
            widget.destroy()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Confusion Matrix
        sns.heatmap(
            self.wizard.cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'],
            ax=axes[0]
        )
        axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Actual')
        axes[0].set_xlabel('Predicted')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(self.wizard.y_test, self.wizard.y_pred_proba)
        axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {self.wizard.roc_auc:.4f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve', fontsize=14, fontweight='bold')
        axes[1].legend(loc="lower right")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.viz_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def save_report(self):
        """Αποθήκευση αναφοράς"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=f"DDoS_Analysis_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.metrics_text.get(1.0, tk.END))
            messagebox.showinfo("Επιτυχία", f"Η αναφορά αποθηκεύτηκε στο:\n{filename}")


class DDoSWizard(tk.Tk):
    """Κύριο Wizard παράθυρο"""
    def __init__(self):
        super().__init__()
        
        self.title("DDoS Detection System - Wizard")
        self.geometry("1000x700")
        self.configure(bg=COLORS['white'])
        
        # Center window
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (1000 // 2)
        y = (self.winfo_screenheight() // 2) - (700 // 2)
        self.geometry(f"1000x700+{x}+{y}")
        
        # Pages
        self.pages = []
        self.current_page = 0
        
        # Container για pages
        self.page_container = tk.Frame(self, bg=COLORS['white'])
        self.page_container.pack(fill=tk.BOTH, expand=True)
        
        # Navigation
        nav_frame = tk.Frame(self, bg=COLORS['light'], height=70)
        nav_frame.pack(fill=tk.X, side=tk.BOTTOM)
        nav_frame.pack_propagate(False)
        
        button_frame = tk.Frame(nav_frame, bg=COLORS['light'])
        button_frame.pack(pady=15)
        
        self.back_btn = tk.Button(
            button_frame,
            text="← Πίσω",
            command=self.previous_page,
            font=("Arial", 11),
            padx=20,
            pady=8,
            state=tk.DISABLED
        )
        self.back_btn.pack(side=tk.LEFT, padx=5)
        
        self.next_btn = tk.Button(
            button_frame,
            text="Επόμενο →",
            command=self.next_page,
            font=("Arial", 11, "bold"),
            bg=COLORS['primary'],
            fg=COLORS['white'],
            padx=20,
            pady=8
        )
        self.next_btn.pack(side=tk.LEFT, padx=5)
        
        self.cancel_btn = tk.Button(
            button_frame,
            text="Ακύρωση",
            command=self.quit,
            font=("Arial", 11),
            padx=20,
            pady=8
        )
        self.cancel_btn.pack(side=tk.LEFT, padx=5)
        
        # Προσθήκη σελίδων
        self.add_page(WelcomePage(self.page_container, self))
        self.add_page(ModelSelectionPage(self.page_container, self))
        self.add_page(OptionsPage(self.page_container, self))
        self.add_page(ProcessingPage(self.page_container, self))
        self.add_page(ResultsPage(self.page_container, self))
        
        self.show_page(0)
    
    def add_page(self, page):
        """Προσθήκη σελίδας"""
        page.pack_forget()
        self.pages.append(page)
    
    def show_page(self, index):
        """Εμφάνιση σελίδας"""
        if 0 <= index < len(self.pages):
            # Hide current
            if self.pages[self.current_page]:
                self.pages[self.current_page].pack_forget()
                self.pages[self.current_page].on_hide()
            
            # Show new
            self.current_page = index
            self.pages[self.current_page].pack(fill=tk.BOTH, expand=True)
            self.pages[self.current_page].on_show()
            
            # Update buttons
            self.back_btn.config(state=tk.NORMAL if index > 0 else tk.DISABLED)
            
            if index == len(self.pages) - 1:
                self.next_btn.config(text="Τέλος", command=self.finish)
            else:
                self.next_btn.config(text="Επόμενο →", command=self.next_page)
    
    def next_page(self):
        """Επόμενη σελίδα"""
        if self.pages[self.current_page].validate():
            self.show_page(self.current_page + 1)
    
    def previous_page(self):
        """Προηγούμενη σελίδα"""
        self.show_page(self.current_page - 1)
    
    def finish(self):
        """Ολοκλήρωση"""
        if messagebox.askyesno("Ολοκλήρωση", "Θέλετε να κλείσετε την εφαρμογή;"):
            self.quit()


if __name__ == "__main__":
    app = DDoSWizard()
    app.mainloop()
