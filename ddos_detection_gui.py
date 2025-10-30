"""
DDoS Detection GUI - Γραφικό Περιβάλλον
========================================
Γραφικό περιβάλλον για την εκπαίδευση και αξιολόγηση μοντέλου DDoS detection
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Ορισμός χρωμάτων
COLORS = {
    'primary': '#2c3e50',
    'secondary': '#3498db',
    'success': '#27ae60',
    'danger': '#e74c3c',
    'warning': '#f39c12',
    'light': '#ecf0f1',
    'dark': '#34495e'
}

class DDoSDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DDoS Detection System - Γραφικό Περιβάλλον")
        self.root.geometry("1400x900")
        self.root.configure(bg=COLORS['light'])
        
        # Μεταβλητές
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.X_train_smote = None
        self.y_train_smote = None
        
        # Μεταβλητές GUI
        self.test_size_var = tk.DoubleVar(value=0.30)
        self.model_exists = tk.BooleanVar(value=self.check_model_exists())
        self.use_existing_model = tk.BooleanVar(value=False)
        
        # Επιλογές διαδικασιών
        self.load_data_var = tk.BooleanVar(value=True)
        self.visualize_var = tk.BooleanVar(value=False)
        self.train_model_var = tk.BooleanVar(value=True)
        self.evaluate_var = tk.BooleanVar(value=True)
        self.feature_importance_var = tk.BooleanVar(value=False)
        
        self.create_widgets()
        self.update_model_status()
        
    def check_model_exists(self):
        """Ελέγχει αν υπάρχει εκπαιδευμένο μοντέλο"""
        model_path = "trained_model/logistic_regression_model.joblib"
        return os.path.exists(model_path)
    
    def create_widgets(self):
        """Δημιουργία των widgets του GUI"""
        
        # Header
        header_frame = tk.Frame(self.root, bg=COLORS['primary'], height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame, 
            text="🛡️ DDoS Detection System",
            font=("Arial", 24, "bold"),
            bg=COLORS['primary'],
            fg='white'
        )
        title_label.pack(pady=20)
        
        # Main container
        main_container = tk.Frame(self.root, bg=COLORS['light'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Ρυθμίσεις
        left_panel = tk.Frame(main_container, bg='white', relief=tk.RAISED, borderwidth=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5), pady=0)
        
        self.create_settings_panel(left_panel)
        
        # Right panel - Αποτελέσματα και Οπτικοποιήσεις
        right_panel = tk.Frame(main_container, bg='white', relief=tk.RAISED, borderwidth=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=0)
        
        self.create_results_panel(right_panel)
        
    def create_settings_panel(self, parent):
        """Δημιουργία panel ρυθμίσεων"""
        
        # Τίτλος
        title = tk.Label(
            parent,
            text="⚙️ ΡΥΘΜΙΣΕΙΣ",
            font=("Arial", 16, "bold"),
            bg='white',
            fg=COLORS['primary']
        )
        title.pack(pady=10)
        
        # Separator
        ttk.Separator(parent, orient='horizontal').pack(fill=tk.X, padx=10, pady=5)
        
        # Κατάσταση Μοντέλου
        model_frame = tk.LabelFrame(
            parent,
            text="📊 Κατάσταση Μοντέλου",
            font=("Arial", 11, "bold"),
            bg='white',
            fg=COLORS['dark']
        )
        model_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.model_status_label = tk.Label(
            model_frame,
            text="",
            font=("Arial", 10),
            bg='white'
        )
        self.model_status_label.pack(pady=5)
        
        # Χρήση υπάρχοντος μοντέλου
        self.use_existing_cb = tk.Checkbutton(
            model_frame,
            text="Χρήση εκπαιδευμένου μοντέλου",
            variable=self.use_existing_model,
            font=("Arial", 10),
            bg='white',
            command=self.on_use_existing_changed
        )
        self.use_existing_cb.pack(pady=5)
        
        # Κουμπί διαγραφής μοντέλου
        delete_btn = tk.Button(
            model_frame,
            text="🗑️ Διαγραφή Μοντέλου",
            command=self.delete_model,
            bg=COLORS['danger'],
            fg='white',
            font=("Arial", 10, "bold"),
            cursor="hand2"
        )
        delete_btn.pack(pady=5)
        
        # Επιλογές Διαδικασιών
        process_frame = tk.LabelFrame(
            parent,
            text="🔧 Επιλογή Διαδικασιών",
            font=("Arial", 11, "bold"),
            bg='white',
            fg=COLORS['dark']
        )
        process_frame.pack(fill=tk.X, padx=10, pady=10)
        
        processes = [
            ("Φόρτωση Δεδομένων", self.load_data_var, True),
            ("Οπτικοποίηση Δεδομένων", self.visualize_var, False),
            ("Εκπαίδευση Μοντέλου", self.train_model_var, True),
            ("Αξιολόγηση Μοντέλου", self.evaluate_var, True),
            ("Ανάλυση Σημαντικότητας", self.feature_importance_var, False)
        ]
        
        for text, var, default in processes:
            cb = tk.Checkbutton(
                process_frame,
                text=text,
                variable=var,
                font=("Arial", 10),
                bg='white'
            )
            cb.pack(anchor=tk.W, padx=10, pady=2)
        
        # Ρυθμίσεις Train/Test Split
        split_frame = tk.LabelFrame(
            parent,
            text="📊 Train/Test Split",
            font=("Arial", 11, "bold"),
            bg='white',
            fg=COLORS['dark']
        )
        split_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Label για το ποσοστό
        self.split_label = tk.Label(
            split_frame,
            text=f"Test Size: {int(self.test_size_var.get() * 100)}%",
            font=("Arial", 10),
            bg='white'
        )
        self.split_label.pack(pady=5)
        
        # Slider (επιστρέφει ποσοστό 10-40, αλλά αποθηκεύεται ως 0.10-0.40)
        split_slider = tk.Scale(
            split_frame,
            from_=10,
            to=40,
            orient=tk.HORIZONTAL,
            resolution=5,
            length=200,
            command=self.update_split_label,
            bg='white'
        )
        split_slider.set(30)
        split_slider.pack(pady=5)
        
        info_label = tk.Label(
            split_frame,
            text="(Train: 90%-60%)",
            font=("Arial", 9, "italic"),
            bg='white',
            fg='gray'
        )
        info_label.pack()
        
        # Κουμπιά Εκτέλεσης
        buttons_frame = tk.Frame(parent, bg='white')
        buttons_frame.pack(fill=tk.X, padx=10, pady=20)
        
        start_btn = tk.Button(
            buttons_frame,
            text="▶️ ΕΚΚΙΝΗΣΗ",
            command=self.start_analysis,
            bg=COLORS['success'],
            fg='white',
            font=("Arial", 12, "bold"),
            height=2,
            cursor="hand2"
        )
        start_btn.pack(fill=tk.X, pady=5)
        
        clear_btn = tk.Button(
            buttons_frame,
            text="🗑️ Καθαρισμός Αποτελεσμάτων",
            command=self.clear_results,
            bg=COLORS['warning'],
            fg='white',
            font=("Arial", 10, "bold"),
            cursor="hand2"
        )
        clear_btn.pack(fill=tk.X, pady=5)
        
        # Progress Bar (Indeterminate)
        self.progress = ttk.Progressbar(
            parent,
            mode='indeterminate',
            length=250
        )
        self.progress.pack(pady=5)
        
        # Progress Bar (Determinate)
        self.progress_determinate = ttk.Progressbar(
            parent,
            mode='determinate',
            length=250,
            maximum=100
        )
        self.progress_determinate.pack(pady=5)
        
        # Progress Percentage Label
        self.progress_percent_label = tk.Label(
            parent,
            text="",
            font=("Arial", 9),
            bg='white',
            fg=COLORS['dark']
        )
        self.progress_percent_label.pack(pady=2)
        
        # Status Label
        self.status_label = tk.Label(
            parent,
            text="Έτοιμο",
            font=("Arial", 10),
            bg='white',
            fg=COLORS['success']
        )
        self.status_label.pack(pady=5)
        
    def create_results_panel(self, parent):
        """Δημιουργία panel αποτελεσμάτων"""
        
        # Τίτλος
        title = tk.Label(
            parent,
            text="📊 ΑΠΟΤΕΛΕΣΜΑΤΑ & ΟΠΤΙΚΟΠΟΙΗΣΗ",
            font=("Arial", 16, "bold"),
            bg='white',
            fg=COLORS['primary']
        )
        title.pack(pady=10)
        
        # Separator
        ttk.Separator(parent, orient='horizontal').pack(fill=tk.X, padx=10, pady=5)
        
        # Notebook για tabs
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Δεδομένα
        self.data_tab = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.data_tab, text="📋 Δεδομένα")
        
        self.data_text = scrolledtext.ScrolledText(
            self.data_tab,
            wrap=tk.WORD,
            width=70,
            height=20,
            font=("Courier", 10)
        )
        self.data_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 2: Αποτελέσματα
        self.results_tab = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.results_tab, text="📈 Αποτελέσματα")
        
        self.results_text = scrolledtext.ScrolledText(
            self.results_tab,
            wrap=tk.WORD,
            width=70,
            height=20,
            font=("Courier", 10)
        )
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 3: Γραφήματα
        self.plots_tab = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.plots_tab, text="📊 Γραφήματα")
        
        # Canvas για γραφήματα
        self.plot_canvas_frame = tk.Frame(self.plots_tab, bg='white')
        self.plot_canvas_frame.pack(fill=tk.BOTH, expand=True)
        
    def update_split_label(self, value):
        """Ενημέρωση label για test size"""
        # Μετατροπή του value σε float (π.χ. 30 -> 0.30)
        test_pct = int(float(value))
        train_pct = 100 - test_pct
        
        # Αποθήκευση ως float (0.10 - 0.40)
        self.test_size_var.set(test_pct / 100.0)
        
        self.split_label.config(text=f"Train: {train_pct}% | Test: {test_pct}%")
        
    def update_model_status(self):
        """Ενημέρωση κατάστασης μοντέλου"""
        exists = self.check_model_exists()
        self.model_exists.set(exists)
        
        if exists:
            self.model_status_label.config(
                text="✅ Εκπαιδευμένο μοντέλο διαθέσιμο",
                fg=COLORS['success']
            )
            # Ενεργοποίηση checkbox
            self.use_existing_cb.config(state=tk.NORMAL)
        else:
            self.model_status_label.config(
                text="❌ Δεν υπάρχει εκπαιδευμένο μοντέλο",
                fg=COLORS['danger']
            )
            # Απενεργοποίηση checkbox και unchecked
            self.use_existing_model.set(False)
            self.use_existing_cb.config(state=tk.DISABLED)
    
    def on_use_existing_changed(self):
        """Όταν αλλάζει η επιλογή χρήσης υπάρχοντος μοντέλου"""
        if self.use_existing_model.get():
            self.train_model_var.set(False)
        else:
            self.train_model_var.set(True)
    
    def delete_model(self):
        """Διαγραφή εκπαιδευμένου μοντέλου"""
        if not self.check_model_exists():
            messagebox.showinfo("Πληροφορία", "Δεν υπάρχει εκπαιδευμένο μοντέλο για διαγραφή.")
            return
        
        response = messagebox.askyesno(
            "Επιβεβαίωση Διαγραφής",
            "Είστε σίγουροι ότι θέλετε να διαγράψετε το εκπαιδευμένο μοντέλο;"
        )
        
        if response:
            try:
                if os.path.exists("trained_model"):
                    shutil.rmtree("trained_model")
                messagebox.showinfo("Επιτυχία", "Το μοντέλο διαγράφηκε επιτυχώς!")
                self.update_model_status()
            except Exception as e:
                messagebox.showerror("Σφάλμα", f"Αποτυχία διαγραφής: {str(e)}")
    
    def update_progress(self, percent, status_text=""):
        """Ενημέρωση progress bar"""
        self.progress_determinate['value'] = percent
        if status_text:
            self.progress_percent_label.config(text=f"{int(percent)}% - {status_text}")
        else:
            self.progress_percent_label.config(text=f"{int(percent)}%")
        self.root.update()
    
    def reset_progress(self):
        """Reset progress bar"""
        self.progress_determinate['value'] = 0
        self.progress_percent_label.config(text="")
        self.root.update()
    
    def log_message(self, message, text_widget=None):
        """Καταγραφή μηνύματος"""
        if text_widget is None:
            text_widget = self.results_text
        
        text_widget.insert(tk.END, message + "\n")
        text_widget.see(tk.END)
        self.root.update()
    
    def clear_results(self):
        """Καθαρισμός αποτελεσμάτων"""
        self.data_text.delete(1.0, tk.END)
        self.results_text.delete(1.0, tk.END)
        
        # Καθαρισμός γραφημάτων
        for widget in self.plot_canvas_frame.winfo_children():
            widget.destroy()
        
        # Reset progress
        self.reset_progress()
        self.status_label.config(text="Έτοιμο", fg=COLORS['success'])
    
    def start_analysis(self):
        """Εκκίνηση ανάλυσης σε ξεχωριστό thread"""
        # Έλεγχος αν τουλάχιστον μία διαδικασία είναι επιλεγμένη
        if not any([
            self.load_data_var.get(),
            self.visualize_var.get(),
            self.train_model_var.get(),
            self.evaluate_var.get(),
            self.feature_importance_var.get()
        ]):
            messagebox.showwarning("Προειδοποίηση", "Επιλέξτε τουλάχιστον μία διαδικασία!")
            return
        
        # Έλεγχος για χρήση υπάρχοντος μοντέλου
        if self.use_existing_model.get() and not self.check_model_exists():
            messagebox.showerror(
                "Σφάλμα",
                "Δεν υπάρχει εκπαιδευμένο μοντέλο!\nΑπενεργοποιήστε την επιλογή ή εκπαιδεύστε νέο μοντέλο."
            )
            return
        
        # Καθαρισμός προηγούμενων αποτελεσμάτων
        self.clear_results()
        
        # Εκκίνηση σε thread
        thread = threading.Thread(target=self.run_analysis)
        thread.daemon = True
        thread.start()
    
    def run_analysis(self):
        """Εκτέλεση ανάλυσης"""
        try:
            self.progress.start()
            self.reset_progress()
            self.status_label.config(text="Σε εξέλιξη...", fg=COLORS['warning'])
            
            # Υπολογισμός συνολικών βημάτων
            total_steps = 0
            if self.load_data_var.get():
                total_steps += 1
            if self.visualize_var.get():
                total_steps += 1
            if self.train_model_var.get() and not self.use_existing_model.get():
                total_steps += 1
            elif self.use_existing_model.get():
                total_steps += 1
            if self.evaluate_var.get():
                total_steps += 1
            if self.feature_importance_var.get():
                total_steps += 1
            
            current_step = 0
            step_percentage = 100 / total_steps if total_steps > 0 else 100
            
            # 1. Φόρτωση Δεδομένων
            if self.load_data_var.get():
                current_step += 1
                self.update_progress(current_step * step_percentage, "Φόρτωση δεδομένων...")
                self.log_message("=" * 70)
                self.log_message("📂 ΦΟΡΤΩΣΗ ΔΕΔΟΜΕΝΩΝ")
                self.log_message("=" * 70)
                self.load_data()
            
            # 2. Οπτικοποίηση
            if self.visualize_var.get() and self.data is not None:
                current_step += 1
                self.update_progress(current_step * step_percentage, "Οπτικοποίηση δεδομένων...")
                self.log_message("\n" + "=" * 70)
                self.log_message("📊 ΟΠΤΙΚΟΠΟΙΗΣΗ ΔΕΔΟΜΕΝΩΝ")
                self.log_message("=" * 70)
                self.visualize_data()
            
            # 3. Εκπαίδευση ή Φόρτωση Μοντέλου
            if self.use_existing_model.get():
                current_step += 1
                self.update_progress(current_step * step_percentage, "Φόρτωση μοντέλου...")
                self.log_message("\n" + "=" * 70)
                self.log_message("📥 ΦΟΡΤΩΣΗ ΕΚΠΑΙΔΕΥΜΕΝΟΥ ΜΟΝΤΕΛΟΥ")
                self.log_message("=" * 70)
                self.load_existing_model()
            elif self.train_model_var.get() and self.data is not None:
                current_step += 1
                self.update_progress(current_step * step_percentage, "Εκπαίδευση μοντέλου...")
                self.log_message("\n" + "=" * 70)
                self.log_message("🔧 ΕΚΠΑΙΔΕΥΣΗ ΜΟΝΤΕΛΟΥ")
                self.log_message("=" * 70)
                self.train_model()
            
            # 4. Αξιολόγηση
            if self.evaluate_var.get() and self.model is not None:
                current_step += 1
                self.update_progress(current_step * step_percentage, "Αξιολόγηση μοντέλου...")
                self.log_message("\n" + "=" * 70)
                self.log_message("📈 ΑΞΙΟΛΟΓΗΣΗ ΜΟΝΤΕΛΟΥ")
                self.log_message("=" * 70)
                self.evaluate_model()
            
            # 5. Feature Importance
            if self.feature_importance_var.get() and self.model is not None:
                current_step += 1
                self.update_progress(current_step * step_percentage, "Ανάλυση χαρακτηριστικών...")
                self.log_message("\n" + "=" * 70)
                self.log_message("🔍 ΑΝΑΛΥΣΗ ΣΗΜΑΝΤΙΚΟΤΗΤΑΣ ΧΑΡΑΚΤΗΡΙΣΤΙΚΩΝ")
                self.log_message("=" * 70)
                self.analyze_feature_importance()
            
            self.update_progress(100, "Ολοκληρώθηκε!")
            self.log_message("\n" + "=" * 70)
            self.log_message("✅ ΟΛΟΚΛΗΡΩΘΗΚΕ ΕΠΙΤΥΧΩΣ!")
            self.log_message("=" * 70)
            
            self.status_label.config(text="Ολοκληρώθηκε", fg=COLORS['success'])
            messagebox.showinfo("Επιτυχία", "Η ανάλυση ολοκληρώθηκε επιτυχώς!")
            
        except Exception as e:
            self.log_message(f"\n❌ ΣΦΑΛΜΑ: {str(e)}")
            self.status_label.config(text="Σφάλμα", fg=COLORS['danger'])
            self.update_progress(0, "Σφάλμα!")
            messagebox.showerror("Σφάλμα", f"Προέκυψε σφάλμα:\n{str(e)}")
        
        finally:
            self.progress.stop()
    
    def load_data(self):
        """Φόρτωση δεδομένων"""
        self.log_message("📂 Φόρτωση αρχείου DrDoS_DNS.csv...")
        self.log_message("   (Χρησιμοποιείται chunk reading για μεγάλα αρχεία...)")
        
        # Μέτρηση γραμμών αρχείου για progress bar
        try:
            self.log_message("🔍 Μέτρηση γραμμών αρχείου...")
            with open('DrDoS_DNS.csv', 'r') as f:
                total_lines = sum(1 for _ in f) - 1  # -1 για το header
            self.log_message(f"✓ Αναμένονται ~{total_lines:,} εγγραφές")
        except:
            total_lines = 5000000  # Default estimate
        
        # Φόρτωση με chunks και progress bar
        chunk_size = 500000
        chunks = []
        lines_read = 0
        
        self.log_message(f"📥 Φόρτωση δεδομένων σε chunks των {chunk_size:,}...")
        
        try:
            for i, chunk in enumerate(pd.read_csv('DrDoS_DNS.csv', 
                                                   chunksize=chunk_size, 
                                                   low_memory=False)):
                chunks.append(chunk)
                lines_read += len(chunk)
                
                # Ενημέρωση progress
                progress_pct = min(95, (lines_read / total_lines) * 100)
                self.log_message(f"   Chunk {i+1}: {len(chunk):,} εγγραφές (Σύνολο: {lines_read:,})")
                self.root.update()
            
            self.log_message("🔗 Συγχώνευση chunks...")
            self.data = pd.concat(chunks, ignore_index=True)
            self.log_message(f"✓ Φορτώθηκαν {len(self.data):,} εγγραφές")
            
        except Exception as e:
            self.log_message(f"⚠️ Chunk reading απέτυχε, χρήση standard loading...")
            self.data = pd.read_csv('DrDoS_DNS.csv', low_memory=False)
            self.log_message(f"✓ Φορτώθηκαν {len(self.data):,} εγγραφές")
        
        self.log_message(f"✓ Χαρακτηριστικά: {self.data.shape[1]}")
        
        # Εμφάνιση πληροφοριών στο data tab
        info = f"ΠΛΗΡΟΦΟΡΙΕΣ DATASET\n{'='*70}\n\n"
        info += f"Διαστάσεις: {self.data.shape}\n"
        info += f"Εγγραφές: {len(self.data):,}\n"
        info += f"Χαρακτηριστικά: {self.data.shape[1]}\n\n"
        info += f"Στήλες:\n{'-'*70}\n"
        info += "\n".join(self.data.columns.tolist()[:20])
        if len(self.data.columns) > 20:
            info += f"\n... και {len(self.data.columns) - 20} ακόμα\n"
        
        info += f"\n\n{'='*70}\n"
        info += "ΠΡΩΤΕΣ 10 ΕΓΓΡΑΦΕΣ:\n"
        info += f"{'='*70}\n\n"
        info += self.data.head(10).to_string()
        
        self.data_text.delete(1.0, tk.END)
        self.data_text.insert(tk.END, info)
        
        # Καθαρισμός δεδομένων
        self.log_message("\n🔄 Καθαρισμός δεδομένων...")
        initial_len = len(self.data)
        
        # Αφαίρεση μη-αριθμητικών στηλών (εκτός από Label)
        self.log_message("🔍 Έλεγχος τύπων δεδομένων...")
        
        # Κρατάμε το Label ξεχωριστά
        label_col = self.data[' Label'] if ' Label' in self.data.columns else None
        
        # Επιλογή μόνο αριθμητικών στηλών
        numeric_data = self.data.select_dtypes(include=[np.number])
        self.log_message(f"✓ Βρέθηκαν {numeric_data.shape[1]} αριθμητικές στήλες")
        
        # Αν υπάρχει Label, το προσθέτουμε πίσω
        if label_col is not None:
            self.data = numeric_data.copy()
            self.data[' Label'] = label_col
        else:
            self.data = numeric_data
        
        # Καθαρισμός πιο έξυπνα - αντικατάσταση αντί διαγραφής
        self.log_message("🧹 Καθαρισμός ελλιπών τιμών και infinity...")
        
        # Αντικατάσταση inf με NaN
        self.data = self.data.replace([np.inf, -np.inf], np.nan)
        
        # Αντί να διαγράψουμε γραμμές, γεμίζουμε με τη μέση τιμή κάθε στήλης
        for col in self.data.columns:
            if col != ' Label' and self.data[col].dtype in [np.float64, np.int64]:
                if self.data[col].isnull().any():
                    median_val = self.data[col].median()
                    self.data[col].fillna(median_val, inplace=True)
        
        # Τώρα διαγράφουμε μόνο γραμμές που έχουν NaN στο Label ή παντού
        self.data = self.data.dropna(subset=[' Label'])
        
        removed = initial_len - len(self.data)
        self.log_message(f"✓ Αφαιρέθηκαν {removed:,} γραμμές")
        self.log_message(f"✓ Τελικό μέγεθος: {len(self.data):,} εγγραφές")
        self.log_message(f"✓ Τελικές στήλες: {self.data.shape[1]}")
        
        # Κατανομή κλάσεων
        if ' Label' in self.data.columns:
            self.log_message("\n📊 Κατανομή Κλάσεων:")
            label_counts = self.data[' Label'].value_counts()
            for label, count in label_counts.items():
                pct = (count / len(self.data)) * 100
                self.log_message(f"   {label}: {count:,} ({pct:.2f}%)")
    
    def visualize_data(self):
        """Οπτικοποίηση δεδομένων"""
        self.log_message("📊 Δημιουργία οπτικοποιήσεων...")
        
        # Καθαρισμός προηγούμενων γραφημάτων
        for widget in self.plot_canvas_frame.winfo_children():
            widget.destroy()
        
        # Δημιουργία figure με subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Ανάλυση Δεδομένων DDoS', fontsize=16, fontweight='bold')
        
        # 1. Κατανομή κλάσεων
        if ' Label' in self.data.columns:
            label_counts = self.data[' Label'].value_counts()
            axes[0, 0].pie(
                label_counts.values,
                labels=label_counts.index,
                autopct='%1.1f%%',
                colors=['#27ae60', '#e74c3c']
            )
            axes[0, 0].set_title('Κατανομή Κλάσεων')
        
        # 2. Histogram για επιλεγμένα features
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns[:5]
        for i, col in enumerate(numeric_cols[:4]):
            row = (i + 1) // 2
            col_idx = (i + 1) % 2
            if row < 2 and col_idx < 2:
                self.data[col].hist(ax=axes[row, col_idx], bins=50)
                axes[row, col_idx].set_title(f'Κατανομή: {col[:30]}')
                axes[row, col_idx].set_xlabel('Τιμή')
                axes[row, col_idx].set_ylabel('Συχνότητα')
        
        plt.tight_layout()
        
        # Εμφάνιση στο GUI
        canvas = FigureCanvasTkAgg(fig, master=self.plot_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.log_message("✓ Οπτικοποιήσεις δημιουργήθηκαν επιτυχώς")
    
    def train_model(self):
        """Εκπαίδευση μοντέλου"""
        if self.data is None:
            self.load_data()
        
        self.log_message("🔧 Προετοιμασία δεδομένων...")
        
        # Διαχωρισμός features και target
        X = self.data.drop(' Label', axis=1)
        
        # Σωστό mapping των labels - ελέγχουμε τις πραγματικές τιμές
        unique_labels = self.data[' Label'].unique()
        self.log_message(f"🔍 Unique labels βρέθηκαν: {unique_labels}")
        
        # Mapping: BENIGN/Normal -> 0, DrDoS_DNS/Attack -> 1
        y = self.data[' Label'].apply(
            lambda x: 0 if x in ['Normal', 'BENIGN'] else 1
        )
        
        # Έλεγχος κατανομής κλάσεων
        self.log_message("\n📊 Κατανομή κλάσεων πριν SMOTE:")
        class_counts = y.value_counts()
        for cls, count in class_counts.items():
            label_name = "Normal" if cls == 0 else "Attack"
            pct = (count / len(y)) * 100
            self.log_message(f"   {label_name} ({cls}): {count:,} ({pct:.2f}%)")
        
        if len(class_counts) < 2:
            raise Exception("Το dataset έχει μόνο μία κλάση! Ελέγξτε τον καθαρισμό δεδομένων.")
        
        self.feature_names = X.columns.tolist()
        
        # Κανονικοποίηση ΠΡΙΝ το SMOTE
        self.log_message("\n🔄 Κανονικοποίηση δεδομένων...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Έλεγχος για inf/nan μετά την κανονικοποίηση
        self.log_message("🔍 Έλεγχος για inf/nan μετά την κανονικοποίηση...")
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        self.log_message("✓ Κανονικοποίηση ολοκληρώθηκε")
        
        # SMOTE - Εφαρμογή σε ΟΛΑ τα δεδομένα πρώτα
        self.log_message("\n⚙️ Εφαρμογή SMOTE σε όλο το dataset...")
        
        # Έλεγχος για minimum samples
        min_samples = class_counts.min()
        k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
        
        if k_neighbors < 1:
            self.log_message("⚠️ Η μειοψηφική κλάση έχει πολύ λίγα samples, παράλειψη SMOTE...")
            X_balanced = X_scaled
            y_balanced = y.values
        else:
            smote = SMOTE(sampling_strategy=1.0, random_state=42, k_neighbors=k_neighbors)
            X_balanced, y_balanced = smote.fit_resample(X_scaled, y)
            
            self.log_message(f"✓ Balanced Dataset: {len(X_balanced):,} εγγραφές")
            self.log_message(f"   Normal: {sum(y_balanced == 0):,}")
            self.log_message(f"   Attack: {sum(y_balanced == 1):,}")
        
        # Τώρα χωρισμός σε Train/Test από τα balanced δεδομένα
        test_size = self.test_size_var.get()
        self.log_message(f"\n📊 Χωρισμός balanced δεδομένων (Test: {int(test_size*100)}%)...")
        
        # Stratified split για να διατηρηθεί η 50-50 ισορροπία
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_balanced, y_balanced, 
            test_size=test_size, 
            random_state=42, 
            stratify=y_balanced
        )
        
        self.log_message(f"   Train set: {len(self.X_train):,} εγγραφές")
        self.log_message(f"     - Normal: {sum(self.y_train == 0):,} ({sum(self.y_train == 0)/len(self.y_train)*100:.1f}%)")
        self.log_message(f"     - Attack: {sum(self.y_train == 1):,} ({sum(self.y_train == 1)/len(self.y_train)*100:.1f}%)")
        
        self.log_message(f"   Test set: {len(self.X_test):,} εγγραφές")
        self.log_message(f"     - Normal: {sum(self.y_test == 0):,} ({sum(self.y_test == 0)/len(self.y_test)*100:.1f}%)")
        self.log_message(f"     - Attack: {sum(self.y_test == 1):,} ({sum(self.y_test == 1)/len(self.y_test)*100:.1f}%)")
        
        # Τα train δεδομένα είναι ήδη balanced, δεν χρειάζεται ξανά SMOTE
        self.X_train_smote = self.X_train
        self.y_train_smote = self.y_train
        
        # Εκπαίδευση
        self.log_message("\n🔧 Εκπαίδευση Logistic Regression...")
        self.log_message("   (Αυτό μπορεί να πάρει λίγο χρόνο...)")
        self.model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        self.model.fit(self.X_train_smote, self.y_train_smote)
        
        self.log_message("✓ Η εκπαίδευση ολοκληρώθηκε!")
        
        # Αποθήκευση
        self.log_message("\n💾 Αποθήκευση μοντέλου...")
        os.makedirs('trained_model', exist_ok=True)
        
        joblib.dump(self.model, 'trained_model/logistic_regression_model.joblib')
        joblib.dump(self.scaler, 'trained_model/scaler.joblib')
        pd.DataFrame({'feature': self.feature_names}).to_csv(
            'trained_model/feature_names.csv', index=False
        )
        
        self.log_message("✓ Μοντέλο αποθηκεύτηκε στο 'trained_model/'")
        self.update_model_status()
    
    def load_existing_model(self):
        """Φόρτωση υπάρχοντος μοντέλου"""
        self.log_message("📥 Φόρτωση εκπαιδευμένου μοντέλου...")
        
        try:
            self.model = joblib.load('trained_model/logistic_regression_model.joblib')
            self.scaler = joblib.load('trained_model/scaler.joblib')
            self.feature_names = pd.read_csv('trained_model/feature_names.csv')['feature'].tolist()
            
            self.log_message("✓ Μοντέλο φορτώθηκε επιτυχώς!")
            
            # Προετοιμασία δεδομένων για αξιολόγηση
            if self.data is None:
                self.load_data()
            
            X = self.data.drop(' Label', axis=1)
            # Σωστό mapping
            y = self.data[' Label'].apply(
                lambda x: 0 if x in ['Normal', 'BENIGN'] else 1
            )
            
            # Έλεγχος κλάσεων
            unique_classes = np.unique(y)
            if len(unique_classes) < 2:
                raise Exception("Το dataset έχει μόνο μία κλάση μετά τον καθαρισμό!")
            
            # Κανονικοποίηση
            self.log_message("🔄 Κανονικοποίηση δεδομένων...")
            X_scaled = self.scaler.transform(X)
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Εφαρμογή SMOTE για ισορροπημένο test set
            self.log_message("⚙️ Δημιουργία ισορροπημένου test set...")
            
            # ΔΕΝ χρειάζεται SMOTE σε όλα τα δεδομένα - χρησιμοποιούμε sample
            # Παίρνουμε ισορροπημένο sample για test
            class_counts = pd.Series(y).value_counts()
            min_samples = class_counts.min()
            
            self.log_message(f"📊 Κατανομή δεδομένων:")
            self.log_message(f"   BENIGN: {class_counts.get(0, 0):,}")
            self.log_message(f"   Attack: {class_counts.get(1, 0):,}")
            
            # Παίρνουμε balanced sample για test (όχι SMOTE σε όλο το dataset)
            test_size_count = int(len(X_scaled) * self.test_size_var.get())
            samples_per_class = min(min_samples, test_size_count // 2)
            
            self.log_message(f"🎯 Δημιουργία balanced test set με {samples_per_class:,} samples/class...")
            
            # Sample από κάθε κλάση
            benign_idx = y[y == 0].index[:samples_per_class]
            attack_idx = y[y == 1].index[:samples_per_class]
            test_idx = np.concatenate([benign_idx, attack_idx])
            
            self.X_test = X_scaled[test_idx]
            self.y_test = y.iloc[test_idx].values
            
            self.log_message(f"✓ Test set: {len(self.X_test):,} εγγραφές")
            self.log_message(f"   - Normal: {sum(self.y_test == 0):,} ({sum(self.y_test == 0)/len(self.y_test)*100:.1f}%)")
            self.log_message(f"   - Attack: {sum(self.y_test == 1):,} ({sum(self.y_test == 1)/len(self.y_test)*100:.1f}%)")
            
        except Exception as e:
            raise Exception(f"Αποτυχία φόρτωσης μοντέλου: {str(e)}")
    
    def evaluate_model(self):
        """Αξιολόγηση μοντέλου"""
        self.log_message("🔍 Υπολογισμός προβλέψεων...")
        self.log_message("   (Αυτό μπορεί να πάρει λίγο χρόνο...)")
        
        y_pred = self.model.predict(self.X_test)
        self.log_message("✓ Προβλέψεις ολοκληρώθηκαν")
        
        self.log_message("🔍 Υπολογισμός πιθανοτήτων...")
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Classification Report
        self.log_message("\n📊 Classification Report:")
        self.log_message("-" * 70)
        report = classification_report(
            self.y_test,
            y_pred,
            target_names=['Normal', 'Attack'],
            digits=4
        )
        self.log_message(report)
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        self.log_message("\n📊 Confusion Matrix:")
        self.log_message(f"                 Predicted")
        self.log_message(f"                Normal  Attack")
        self.log_message(f"Actual Normal   {cm[0,0]:6d}  {cm[0,1]:6d}")
        self.log_message(f"       Attack   {cm[1,0]:6d}  {cm[1,1]:6d}")
        
        # ROC AUC
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        self.log_message(f"\n📈 ROC AUC Score: {roc_auc:.4f}")
        
        # Accuracy
        accuracy = (y_pred == self.y_test).mean()
        self.log_message(f"✓ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Οπτικοποίηση αποτελεσμάτων
        self.log_message("\n📊 Δημιουργία γραφημάτων...")
        self.visualize_results(cm, self.y_test, y_pred_proba)
        self.log_message("✓ Γραφήματα δημιουργήθηκαν")
    
    def visualize_results(self, cm, y_test, y_pred_proba):
        """Οπτικοποίηση αποτελεσμάτων"""
        # Καθαρισμός προηγούμενων γραφημάτων
        for widget in self.plot_canvas_frame.winfo_children():
            widget.destroy()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Αποτελέσματα Μοντέλου', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'],
            ax=axes[0]
        )
        axes[0].set_title('Confusion Matrix')
        axes[0].set_ylabel('Actual')
        axes[0].set_xlabel('Predicted')
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend(loc="lower right")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Εμφάνιση στο GUI
        canvas = FigureCanvasTkAgg(fig, master=self.plot_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def analyze_feature_importance(self):
        """Ανάλυση σημαντικότητας χαρακτηριστικών"""
        self.log_message("🔍 Υπολογισμός σημαντικότητας χαρακτηριστικών...")
        
        # Λήψη coefficients
        coefficients = self.model.coef_[0]
        
        # Δημιουργία DataFrame
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)
        
        # Εμφάνιση top 15
        self.log_message("\n📊 Top 15 Σημαντικότερα Χαρακτηριστικά:")
        self.log_message("-" * 70)
        
        for idx, row in feature_importance.head(15).iterrows():
            sign = '+' if row['coefficient'] > 0 else '-'
            self.log_message(f"{row['feature']:50s} {row['coefficient']:8.4f} ({sign})")
        
        # Οπτικοποίηση
        for widget in self.plot_canvas_frame.winfo_children():
            widget.destroy()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        top_features = feature_importance.head(15)
        colors = ['#27ae60' if x > 0 else '#e74c3c' for x in top_features['coefficient']]
        
        ax.barh(range(len(top_features)), top_features['coefficient'], color=colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Coefficient Value')
        ax.set_title('Top 15 Σημαντικότερα Χαρακτηριστικά', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.plot_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.log_message("\n✓ Ανάλυση ολοκληρώθηκε!")


def main():
    """Main function"""
    root = tk.Tk()
    app = DDoSDetectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
