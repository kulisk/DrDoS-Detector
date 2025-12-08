# 🛡️ DrDoS-Detector

**Comprehensive DNS Security Suite with Machine Learning**

A complete DNS security analysis framework featuring four specialized detection modules for DNS-based attacks and anomalies.

---

## 📊 Overview

DrDoS-Detector is a modular DNS security project that implements machine learning models for detecting and analyzing various DNS-based threats:

| Module | Purpose | Accuracy | Dataset |
|--------|---------|----------|---------|
| **DDoS Detector** | DNS amplification DDoS detection | 99.99% | CICDDoS2019 |
| **Exfiltration Detection** | DNS tunneling & data exfiltration | 96.29% | CIC-Bell-DNS-EXF-2021 |
| **DoH Detection** | Malicious DNS-over-HTTPS traffic | 99.94% | CIRA-CIC-DoHBrw-2020 |
| **Reflector Analysis** | Statistical domain analysis | N/A | CIC-Bell-DNS 2021 |

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip install -r requirements.txt
```

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/kulisk/DrDoS-Detector.git
cd DrDoS-Detector
```

2. **Download datasets:**

The datasets are available separately due to their size. Download them from:
**[GitHub Releases - Datasets](https://github.com/kulisk/DrDoS-Detector/releases/tag/dataset)**

Extract the `datasets/` folder to the project root directory.

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Running Individual Modules

```bash
# DDoS Detection
cd ddos_detector
python main.py

# Exfiltration Detection
cd exfiltration_detection
python main.py

# DoH Detection
cd doh_detection
python main.py

# Reflector Analysis
cd reflector_analysis
python main.py
```

### Running All Modules

```bash
python run_all.py
```

---

## 📁 Project Structure

```
DrDoS-Detector/
├── datasets/                          # All datasets (CSV & TXT files)
│   ├── DrDoS_DNS.csv                 # DDoS detection dataset
│   ├── l1-doh.csv                    # DoH traffic dataset
│   ├── l1-nondoh.csv                 # Non-DoH traffic dataset
│   ├── stateless_features-benign_*.csv  # Exfiltration benign samples
│   ├── stateful_features-benign_*.csv   # Exfiltration benign samples
│   ├── domains_FirstDay.txt          # Domain analysis data
│   ├── domains_SecondDay.txt         # Domain analysis data
│   └── domains_ThirdDay.txt          # Domain analysis data
│
├── ddos_detector/                     # Module 1: DDoS Amplification Detection
│   ├── main.py                       # Main execution (config + pipeline)
│   └── utils.py                      # Helper functions (organized by stages)
│
├── exfiltration_detection/           # Module 2: DNS Exfiltration Detection
│   ├── main.py                       # Main execution (config + pipeline)
│   └── utils.py                      # Helper functions (organized by stages)
│
├── doh_detection/                    # Module 3: DoH Detection
│   ├── main.py                       # Main execution (config + pipeline)
│   └── utils.py                      # Helper functions (organized by stages)
│
├── reflector_analysis/               # Module 4: Domain Analysis
│   ├── main.py                       # Main execution (config + pipeline)
│   └── utils.py                      # Helper functions (organized by stages)
│
├── run_all.py                        # Execute all modules sequentially
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

### Module Architecture

Each module follows a clean **2-file structure**:

- **`main.py`**: Configuration, parameters, and main execution pipeline
- **`utils.py`**: Helper functions organized by pipeline stages:
  - **Stage Functions** (top-level): `preprocessing_stage()`, `training_stage()`, `evaluation_stage()`
  - **Helper Functions** (internal): `_load_data()`, `_clean_data()`, `_train_models()`, etc.

---

## 🔧 Module Details

### 1️⃣ DDoS Detector (`ddos_detector/`)

Detects DNS amplification DDoS attacks using multiple ML classifiers.

**Features:**
- Multi-model training (Logistic Regression, Random Forest, Decision Tree, SVM, KNN)
- SMOTE-based class balancing (10x multiplication of benign samples)
- Automated model selection based on accuracy
- Model persistence for production deployment

**Configuration (main.py):**
```python
RANDOM_STATE = 42
CSV_PATH = '../datasets/DrDoS_DNS.csv'
TEST_SIZE = 0.20
SMOTE_TARGET_RATIO = 10

ENABLE_MODELS = {
    'Logistic Regression': True,
    'Random Forest': True,
    'Decision Tree': True,
    'SVM': False,   # Slow
    'KNN': False    # Very slow
}
```

**Results:**
- Accuracy: 99.99% (Random Forest & Decision Tree)
- Training time: ~2-5 seconds per model
- Dataset: CICDDoS2019 - DrDoS_DNS.csv

**Pipeline Stages:**
1. `preprocessing_stage()`: Load → Clean → Encode → Balance (SMOTE) → Split → Scale
2. `training_stage()`: Train selected models
3. `evaluation_stage()`: Evaluate and compare models
4. `persistence_stage()`: Save best model

---

### 2️⃣ Exfiltration Detection (`exfiltration_detection/`)

Detects DNS tunneling and data exfiltration attempts.

**Features:**
- Analyzes stateless and stateful DNS features
- Detects multiple exfiltration types (audio, video, text, compressed, exe, image)
- Multi-class classification (Benign vs Malicious types)

**Configuration (main.py):**
```python
RANDOM_STATE = 42
TEST_SIZE = 0.20

STATELESS_FILES = [
    '../datasets/stateless_features-benign_1.pcap.csv',
    '../datasets/stateless_features-benign_2.pcap.csv',
]

STATEFUL_FILES = [
    '../datasets/stateful_features-benign_1.pcap.csv',
    '../datasets/stateful_features-benign_2.pcap.csv',
]

ATTACK_FOLDERS = [
    '../CIC-Bell-DNS-EXF-2021 dataset/Attack_heavy_Benign/Attacks',
    '../CIC-Bell-DNS-EXF-2021 dataset/Attack_Light_Benign/Attacks',
]
```

**Results:**
- Accuracy: 96.29% (Decision Tree)
- Processed samples: 685,747
- Dataset: CIC-Bell-DNS-EXF-2021

**Pipeline Stages:**
1. `data_loading_stage()`: Load benign + attack samples
2. `preprocessing_stage()`: Clean → Encode → Split
3. `training_stage()`: Train Random Forest, Decision Tree, Logistic Regression
4. `evaluation_stage()`: Compare and select best model

---

### 3️⃣ DoH Detection (`doh_detection/`)

Identifies malicious DNS-over-HTTPS traffic.

**Features:**
- Binary classification (DoH vs non-DoH)
- High-speed detection with minimal false positives
- Suitable for real-time traffic analysis

**Configuration (main.py):**
```python
RANDOM_STATE = 42
TEST_SIZE = 0.20
SAMPLE_LIMIT = 50000

STAGE1_DOH = '../datasets/l1-doh.csv'
STAGE1_NONDOH = '../datasets/l1-nondoh.csv'

MODEL_PARAMS = {
    'Random Forest': {
        'n_estimators': 100,
        'max_depth': 20,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    },
    'Decision Tree': {
        'max_depth': 20,
        'random_state': RANDOM_STATE
    }
}
```

**Results:**
- Accuracy: 99.94% (Random Forest)
- Processing: 100,000 samples
- Dataset: CIRA-CIC-DoHBrw-2020

**Pipeline Stages:**
1. `data_loading_stage()`: Load DoH and non-DoH samples
2. `preprocessing_stage()`: Clean → Encode → Split
3. `training_stage()`: Train models with scaling
4. `evaluation_stage()`: Performance analysis

---

### 4️⃣ Reflector Analysis (`reflector_analysis/`)

Statistical analysis of DNS domains used in amplification attacks.

**Features:**
- TLD (Top-Level Domain) distribution analysis
- Domain length and uniqueness metrics
- Pattern identification in abused domains
- Report generation

**Configuration (main.py):**
```python
OUTPUT_REPORT = 'dns_reflector_analysis_report.txt'

DATASET_PATHS = [
    '../datasets/domains_FirstDay.txt',
    '../datasets/domains_SecondDay.txt',
    '../datasets/domains_ThirdDay.txt',
]
```

**Results:**
- Analyzed: 33,313 domains
- Top TLD: .com (52.93%)
- Average domain length: 12.49 characters
- Dataset: CIC-Bell-DNS 2021

**Pipeline Stages:**
1. `data_loading_stage()`: Load domain lists
2. `analysis_stage()`: Calculate statistics and TLD distribution
3. `reporting_stage()`: Generate and save report

---

## 📦 Dependencies

```txt
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
imbalanced-learn>=0.10.0
```

Install all dependencies:
```bash
pip install pandas numpy scikit-learn imbalanced-learn
```

---

## 🎯 Usage Examples

### Example 1: Run DDoS Detector

```bash
cd ddos_detector
python main.py
```

**Output:**
```
================================================================================
DrDoS DNS ATTACK DETECTION
================================================================================

[1/7] Loading dataset...
   Total samples: 1,264,154
   Total features: 87

[2/7] Cleaning data...
   Features: 85
   Samples: 1,264,154

[3/7] Applying SMOTE to BENIGN class...
   Original BENIGN: 120,794
   Original DDoS: 1,143,360
   After SMOTE BENIGN: 1,207,940 (multiplied by 10x)

[4/7] Splitting data...
   Train samples: 1,880,240
   Test samples: 470,060

[5/7] Scaling features...

[6/7] Training models...
   Training Random Forest...
      Accuracy: 0.9999
      Train time: 3.45s

[7/7] Model comparison...
BEST MODEL: Random Forest (0.9999)

Model saved: best_model_random_forest.pkl
```

### Example 2: Run All Modules

```bash
python run_all.py
```

**Features:**
- Sequential execution of all 4 modules
- Automatic error handling
- Summary report with timings
- Individual module results

---

## 📊 Performance Metrics

| Module | Accuracy | Precision | Recall | F1-Score | Dataset Size |
|--------|----------|-----------|--------|----------|--------------|
| DDoS Detector | 99.99% | 1.00 | 1.00 | 1.00 | 1.26M samples |
| Exfiltration Detection | 96.29% | 0.97 | 0.96 | 0.96 | 685K samples |
| DoH Detection | 99.94% | 1.00 | 1.00 | 1.00 | 100K samples |
| Reflector Analysis | N/A | N/A | N/A | N/A | 33K domains |

---

## 🔬 Technical Details

### Data Preprocessing Pipeline

All modules follow a consistent preprocessing approach:

1. **Loading**: Read CSV/TXT files with error handling
2. **Cleaning**: Remove NaN, handle infinite values, drop unnamed columns
3. **Encoding**: LabelEncoder for categorical features and labels
4. **Balancing**: SMOTE for class imbalance (DDoS module only)
5. **Splitting**: Stratified train/test split (80/20)
6. **Scaling**: StandardScaler for feature normalization

### Model Training Strategy

**Stage-based architecture:**
```python
# Stage 1: Preprocessing
X_train, X_test, y_train, y_test, scaler, le = preprocessing_stage(...)

# Stage 2: Training
results, timing = training_stage(X_train, y_train, params)

# Stage 3: Evaluation
best_model, metrics = evaluation_stage(results, X_test, y_test)

# Stage 4: Persistence (optional)
persistence_stage(best_model, scaler, le, filename)
```

### Code Organization

**utils.py structure:**
```python
# ============================================================================
# STAGE FUNCTIONS (Top-level pipeline stages)
# ============================================================================

def preprocessing_stage(...):
    """Complete preprocessing pipeline"""
    # Calls internal helpers
    
def training_stage(...):
    """Train all models"""
    # Calls internal helpers
    
def evaluation_stage(...):
    """Evaluate and compare models"""
    # Calls internal helpers

# ============================================================================
# HELPER FUNCTIONS (Internal implementation)
# ============================================================================

def _load_data(...):
    """Load dataset from file"""
    
def _clean_data(...):
    """Clean and preprocess data"""
    
def _train_model(...):
    """Train a single model"""
```

---

## 📈 Datasets

### Download Datasets

**⚠️ Important:** The datasets are hosted separately due to their large size.

**Download from:** [GitHub Releases - Datasets](https://github.com/kulisk/DrDoS-Detector/releases/tag/dataset)

After downloading, extract the `datasets/` folder to the project root:
```
DrDoS-Detector/
├── datasets/          # <- Extract here
│   ├── DrDoS_DNS.csv
│   ├── l1-doh.csv
│   └── ...
├── ddos_detector/
├── exfiltration_detection/
└── ...
```

### Dataset Details

Place datasets in the `datasets/` folder:

| File | Size | Source | Purpose |
|------|------|--------|---------|
| `DrDoS_DNS.csv` | ~500MB | CICDDoS2019 | DDoS detection training |
| `l1-doh.csv` | ~200MB | CIRA-CIC-DoHBrw-2020 | DoH traffic samples |
| `l1-nondoh.csv` | ~200MB | CIRA-CIC-DoHBrw-2020 | Non-DoH traffic samples |
| `stateless_features-benign_*.csv` | ~100MB | CIC-Bell-DNS-EXF-2021 | Benign DNS samples |
| `stateful_features-benign_*.csv` | ~50MB | CIC-Bell-DNS-EXF-2021 | Benign DNS samples |
| `domains_*.txt` | ~1MB | CIC-Bell-DNS 2021 | Domain lists |

### Additional Dataset Folders

Some modules also require full dataset folders:
- `CIC-Bell-DNS-EXF-2021 dataset/` (exfiltration attacks)
- `CIC-Bell-DNS 2021 dataset/` (domain analysis)
- `CIRA-CIC-DoHBrw-2020/` (DoH detection)

---

## 🛠️ Development

### Adding a New Module

1. Create module folder: `mkdir new_module`
2. Create `main.py` with configuration and pipeline
3. Create `utils.py` with stage functions and helpers
4. Add datasets to `datasets/` folder
5. Update `run_all.py` to include new module

### Code Style

- Follow PEP 8 guidelines
- Use docstrings for all functions
- Organize functions by pipeline stage
- Keep `main.py` clean (config + main only)
- Use descriptive variable names

---

## 🐛 Troubleshooting

### Common Issues

**Issue: "File not found" error**
```
Solution: Ensure datasets are in datasets/ folder with correct names
```

**Issue: "Memory error" during SMOTE**
```
Solution: Reduce SMOTE_TARGET_RATIO in ddos_detector/main.py
```

**Issue: "Module not found" error**
```
Solution: Install dependencies: pip install -r requirements.txt
```

**Issue: Slow training (SVM/KNN)**
```
Solution: Disable slow models in ENABLE_MODELS configuration
```

---

## 📝 License

This project is part of a thesis work on DNS security.

---

## 👥 Contributors

- **Kulis** - Initial work and development

---

## 🙏 Acknowledgments

- **CIC (Canadian Institute for Cybersecurity)** for providing comprehensive DNS security datasets
- **CIRA (Canadian Internet Registration Authority)** for DoH detection datasets
- **scikit-learn** and **imbalanced-learn** communities for ML tools

---

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Last Updated:** December 8, 2025
