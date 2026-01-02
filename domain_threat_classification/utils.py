# -*- coding: utf-8 -*-
"""DNS Domain Threat Classification (ML) - Utility Functions
========================================
Pipeline-style utilities for training a classifier that distinguishes domains
across 4 categories (Benign/Malware/Phishing/Spam).

Important: This module does NOT modify any datasets on disk.
"""

from __future__ import annotations

import ast
import csv
import math
import os
import pickle
import re
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


_DOMAIN_RE = re.compile(r"([A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,}")


# ============================================================================
# STAGE FUNCTIONS (Top-level pipeline stages)
# ============================================================================


def data_loading_stage(labeled_domain_csvs, max_samples_per_class=50_000, random_state=42):
    """Load labeled samples from CSVs only.

    Args:
        labeled_domain_csvs: dict[label -> Path/str]
        max_samples_per_class: cap per class after dedupe (keeps runtime reasonable)
        random_state: sampling seed

    Returns:
        DataFrame with columns: domain, Label
    """
    if not isinstance(labeled_domain_csvs, dict):
        raise ValueError(
            "data_loading_stage expects a dict[label -> CSV path]. "
            "TXT lists are no longer supported; use only the CIC-Bell-DNS 2021 CSV files."
        )

    return _load_labeled_domains_from_csvs(
        labeled_domain_csvs,
        max_samples_per_class=max_samples_per_class,
        random_state=random_state,
    )


def preprocessing_stage(df, test_size, random_state):
    """Extract lexical features -> encode labels -> split."""
    X, y, le_label = _build_feature_matrix(df)
    X_train, X_test, y_train, y_test = _split_data(X, y, test_size, random_state)
    return X_train, X_test, y_train, y_test, le_label


def training_stage(X_train, X_test, y_train, model_params, random_state):
    """Scale features and train models."""
    X_train_scaled, X_test_scaled, preprocessor = _scale_features(X_train, X_test)
    results = _train_models(X_train_scaled, X_test_scaled, y_train, model_params)
    return results, preprocessor, X_test_scaled


def evaluation_stage(results, y_test, label_encoder):
    """Evaluate all models and return best."""
    return _evaluate_models(results, y_test, label_encoder)


def save_report(results, y_test, le_label, output_path):
    """Save a text report with per-model metrics."""
    lines = []
    lines.append("=" * 80)
    lines.append("DNS DOMAIN THREAT CLASSIFICATION (ML) - REPORT")
    lines.append("=" * 80)
    lines.append("")

    for name, result in results.items():
        y_pred = result['predictions']
        acc = accuracy_score(y_test, y_pred)
        lines.append("=" * 80)
        lines.append(f"Model: {name}")
        lines.append("=" * 80)
        lines.append(f"Train Accuracy: {result['accuracy']:.4f}")
        lines.append(f"Test Accuracy: {acc:.4f}")
        lines.append("")
        lines.append("Confusion Matrix:")
        lines.append(str(confusion_matrix(y_test, y_pred)))
        lines.append("")
        lines.append("Classification Report:")
        lines.append(classification_report(y_test, y_pred, target_names=le_label.classes_))
        lines.append("")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


def save_model(
    model,
    preprocessor,
    label_encoder,
    feature_names,
    filepath: str,
    model_name: str | None = None,
) -> None:
    """Persist best model and preprocessing objects to a pickle file."""
    payload = {
        'model': model,
        'preprocessor': preprocessor,
        'label_encoder': label_encoder,
        'feature_names': feature_names,
        'model_name': model_name,
    }
    with open(filepath, 'wb') as f:
        pickle.dump(payload, f)


def load_model(filepath: str) -> dict:
    """Load a persisted model payload."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# ============================================================================
# DATA LOADING HELPERS
# ============================================================================


def _load_labeled_domains_from_csvs(labeled_domain_csvs, max_samples_per_class=50_000, random_state=42):
    print("\n[1/5] Loading labeled domains from CSVs...")

    if not labeled_domain_csvs:
        return None

    rng = np.random.default_rng(random_state)

    frames: list[pd.DataFrame] = []

    for label, path in labeled_domain_csvs.items():
        p = str(path)
        if not os.path.exists(p):
            print(f"  [SKIP] Missing CSV for {label}: {p}")
            continue

        df_label = _load_samples_from_cic_bell_dns_2021_csv(
            p,
            max_rows=max_samples_per_class,
            rng=rng,
        )

        if df_label is None or df_label.empty:
            print(f"  [SKIP] No usable rows extracted for {label}: {os.path.basename(p)}")
            continue

        df_label = df_label.copy()
        df_label['Label'] = label
        frames.append(df_label)
        print(f"  [OK] {label}: {len(df_label):,} samples ({os.path.basename(p)})")

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    df['domain'] = df['domain'].astype(str)
    df = df.dropna(subset=['domain'])
    df = df.drop_duplicates(subset=['domain', 'Label'])

    print(f"\n  Total samples: {len(df):,}")
    print("  Class distribution:")
    for label, count in df['Label'].value_counts().items():
        pct = 100 * count / len(df)
        print(f"    - {label}: {count:,} ({pct:.2f}%)")

    return df


def _extract_domains_from_text(path: str) -> list[str]:
    """Extract domains from an arbitrary text/CSV file using regex (streaming)."""
    results: list[str] = []
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                for match in _DOMAIN_RE.findall(line):
                    d = match.strip().rstrip('.').lower()
                    if d:
                        results.append(d)
    except Exception:
        return []
    return results


def _extract_unique_domains_from_text(path: str, soft_cap: int | None) -> list[str]:
    """Stream file and extract unique normalized domains up to a soft cap.

    soft_cap:
      - If provided, stops once we have collected ~soft_cap*5 unique domains.
        (We may still sample down later for randomness.)
      - If None/0, reads the whole file.
    """
    max_unique = None
    if soft_cap and soft_cap > 0:
        max_unique = int(soft_cap) * 5

    seen: set[str] = set()
    out: list[str] = []
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                for match in _DOMAIN_RE.findall(line):
                    d = _normalize_domain(match)
                    if not d:
                        continue
                    if d in seen:
                        continue
                    seen.add(d)
                    out.append(d)
                    if max_unique and len(out) >= max_unique:
                        return out
    except Exception:
        return []

    return out


def _get_csv_header_columns(path: str) -> list[str]:
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore', newline='') as f:
            header = f.readline()
        if not header:
            return []
        cols = next(csv.reader([header]))
        return _dedupe_columns(cols)
    except Exception:
        return []


def _dedupe_columns(cols: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    out: list[str] = []
    for c in cols:
        c = c.strip()
        n = counts.get(c, 0)
        if n == 0:
            out.append(c)
        else:
            out.append(f"{c}_{n+1}")
        counts[c] = n + 1
    return out


def _split_nested_csv_line(line: str) -> list[str]:
    """Split a CSV line on commas, but ignore commas inside (), [], {}.

    The CIC-Bell-DNS 2021 CSVs contain unquoted list/dict-like fields that include
    commas (e.g., defaultdict(..., {...}), ['a','b']). Standard CSV parsers will
    mis-split these rows; this function keeps them intact.
    """
    parts: list[str] = []
    buf: list[str] = []
    depth_round = 0
    depth_square = 0
    depth_curly = 0

    for ch in line:
        if ch == '(':
            depth_round += 1
        elif ch == ')':
            depth_round = max(0, depth_round - 1)
        elif ch == '[':
            depth_square += 1
        elif ch == ']':
            depth_square = max(0, depth_square - 1)
        elif ch == '{':
            depth_curly += 1
        elif ch == '}':
            depth_curly = max(0, depth_curly - 1)

        if ch == ',' and depth_round == 0 and depth_square == 0 and depth_curly == 0:
            parts.append(''.join(buf).strip())
            buf = []
            continue

        buf.append(ch)

    parts.append(''.join(buf).strip())
    return parts


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {'nan', 'none'}:
        return None
    # Some fields look like "4277 days, 21:07:56.450015"; those should stay None.
    try:
        return float(s)
    except Exception:
        return None


def _load_samples_from_cic_bell_dns_2021_csv(path: str, max_rows: int | None, rng: np.random.Generator):
    """Load samples from a CIC-Bell-DNS 2021 CSV, using existing numeric features.

    These CSVs may contain malformed rows (unquoted commas inside list/dict fields).
    We use pandas with `engine='python'` and `on_bad_lines='skip'` to salvage rows.
    If parsing yields too few rows, we fall back to regex-extracted domains (lexical-only).
    """

    header_cols = _get_csv_header_columns(path)
    if not header_cols:
        return None

    domain_candidates = ['Domain', 'Domain_Name']
    domain_col = next((c for c in domain_candidates if c in header_cols), None)
    if not domain_col:
        return None

    # Prefer a small set of numeric columns that should exist across files.
    preferred_cols = [
        'ASN', 'TTL',
        'entropy', 'Name_Server_Count',
        'Page_Rank', 'Alexa_Rank',
        'len', 'numeric_percentage',
        'oc_8', 'oc_32',
        'hex_8', 'hex_32',
        'dec_8', 'dec_32',
        'subdomain', 'puny_coded',
        'shortened', 'obfuscate_at_sign',
    ]
    usecols = [c for c in preferred_cols if c in header_cols]

    max_unique = max_rows if (max_rows and max_rows > 0) else None
    rows: list[dict] = []
    seen: set[str] = set()

    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            _ = f.readline()  # header
            for line in f:
                line = line.rstrip('\n')
                if not line:
                    continue

                fields = _split_nested_csv_line(line)
                if len(fields) != len(header_cols):
                    continue

                record = dict(zip(header_cols, fields))
                d = _normalize_domain(record.get(domain_col))
                if not d or d in seen:
                    continue
                seen.add(d)

                row: dict = {'domain': d}
                for col in usecols:
                    row[col] = _to_float(record.get(col))
                rows.append(row)

                if max_unique and len(rows) >= max_unique:
                    break
    except Exception:
        rows = []

    if rows:
        return pd.DataFrame(rows)

    # Fallback: regex domains only (no numeric features).
    domains = _extract_unique_domains_from_text(path, soft_cap=max_rows)
    if not domains:
        return None
    if max_rows and max_rows > 0 and len(domains) > max_rows:
        idx = rng.choice(len(domains), size=max_rows, replace=False)
        domains = [domains[i] for i in idx]
    return pd.DataFrame({'domain': domains})


# ============================================================================
# FEATURE ENGINEERING / PREPROCESSING
# ============================================================================


def _build_feature_matrix(df: pd.DataFrame):
    print("\n[2/5] Building feature matrix (lexical + CSV numeric)...")

    domains = df['domain'].astype(str).tolist()
    labels = df['Label'].astype(str)

    lexical = pd.DataFrame([_domain_features(d) for d in domains]).add_prefix('lex_')

    extra_cols = [c for c in df.columns if c not in {'domain', 'Label'}]
    if extra_cols:
        numeric = df[extra_cols].apply(pd.to_numeric, errors='coerce').add_prefix('csv_')
        X = pd.concat([lexical, numeric], axis=1)
    else:
        X = lexical

    # Encode categorical columns (tld)
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    le_label = LabelEncoder()
    y = le_label.fit_transform(labels)

    print(f"  Samples: {len(X):,}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes: {list(le_label.classes_)}")

    return X, y, le_label


def _split_data(X, y, test_size, random_state):
    print("\n[3/5] Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


def _scale_features(X_train, X_test):
    # Handle NaNs introduced by numeric coercion / missing columns
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    preprocessor = {
        'imputer': imputer,
        'scaler': scaler,
    }
    return X_train_scaled, X_test_scaled, preprocessor


# ============================================================================
# TRAINING / EVALUATION
# ============================================================================


def _train_models(X_train_scaled, X_test_scaled, y_train, model_params):
    print("\n[4/5] Training models...")

    models = {
        'Random Forest': RandomForestClassifier(**model_params['Random Forest']),
        'Decision Tree': DecisionTreeClassifier(**model_params['Decision Tree']),
        'Logistic Regression': LogisticRegression(**model_params['Logistic Regression']),
    }

    results = {}
    for name, clf in models.items():
        print(f"\n  Training {name}...")
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        train_acc = accuracy_score(y_train, clf.predict(X_train_scaled))
        results[name] = {
            'model': clf,
            'accuracy': train_acc,
            'predictions': y_pred,
        }
        print(f"    Train Accuracy: {train_acc:.4f}")

    return results


def _evaluate_models(results, y_test, le_label):
    print("\n[5/5] Evaluating...")
    best_name = None
    best_acc = -1.0

    for name, result in results.items():
        y_pred = result['predictions']
        acc = accuracy_score(y_test, y_pred)

        print("\n" + "=" * 80)
        print(name)
        print("=" * 80)
        print(f"Test Accuracy: {acc:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=le_label.classes_))

        if acc > best_acc:
            best_acc = acc
            best_name = name

    print("\n" + "=" * 80)
    print(f"Best Model: {best_name} ({best_acc:.4f})")
    print("=" * 80)
    return best_name, best_acc


# ============================================================================
# DOMAIN NORMALIZATION / FEATURES
# ============================================================================


def _normalize_domain(value) -> str | None:
    """Normalize various domain string representations into a plain domain."""
    if value is None:
        return None

    s = str(value).strip()
    if not s or s.lower() == 'nan':
        return None

    # Handle Python-bytes literal style: b'example.com.'
    if s.startswith("b'") or s.startswith('b"'):
        s = s[2:]
        if s.endswith("'") or s.endswith('"'):
            s = s[:-1]

    # Handle list-like strings, e.g. "['GOOGLE.COM', 'google.com']"
    if s.startswith('[') and s.endswith(']'):
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple)) and obj:
                s = str(obj[-1] if obj[-1] else obj[0]).strip()
        except Exception:
            pass

    s = s.strip().strip("'\"")
    s = s.rstrip('.')

    m = _DOMAIN_RE.search(s)
    if m:
        s = m.group(0)

    s = s.strip().rstrip('.')
    if not s:
        return None
    return s.lower()


def _tld(domain: str) -> str:
    d = _normalize_domain(domain)
    if not d or '.' not in d:
        return 'unknown'
    return d.split('.')[-1]


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    total = len(s)
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)
    return float(ent)


def _domain_features(domain: str) -> dict:
    d = _normalize_domain(domain) or ''
    parts = d.split('.') if d else []
    tld = _tld(d)
    sld = parts[-2] if len(parts) >= 2 else ''

    letters = sum(ch.isalpha() for ch in d)
    digits = sum(ch.isdigit() for ch in d)
    hyphens = d.count('-')
    dots = d.count('.')
    length = len(d)

    return {
        'len': length,
        'dots': dots,
        'subdomain_count': max(0, dots - 1),
        'hyphens': hyphens,
        'digits': digits,
        'letters': letters,
        'digit_ratio': (digits / length) if length else 0.0,
        'entropy': _shannon_entropy(d),
        'tld': tld,
        'sld_len': len(sld),
        'starts_with_xn': 1 if d.startswith('xn--') else 0,
    }
