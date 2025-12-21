# -*- coding: utf-8 -*-
"""
DNS Reflector Analysis - Utility Functions
==========================================
Helper functions organized by analysis stages for DNS amplification reflector analysis.

Author: DrDoS-Detector Team
"""

import pandas as pd
import os
from collections import Counter
import ast
import re


# ============================================================================
# STAGE FUNCTIONS (Top-level pipeline stages)
# ============================================================================

def data_loading_stage(dataset_paths):
    """
    Load all domain data from multiple CSV files
    
    Returns:
        list: All domains from datasets
    """
    return _load_domains(dataset_paths)


def analysis_stage(domains):
    """
    Analyze domain characteristics (TLD distribution, length, uniqueness)
    
    Returns:
        dict: Analysis results
    """
    return _analyze_characteristics(domains)


def reporting_stage(analysis_results, output_file):
    """
    Generate and save analysis report
    """
    _save_report(analysis_results, output_file)


# ============================================================================
# DATA LOADING HELPERS
# ============================================================================

def _load_domains(dataset_paths):
    """Load all domains from the specified CSV files"""
    print("\n[1/3] Loading domains from datasets...")
    
    all_domains = []
    
    for path in dataset_paths:
        if not os.path.exists(path):
            print(f"  [SKIP] File not found: {path}")
            continue
        
        try:
            # Determine candidate domain columns without loading the full file
            df_head = pd.read_csv(path, nrows=0)
            domain_cols = []
            for col in df_head.columns:
                col_l = col.lower()
                if not ('domain' in col_l or 'query' in col_l):
                    continue
                # Exclude non-domain fields that happen to contain the substring
                if 'age' in col_l:
                    continue
                if col_l == 'subdomain':
                    continue
                domain_cols.append(col)

            # Load only the relevant columns (important for large CSVs)
            if domain_cols:
                df = pd.read_csv(path, usecols=domain_cols, low_memory=False)
            else:
                df = pd.read_csv(path, low_memory=False)
            
            domains_added_for_file = 0

            if domain_cols:
                for col in domain_cols:
                    # Heuristic: only keep columns that actually look like domain values
                    sample_raw = df[col].dropna().astype(str).head(500).tolist()
                    sample_norm = [d for d in (_normalize_domain(v) for v in sample_raw) if d]
                    if sample_norm:
                        domain_like = sum(1 for d in sample_norm if _DOMAIN_RE.fullmatch(d))
                        ratio = domain_like / len(sample_norm)
                        if ratio < 0.30:
                            print(f"  [SKIP] {os.path.basename(path)}[{col}] does not look like domains (sample ratio={ratio:.2f})")
                            continue

                    raw_values = df[col].dropna().astype(str).tolist()
                    domains = [d for d in (_normalize_domain(v) for v in raw_values) if d]
                    all_domains.extend(domains)
                        domains_added_for_file += len(domains)
                        print(f"  [OK] {os.path.basename(path)}[{col}]: {len(domains):,} domains")
            else:
                # Try first text column
                text_cols = df.select_dtypes(include=['object']).columns
                if len(text_cols) > 0:
                    raw_values = df[text_cols[0]].dropna().astype(str).tolist()
                    domains = [d for d in (_normalize_domain(v) for v in raw_values) if d]
                    all_domains.extend(domains)
                    domains_added_for_file += len(domains)
                    print(f"  [OK] {os.path.basename(path)}[{text_cols[0]}]: {len(domains):,} domains")

            # Fallback for malformed CSVs: extract domains directly from raw text
            if domains_added_for_file == 0 and str(path).lower().endswith('.csv'):
                extracted = _extract_domains_from_text(path)
                if extracted:
                    all_domains.extend(extracted)
                    print(f"  [OK] {os.path.basename(path)}[regex]: {len(extracted):,} domains")
        
        except Exception as e:
            print(f"  [ERROR] {os.path.basename(path)}: {str(e)}")
    
    print(f"\n  Total domains loaded: {len(all_domains):,}")
    
    return all_domains


_DOMAIN_RE = re.compile(r"([A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,}")


def _extract_domains_from_text(path):
    """Extract domains from an arbitrary text/CSV file using regex (streaming)."""
    results = []
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


def _normalize_domain(value):
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
                # Prefer the last element (often lowercase), fallback to first
                s = str(obj[-1] if obj[-1] else obj[0]).strip()
        except Exception:
            pass

    s = s.strip().strip("'\"")
    s = s.rstrip('.')

    # Extract a domain substring if the value contains extra text
    m = _DOMAIN_RE.search(s)
    if m:
        s = m.group(0)

    return s.lower()


# ============================================================================
# ANALYSIS HELPERS
# ============================================================================

def _extract_tld(domain):
    """Extract top-level domain (TLD) from domain name"""
    try:
        normalized = _normalize_domain(domain)
        if not normalized:
            return 'unknown'
        parts = normalized.split('.')
        if len(parts) >= 2:
            return parts[-1].lower()
        return 'unknown'
    except:
        return 'unknown'


def _analyze_characteristics(domains):
    """Analyze domain characteristics"""
    print("\n[2/3] Analyzing domain characteristics...")
    
    # Extract TLDs
    tlds = [_extract_tld(d) for d in domains]
    tld_counts = Counter(tlds)
    
    # Calculate statistics
    domain_lengths = [len(d) for d in domains]
    unique_domains = len(set(domains))
    
    avg_length = sum(domain_lengths) / len(domain_lengths) if domain_lengths else 0
    
    results = {
        'total_domains': len(domains),
        'unique_domains': unique_domains,
        'avg_length': avg_length,
        'tld_distribution': tld_counts.most_common(20),
        'top_tlds': tld_counts.most_common(10)
    }
    
    # Display summary
    print("\n  Statistics:")
    print(f"    Total domains: {results['total_domains']:,}")
    print(f"    Unique domains: {results['unique_domains']:,}")
    print(f"    Average length: {results['avg_length']:.2f} characters")
    
    print("\n  Top 10 TLDs:")
    total = results['total_domains']
    for tld, count in results['top_tlds']:
        pct = 100 * count / total
        print(f"    .{tld}: {count:,} ({pct:.2f}%)")
    
    return results


# ============================================================================
# REPORTING HELPERS
# ============================================================================

def _save_report(results, output_file):
    """Save analysis report to file"""
    print(f"\n[3/3] Saving report to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("DNS REFLECTOR ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("DOMAIN STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total domains analyzed: {results['total_domains']:,}\n")
        f.write(f"Unique domains: {results['unique_domains']:,}\n")
        f.write(f"Average domain length: {results['avg_length']:.2f} characters\n\n")
        
        f.write("TOP-LEVEL DOMAIN (TLD) DISTRIBUTION\n")
        f.write("-"*80 + "\n")
        f.write("{'TLD':<20} {'Count':<15} {'Percentage'}\n")
        f.write("-"*80 + "\n")
        
        total = results['total_domains']
        for tld, count in results['tld_distribution']:
            pct = 100 * count / total
            f.write(f".{tld:<19} {count:<15,} {pct:>6.2f}%\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print("  Report saved successfully!")
