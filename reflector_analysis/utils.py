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
            print("  [SKIP] File not found: {path}")
            continue
        
        try:
            # Try different CSV structures
            df = pd.read_csv(path)
            
            # Check for domain columns (various naming conventions)
            domain_cols = [col for col in df.columns if 'domain' in col.lower() or 'query' in col.lower()]
            
            if domain_cols:
                for col in domain_cols:
                    domains = df[col].dropna().astype(str).tolist()
                    all_domains.extend(domains)
                    print("  [OK] {os.path.basename(path)}: {len(domains):,} domains")
            else:
                # Try first text column
                text_cols = df.select_dtypes(include=['object']).columns
                if len(text_cols) > 0:
                    domains = df[text_cols[0]].dropna().astype(str).tolist()
                    all_domains.extend(domains)
                    print("  [OK] {os.path.basename(path)}: {len(domains):,} domains")
        
        except Exception as e:
            print("  [ERROR] {os.path.basename(path)}: {str(e)}")
    
    print("\n  Total domains loaded: {len(all_domains):,}")
    
    return all_domains


# ============================================================================
# ANALYSIS HELPERS
# ============================================================================

def _extract_tld(domain):
    """Extract top-level domain (TLD) from domain name"""
    try:
        parts = domain.strip().split('.')
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
    print("    Total domains: {results['total_domains']:,}")
    print("    Unique domains: {results['unique_domains']:,}")
    print("    Average length: {results['avg_length']:.2f} characters")
    
    print("\n  Top 10 TLDs:")
    total = results['total_domains']
    for tld, count in results['top_tlds']:
        pct = 100 * count / total
        print("    .{tld}: {count:,} ({pct:.2f}%)")
    
    return results


# ============================================================================
# REPORTING HELPERS
# ============================================================================

def _save_report(results, output_file):
    """Save analysis report to file"""
    print("\n[3/3] Saving report to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("DNS REFLECTOR ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("DOMAIN STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write("Total domains analyzed: {results['total_domains']:,}\n")
        f.write("Unique domains: {results['unique_domains']:,}\n")
        f.write("Average domain length: {results['avg_length']:.2f} characters\n\n")
        
        f.write("TOP-LEVEL DOMAIN (TLD) DISTRIBUTION\n")
        f.write("-"*80 + "\n")
        f.write("{'TLD':<20} {'Count':<15} {'Percentage'}\n")
        f.write("-"*80 + "\n")
        
        total = results['total_domains']
        for tld, count in results['tld_distribution']:
            pct = 100 * count / total
            f.write(".{tld:<19} {count:<15,} {pct:>6.2f}%\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print("  Report saved successfully!")
