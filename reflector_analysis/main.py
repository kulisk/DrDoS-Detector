# -*- coding: utf-8 -*-
"""
DNS Reflector Analysis - Main Script
====================================
Analyzes DNS amplification reflector characteristics.

Author: DrDoS-Detector Team
"""

import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

from utils import data_loading_stage, analysis_stage, reporting_stage

print("="*80)
print("DNS REFLECTOR ANALYSIS")
print("="*80)
print("\nAnalyzing domain characteristics from DNS amplification attacks\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / 'reports'
OUTPUT_REPORT = REPORTS_DIR / 'dns_reflector_analysis_report.txt'

# ============================================================================
# DATASET PATHS
# ============================================================================

DATASET_PATHS = [
    BASE_DIR / 'datasets' / 'domains_FirstDay.txt',
    BASE_DIR / 'datasets' / 'domains_SecondDay.txt',
    BASE_DIR / 'datasets' / 'domains_ThirdDay.txt',
]


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline with stage calls"""
    
    # Stage 1: Data Loading
    domains = data_loading_stage(DATASET_PATHS)
    
    if not domains:
        print("\n[ERROR] No domains found!")
        return
    
    # Stage 2: Analysis
    results = analysis_stage(domains)
    
    # Stage 3: Reporting
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    reporting_stage(results, OUTPUT_REPORT)
    
    print("\n" + "="*80)
    print("DNS REFLECTOR ANALYSIS - COMPLETED")
    print("="*80)
    print("\nReport saved: {}".format(OUTPUT_REPORT))
    print("\nKey Findings:")
    print("  - Analyzed {:,} total domains".format(results['total_domains']))
    print("  - Found {:,} unique domains".format(results['unique_domains']))
    print("  - Average domain length: {:.1f} characters".format(results['avg_length']))
    if results['top_tlds']:
        top_tld, top_count = results['top_tlds'][0]
        pct = 100 * top_count / results['total_domains']
        print("  - Most common TLD: .{} ({:.1f}%)".format(top_tld, pct))
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
