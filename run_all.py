# -*- coding: utf-8 -*-
"""
DrDoS-Detector - Master Execution Script
========================================
Runs all DNS security modules sequentially.

Modules:
1. DrDoS Amplification Detection
2. DNS Exfiltration Detection
3. Malicious DoH Detection
4. DNS Reflector Analysis

Author: DrDoS-Detector Team
"""

import subprocess
import sys
import time
from datetime import datetime

# Get Python executable path - force Python 3
PYTHON_EXE = 'py' if sys.platform == 'win32' else 'python3'
PYTHON_ARGS = ['-3'] if sys.platform == 'win32' else []

# Module paths
MODULES = [
    {
        'name': 'DrDoS Amplification Detection',
        'path': 'ddos_detector/main.py',
        'description': 'Detecting DNS amplification DDoS attacks'
    },
    {
        'name': 'DNS Exfiltration Detection',
        'path': 'exfiltration_detection/main.py',
        'description': 'Detecting data exfiltration via DNS tunneling'
    },
    {
        'name': 'Malicious DoH Detection',
        'path': 'doh_detection/main.py',
        'description': 'Detecting malicious DNS-over-HTTPS traffic'
    },
    {
        'name': 'DNS Reflector Analysis',
        'path': 'reflector_analysis/main.py',
        'description': 'Statistical analysis of attacked domains'
    }
]


def print_header():
    """Print welcome banner"""
    print("\n" + "="*80)
    print(" " * 20 + "DrDoS-DETECTOR - MASTER SCRIPT")
    print("="*80)
    print("\nThis script will run all DNS security modules sequentially.")
    print("Start time: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print("\nModules to run:")
    for i, module in enumerate(MODULES, 1):
        print("  {}. {}".format(i, module['name']))
        print("     {}".format(module['description']))
    print("\n" + "="*80 + "\n")


def run_module(module_info, module_num, total_modules):
    """Run a single module and track execution"""
    print("\n" + "="*80)
    print("MODULE {}/{}: {}".format(module_num, total_modules, module_info['name']))
    print("="*80)
    print("Running: {}".format(module_info['path']))
    print("Description: {}".format(module_info['description']))
    print("-"*80 + "\n")
    
    start_time = time.time()
    
    try:
        # Run the module (extract directory from path)
        module_dir = '/'.join(module_info['path'].split('/')[:-1])  # e.g., 'ddos_detector' from 'ddos_detector/main.py'
        module_file = module_info['path'].split('/')[-1]  # e.g., 'main.py'
        
        cmd = [PYTHON_EXE] + PYTHON_ARGS + [module_file]
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            check=True,
            cwd=module_dir if module_dir else None
        )
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "-"*80)
        print("[SUCCESS] {} completed in {:.2f} seconds".format(module_info['name'], elapsed_time))
        print("-"*80)
        
        return True, elapsed_time
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        
        print("\n" + "-"*80)
        print("[ERROR] {} failed after {:.2f} seconds".format(module_info['name'], elapsed_time))
        print("Error code: {}".format(e.returncode))
        print("-"*80)
        
        return False, elapsed_time
    
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] User cancelled execution")
        return False, time.time() - start_time


def print_summary(results):
    """Print execution summary"""
    print("\n\n" + "="*80)
    print(" " * 25 + "EXECUTION SUMMARY")
    print("="*80)
    
    total_time = sum(r['time'] for r in results)
    successful = sum(1 for r in results if r['status'])
    failed = len(results) - successful
    
    print("\nTotal modules: {}".format(len(results)))
    print("Successful: {}".format(successful))
    print("Failed: {}".format(failed))
    print("Total time: {:.2f} seconds ({:.2f} minutes)".format(total_time, total_time/60))
    
    print("\nModule Results:")
    print("-"*80)
    
    for result in results:
        status = "[OK]" if result['status'] else "[FAIL]"
        print("{} {:<40} {:>8.2f}s".format(status, result['name'], result['time']))
    
    print("="*80)
    print("Completed: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print("="*80 + "\n")


def main():
    """Main execution function"""
    print_header()
    
    results = []
    
    # Run each module
    for i, module in enumerate(MODULES, 1):
        success, elapsed = run_module(module, i, len(MODULES))
        
        results.append({
            'name': module['name'],
            'status': success,
            'time': elapsed
        })
        
        # Small pause between modules
        if i < len(MODULES):
            print("\nPausing for 2 seconds before next module...")
            time.sleep(2)
    
    # Print summary
    print_summary(results)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Execution cancelled by user")
        sys.exit(1)
