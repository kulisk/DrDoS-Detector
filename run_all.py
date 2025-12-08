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

# Get Python executable path
PYTHON_EXE = sys.executable

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
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nModules to run:")
    for i, module in enumerate(MODULES, 1):
        print(f"  {i}. {module['name']}")
        print(f"     {module['description']}")
    print("\n" + "="*80 + "\n")


def run_module(module_info, module_num, total_modules):
    """Run a single module and track execution"""
    print("\n" + "="*80)
    print(f"MODULE {module_num}/{total_modules}: {module_info['name']}")
    print("="*80)
    print(f"Running: {module_info['path']}")
    print(f"Description: {module_info['description']}")
    print("-"*80 + "\n")
    
    start_time = time.time()
    
    try:
        # Run the module
        result = subprocess.run(
            [PYTHON_EXE, module_info['path']],
            capture_output=False,
            text=True,
            check=True
        )
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "-"*80)
        print(f"[SUCCESS] {module_info['name']} completed in {elapsed_time:.2f} seconds")
        print("-"*80)
        
        return True, elapsed_time
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        
        print("\n" + "-"*80)
        print(f"[ERROR] {module_info['name']} failed after {elapsed_time:.2f} seconds")
        print(f"Error code: {e.returncode}")
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
    
    print(f"\nTotal modules: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    print("\nModule Results:")
    print("-"*80)
    
    for result in results:
        status = "[OK]" if result['status'] else "[FAIL]"
        print(f"{status} {result['name']:<40} {result['time']:>8.2f}s")
    
    print("="*80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")


def main():
    """Main execution function"""
    print_header()
    
    # Ask for confirmation
    response = input("Do you want to continue? (y/n): ").strip().lower()
    if response != 'y':
        print("\nExecution cancelled.")
        return
    
    results = []
    
    # Run each module
    for i, module in enumerate(MODULES, 1):
        success, elapsed = run_module(module, i, len(MODULES))
        
        results.append({
            'name': module['name'],
            'status': success,
            'time': elapsed
        })
        
        # If a module fails, ask if we should continue
        if not success:
            response = input("\nModule failed. Continue with next module? (y/n): ").strip().lower()
            if response != 'y':
                print("\nExecution stopped by user.")
                break
        
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
