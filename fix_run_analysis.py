#!/usr/bin/env python3
"""
Script to fix the return value handling in app_v5_automated.py
This script updates all calls to run_analysis to handle the new 3-value return format
"""

import re
import os

def fix_run_analysis_calls(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern to match run_analysis calls that expect 2 return values
    pattern = r'(\s*)([a-zA-Z_][a-zA-Z0-9_]*),\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*run_analysis\('
    
    # Replace with pattern that handles 3 return values
    replacement = r'\1\2, \3, _ = run_analysis('
    
    # Apply the replacement
    modified_content = re.sub(pattern, replacement, content)
    
    # Write the modified content back to the file
    with open(file_path, 'w') as f:
        f.write(modified_content)
    
    print(f"Updated run_analysis calls in {file_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_file = os.path.join(script_dir, 'app_v5_automated.py')
    fix_run_analysis_calls(app_file)
