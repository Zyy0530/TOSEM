#!/usr/bin/env python3
"""
Quick setup verification for RQ2 experiment
"""

import os
import sys

def check_file_exists(filepath, description):
    if os.path.exists(filepath):
        print(f"✅ {description}")
        return True
    else:
        print(f"❌ {description} - MISSING: {filepath}")
        return False

def main():
    print("🔍 RQ2 Experiment Setup Verification")
    print("=" * 40)
    
    base_dir = "/Users/mac/ResearchSpace/TOSEM/experiments/RQ2"
    parent_dir = "/Users/mac/ResearchSpace/TOSEM"
    
    all_good = True
    
    # Check essential files
    files_to_check = [
        (f"{base_dir}/rq2_factory_detection.py", "Main experiment script"),
        (f"{base_dir}/test_setup.py", "Setup validation script"),
        (f"{base_dir}/config.json", "Configuration file"),
        (f"{base_dir}/README.md", "Documentation"),
        (f"{parent_dir}/factory_detector.py", "Factory detector module"),
    ]
    
    for filepath, description in files_to_check:
        if not check_file_exists(filepath, description):
            all_good = False
    
    # Check logs directory
    logs_dir = f"{base_dir}/logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
        print(f"✅ Created logs directory: {logs_dir}")
    else:
        print(f"✅ Logs directory exists")
    
    print("\n" + "=" * 40)
    
    if all_good:
        print("🎉 Setup verification complete!")
        print("\n📋 Next steps:")
        print("1. Run tests: python test_setup.py")
        print("2. Start experiment: python rq2_factory_detection.py")
        print("3. Monitor progress in logs/ directory")
        print("\n💡 See README.md for detailed instructions")
    else:
        print("⚠️  Setup incomplete - missing files detected")
        print("Please ensure all required files are present")
    
    return all_good

if __name__ == "__main__":
    main()