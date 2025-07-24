#!/usr/bin/env python3
"""
Helper script to run GPU Worker Pool examples with proper setup.

This script ensures the examples run with the latest local code.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_with_pythonpath(script_path):
    """Run a script with PYTHONPATH set to current directory."""
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              env=env, 
                              cwd=Path(__file__).parent,
                              check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")
        return False

def main():
    """Run all available examples."""
    examples_dir = Path(__file__).parent / "examples"
    
    # List of examples to run
    examples = [
        "multi_endpoint_demo.py",
        "basic_client_usage.py",
        "real_endpoint_usage.py",  # Real endpoint example (requires GPU server running)
        # Note: multi_endpoint_usage.py might have issues with mocking
    ]
    
    print("GPU Worker Pool - Running Examples")
    print("=" * 50)
    print("Using local development code (PYTHONPATH=.)")
    print()
    
    for example in examples:
        example_path = examples_dir / example
        if example_path.exists():
            print(f"Running {example}...")
            print("-" * 30)
            
            success = run_with_pythonpath(str(example_path))
            
            if success:
                print(f"✓ {example} completed successfully")
            else:
                print(f"✗ {example} failed")
            print()
        else:
            print(f"Example {example} not found, skipping...")
    
    print("=" * 50)
    print("Examples completed!")
    print()
    print("For more information:")
    print("- See docs/API.md for detailed API documentation")
    print("- See docs/MIGRATION_GUIDE.md for migration instructions")
    print("- See docs/TROUBLESHOOTING.md for common issues")

if __name__ == "__main__":
    main()