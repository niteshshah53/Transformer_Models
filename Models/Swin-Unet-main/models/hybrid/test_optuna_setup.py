#!/usr/bin/env python3
"""
Quick test script to verify Optuna setup is working correctly
Run this before submitting the full job
"""

import sys
import os

print("="*80)
print("OPTUNA SETUP VERIFICATION TEST")
print("="*80)
print()

# Test 1: Python version
print("1. Checking Python version...")
print(f"   Python {sys.version}")
if sys.version_info >= (3, 8):
    print("   ✓ Python version OK (>=3.8)")
else:
    print("   ✗ Python version too old. Need >=3.8")
    sys.exit(1)

# Test 2: PyTorch
print("\n2. Checking PyTorch...")
try:
    import torch
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
        print(f"   GPU name: {torch.cuda.get_device_name(0)}")
    print("   ✓ PyTorch OK")
except ImportError as e:
    print(f"   ✗ PyTorch not found: {e}")
    sys.exit(1)

# Test 3: Optuna
print("\n3. Checking Optuna...")
try:
    import optuna
    print(f"   Optuna version: {optuna.__version__}")
    print("   ✓ Optuna OK")
except ImportError:
    print("   ✗ Optuna not found!")
    print("   Install with: pip install optuna")
    sys.exit(1)

# Test 4: Plotly (for visualizations)
print("\n4. Checking Plotly (optional)...")
try:
    import plotly
    print(f"   Plotly version: {plotly.__version__}")
    print("   ✓ Plotly OK")
except ImportError:
    print("   ⚠ Plotly not found (visualizations won't work)")
    print("   Install with: pip install plotly kaleido")

# Test 5: Dataset paths
print("\n5. Checking dataset paths...")
udiadsbib_path = "../../U-DIADS-Bib-MS_patched"
divahisdb_path = "../../DivaHisDB_patched"

if os.path.exists(udiadsbib_path):
    print(f"   ✓ UDIADS_BIB path exists: {os.path.abspath(udiadsbib_path)}")
else:
    print(f"   ⚠ UDIADS_BIB path not found: {os.path.abspath(udiadsbib_path)}")

if os.path.exists(divahisdb_path):
    print(f"   ✓ DIVAHISDB path exists: {os.path.abspath(divahisdb_path)}")
else:
    print(f"   ⚠ DIVAHISDB path not found: {os.path.abspath(divahisdb_path)}")

# Test 6: Required modules
print("\n6. Checking required modules...")
required_modules = [
    'numpy',
    'torch.nn',
    'torch.optim',
    'tensorboard',
]

all_ok = True
for module_name in required_modules:
    try:
        if '.' in module_name:
            # Handle submodules
            parts = module_name.split('.')
            mod = __import__(parts[0])
            for part in parts[1:]:
                mod = getattr(mod, part)
        else:
            __import__(module_name)
        print(f"   ✓ {module_name}")
    except ImportError:
        print(f"   ✗ {module_name} not found")
        all_ok = False

# Test 7: Check if model files exist
print("\n7. Checking model files...")
model_files = [
    'hybrid1/hybrid_model.py',
    'hybrid2/hybrid_model.py',
    'trainer.py',
    'trainer_optuna.py',
    'optuna_tune.py'
]

for file_path in model_files:
    if os.path.exists(file_path):
        print(f"   ✓ {file_path}")
    else:
        print(f"   ✗ {file_path} not found")
        all_ok = False

# Test 8: Simple Optuna test
print("\n8. Running simple Optuna test...")
try:
    def objective(trial):
        x = trial.suggest_float('x', -10, 10)
        return (x - 2) ** 2
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5, show_progress_bar=False)
    
    print(f"   Test optimization completed")
    print(f"   Best value: {study.best_value:.4f}")
    print(f"   Best params: {study.best_params}")
    print("   ✓ Optuna functionality OK")
except Exception as e:
    print(f"   ✗ Optuna test failed: {e}")
    all_ok = False

# Final summary
print("\n" + "="*80)
if all_ok:
    print("✓ ALL CHECKS PASSED - Ready to run Optuna tuning!")
    print()
    print("Next steps:")
    print("  1. Review configuration in run_optuna.sh")
    print("  2. Submit job: sbatch run_optuna.sh")
    print("  3. Monitor: tail -f optuna_results/optuna_tune_*.out")
else:
    print("⚠ SOME CHECKS FAILED - Please fix issues before running")
    print()
    print("Common fixes:")
    print("  - Install Optuna: bash install_optuna.sh")
    print("  - Check dataset paths in run_optuna.sh")

print("="*80)

