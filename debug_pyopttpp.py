#!/usr/bin/env python3
"""
Debug script to check what's available in pyopttpp module.
"""

import sys
import os

# Add the source directory to the path
src_path = os.path.join(os.path.dirname(__file__), "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Try to import pyopttpp and list available attributes
try:
    from everest_optimizers.optqnewton import _import_pyopttpp
    pyopttpp = _import_pyopttpp()
    
    print("Available attributes in pyopttpp:")
    for attr in sorted(dir(pyopttpp)):
        if not attr.startswith('_'):
            print(f"  {attr}: {type(getattr(pyopttpp, attr))}")
    
    # Test specific classes
    classes_to_test = [
        'BoundConstraint', 'LinearEquation', 'LinearInequality', 
        'NonLinearConstraint', 'NonLinearInequality', 'NonLinearEquation',
        'CompoundConstraint', 'Constraint'
    ]
    
    print("\nTesting specific constraint classes:")
    for cls_name in classes_to_test:
        if hasattr(pyopttpp, cls_name):
            print(f"  ✓ {cls_name}: Available")
        else:
            print(f"  ✗ {cls_name}: NOT Available")
    
    # Test the create_compound_constraint functions
    print("\nTesting compound constraint creation functions:")
    create_funcs = [name for name in dir(pyopttpp) if 'create_compound_constraint' in name]
    for func_name in create_funcs:
        func = getattr(pyopttpp, func_name)
        print(f"  {func_name}: {func.__doc__ or 'No docstring'}")
        
    
except Exception as e:
    print(f"Error importing pyopttpp: {e}")
    import traceback
    traceback.print_exc()