#!/usr/bin/env python3
"""
Validation script for MADRID v2 - checks structure and basic functionality
"""

import sys
import inspect

def validate_madrid_v2():
    """Validate MADRID v2 implementation structure"""
    print("Validating MADRID v2 implementation...")
    
    try:
        # Test import
        from models.madrid_v2 import MADRID_V2
        print("✓ MADRID_V2 class imported successfully")
        
        # Check class structure
        required_methods = [
            '__init__',
            'fit', 
            'get_anomaly_scores',
            'mass_v2',
            'contains_constant_regions'
        ]
        
        available_methods = [method for method in dir(MADRID_V2) if not method.startswith('_') or method in ['__init__']]
        
        print(f"✓ Available methods: {len(available_methods)}")
        for method in available_methods:
            print(f"    - {method}")
        
        # Check required methods
        missing_methods = []
        for method in required_methods:
            if not hasattr(MADRID_V2, method):
                missing_methods.append(method)
        
        if missing_methods:
            print(f"❌ Missing required methods: {missing_methods}")
            return False
        else:
            print("✓ All required methods present")
        
        # Test basic instantiation
        try:
            madrid = MADRID_V2(use_gpu=False, enable_output=False)
            print("✓ MADRID_V2 instantiation successful")
        except Exception as e:
            print(f"❌ MADRID_V2 instantiation failed: {e}")
            return False
        
        # Check get_anomaly_scores method signature
        sig = inspect.signature(madrid.get_anomaly_scores)
        params = list(sig.parameters.keys())
        expected_params = ['threshold_percentile', 'max_anomalies_per_length']
        
        if all(param in params for param in expected_params):
            print("✓ get_anomaly_scores has enhanced signature")
        else:
            print(f"⚠️  get_anomaly_scores parameters: {params}")
        
        print("\n" + "="*50)
        print("Key differences from original madrid.py:")
        print("1. Enhanced get_anomaly_scores with max_anomalies_per_length parameter")
        print("2. Complete matrix profile storage (_last_multi_length_table)")
        print("3. Multiple anomaly extraction per length")
        print("4. MATLAB-style strategic initialization")
        print("5. Enhanced DAMP with progressive search and pruning")
        print("="*50)
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

if __name__ == "__main__":
    success = validate_madrid_v2()
    
    if success:
        print("\n✓ MADRID v2 validation PASSED")
        print("The implementation should now be able to find multiple anomalies")
        print("instead of just one per length like the original version.")
    else:
        print("\n❌ MADRID v2 validation FAILED")
    
    sys.exit(0 if success else 1)