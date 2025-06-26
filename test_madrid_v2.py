#!/usr/bin/env python3
"""
Test script for MADRID v2 implementation
"""

try:
    import numpy as np
    print("✓ NumPy available")
except ImportError:
    print("❌ NumPy not available")
    exit(1)

try:
    from models.madrid_v2 import MADRID_V2
    print("✓ MADRID_V2 imported successfully")
except ImportError as e:
    print(f"❌ Failed to import MADRID_V2: {e}")
    exit(1)

def test_madrid_v2():
    """Test MADRID v2 with synthetic data"""
    print("\n" + "="*50)
    print("Testing MADRID v2 implementation")
    print("="*50)
    
    # Create test data with multiple anomalies
    np.random.seed(42)
    n = 1000
    T = np.random.randn(n) * 0.5
    
    # Add multiple anomalies
    T[200:250] += np.random.randn(50) * 1.5  # Anomaly 1
    T[500:580] += np.random.randn(80) * 1.2  # Anomaly 2  
    T[750:800] += np.random.randn(50) * 1.8  # Anomaly 3
    
    print(f"Created time series with {n} points and 3 synthetic anomalies")
    print("Anomaly locations: 200-250, 500-580, 750-800")
    
    # Test MADRID v2
    print("\nInitializing MADRID v2...")
    madrid = MADRID_V2(use_gpu=False, enable_output=True)
    
    print("\nRunning MADRID v2 fit...")
    try:
        results = madrid.fit(
            T=T, 
            min_length=30, 
            max_length=100, 
            step_size=20, 
            train_test_split=150
        )
        print("✓ MADRID v2 fit completed successfully")
    except Exception as e:
        print(f"❌ MADRID v2 fit failed: {e}")
        return False
    
    # Get anomalies
    print("\nExtracting anomalies...")
    try:
        anomaly_info = madrid.get_anomaly_scores(threshold_percentile=85)
        print(f"✓ Found {len(anomaly_info['anomalies'])} anomalies")
        
        if len(anomaly_info['anomalies']) > 0:
            print("\nTop anomalies:")
            for i, anomaly in enumerate(anomaly_info['anomalies'][:10]):
                print(f"  {i+1:2d}. Score: {anomaly['score']:.3f}, "
                      f"Location: {anomaly['location']:4d}, "
                      f"Length: {anomaly['length']:3d}")
            
            # Check if we found anomalies near the synthetic ones
            found_near_200 = any(180 <= a['location'] <= 270 for a in anomaly_info['anomalies'])
            found_near_500 = any(480 <= a['location'] <= 600 for a in anomaly_info['anomalies'])
            found_near_750 = any(730 <= a['location'] <= 820 for a in anomaly_info['anomalies'])
            
            print(f"\nDetection results:")
            print(f"  Anomaly 1 (200-250): {'✓' if found_near_200 else '❌'}")
            print(f"  Anomaly 2 (500-580): {'✓' if found_near_500 else '❌'}")
            print(f"  Anomaly 3 (750-800): {'✓' if found_near_750 else '❌'}")
            
            return len(anomaly_info['anomalies']) > 1  # Success if multiple anomalies found
        else:
            print("❌ No anomalies detected")
            return False
            
    except Exception as e:
        print(f"❌ Anomaly extraction failed: {e}")
        return False

if __name__ == "__main__":
    success = test_madrid_v2()
    
    print("\n" + "="*50)
    if success:
        print("✓ MADRID v2 test PASSED - Multiple anomalies detected!")
    else:
        print("❌ MADRID v2 test FAILED")
    print("="*50)