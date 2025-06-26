"""
MADRID Usage Examples for ECG Seizure Detection

This script demonstrates various ways to use the MADRID algorithm
for ECG-based seizure detection tasks.

Run this script to see MADRID in action with different configurations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from models.madrid import MADRID

def example_1_basic_usage():
    """
    Example 1: Basic MADRID usage with synthetic data
    """
    print("Example 1: Basic MADRID Usage")
    print("-" * 40)
    
    # Create synthetic time series with anomalies
    np.random.seed(42)
    n = 5000
    
    # Base signal
    t = np.linspace(0, 50, n)
    signal = np.sin(0.5 * t) + 0.5 * np.sin(2 * t) + 0.1 * np.random.randn(n)
    
    # Add anomalies
    # Short anomaly
    signal[1000:1050] += 2 * np.random.randn(50)
    # Long anomaly  
    signal[3000:3200] *= 0.3
    
    # Initialize MADRID
    madrid = MADRID(use_gpu=True, enable_output=True)
    
    # Run detection
    results = madrid.fit(
        T=signal,
        min_length=25,    # Minimum pattern length
        max_length=200,   # Maximum pattern length
        step_size=25,     # Step between lengths
        train_test_split=2500  # Use first half for training
    )
    
    # Get anomaly information
    anomalies = madrid.get_anomaly_scores(threshold_percentile=95)
    
    print(f"Detected {len(anomalies['anomalies'])} anomalies")
    for i, anomaly in enumerate(anomalies['anomalies'][:3]):
        print(f"  Anomaly {i+1}: Score={anomaly['score']:.3f}, "
              f"Location={anomaly['location']}, Length={anomaly['length']}")
    
    # Plot results
    madrid.plot_results()
    
    return madrid, results, anomalies

def example_2_ecg_seizure_detection():
    """
    Example 2: ECG seizure detection simulation
    """
    print("\nExample 2: ECG Seizure Detection Simulation")
    print("-" * 50)
    
    # Simulate ECG data at 250Hz
    fs = 250  # Sampling frequency
    duration = 120  # 2 minutes
    n = fs * duration
    t = np.linspace(0, duration, n)
    
    # Simulate ECG-like signal
    np.random.seed(123)
    
    # Heart rate variability
    hr_base = 70  # BPM
    hr_variation = 10 * np.sin(0.1 * t) + 5 * np.random.randn(n) * 0.01
    hr = hr_base + hr_variation
    
    # Generate R-peaks
    rr_intervals = 60 / hr  # RR intervals in seconds
    ecg_signal = np.zeros(n)
    
    # Simple R-peak simulation
    peak_times = np.cumsum(rr_intervals * fs).astype(int)
    peak_times = peak_times[peak_times < n]
    
    for peak_time in peak_times:
        if peak_time < n - 5:
            # Simple R-peak shape
            ecg_signal[peak_time-2:peak_time+3] = [0.2, 0.5, 1.0, 0.5, 0.2]
    
    # Add baseline noise
    ecg_signal += 0.05 * np.random.randn(n)
    
    # Simulate seizure events (cardiac changes)
    # Seizure 1: Tachycardia (increased heart rate)
    seizure_start_1 = 30 * fs  # 30 seconds
    seizure_end_1 = 45 * fs    # 45 seconds
    seizure_factor = 1.5
    ecg_signal[seizure_start_1:seizure_end_1] *= seizure_factor
    
    # Seizure 2: Bradycardia (decreased heart rate)  
    seizure_start_2 = 80 * fs  # 80 seconds
    seizure_end_2 = 95 * fs    # 95 seconds
    ecg_signal[seizure_start_2:seizure_end_2] *= 0.6
    
    print(f"Generated {duration}s of ECG data at {fs}Hz")
    print(f"Simulated seizures at: {seizure_start_1/fs:.1f}-{seizure_end_1/fs:.1f}s "
          f"and {seizure_start_2/fs:.1f}-{seizure_end_2/fs:.1f}s")
    
    # Configure MADRID for seizure detection
    madrid_ecg = MADRID(use_gpu=True, enable_output=True)
    
    # Parameters optimized for seizure detection
    min_seizure_duration = 0.5  # 0.5 seconds
    max_seizure_duration = 30   # 30 seconds
    step_duration = 0.5         # 0.5 second steps
    
    results = madrid_ecg.fit(
        T=ecg_signal,
        min_length=int(min_seizure_duration * fs),  # 125 samples
        max_length=int(max_seizure_duration * fs),  # 7500 samples
        step_size=int(step_duration * fs),          # 125 samples
        train_test_split=15 * fs  # Use first 15 seconds for training
    )
    
    # Analyze results
    anomalies = madrid_ecg.get_anomaly_scores(threshold_percentile=90)
    
    print(f"\nMADRID detected {len(anomalies['anomalies'])} potential seizure events")
    
    # Check detection accuracy
    detected_seizures = []
    for anomaly in anomalies['anomalies']:
        location_seconds = anomaly['location'] / fs
        length_seconds = anomaly['length'] / fs
        
        print(f"  Detected event: {location_seconds:.1f}s "
              f"(duration: {length_seconds:.1f}s, score: {anomaly['score']:.3f})")
        
        # Check if detection overlaps with true seizures
        seizure_1_range = (seizure_start_1/fs, seizure_end_1/fs)
        seizure_2_range = (seizure_start_2/fs, seizure_end_2/fs)
        
        if (seizure_1_range[0] <= location_seconds <= seizure_1_range[1] or
            seizure_2_range[0] <= location_seconds <= seizure_2_range[1]):
            detected_seizures.append(anomaly)
    
    print(f"\nCorrectly detected {len(detected_seizures)} out of 2 seizures")
    
    # Plot results
    madrid_ecg.plot_results()
    
    # Additional ECG-specific plot
    plt.figure(figsize=(15, 8))
    
    # Plot ECG signal
    plt.subplot(2, 1, 1)
    time_axis = np.arange(len(ecg_signal)) / fs
    plt.plot(time_axis, ecg_signal, 'b-', alpha=0.7, label='ECG Signal')
    
    # Mark true seizures
    plt.axvspan(seizure_start_1/fs, seizure_end_1/fs, alpha=0.3, color='red', label='True Seizure')
    plt.axvspan(seizure_start_2/fs, seizure_end_2/fs, alpha=0.3, color='red')
    
    # Mark detected anomalies
    for anomaly in anomalies['anomalies'][:5]:  # Top 5 detections
        loc_sec = anomaly['location'] / fs
        plt.axvline(x=loc_sec, color='orange', linestyle='--', alpha=0.8, 
                   label='MADRID Detection' if anomaly == anomalies['anomalies'][0] else '')
    
    plt.title('ECG Signal with Seizure Detection')
    plt.xlabel('Time (seconds)')
    plt.ylabel('ECG Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot anomaly scores over time
    plt.subplot(2, 1, 2)
    multi_length_table, bsf, bsf_loc = results
    
    # Average anomaly score across all lengths
    avg_scores = np.nanmean(multi_length_table, axis=0)
    valid_scores = avg_scores[~np.isnan(avg_scores)]
    valid_time = time_axis[:len(valid_scores)]
    
    plt.plot(valid_time, valid_scores, 'g-', label='Average Anomaly Score')
    plt.axhline(y=np.percentile(valid_scores, 90), color='red', 
               linestyle=':', label='90th Percentile Threshold')
    
    plt.title('MADRID Anomaly Scores Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Anomaly Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return madrid_ecg, results, anomalies

def example_3_parameter_tuning():
    """
    Example 3: Parameter tuning for different scenarios
    """
    print("\nExample 3: Parameter Tuning Examples")
    print("-" * 40)
    
    # Generate test data
    madrid_test = MADRID(use_gpu=False, enable_output=False)
    test_data = madrid_test.generate_test_data()
    
    # Different parameter configurations
    configs = [
        {
            'name': 'Short Anomalies',
            'min_length': 32,
            'max_length': 128,
            'step_size': 16,
            'description': 'Optimized for detecting short-duration anomalies'
        },
        {
            'name': 'Long Anomalies', 
            'min_length': 128,
            'max_length': 512,
            'step_size': 32,
            'description': 'Optimized for detecting long-duration anomalies'
        },
        {
            'name': 'Multi-Scale',
            'min_length': 64,
            'max_length': 256,
            'step_size': 32,
            'description': 'Balanced approach for various anomaly lengths'
        }
    ]
    
    results_comparison = {}
    
    for config in configs:
        print(f"\nTesting {config['name']} configuration...")
        print(f"  {config['description']}")
        
        madrid = MADRID(use_gpu=True, enable_output=False)
        results = madrid.fit(
            T=test_data,
            min_length=config['min_length'],
            max_length=config['max_length'],
            step_size=config['step_size'],
            train_test_split=len(test_data)//2
        )
        
        anomalies = madrid.get_anomaly_scores(threshold_percentile=95)
        results_comparison[config['name']] = {
            'config': config,
            'results': results,
            'anomalies': anomalies,
            'madrid': madrid
        }
        
        print(f"  Detected {len(anomalies['anomalies'])} anomalies")
        if anomalies['anomalies']:
            print(f"  Top score: {anomalies['anomalies'][0]['score']:.3f}")
    
    # Compare results
    print(f"\nComparison Summary:")
    print(f"{'Configuration':<15} {'Detections':<12} {'Top Score':<12} {'Avg Score':<12}")
    print("-" * 55)
    
    for name, data in results_comparison.items():
        anomalies = data['anomalies']['anomalies']
        n_detections = len(anomalies)
        top_score = anomalies[0]['score'] if anomalies else 0
        avg_score = np.mean([a['score'] for a in anomalies]) if anomalies else 0
        
        print(f"{name:<15} {n_detections:<12} {top_score:<12.3f} {avg_score:<12.3f}")
    
    return results_comparison

def example_4_gpu_performance():
    """
    Example 4: GPU vs CPU performance comparison
    """
    print("\nExample 4: GPU vs CPU Performance Comparison")
    print("-" * 50)
    
    # Generate larger dataset for performance testing
    np.random.seed(42)
    large_data = np.random.randn(20000)
    
    # Add some anomalies
    large_data[5000:5100] += 3 * np.random.randn(100)
    large_data[15000:15200] *= 0.3
    
    configs = [
        {'use_gpu': False, 'name': 'CPU'},
        {'use_gpu': True, 'name': 'GPU'}
    ]
    
    performance_results = {}
    
    for config in configs:
        if config['use_gpu'] and not MADRID(use_gpu=True).use_gpu:
            print(f"Skipping {config['name']} test - GPU not available")
            continue
            
        print(f"\nTesting {config['name']} performance...")
        
        madrid = MADRID(use_gpu=config['use_gpu'], enable_output=False)
        
        # Time the execution
        import time
        start_time = time.time()
        
        results = madrid.fit(
            T=large_data,
            min_length=50,
            max_length=200,
            step_size=25,
            train_test_split=10000
        )
        
        execution_time = time.time() - start_time
        
        anomalies = madrid.get_anomaly_scores()
        
        performance_results[config['name']] = {
            'execution_time': execution_time,
            'detections': len(anomalies['anomalies']),
            'device': 'GPU' if madrid.use_gpu else 'CPU'
        }
        
        print(f"  Execution time: {execution_time:.2f} seconds")
        print(f"  Detections: {len(anomalies['anomalies'])}")
        print(f"  Device used: {performance_results[config['name']]['device']}")
    
    # Performance summary
    if len(performance_results) > 1:
        cpu_time = performance_results.get('CPU', {}).get('execution_time', 0)
        gpu_time = performance_results.get('GPU', {}).get('execution_time', 0)
        
        if cpu_time > 0 and gpu_time > 0:
            speedup = cpu_time / gpu_time
            print(f"\nSpeedup: {speedup:.2f}x faster with GPU")
    
    return performance_results

def main():
    """
    Run all examples
    """
    print("MADRID - Multi-Length Anomaly Detection Examples")
    print("=" * 60)
    
    try:
        # Run examples
        example_1_basic_usage()
        example_2_ecg_seizure_detection()
        example_3_parameter_tuning()
        example_4_gpu_performance()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("\nKey takeaways:")
        print("1. MADRID can detect anomalies of different lengths simultaneously")
        print("2. Parameter tuning is crucial for different types of anomalies")
        print("3. GPU acceleration can significantly improve performance")
        print("4. The algorithm is well-suited for ECG seizure detection")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()