#!/usr/bin/env python3
"""
MERLIN Anomaly Detection on SeizeIT2 Data

This script demonstrates MERLIN (Parameter-Free Discovery of Arbitrary Length Anomalies) 
on real preprocessed ECG data from the SeizeIT2 dataset.

Usage:
    python merlin_seizeit_analysis.py [--data-path DATA_PATH] [--config CONFIG_NAME] [--batch-size BATCH_SIZE]

Author: Generated from Jupyter notebook
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import sys
import os
import argparse
from pathlib import Path
import warnings
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats

warnings.filterwarnings('ignore')

# Add the Information/Merlin directory to path to import MERLIN
sys.path.append(os.path.join('..', 'Information', 'Merlin'))

try:
    from _merlin import MERLIN
    print("✓ MERLIN successfully imported from Information/Merlin")
    MERLIN_AVAILABLE = True
except ImportError:
    print("❌ MERLIN import failed")
    print("Please ensure the MERLIN implementation is available in ../Information/Merlin/")
    MERLIN_AVAILABLE = False


def load_seizeit_data(data_path: str) -> Optional[Dict[str, Any]]:
    """
    Load preprocessed SeizeIT2 data from pickle file.
    
    Args:
        data_path: Path to the preprocessed data file
        
    Returns:
        Dictionary containing the loaded data or None if loading failed
    """
    data_path = Path(data_path)
    print(f"Loading preprocessed data from: {data_path}")
    
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        print("✓ Data loaded successfully")
        print(f"Subject: {data['subject_id']} {data['run_id']}")
        print(f"Recording duration: {data['recording_duration']:.1f} seconds ({data['recording_duration']/3600:.1f} hours)")
        print(f"Total seizures: {data['total_seizures']}")
        print(f"Number of channels: {len(data['channels'])}")
        
        # Get first channel info
        channel = data['channels'][0]
        print(f"\\nChannel: {channel['channel_name']}")
        print(f"Number of windows: {channel['n_windows']}")
        print(f"Seizure windows: {channel['n_seizure_windows']}")
        print(f"Original sampling rate: {channel['original_fs']} Hz")
        print(f"Processed sampling rate: {channel['processed_fs']} Hz")
        print(f"Window size: {len(channel['windows'][0])} samples ({len(channel['windows'][0])/channel['processed_fs']} seconds)")
        
        # Display preprocessing parameters
        print(f"\\nPreprocessing parameters:")
        for key, value in data['preprocessing_params'].items():
            print(f"  {key}: {value}")
            
        return data
        
    except FileNotFoundError:
        print(f"❌ Data file not found: {data_path}")
        print("Please ensure the data has been preprocessed first.")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


def analyze_data_structure(data: Dict[str, Any]) -> None:
    """
    Analyze and display data structure and labels.
    
    Args:
        data: Loaded SeizeIT2 data dictionary
    """
    channel = data['channels'][0]  # Use first ECG channel
    
    # Analyze label distribution
    labels = np.array(channel['labels'])
    normal_windows = np.sum(labels == 0)
    seizure_windows = np.sum(labels == 1)
    
    print(f"\\nLabel Distribution:")
    print(f"  Normal windows: {normal_windows} ({normal_windows/len(labels)*100:.2f}%)")
    print(f"  Seizure windows: {seizure_windows} ({seizure_windows/len(labels)*100:.2f}%)")
    
    # Find seizure window indices
    seizure_indices = np.where(labels == 1)[0]
    print(f"  Seizure window indices: {seizure_indices}")
    
    # Sample data statistics
    sample_window = channel['windows'][0]
    print(f"\\nSample window statistics:")
    print(f"  Shape: {sample_window.shape}")
    print(f"  Data type: {sample_window.dtype}")
    print(f"  Range: {np.min(sample_window):.2f} to {np.max(sample_window):.2f}")
    print(f"  Mean: {np.mean(sample_window):.2f}")
    print(f"  Std: {np.std(sample_window):.2f}")
    
    # Time information from metadata
    if seizure_indices.size > 0:
        seizure_metadata = channel['metadata'][seizure_indices[0]]
        print(f"\\nFirst seizure window metadata:")
        for key, value in seizure_metadata.items():
            print(f"  {key}: {value}")


def create_merlin_configs(fs: int, window_samples: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Create MERLIN configuration parameters for ECG analysis.
    
    Args:
        fs: Sampling frequency in Hz
        window_samples: Number of samples per window
        
    Returns:
        Tuple of (all_configs, valid_configs)
    """
    # MERLIN parameter configurations for ECG analysis
    merlin_configs = [
        {
            "name": "Short-term (1-5s)",
            "min_length": int(1 * fs),    # 1 second
            "max_length": int(5 * fs),    # 5 seconds
            "description": "Detect rapid ECG changes and artifacts"
        },
        {
            "name": "Medium-term (2-10s)",
            "min_length": int(2 * fs),    # 2 seconds
            "max_length": int(10 * fs),   # 10 seconds
            "description": "Capture seizure-related rhythm changes"
        },
        {
            "name": "Long-term (5-15s)",
            "min_length": int(5 * fs),    # 5 seconds
            "max_length": int(15 * fs),   # 15 seconds
            "description": "Detect sustained seizure patterns"
        }
    ]
    
    print(f"\\nMERLIN Configuration for ECG Data (fs = {fs} Hz):")
    print(f"Window size: {window_samples} samples ({window_samples/fs:.1f} seconds)")
    print(f"\\nProposed MERLIN configurations:")
    
    for i, config in enumerate(merlin_configs):
        min_sec = config['min_length'] / fs
        max_sec = config['max_length'] / fs
        print(f"\\n{i+1}. {config['name']}:")
        print(f"   Min length: {config['min_length']} samples ({min_sec:.1f}s)")
        print(f"   Max length: {config['max_length']} samples ({max_sec:.1f}s)")
        print(f"   Purpose: {config['description']}")
    
    # Check if configurations are valid for our window size
    print(f"\\nConfiguration Validation:")
    valid_configs = []
    
    for config in merlin_configs:
        max_length = config['max_length']
        min_required = max_length * 2  # MERLIN requires at least 2x max_length
        
        if window_samples >= min_required:
            print(f"✓ {config['name']}: Valid (requires {min_required}, have {window_samples})")
            valid_configs.append(config)
        else:
            print(f"❌ {config['name']}: Invalid (requires {min_required}, have {window_samples})")
    
    print(f"\\nValid configurations: {len(valid_configs)}/{len(merlin_configs)}")
    return merlin_configs, valid_configs


def run_merlin_on_window(window_data: np.ndarray, config: Dict[str, Any], fs: int = 125) -> Dict[str, Any]:
    """
    Run MERLIN anomaly detection on a single ECG window.
    
    Args:
        window_data: ECG window data
        config: MERLIN configuration dictionary
        fs: Sampling frequency
        
    Returns:
        Dictionary containing detection results
    """
    start_time = time.time()
    
    # Initialize MERLIN
    detector = MERLIN(
        min_length=config['min_length'],
        max_length=config['max_length'],
        max_iterations=500
    )
    
    try:
        # Run detection
        anomalies = detector.fit_predict(window_data.astype(np.float64))
        detection_time = time.time() - start_time
        
        # Calculate statistics
        n_anomalies = np.sum(anomalies)
        anomaly_rate = n_anomalies / len(window_data) * 100
        
        # Find anomaly regions (consecutive anomaly points)
        anomaly_regions = []
        if n_anomalies > 0:
            anomaly_indices = np.where(anomalies)[0]
            
            if len(anomaly_indices) > 0:
                # Group consecutive indices
                region_start = anomaly_indices[0]
                region_end = anomaly_indices[0]
                
                for i in range(1, len(anomaly_indices)):
                    if anomaly_indices[i] == region_end + 1:
                        region_end = anomaly_indices[i]
                    else:
                        # Add completed region
                        anomaly_regions.append({
                            'start_sample': region_start,
                            'end_sample': region_end,
                            'start_time': region_start / fs,
                            'end_time': region_end / fs,
                            'duration': (region_end - region_start + 1) / fs
                        })
                        region_start = anomaly_indices[i]
                        region_end = anomaly_indices[i]
                
                # Add final region
                anomaly_regions.append({
                    'start_sample': region_start,
                    'end_sample': region_end,
                    'start_time': region_start / fs,
                    'end_time': region_end / fs,
                    'duration': (region_end - region_start + 1) / fs
                })
        
        return {
            'success': True,
            'anomalies': anomalies,
            'n_anomalies': n_anomalies,
            'anomaly_rate': anomaly_rate,
            'anomaly_regions': anomaly_regions,
            'detection_time': detection_time,
            'config': config
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'detection_time': time.time() - start_time,
            'config': config
        }


def test_merlin_configurations(data: Dict[str, Any], valid_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Test MERLIN with different configurations on sample windows.
    
    Args:
        data: Loaded SeizeIT2 data
        valid_configs: List of valid MERLIN configurations
        
    Returns:
        List of results for each configuration
    """
    channel = data['channels'][0]
    labels = np.array(channel['labels'])
    fs = channel['processed_fs']
    
    # Select test windows: normal and seizure (if available)
    normal_indices = np.where(labels == 0)[0]
    seizure_indices = np.where(labels == 1)[0]
    
    # Test windows to analyze
    test_windows = []
    
    # Add normal window
    test_windows.append({
        'index': normal_indices[0],
        'type': 'normal',
        'data': channel['windows'][normal_indices[0]],
        'label': labels[normal_indices[0]]
    })
    
    # Add seizure window if available
    if len(seizure_indices) > 0:
        test_windows.append({
            'index': seizure_indices[0],
            'type': 'seizure',
            'data': channel['windows'][seizure_indices[0]],
            'label': labels[seizure_indices[0]]
        })
    
    print(f"\\nTesting MERLIN on {len(test_windows)} windows:")
    for window in test_windows:
        print(f"  - Window #{window['index']}: {window['type']} (label: {window['label']})")
    
    # Run MERLIN with different configurations
    results_summary = []
    
    print(f"\\n{'='*80}")
    print(f"MERLIN DETECTION RESULTS")
    print(f"{'='*80}")
    
    for config in valid_configs:
        print(f"\\nConfiguration: {config['name']}")
        print(f"  Window range: {config['min_length']}-{config['max_length']} samples")
        print(f"  Time range: {config['min_length']/fs:.1f}-{config['max_length']/fs:.1f} seconds")
        
        config_results = []
        
        for window in test_windows:
            print(f"\\n  Testing on {window['type']} window #{window['index']}:")
            
            result = run_merlin_on_window(window['data'], config, fs)
            config_results.append(result)
            
            if result['success']:
                print(f"    ✓ Detection completed in {result['detection_time']:.3f}s")
                print(f"    ✓ Found {result['n_anomalies']} anomaly points ({result['anomaly_rate']:.2f}%)")
                print(f"    ✓ Found {len(result['anomaly_regions'])} anomaly regions")
                
                # Show first few regions
                if result['anomaly_regions']:
                    print(f"    ✓ Sample regions:")
                    for i, region in enumerate(result['anomaly_regions'][:3]):
                        print(f"      {i+1}. {region['start_time']:.2f}s - {region['end_time']:.2f}s ({region['duration']:.3f}s)")
                    if len(result['anomaly_regions']) > 3:
                        print(f"      ... and {len(result['anomaly_regions']) - 3} more")
            else:
                print(f"    ❌ Detection failed: {result['error']}")
        
        results_summary.append({
            'config': config,
            'results': config_results
        })
    
    print(f"\\n✓ MERLIN testing completed on {len(test_windows)} windows with {len(valid_configs)} configurations")
    return results_summary


def analyze_performance(results_summary: List[Dict[str, Any]], test_windows: List[Dict[str, Any]]) -> None:
    """
    Analyze and display MERLIN performance results.
    
    Args:
        results_summary: Results from MERLIN testing
        test_windows: List of test windows used
    """
    print(f"\\n{'='*80}")
    print(f"MERLIN PERFORMANCE ANALYSIS")
    print(f"{'='*80}")
    
    # Create comparison table
    print(f"\\nConfiguration Comparison:")
    print(f"{'Config':<18} | {'Window':<8} | {'Anomalies':<10} | {'Rate %':<8} | {'Regions':<8} | {'Time (s)':<8}")
    print("-" * 80)
    
    for config_result in results_summary:
        config = config_result['config']
        results = config_result['results']
        
        for window, result in zip(test_windows, results):
            if result['success']:
                window_type = f"{window['type'][:4]}#{window['index']}"
                print(f"{config['name']:<18} | {window_type:<8} | {result['n_anomalies']:<10} | "
                      f"{result['anomaly_rate']:<8.2f} | {len(result['anomaly_regions']):<8} | "
                      f"{result['detection_time']:<8.3f}")
            else:
                window_type = f"{window['type'][:4]}#{window['index']}"
                print(f"{config['name']:<18} | {window_type:<8} | {'FAILED':<10} | {'N/A':<8} | {'N/A':<8} | "
                      f"{result['detection_time']:<8.3f}")
    
    # Analysis by window type
    print(f"\\nAnalysis by Window Type:")
    
    # Group results by window type
    window_types = {}
    
    for config_result in results_summary:
        config = config_result['config']
        results = config_result['results']
        
        for window, result in zip(test_windows, results):
            if result['success']:
                window_type = window['type']
                if window_type not in window_types:
                    window_types[window_type] = []
                
                window_types[window_type].append({
                    'config': config['name'],
                    'anomaly_rate': result['anomaly_rate'],
                    'n_regions': len(result['anomaly_regions']),
                    'detection_time': result['detection_time']
                })
    
    for window_type, type_results in window_types.items():
        print(f"\\n{window_type.upper()} Windows:")
        
        avg_anomaly_rate = np.mean([r['anomaly_rate'] for r in type_results])
        avg_regions = np.mean([r['n_regions'] for r in type_results])
        avg_time = np.mean([r['detection_time'] for r in type_results])
        
        print(f"  Average anomaly rate: {avg_anomaly_rate:.2f}%")
        print(f"  Average regions per window: {avg_regions:.1f}")
        print(f"  Average detection time: {avg_time:.3f}s")
        
        # Best configuration for this window type
        if window_type == 'seizure':
            # For seizures, prefer higher anomaly rates
            best = max(type_results, key=lambda x: x['anomaly_rate'])
            print(f"  Best config (highest anomaly rate): {best['config']} ({best['anomaly_rate']:.2f}%)")
        else:
            # For normal, prefer lower anomaly rates
            best = min(type_results, key=lambda x: x['anomaly_rate'])
            print(f"  Best config (lowest anomaly rate): {best['config']} ({best['anomaly_rate']:.2f}%)")
    
    # Overall recommendations
    print(f"\\nRecommendations:")
    
    if 'seizure' in window_types and 'normal' in window_types:
        seizure_results = window_types['seizure']
        normal_results = window_types['normal']
        
        # Calculate discrimination ability
        for config_name in [c['config']['name'] for c in results_summary]:
            seizure_rate = next((r['anomaly_rate'] for r in seizure_results if r['config'] == config_name), 0)
            normal_rate = next((r['anomaly_rate'] for r in normal_results if r['config'] == config_name), 0)
            
            discrimination = seizure_rate - normal_rate
            print(f"  {config_name}: Discrimination = {discrimination:.2f}% (seizure: {seizure_rate:.2f}%, normal: {normal_rate:.2f}%)")
        
        # Find best discriminating configuration
        best_discrimination = -float('inf')
        best_config_name = None
        
        for config_name in [c['config']['name'] for c in results_summary]:
            seizure_rate = next((r['anomaly_rate'] for r in seizure_results if r['config'] == config_name), 0)
            normal_rate = next((r['anomaly_rate'] for r in normal_results if r['config'] == config_name), 0)
            discrimination = seizure_rate - normal_rate
            
            if discrimination > best_discrimination:
                best_discrimination = discrimination
                best_config_name = config_name
        
        print(f"\\n✓ Best configuration for seizure detection: {best_config_name}")
        print(f"  Discrimination: {best_discrimination:.2f}% (higher is better)")
    
    else:
        print(f"  Limited analysis - need both normal and seizure windows for comparison")


def batch_merlin_analysis(channel_data: Dict[str, Any], config: Dict[str, Any], 
                         n_windows: int = 50, fs: int = 125) -> List[Dict[str, Any]]:
    """
    Run MERLIN on multiple windows for statistical analysis.
    
    Args:
        channel_data: Channel data dictionary
        config: MERLIN configuration
        n_windows: Number of windows to process
        fs: Sampling frequency
        
    Returns:
        List of analysis results
    """
    labels = np.array(channel_data['labels'])
    windows = channel_data['windows']
    
    # Select windows to process
    normal_indices = np.where(labels == 0)[0]
    seizure_indices = np.where(labels == 1)[0]
    
    # Sample windows
    n_normal = min(n_windows - len(seizure_indices), len(normal_indices))
    selected_normal = np.random.choice(normal_indices, n_normal, replace=False)
    
    selected_indices = np.concatenate([selected_normal, seizure_indices])
    selected_labels = labels[selected_indices]
    
    print(f"\\nProcessing {len(selected_indices)} windows:")
    print(f"  Normal: {np.sum(selected_labels == 0)}")
    print(f"  Seizure: {np.sum(selected_labels == 1)}")
    
    # Initialize MERLIN
    detector = MERLIN(
        min_length=config['min_length'],
        max_length=config['max_length'],
        max_iterations=500
    )
    
    results = []
    processing_times = []
    
    print(f"\\nRunning MERLIN with {config['name']} configuration...")
    
    for i, idx in enumerate(selected_indices):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(selected_indices)} windows processed")
        
        window_data = windows[idx].astype(np.float64)
        label = selected_labels[i]
        
        start_time = time.time()
        
        try:
            anomalies = detector.fit_predict(window_data)
            detection_time = time.time() - start_time
            
            n_anomalies = np.sum(anomalies)
            anomaly_rate = n_anomalies / len(window_data) * 100
            
            results.append({
                'window_idx': idx,
                'label': label,
                'n_anomalies': n_anomalies,
                'anomaly_rate': anomaly_rate,
                'detection_time': detection_time,
                'success': True
            })
            
            processing_times.append(detection_time)
            
        except Exception as e:
            results.append({
                'window_idx': idx,
                'label': label,
                'error': str(e),
                'success': False
            })
    
    print(f"\\n✓ Batch processing completed")
    return results


def analyze_batch_results(batch_results: List[Dict[str, Any]]) -> None:
    """
    Analyze and display batch processing results.
    
    Args:
        batch_results: Results from batch processing
    """
    # Analyze results
    successful_results = [r for r in batch_results if r['success']]
    failed_results = [r for r in batch_results if not r['success']]
    
    print(f"\\nBatch Results Summary:")
    print(f"  Successful: {len(successful_results)}/{len(batch_results)}")
    print(f"  Failed: {len(failed_results)}/{len(batch_results)}")
    
    if successful_results:
        # Separate by label
        normal_results = [r for r in successful_results if r['label'] == 0]
        seizure_results = [r for r in successful_results if r['label'] == 1]
        
        print(f"\\nStatistical Analysis:")
        
        if normal_results:
            normal_rates = [r['anomaly_rate'] for r in normal_results]
            print(f"  Normal windows ({len(normal_results)}):")
            print(f"    Anomaly rate: {np.mean(normal_rates):.2f}% ± {np.std(normal_rates):.2f}%")
            print(f"    Range: {np.min(normal_rates):.2f}% - {np.max(normal_rates):.2f}%")
        
        if seizure_results:
            seizure_rates = [r['anomaly_rate'] for r in seizure_results]
            print(f"  Seizure windows ({len(seizure_results)}):")
            print(f"    Anomaly rate: {np.mean(seizure_rates):.2f}% ± {np.std(seizure_rates):.2f}%")
            print(f"    Range: {np.min(seizure_rates):.2f}% - {np.max(seizure_rates):.2f}%")
        
        # Processing time analysis
        processing_times = [r['detection_time'] for r in successful_results]
        print(f"\\n  Processing Performance:")
        print(f"    Average time per window: {np.mean(processing_times):.3f}s")
        print(f"    Total processing time: {np.sum(processing_times):.2f}s")
        print(f"    Throughput: {len(successful_results)/np.sum(processing_times):.1f} windows/second")
        
        # Statistical test if both window types available
        if normal_results and seizure_results:
            normal_rates = [r['anomaly_rate'] for r in normal_results]
            seizure_rates = [r['anomaly_rate'] for r in seizure_results]
            
            t_stat, p_value = stats.ttest_ind(seizure_rates, normal_rates)
            print(f"\\n  Statistical Test (t-test):")
            print(f"    t-statistic: {t_stat:.3f}")
            print(f"    p-value: {p_value:.6f}")
            print(f"    Significant difference: {'Yes' if p_value < 0.05 else 'No'} (α = 0.05)")
    
    if failed_results:
        print(f"\\nFailed Analyses:")
        for result in failed_results[:5]:  # Show first 5 failures
            print(f"  Window #{result['window_idx']} (label: {result['label']}): {result['error']}")


def print_summary(data: Dict[str, Any], valid_configs: List[Dict[str, Any]], 
                 results_summary: List[Dict[str, Any]], batch_results: Optional[List[Dict[str, Any]]] = None) -> None:
    """
    Print comprehensive summary and conclusions.
    
    Args:
        data: Loaded SeizeIT2 data
        valid_configs: Valid MERLIN configurations
        results_summary: Configuration test results
        batch_results: Batch processing results (optional)
    """
    print(f"\\n{'='*80}")
    print(f"MERLIN ON SEIZEIT2 DATA - SUMMARY AND CONCLUSIONS")
    print(f"{'='*80}")
    
    print(f"\\nDataset Information:")
    print(f"  Subject: {data['subject_id']} {data['run_id']}")
    print(f"  Recording Duration: {data['recording_duration']:.1f} seconds ({data['recording_duration']/3600:.1f} hours)")
    print(f"  ECG Channel: {data['channels'][0]['channel_name']}")
    print(f"  Sampling Rate: {data['channels'][0]['processed_fs']} Hz")
    print(f"  Window Size: 30 seconds ({len(data['channels'][0]['windows'][0])} samples)")
    print(f"  Total Windows: {len(data['channels'][0]['windows'])}")
    print(f"  Seizure Windows: {data['channels'][0]['n_seizure_windows']}")
    
    if valid_configs:
        print(f"\\nMERLIN Configuration Testing:")
        print(f"  Valid Configurations: {len(valid_configs)}")
        for config in valid_configs:
            print(f"    ✓ {config['name']}: {config['min_length']}-{config['max_length']} samples")
        
        if results_summary:
            print(f"\\nKey Findings:")
            
            # Success rate
            total_tests = sum(len(cr['results']) for cr in results_summary)
            successful_tests = sum(sum(1 for r in cr['results'] if r['success']) for cr in results_summary)
            print(f"  Detection Success Rate: {successful_tests}/{total_tests} ({successful_tests/total_tests*100:.1f}%)")
            
            # Performance summary
            all_successful = []
            for cr in results_summary:
                all_successful.extend([r for r in cr['results'] if r['success']])
            
            if all_successful:
                avg_detection_time = np.mean([r['detection_time'] for r in all_successful])
                avg_anomaly_rate = np.mean([r['anomaly_rate'] for r in all_successful])
                print(f"  Average Detection Time: {avg_detection_time:.3f} seconds per window")
                print(f"  Average Anomaly Rate: {avg_anomaly_rate:.2f}%")
                
                # Real-time capability assessment
                window_duration = 30  # seconds
                real_time_capable = avg_detection_time < window_duration
                print(f"  Real-time Capable: {'Yes' if real_time_capable else 'No'} (need < {window_duration}s per window)")
        
        if batch_results:
            print(f"\\nBatch Analysis Results:")
            successful_batch = [r for r in batch_results if r['success']]
            normal_batch = [r for r in successful_batch if r['label'] == 0]
            seizure_batch = [r for r in successful_batch if r['label'] == 1]
            
            if normal_batch and seizure_batch:
                normal_rate = np.mean([r['anomaly_rate'] for r in normal_batch])
                seizure_rate = np.mean([r['anomaly_rate'] for r in seizure_batch])
                discrimination = seizure_rate - normal_rate
                
                print(f"  Normal Windows: {normal_rate:.2f}% average anomaly rate")
                print(f"  Seizure Windows: {seizure_rate:.2f}% average anomaly rate")
                print(f"  Discrimination Ability: {discrimination:.2f}% (higher is better)")
                
                if discrimination > 5:
                    print(f"  ✓ Good discrimination between normal and seizure windows")
                elif discrimination > 0:
                    print(f"  ⚠️  Moderate discrimination - parameter tuning may help")
                else:
                    print(f"  ❌ Poor discrimination - different approach may be needed")
    
    print(f"\\nConclusions:")
    
    if valid_configs and results_summary:
        print(f"  ✓ MERLIN successfully applied to preprocessed SeizeIT2 ECG data")
        print(f"  ✓ Compatible with 30-second ECG windows at 125 Hz sampling rate")
        
        if batch_results:
            successful_batch = [r for r in batch_results if r['success']]
            if successful_batch:
                avg_time = np.mean([r['detection_time'] for r in successful_batch])
                if avg_time < 30:
                    print(f"  ✓ Real-time processing capability demonstrated")
                
                normal_batch = [r for r in successful_batch if r['label'] == 0]
                seizure_batch = [r for r in successful_batch if r['label'] == 1]
                
                if normal_batch and seizure_batch:
                    discrimination = np.mean([r['anomaly_rate'] for r in seizure_batch]) - np.mean([r['anomaly_rate'] for r in normal_batch])
                    if discrimination > 0:
                        print(f"  ✓ MERLIN shows potential for seizure detection (discrimination: {discrimination:.2f}%)")
                    else:
                        print(f"  ⚠️  Limited seizure detection capability - further optimization needed")
    
    print(f"\\nRecommendations for Future Work:")
    print(f"  1. Test MERLIN on multiple patients and seizure types")
    print(f"  2. Optimize parameters (min_length, max_length, max_iterations) for ECG seizure detection")
    print(f"  3. Compare with other anomaly detection methods (Matrix Profile, TimeVQVAE-AD)")
    print(f"  4. Implement ensemble methods combining multiple configurations")
    print(f"  5. Evaluate on longer continuous recordings with multiple seizures")
    print(f"  6. Consider preprocessing variations (filtering, normalization, feature extraction)")
    print(f"  7. Develop adaptive thresholding for anomaly classification")


def main():
    """Main function to run MERLIN analysis on SeizeIT2 data."""
    parser = argparse.ArgumentParser(description='MERLIN Anomaly Detection on SeizeIT2 Data')
    parser.add_argument('--data-path', type=str, 
                       default='../results/preprocessed_all/sub-001_run-03_preprocessed.pkl',
                       help='Path to preprocessed SeizeIT2 data file')
    parser.add_argument('--config', type=str, choices=['short', 'medium', 'long', 'all'],
                       default='all', help='MERLIN configuration to use')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Number of windows for batch analysis')
    parser.add_argument('--no-batch', action='store_true',
                       help='Skip batch processing analysis')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible results')
    
    args = parser.parse_args()
    
    # Set random seed for reproducible results
    np.random.seed(args.seed)
    
    print("MERLIN Anomaly Detection on SeizeIT2 Data")
    print("=" * 50)
    print(f"MERLIN available: {MERLIN_AVAILABLE}")
    
    if not MERLIN_AVAILABLE:
        print("Cannot proceed without MERLIN implementation.")
        return 1
    
    # Load data
    data = load_seizeit_data(args.data_path)
    if data is None:
        return 1
    
    # Analyze data structure
    analyze_data_structure(data)
    
    # Create MERLIN configurations
    channel = data['channels'][0]
    fs = channel['processed_fs']
    window_samples = len(channel['windows'][0])
    
    merlin_configs, valid_configs = create_merlin_configs(fs, window_samples)
    
    if not valid_configs:
        print("\\n❌ No valid MERLIN configurations found!")
        return 1
    
    # Filter configurations based on user choice
    if args.config != 'all':
        config_map = {'short': 0, 'medium': 1, 'long': 2}
        if args.config in config_map and config_map[args.config] < len(valid_configs):
            valid_configs = [valid_configs[config_map[args.config]]]
            print(f"\\nUsing only {args.config}-term configuration")
    
    # Test MERLIN configurations
    results_summary = test_merlin_configurations(data, valid_configs)
    
    if results_summary:
        # Analyze performance
        # Create test_windows for analysis (recreate from test function)
        labels = np.array(channel['labels'])
        normal_indices = np.where(labels == 0)[0]
        seizure_indices = np.where(labels == 1)[0]
        
        test_windows = [{'index': normal_indices[0], 'type': 'normal', 'label': 0}]
        if len(seizure_indices) > 0:
            test_windows.append({'index': seizure_indices[0], 'type': 'seizure', 'label': 1})
        
        analyze_performance(results_summary, test_windows)
        
        # Batch processing analysis
        batch_results = None
        if not args.no_batch and valid_configs:
            print(f"\\n{'='*60}")
            print(f"BATCH PROCESSING ANALYSIS")
            print(f"{'='*60}")
            
            # Use first valid configuration for batch analysis
            best_config = valid_configs[0]
            print(f"Using configuration: {best_config['name']}")
            
            batch_results = batch_merlin_analysis(channel, best_config, args.batch_size, fs)
            analyze_batch_results(batch_results)
    
    # Print summary
    print_summary(data, valid_configs, results_summary, batch_results)
    
    print(f"\\n{'='*80}")
    print(f"MERLIN ANALYSIS COMPLETED")
    print(f"{'='*80}")
    
    return 0


if __name__ == "__main__":
    exit(main())