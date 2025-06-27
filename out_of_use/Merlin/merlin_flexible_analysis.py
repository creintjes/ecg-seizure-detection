#!/usr/bin/env python3
"""
Flexible MERLIN Anomaly Detection for Various ECG Data Formats

This script provides a flexible interface for running MERLIN anomaly detection
on ECG data with different sampling rates, window sizes, and custom configurations.

Usage:
    python merlin_flexible_analysis.py --data-path DATA_PATH --fs 8 --window-size 3600
    python merlin_flexible_analysis.py --config-file custom_config.json
    python merlin_flexible_analysis.py --min-length 8 --max-length 400 --fs 8

Author: Generated for flexible MERLIN analysis
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import sys
import os
import argparse
import json
from pathlib import Path
import warnings
from typing import Dict, List, Any, Optional, Tuple, Union
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


class FlexibleMerlinAnalyzer:
    """
    Flexible MERLIN analyzer that can handle different data formats and configurations.
    """
    
    def __init__(self, fs: int, window_size_seconds: float, min_length_seconds: float = None, 
                 max_length_seconds: float = None, custom_configs: List[Dict[str, Any]] = None):
        """
        Initialize the flexible MERLIN analyzer.
        
        Args:
            fs: Sampling frequency in Hz
            window_size_seconds: Window size in seconds
            min_length_seconds: Minimum MERLIN search length in seconds
            max_length_seconds: Maximum MERLIN search length in seconds
            custom_configs: List of custom MERLIN configurations
        """
        self.fs = fs
        self.window_size_seconds = window_size_seconds
        self.window_size_samples = int(window_size_seconds * fs)
        
        if custom_configs:
            self.configs = custom_configs
        else:
            # Generate default configurations based on window size
            self.configs = self._generate_default_configs(min_length_seconds, max_length_seconds)
        
        print(f"Initialized FlexibleMerlinAnalyzer:")
        print(f"  Sampling rate: {self.fs} Hz")
        print(f"  Window size: {self.window_size_seconds}s ({self.window_size_samples} samples)")
        print(f"  Number of configurations: {len(self.configs)}")
    
    def _generate_default_configs(self, min_length_seconds: float = None, 
                                max_length_seconds: float = None) -> List[Dict[str, Any]]:
        """Generate default MERLIN configurations based on window size and sampling rate."""
        
        # Set reasonable defaults based on window size
        if min_length_seconds is None:
            if self.window_size_seconds >= 3600:  # 1 hour or more
                min_length_seconds = 60  # 1 minute minimum
            elif self.window_size_seconds >= 300:  # 5 minutes or more
                min_length_seconds = 10  # 10 seconds minimum
            else:
                min_length_seconds = 1   # 1 second minimum
        
        if max_length_seconds is None:
            if self.window_size_seconds >= 3600:  # 1 hour or more
                max_length_seconds = 90  # 1,5 minutes maximum
            elif self.window_size_seconds >= 300:  # 5 minutes or more
                max_length_seconds = 60   # 1 minute maximum
            else:
                max_length_seconds = 15   # 15 seconds maximum
        
        # Ensure max_length doesn't exceed half the window size (MERLIN requirement)
        max_allowed = self.window_size_seconds / 2
        if max_length_seconds > max_allowed:
            max_length_seconds = max_allowed
            print(f"⚠️  Adjusted max_length to {max_length_seconds}s (half of window size)")
        
        # Generate configurations
        configs = []
        
        if self.window_size_seconds >= 3600:  # Long windows (hours)
            configs = [
                {
                    "name": f"Short-term ({min_length_seconds}s-{min_length_seconds*1.5}s)",
                    "min_length": int(min_length_seconds * self.fs),
                    "max_length": int(min_length_seconds * 1.5 * self.fs),
                    "description": f"Detect short anomalies ({min_length_seconds}-{min_length_seconds*5}s)"
                },
                {
                    "name": f"Medium-term ({min_length_seconds*2}s-{min_length_seconds*15}s)",
                    "min_length": int(min_length_seconds * 2 * self.fs),
                    "max_length": int(min_length_seconds * 15 * self.fs),
                    "description": f"Detect medium anomalies ({min_length_seconds*2}-{min_length_seconds*15}s)"
                },
                {
                    "name": f"Long-term ({min_length_seconds*5}s-{max_length_seconds}s)",
                    "min_length": int(min_length_seconds * 5 * self.fs),
                    "max_length": int(max_length_seconds * self.fs),
                    "description": f"Detect long anomalies ({min_length_seconds*5}-{max_length_seconds}s)"
                }
            ]
        else:  # Shorter windows
            configs = [
                {
                    "name": f"Short-term ({min_length_seconds}s-{min_length_seconds*5}s)",
                    "min_length": int(min_length_seconds * self.fs),
                    "max_length": int(min_length_seconds * 5 * self.fs),
                    "description": f"Detect rapid changes ({min_length_seconds}-{min_length_seconds*5}s)"
                },
                {
                    "name": f"Long-term ({min_length_seconds*2}s-{max_length_seconds}s)",
                    "min_length": int(min_length_seconds * 2 * self.fs),
                    "max_length": int(max_length_seconds * self.fs),
                    "description": f"Detect sustained patterns ({min_length_seconds*2}-{max_length_seconds}s)"
                }
            ]
        
        return configs
    
    def validate_configs(self) -> List[Dict[str, Any]]:
        """Validate MERLIN configurations against window size."""
        valid_configs = []
        
        print(f"\\nValidating MERLIN configurations:")
        
        for config in self.configs:
            max_length = config['max_length']
            min_required = max_length * 2  # MERLIN requires at least 2x max_length
            
            if self.window_size_samples >= min_required:
                print(f"✓ {config['name']}: Valid (requires {min_required}, have {self.window_size_samples})")
                valid_configs.append(config)
            else:
                print(f"❌ {config['name']}: Invalid (requires {min_required}, have {self.window_size_samples})")
        
        print(f"\\nValid configurations: {len(valid_configs)}/{len(self.configs)}")
        return valid_configs
    
    def load_data(self, data_path: str) -> Optional[Dict[str, Any]]:
        """Load preprocessed data with flexible format handling."""
        data_path = Path(data_path)
        print(f"\\nLoading data from: {data_path}")
        
        try:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            
            # Validate data structure
            if not isinstance(data, dict):
                raise ValueError("Data must be a dictionary")
            
            if 'channels' not in data or not data['channels']:
                raise ValueError("No channels found in data")
            
            channel = data['channels'][0]
            
            # Auto-detect sampling rate if not specified
            detected_fs = channel.get('processed_fs', channel.get('original_fs', self.fs))
            if detected_fs != self.fs:
                print(f"⚠️  Detected sampling rate ({detected_fs} Hz) differs from specified ({self.fs} Hz)")
                print(f"   Using detected rate: {detected_fs} Hz")
                self.fs = detected_fs
                # Recalculate window size in samples
                self.window_size_samples = int(self.window_size_seconds * self.fs)
                # Regenerate configs with new sampling rate
                self.configs = self._generate_default_configs()
            
            print("✓ Data loaded successfully")
            print(f"  Subject: {data.get('subject_id', 'Unknown')}")
            print(f"  Run: {data.get('run_id', 'Unknown')}")
            print(f"  Recording duration: {data.get('recording_duration', 0):.1f} seconds")
            print(f"  Channels: {len(data['channels'])}")
            print(f"  Channel name: {channel.get('channel_name', 'Unknown')}")
            print(f"  Windows: {channel.get('n_windows', len(channel.get('windows', [])))} ")
            print(f"  Seizure windows: {channel.get('n_seizure_windows', 0)}")
            print(f"  Sampling rate: {detected_fs} Hz")
            
            # Check window size compatibility
            if channel.get('windows') and len(channel['windows']) > 0:
                actual_window_size = len(channel['windows'][0])
                expected_window_size = self.window_size_samples
                
                if actual_window_size != expected_window_size:
                    print(f"⚠️  Window size mismatch:")
                    print(f"     Expected: {expected_window_size} samples ({self.window_size_seconds}s)")
                    print(f"     Actual: {actual_window_size} samples ({actual_window_size/self.fs:.1f}s)")
                    
                    # Update window size to match data
                    self.window_size_samples = actual_window_size
                    self.window_size_seconds = actual_window_size / self.fs
                    print(f"     Updated to match data: {self.window_size_seconds:.1f}s")
            
            return data
            
        except FileNotFoundError:
            print(f"❌ Data file not found: {data_path}")
            return None
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def run_merlin_on_window(self, window_data: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run MERLIN anomaly detection on a single window."""
        start_time = time.time()
        
        # Initialize MERLIN
        detector = MERLIN(
            min_length=config['min_length'],
            #max_length=config['max_length'],
            max_length=int(config['min_length'] * 1.5),
            max_iterations=10
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
                                'start_time': region_start / self.fs,
                                'end_time': region_end / self.fs,
                                'duration': (region_end - region_start + 1) / self.fs
                            })
                            region_start = anomaly_indices[i]
                            region_end = anomaly_indices[i]
                    
                    # Add final region
                    anomaly_regions.append({
                        'start_sample': region_start,
                        'end_sample': region_end,
                        'start_time': region_start / self.fs,
                        'end_time': region_end / self.fs,
                        'duration': (region_end - region_start + 1) / self.fs
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
    
    def test_configurations(self, data: Dict[str, Any], valid_configs: List[Dict[str, Any]], 
                          max_test_windows: int = 2) -> List[Dict[str, Any]]:
        """Test MERLIN configurations on sample windows."""
        channel = data['channels'][0]
        labels = np.array(channel.get('labels', []))
        windows = channel.get('windows', [])
        
        if not windows:
            print("❌ No windows found in data")
            return []
        
        # Select test windows
        test_windows = []
        
        # Add normal windows
        if len(labels) > 0:
            normal_indices = np.where(labels == 0)[0]
            seizure_indices = np.where(labels == 1)[0]
            
            # Add normal window
            if len(normal_indices) > 0:
                test_windows.append({
                    'index': normal_indices[0],
                    'type': 'normal',
                    'data': windows[normal_indices[0]],
                    'label': 0
                })
            
            # Add seizure window if available
            if len(seizure_indices) > 0:
                test_windows.append({
                    'index': seizure_indices[0],
                    'type': 'seizure',
                    'data': windows[seizure_indices[0]],
                    'label': 1
                })
        else:
            # No labels available, just use first few windows
            for i in range(min(max_test_windows, len(windows))):
                test_windows.append({
                    'index': i,
                    'type': 'unknown',
                    'data': windows[i],
                    'label': -1  # Unknown label
                })
        
        print(f"\\nTesting MERLIN on {len(test_windows)} windows:")
        for window in test_windows:
            print(f"  - Window #{window['index']}: {window['type']}")
        
        # Run MERLIN with different configurations
        results_summary = []
        
        print(f"\\n{'='*80}")
        print(f"MERLIN DETECTION RESULTS")
        print(f"{'='*80}")
        
        for config in valid_configs:
            print(f"\\nConfiguration: {config['name']}")
            print(f"  Window range: {config['min_length']}-{config['max_length']} samples")
            print(f"  Time range: {config['min_length']/self.fs:.1f}-{config['max_length']/self.fs:.1f} seconds")
            
            config_results = []
            
            for window in test_windows:
                print(f"\\n  Testing on {window['type']} window #{window['index']}:")
                
                result = self.run_merlin_on_window(window['data'], config)
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
    
    def batch_analysis(self, data: Dict[str, Any], config: Dict[str, Any], 
                      n_windows: int = 50) -> List[Dict[str, Any]]:
        """Run MERLIN on multiple windows for statistical analysis."""
        channel = data['channels'][0]
        labels = np.array(channel.get('labels', []))
        windows = channel.get('windows', [])
        
        if not windows:
            print("❌ No windows available for batch analysis")
            return []
        
        # Select windows to process
        if len(labels) > 0:
            normal_indices = np.where(labels == 0)[0]
            seizure_indices = np.where(labels == 1)[0]
            
            # Sample windows
            n_normal = min(n_windows - len(seizure_indices), len(normal_indices))
            if n_normal > 0:
                selected_normal = np.random.choice(normal_indices, n_normal, replace=False)
                selected_indices = np.concatenate([selected_normal, seizure_indices])
                selected_labels = labels[selected_indices]
            else:
                selected_indices = seizure_indices
                selected_labels = labels[selected_indices]
        else:
            # No labels, just sample random windows
            n_sample = min(n_windows, len(windows))
            selected_indices = np.random.choice(len(windows), n_sample, replace=False)
            selected_labels = np.full(len(selected_indices), -1)  # Unknown labels
        
        print(f"\\nProcessing {len(selected_indices)} windows:")
        if len(labels) > 0:
            print(f"  Normal: {np.sum(selected_labels == 0)}")
            print(f"  Seizure: {np.sum(selected_labels == 1)}")
        else:
            print(f"  Unknown labels: {len(selected_indices)}")
        
        # Initialize MERLIN
        detector = MERLIN(
            min_length=config['min_length'],
            max_length=int(config['min_length'] * 1.5),
            max_iterations=10
        )
        
        results = []
        
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
                
            except Exception as e:
                results.append({
                    'window_idx': idx,
                    'label': label,
                    'error': str(e),
                    'success': False
                })
        
        print(f"\\n✓ Batch processing completed")
        return results
    
    def analyze_results(self, results_summary: List[Dict[str, Any]], batch_results: List[Dict[str, Any]] = None):
        """Analyze and print comprehensive results."""
        print(f"\\n{'='*80}")
        print(f"FLEXIBLE MERLIN ANALYSIS RESULTS")
        print(f"{'='*80}")
        
        # Configuration comparison
        if results_summary:
            print(f"\\nConfiguration Performance:")
            print(f"{'Config':<25} | {'Avg Anomalies':<12} | {'Avg Rate %':<10} | {'Avg Time (s)':<12}")
            print("-" * 70)
            
            for result in results_summary:
                config = result['config']
                successful_results = [r for r in result['results'] if r['success']]
                
                if successful_results:
                    avg_anomalies = np.mean([r['n_anomalies'] for r in successful_results])
                    avg_rate = np.mean([r['anomaly_rate'] for r in successful_results])
                    avg_time = np.mean([r['detection_time'] for r in successful_results])
                    
                    print(f"{config['name']:<25} | {avg_anomalies:<12.1f} | {avg_rate:<10.2f} | {avg_time:<12.3f}")
                else:
                    print(f"{config['name']:<25} | {'FAILED':<12} | {'N/A':<10} | {'N/A':<12}")
        
        # Batch analysis results
        if batch_results:
            successful_batch = [r for r in batch_results if r['success']]
            
            if successful_batch:
                print(f"\\nBatch Analysis Summary:")
                print(f"  Successful windows: {len(successful_batch)}/{len(batch_results)}")
                
                # Group by label if available
                labeled_results = [r for r in successful_batch if r['label'] != -1]
                if labeled_results:
                    normal_results = [r for r in labeled_results if r['label'] == 0]
                    seizure_results = [r for r in labeled_results if r['label'] == 1]
                    
                    if normal_results:
                        normal_rates = [r['anomaly_rate'] for r in normal_results]
                        print(f"  Normal windows ({len(normal_results)}):")
                        print(f"    Anomaly rate: {np.mean(normal_rates):.2f}% ± {np.std(normal_rates):.2f}%")
                    
                    if seizure_results:
                        seizure_rates = [r['anomaly_rate'] for r in seizure_results]
                        print(f"  Seizure windows ({len(seizure_results)}):")
                        print(f"    Anomaly rate: {np.mean(seizure_rates):.2f}% ± {np.std(seizure_rates):.2f}%")
                    
                    # Statistical test
                    if normal_results and seizure_results:
                        normal_rates = [r['anomaly_rate'] for r in normal_results]
                        seizure_rates = [r['anomaly_rate'] for r in seizure_results]
                        
                        t_stat, p_value = stats.ttest_ind(seizure_rates, normal_rates)
                        discrimination = np.mean(seizure_rates) - np.mean(normal_rates)
                        
                        print(f"\\n  Statistical Analysis:")
                        print(f"    Discrimination: {discrimination:.2f}% (seizure - normal)")
                        print(f"    t-test p-value: {p_value:.6f}")
                        print(f"    Significant difference: {'Yes' if p_value < 0.05 else 'No'} (α = 0.05)")
                
                # Performance metrics
                processing_times = [r['detection_time'] for r in successful_batch]
                print(f"\\n  Performance:")
                print(f"    Average processing time: {np.mean(processing_times):.3f}s per window")
                print(f"    Throughput: {len(successful_batch)/np.sum(processing_times):.1f} windows/second")
                print(f"    Real-time capable: {'Yes' if np.mean(processing_times) < self.window_size_seconds else 'No'}")


def load_config_from_file(config_file: str) -> List[Dict[str, Any]]:
    """Load MERLIN configurations from JSON file."""
    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        if 'configs' in config_data:
            return config_data['configs']
        else:
            return config_data
            
    except Exception as e:
        print(f"❌ Error loading config file: {e}")
        return []


def save_config_template(filename: str = "merlin_config_template.json"):
    """Save a template configuration file."""
    template = {
        "configs": [
            {
                "name": "Short-term (60s-300s)",
                "min_length_seconds": 60,
                "max_length_seconds": 300,
                "description": "Detect short-term anomalies"
            },
            {
                "name": "Long-term (300s-900s)",
                "min_length_seconds": 300,
                "max_length_seconds": 900,
                "description": "Detect long-term anomalies"
            }
        ],
        "fs": 8,
        "window_size_seconds": 3600
    }
    
    with open(filename, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"✓ Configuration template saved to: {filename}")


def main():
    """Main function for flexible MERLIN analysis."""
    parser = argparse.ArgumentParser(description='Flexible MERLIN Anomaly Detection for Various ECG Data Formats')
    
    # Data parameters
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to preprocessed ECG data file')
    parser.add_argument('--fs', type=int, default=125,
                       help='Sampling frequency in Hz (default: 125)')
    parser.add_argument('--window-size', type=float, default=30.0,
                       help='Window size in seconds (default: 30.0)')
    
    # MERLIN configuration
    parser.add_argument('--min-length', type=float, default=None,
                       help='Minimum MERLIN search length in seconds')
    parser.add_argument('--max-length', type=float, default=None,
                       help='Maximum MERLIN search length in seconds')
    parser.add_argument('--config-file', type=str, default=None,
                       help='JSON file with custom MERLIN configurations')
    
    # Analysis parameters
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Number of windows for batch analysis (default: 50)')
    parser.add_argument('--no-batch', action='store_true',
                       help='Skip batch processing analysis')
    parser.add_argument('--max-test-windows', type=int, default=2,
                       help='Maximum number of test windows (default: 2)')
    
    # Other options
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible results (default: 42)')
    parser.add_argument('--save-template', action='store_true',
                       help='Save configuration template file and exit')
    
    args = parser.parse_args()
    
    # Save template and exit if requested
    if args.save_template:
        save_config_template()
        return 0
    
    # Set random seed
    np.random.seed(args.seed)
    
    print("Flexible MERLIN Anomaly Detection")
    print("=" * 50)
    print(f"MERLIN available: {MERLIN_AVAILABLE}")
    
    if not MERLIN_AVAILABLE:
        print("Cannot proceed without MERLIN implementation.")
        return 1
    
    # Load custom configurations if specified
    custom_configs = None
    if args.config_file:
        config_data = load_config_from_file(args.config_file)
        if config_data:
            # Convert time-based configs to sample-based
            custom_configs = []
            for config in config_data:
                if 'min_length_seconds' in config:
                    custom_configs.append({
                        'name': config['name'],
                        'min_length': int(config['min_length_seconds'] * args.fs),
                        'max_length': int(config['max_length_seconds'] * args.fs),
                        'description': config.get('description', '')
                    })
                else:
                    custom_configs.append(config)
    
    # Initialize analyzer
    analyzer = FlexibleMerlinAnalyzer(
        fs=args.fs,
        window_size_seconds=args.window_size,
        min_length_seconds=args.min_length,
        max_length_seconds=args.max_length,
        custom_configs=custom_configs
    )
    
    # Load data
    data = analyzer.load_data(args.data_path)
    if data is None:
        return 1
    
    # Validate configurations
    valid_configs = analyzer.validate_configs()
    if not valid_configs:
        print("\\n❌ No valid MERLIN configurations found!")
        print("Try adjusting --min-length and --max-length parameters or use --save-template")
        return 1
    
    # Test configurations
    results_summary = analyzer.test_configurations(data, valid_configs, args.max_test_windows)
    
    # Batch analysis
    batch_results = None
    if not args.no_batch and valid_configs:
        print(f"\\n{'='*60}")
        print(f"BATCH PROCESSING ANALYSIS")
        print(f"{'='*60}")
        
        # Use first valid configuration for batch analysis
        best_config = valid_configs[0]
        print(f"Using configuration: {best_config['name']}")
        
        batch_results = analyzer.batch_analysis(data, best_config, args.batch_size)
    
    # Analyze and display results
    analyzer.analyze_results(results_summary, batch_results)
    
    print(f"\\n{'='*80}")
    print(f"FLEXIBLE MERLIN ANALYSIS COMPLETED")
    print(f"{'='*80}")
    
    return 0


if __name__ == "__main__":
    exit(main())