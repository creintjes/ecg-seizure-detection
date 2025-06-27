#!/usr/bin/env python3
"""
MERLIN Seizure-Focused Anomaly Detection

This script provides focused MERLIN analysis for seizure-only preprocessed ECG data.
It works with the output of preprocess_seizure_only.py and enables efficient testing
of MERLIN parameters on seizure segments with context.

Key features:
- Load seizure-only preprocessed data with phase labels
- Analyze seizure phases separately (pre-seizure, ictal, post-seizure)
- Multiple MERLIN configurations optimized for seizure detection
- Detailed performance analysis with seizure-specific metrics
- Continuous data analysis (no pre-windowing)

Usage:
    python merlin_seizure_analysis.py --data-path DATA_PATH
    python merlin_seizure_analysis.py --data-dir SEIZURE_DATA_DIR --config seizure_optimize
    python merlin_seizure_analysis.py --subject sub-001 --seizure-idx 0
Example:
    python merlin_seizure_analysis.py --data-path /home/swolf/asim_shared/preprocessed_data/seizure_only/8hz_5min/downsample_8hz_context_5min/sub-002_run-01_seizure_00_preprocessed.pk

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
import pandas as pd
from collections import defaultdict

warnings.filterwarnings('ignore')

# Add the Information/Merlin directory to path to import MERLIN
sys.path.append(os.path.join('..','Information', 'Merlin'))

try:
    from _merlin import MERLIN
    print("✓ MERLIN successfully imported from Information/Merlin")
    MERLIN_AVAILABLE = True
except ImportError:
    print("❌ MERLIN import failed")
    print("Please ensure the MERLIN implementation is available in Information/Merlin/")
    MERLIN_AVAILABLE = False


class SeizureFocusedMerlinAnalyzer:
    """
    MERLIN analyzer specialized for seizure-only preprocessed data.
    
    This analyzer works with continuous ECG segments containing seizures
    and their surrounding context, with detailed phase labeling.
    """
    
    def __init__(self, custom_configs: List[Dict[str, Any]] = None):
        """
        Initialize the seizure-focused MERLIN analyzer.
        
        Args:
            custom_configs: List of custom MERLIN configurations
        """
        self.configs = custom_configs if custom_configs else self._get_seizure_optimized_configs()
        
        print(f"Initialized SeizureFocusedMerlinAnalyzer:")
        print(f"  Number of configurations: {len(self.configs)}")
        for config in self.configs:
            print(f"  - {config['name']}: {config['description']}")
    
    def _get_seizure_optimized_configs(self) -> List[Dict[str, Any]]:
        """Get MERLIN configurations optimized for seizure detection."""
        return [
            {
                "name": "ultra_short",
                "min_length_seconds": 1.0,
                "max_length_seconds": 5.0,
                "description": "Detect very short anomalies (1-5s) - cardiac arrhythmias, brief artifacts"
            },
            {
                "name": "short_term", 
                "min_length_seconds": 5.0,
                "max_length_seconds": 30.0,
                "description": "Detect short-term anomalies (5-30s) - brief seizure patterns, autonomic changes"
            },
            {
                "name": "medium_term",
                "min_length_seconds": 30.0,
                "max_length_seconds": 120.0,
                "description": "Detect medium-term anomalies (30s-2min) - seizure onset/offset, autonomic responses"
            },
            {
                "name": "long_term",
                "min_length_seconds": 120.0,
                "max_length_seconds": 300.0,
                "description": "Detect long-term anomalies (2-5min) - prolonged seizures, post-ictal changes"
            },
            {
                "name": "extended_term",
                "min_length_seconds": 300.0,
                "max_length_seconds": 900.0,
                "description": "Detect extended anomalies (5-15min) - long seizures, recovery patterns"
            }
        ]
    
    def load_seizure_data(self, data_path: str) -> Optional[Dict[str, Any]]:
        """
        Load seizure-only preprocessed data.
        
        Args:
            data_path: Path to seizure segment pickle file
            
        Returns:
            Dictionary with seizure segment data and metadata
        """
        data_path = Path(data_path)
        print(f"\nLoading seizure data from: {data_path}")
        
        try:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            
            # Validate seizure data structure
            required_keys = ['subject_id', 'run_id', 'seizure_index', 'channels', 'metadata']
            for key in required_keys:
                if key not in data:
                    raise ValueError(f"Missing required key: {key}")
            
            if not data['channels']:
                raise ValueError("No channels found in seizure data")
            
            channel = data['channels'][0]
            if 'data' not in channel or 'labels' not in channel:
                raise ValueError("Channel missing data or labels")
            
            # Extract metadata
            fs = data.get('sampling_rate', 125)
            total_duration = data['metadata']['total_duration']
            seizure_duration = data['metadata']['seizure_duration']
            
            print("✓ Seizure data loaded successfully")
            print(f"  Subject: {data['subject_id']}")
            print(f"  Run: {data['run_id']}")
            print(f"  Seizure index: {data['seizure_index']}")
            print(f"  Total duration: {total_duration:.1f}s")
            print(f"  Seizure duration: {seizure_duration:.1f}s")
            print(f"  Sampling rate: {fs} Hz")
            print(f"  Channels: {len(data['channels'])}")
            
            # Analyze phase distribution
            labels = channel['labels']
            n_pre = np.sum(labels == 'pre_seizure')
            n_ictal = np.sum(labels == 'ictal')
            n_post = np.sum(labels == 'post_seizure')
            n_normal = np.sum(labels == 'normal')
            n_total = len(labels)
            
            print(f"  Phase distribution:")
            print(f"    Pre-seizure:  {n_pre:,} samples ({n_pre/n_total*100:.1f}%)")
            print(f"    Ictal:        {n_ictal:,} samples ({n_ictal/n_total*100:.1f}%)")
            print(f"    Post-seizure: {n_post:,} samples ({n_post/n_total*100:.1f}%)")
            print(f"    Normal:       {n_normal:,} samples ({n_normal/n_total*100:.1f}%)")
            
            return data
            
        except FileNotFoundError:
            print(f"❌ Seizure data file not found: {data_path}")
            return None
        except Exception as e:
            print(f"❌ Error loading seizure data: {e}")
            return None
    
    def prepare_merlin_configs(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Prepare MERLIN configurations based on data characteristics.
        
        Args:
            data: Seizure data dictionary
            
        Returns:
            List of validated MERLIN configurations
        """
        fs = data.get('sampling_rate', 125)
        total_duration = data['metadata']['total_duration']
        seizure_duration = data['metadata']['seizure_duration']
        channel = data['channels'][0]
        n_samples = len(channel['data'])
        
        print(f"\nPreparing MERLIN configurations:")
        print(f"  Segment length: {n_samples:,} samples ({total_duration:.1f}s)")
        print(f"  Seizure duration: {seizure_duration:.1f}s")
        print(f"  Sampling rate: {fs} Hz")
        
        valid_configs = []
        
        for config in self.configs:
            # Convert time-based to sample-based parameters
            min_length = int(config['min_length_seconds'] * fs)
            max_length = int(config['max_length_seconds'] * fs)
            
            # MERLIN requires at least 2x max_length samples
            min_required = max_length * 2
            
            if n_samples >= min_required:
                valid_config = {
                    'name': config['name'],
                    'description': config['description'],
                    'min_length': min_length,
                    'max_length': max_length,
                    'min_length_seconds': config['min_length_seconds'],
                    'max_length_seconds': config['max_length_seconds']
                }
                valid_configs.append(valid_config)
                print(f"  ✓ {config['name']}: Valid ({min_length}-{max_length} samples)")
            else:
                print(f"  ❌ {config['name']}: Invalid (requires {min_required}, have {n_samples})")
        
        print(f"\nValid configurations: {len(valid_configs)}/{len(self.configs)}")
        return valid_configs
    
    def create_phase_windows(self, data: Dict[str, Any], window_size_seconds: float) -> Dict[str, List]:
        """
        Create fixed-size windows from continuous data, labeled by seizure phase.
        
        Args:
            data: Seizure data dictionary
            window_size_seconds: Window size in seconds
            
        Returns:
            Dictionary with windows organized by phase
        """
        channel = data['channels'][0]
        signal_data = channel['data']
        labels = channel['labels']
        fs = data.get('sampling_rate', 125)
        
        window_size_samples = int(window_size_seconds * fs)
        stride_samples = window_size_samples // 2  # 50% overlap
        
        print(f"\nCreating phase-labeled windows:")
        print(f"  Window size: {window_size_seconds}s ({window_size_samples} samples)")
        print(f"  Stride: {stride_samples} samples (50% overlap)")
        
        # Calculate number of windows
        n_windows = (len(signal_data) - window_size_samples) // stride_samples + 1
        
        if n_windows <= 0:
            print("❌ Signal too short for windowing")
            return {}
        
        phase_windows = {
            'pre_seizure': [],
            'ictal': [],
            'post_seizure': [],
            'normal': [],
            'mixed': []
        }
        
        # Extract windows and determine dominant phase
        for i in range(n_windows):
            start_idx = i * stride_samples
            end_idx = start_idx + window_size_samples
            
            window_data = signal_data[start_idx:end_idx]
            window_labels = labels[start_idx:end_idx]
            
            # Determine dominant phase (>50% of samples)
            unique_labels, counts = np.unique(window_labels, return_counts=True)
            dominant_idx = np.argmax(counts)
            dominant_label = unique_labels[dominant_idx]
            dominant_ratio = counts[dominant_idx] / len(window_labels)
            
            window_info = {
                'data': window_data,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_time': start_idx / fs,
                'end_time': end_idx / fs,
                'dominant_label': dominant_label,
                'dominant_ratio': dominant_ratio,
                'label_distribution': dict(zip(unique_labels, counts))
            }
            
            if dominant_ratio > 0.5:
                phase_windows[dominant_label].append(window_info)
            else:
                phase_windows['mixed'].append(window_info)
        
        # Print window distribution
        print(f"  Window distribution:")
        for phase, windows in phase_windows.items():
            if windows:
                print(f"    {phase:12s}: {len(windows):3d} windows")
        
        return phase_windows
    
    def run_merlin_on_continuous_data(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run MERLIN on continuous seizure segment.
        
        Args:
            data: Seizure data dictionary
            config: MERLIN configuration
            
        Returns:
            Dictionary with detection results
        """
        channel = data['channels'][0]
        signal_data = channel['data'].astype(np.float64)
        labels = channel['labels']
        fs = data.get('sampling_rate', 125)
        
        print(f"\nRunning MERLIN on continuous data:")
        print(f"  Configuration: {config['name']}")
        print(f"  Search range: {config['min_length_seconds']:.1f}s - {config['max_length_seconds']:.1f}s")
        print(f"  Data length: {len(signal_data):,} samples ({len(signal_data)/fs:.1f}s)")
        
        start_time = time.time()
        
        try:
            # Initialize MERLIN
            detector = MERLIN(
                min_length=config['min_length'],
                max_length=config['max_length'],
                max_iterations=500
            )
            
            # Run detection
            anomalies = detector.fit_predict(signal_data)
            detection_time = time.time() - start_time
            
            # Calculate overall statistics
            n_anomalies = np.sum(anomalies)
            anomaly_rate = n_anomalies / len(signal_data) * 100
            
            print(f"  ✓ Detection completed in {detection_time:.3f}s")
            print(f"  ✓ Found {n_anomalies:,} anomaly points ({anomaly_rate:.2f}%)")
            
            # Analyze anomalies by seizure phase
            phase_analysis = self._analyze_anomalies_by_phase(anomalies, labels, fs)
            
            # Find anomaly regions
            anomaly_regions = self._find_anomaly_regions(anomalies, fs)
            
            print(f"  ✓ Found {len(anomaly_regions)} anomaly regions")
            
            return {
                'success': True,
                'anomalies': anomalies,
                'n_anomalies': n_anomalies,
                'anomaly_rate': anomaly_rate,
                'anomaly_regions': anomaly_regions,
                'phase_analysis': phase_analysis,
                'detection_time': detection_time,
                'config': config,
                'data_info': {
                    'subject_id': data['subject_id'],
                    'run_id': data['run_id'],
                    'seizure_index': data['seizure_index'],
                    'total_duration': data['metadata']['total_duration'],
                    'seizure_duration': data['metadata']['seizure_duration']
                }
            }
            
        except Exception as e:
            print(f"  ❌ Detection failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'detection_time': time.time() - start_time,
                'config': config,
                'data_info': {
                    'subject_id': data['subject_id'],
                    'run_id': data['run_id'],
                    'seizure_index': data['seizure_index']
                }
            }
    
    def _analyze_anomalies_by_phase(self, anomalies: np.ndarray, labels: np.ndarray, fs: int) -> Dict[str, Any]:
        """Analyze anomaly distribution across seizure phases."""
        phases = ['pre_seizure', 'ictal', 'post_seizure', 'normal']
        analysis = {}
        
        for phase in phases:
            phase_mask = (labels == phase)
            if np.any(phase_mask):
                phase_anomalies = anomalies[phase_mask]
                n_phase_samples = np.sum(phase_mask)
                n_phase_anomalies = np.sum(phase_anomalies)
                phase_anomaly_rate = n_phase_anomalies / n_phase_samples * 100
                
                analysis[phase] = {
                    'n_samples': n_phase_samples,
                    'n_anomalies': n_phase_anomalies,
                    'anomaly_rate': phase_anomaly_rate,
                    'duration': n_phase_samples / fs
                }
            else:
                analysis[phase] = {
                    'n_samples': 0,
                    'n_anomalies': 0,
                    'anomaly_rate': 0.0,
                    'duration': 0.0
                }
        
        return analysis
    
    def _find_anomaly_regions(self, anomalies: np.ndarray, fs: int) -> List[Dict]:
        """Find continuous anomaly regions."""
        if np.sum(anomalies) == 0:
            return []
        
        regions = []
        anomaly_indices = np.where(anomalies)[0]
        
        if len(anomaly_indices) == 0:
            return []
        
        # Group consecutive indices
        region_start = anomaly_indices[0]
        region_end = anomaly_indices[0]
        
        for i in range(1, len(anomaly_indices)):
            if anomaly_indices[i] == region_end + 1:
                region_end = anomaly_indices[i]
            else:
                # Add completed region
                regions.append({
                    'start_sample': region_start,
                    'end_sample': region_end,
                    'start_time': region_start / fs,
                    'end_time': region_end / fs,
                    'duration': (region_end - region_start + 1) / fs,
                    'n_samples': region_end - region_start + 1
                })
                region_start = anomaly_indices[i]
                region_end = anomaly_indices[i]
        
        # Add final region
        regions.append({
            'start_sample': region_start,
            'end_sample': region_end,
            'start_time': region_start / fs,
            'end_time': region_end / fs,
            'duration': (region_end - region_start + 1) / fs,
            'n_samples': region_end - region_start + 1
        })
        
        return regions
    
    def run_windowed_analysis(self, data: Dict[str, Any], config: Dict[str, Any], 
                            window_size_seconds: float = 30.0) -> Dict[str, Any]:
        """
        Run MERLIN on windowed data for detailed phase analysis.
        
        Args:
            data: Seizure data dictionary
            config: MERLIN configuration
            window_size_seconds: Window size for analysis
            
        Returns:
            Dictionary with windowed analysis results
        """
        print(f"\n{'='*60}")
        print(f"WINDOWED PHASE ANALYSIS")
        print(f"{'='*60}")
        
        # Create phase-labeled windows
        phase_windows = self.create_phase_windows(data, window_size_seconds)
        
        if not any(phase_windows.values()):
            print("❌ No windows available for analysis")
            return {}
        
        # Initialize MERLIN
        detector = MERLIN(
            min_length=config['min_length'],
            max_length=config['max_length'],
            max_iterations=500
        )
        
        results = {
            'config': config,
            'window_size_seconds': window_size_seconds,
            'phase_results': {},
            'summary': {}
        }
        
        print(f"\nRunning MERLIN on windowed data ({config['name']}):")
        
        # Process each phase
        for phase, windows in phase_windows.items():
            if not windows:
                continue
                
            print(f"\n  Processing {phase} phase ({len(windows)} windows):")
            
            phase_results = []
            successful = 0
            
            for i, window_info in enumerate(windows):
                try:
                    start_time = time.time()
                    anomalies = detector.fit_predict(window_info['data'].astype(np.float64))
                    detection_time = time.time() - start_time
                    
                    n_anomalies = np.sum(anomalies)
                    anomaly_rate = n_anomalies / len(window_info['data']) * 100
                    
                    phase_results.append({
                        'window_idx': i,
                        'start_time': window_info['start_time'],
                        'end_time': window_info['end_time'],
                        'dominant_label': window_info['dominant_label'],
                        'dominant_ratio': window_info['dominant_ratio'],
                        'n_anomalies': n_anomalies,
                        'anomaly_rate': anomaly_rate,
                        'detection_time': detection_time,
                        'success': True
                    })
                    successful += 1
                    
                except Exception as e:
                    phase_results.append({
                        'window_idx': i,
                        'start_time': window_info['start_time'],
                        'end_time': window_info['end_time'],
                        'error': str(e),
                        'success': False
                    })
            
            results['phase_results'][phase] = phase_results
            
            if successful > 0:
                successful_results = [r for r in phase_results if r['success']]
                avg_anomaly_rate = np.mean([r['anomaly_rate'] for r in successful_results])
                std_anomaly_rate = np.std([r['anomaly_rate'] for r in successful_results])
                avg_detection_time = np.mean([r['detection_time'] for r in successful_results])
                
                print(f"    ✓ {successful}/{len(windows)} windows successful")
                print(f"    ✓ Avg anomaly rate: {avg_anomaly_rate:.2f}% ± {std_anomaly_rate:.2f}%")
                print(f"    ✓ Avg detection time: {avg_detection_time:.3f}s")
                
                results['summary'][phase] = {
                    'n_windows': len(windows),
                    'n_successful': successful,
                    'avg_anomaly_rate': avg_anomaly_rate,
                    'std_anomaly_rate': std_anomaly_rate,
                    'avg_detection_time': avg_detection_time
                }
            else:
                print(f"    ❌ All windows failed")
                results['summary'][phase] = {
                    'n_windows': len(windows),
                    'n_successful': 0
                }
        
        return results
    
    def analyze_results(self, continuous_results: List[Dict], windowed_results: List[Dict] = None):
        """
        Analyze and display comprehensive results.
        
        Args:
            continuous_results: Results from continuous analysis
            windowed_results: Results from windowed analysis (optional)
        """
        print(f"\n{'='*80}")
        print(f"SEIZURE-FOCUSED MERLIN ANALYSIS RESULTS")
        print(f"{'='*80}")
        
        if not continuous_results:
            print("No results to analyze")
            return
        
        # Continuous analysis summary
        print(f"\nCONTINUOUS ANALYSIS SUMMARY:")
        print(f"{'Config':<15} | {'Subject':<8} | {'Seizure':<7} | {'Anomaly %':<9} | {'Time (s)':<8} | {'Regions':<7}")
        print("-" * 80)
        
        successful_results = [r for r in continuous_results if r['success']]
        
        for result in continuous_results:
            if result['success']:
                data_info = result['data_info']
                print(f"{result['config']['name']:<15} | "
                      f"{data_info['subject_id']:<8} | "
                      f"{data_info['seizure_index']:<7} | "
                      f"{result['anomaly_rate']:<9.2f} | "
                      f"{result['detection_time']:<8.3f} | "
                      f"{len(result['anomaly_regions']):<7}")
            else:
                print(f"{result['config']['name']:<15} | FAILED")
        
        if successful_results:
            # Phase-wise analysis
            print(f"\nPHASE-WISE ANOMALY ANALYSIS:")
            print(f"{'Config':<15} | {'Pre-seizure':<12} | {'Ictal':<8} | {'Post-seizure':<13} | {'Normal':<8}")
            print("-" * 70)
            
            for result in successful_results:
                phase_analysis = result['phase_analysis']
                print(f"{result['config']['name']:<15} | "
                      f"{phase_analysis['pre_seizure']['anomaly_rate']:<12.2f} | "
                      f"{phase_analysis['ictal']['anomaly_rate']:<8.2f} | "
                      f"{phase_analysis['post_seizure']['anomaly_rate']:<13.2f} | "
                      f"{phase_analysis['normal']['anomaly_rate']:<8.2f}")
            
            # Statistical analysis
            if len(successful_results) > 1:
                print(f"\nSTATISTICAL ANALYSIS:")
                
                # Group results by configuration
                config_groups = defaultdict(list)
                for result in successful_results:
                    config_groups[result['config']['name']].append(result)
                
                for config_name, config_results in config_groups.items():
                    if len(config_results) > 1:
                        ictal_rates = [r['phase_analysis']['ictal']['anomaly_rate'] 
                                     for r in config_results]
                        normal_rates = [r['phase_analysis']['normal']['anomaly_rate'] 
                                      for r in config_results]
                        
                        print(f"\n  {config_name}:")
                        print(f"    Ictal anomaly rate: {np.mean(ictal_rates):.2f}% ± {np.std(ictal_rates):.2f}%")
                        print(f"    Normal anomaly rate: {np.mean(normal_rates):.2f}% ± {np.std(normal_rates):.2f}%")
                        
                        if len(ictal_rates) > 1 and len(normal_rates) > 1:
                            t_stat, p_value = stats.ttest_ind(ictal_rates, normal_rates)
                            discrimination = np.mean(ictal_rates) - np.mean(normal_rates)
                            print(f"    Discrimination: {discrimination:.2f}% (ictal - normal)")
                            print(f"    t-test p-value: {p_value:.6f}")
            
            # Performance summary
            detection_times = [r['detection_time'] for r in successful_results]
            total_durations = [r['data_info']['total_duration'] for r in successful_results]
            
            print(f"\nPERFORMANCE SUMMARY:")
            print(f"  Successful analyses: {len(successful_results)}/{len(continuous_results)}")
            print(f"  Average detection time: {np.mean(detection_times):.3f}s")
            print(f"  Average segment duration: {np.mean(total_durations):.1f}s")
            print(f"  Real-time capable: {'Yes' if np.mean(detection_times) < np.mean(total_durations) else 'No'}")
        
        # Windowed analysis summary
        if windowed_results:
            print(f"\n{'='*60}")
            print(f"WINDOWED ANALYSIS SUMMARY")
            print(f"{'='*60}")
            
            for result in windowed_results:
                if 'summary' in result:
                    print(f"\nConfiguration: {result['config']['name']}")
                    for phase, summary in result['summary'].items():
                        if summary['n_successful'] > 0:
                            print(f"  {phase:12s}: {summary['avg_anomaly_rate']:.2f}% ± "
                                  f"{summary['std_anomaly_rate']:.2f}% "
                                  f"({summary['n_successful']}/{summary['n_windows']} windows)")


def discover_seizure_files(data_dir: str, subject_filter: str = None, 
                          seizure_filter: int = None) -> List[Path]:
    """
    Discover seizure-only preprocessed files.
    
    Args:
        data_dir: Directory containing seizure files
        subject_filter: Filter by subject ID (e.g., 'sub-001')
        seizure_filter: Filter by seizure index
        
    Returns:
        List of seizure file paths
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ Data directory not found: {data_path}")
        return []
    
    # Find all seizure files
    seizure_files = list(data_path.glob("*_seizure_*_preprocessed.pkl"))
    
    print(f"Found {len(seizure_files)} seizure files in {data_path}")
    
    # Apply filters
    if subject_filter:
        seizure_files = [f for f in seizure_files if subject_filter in f.name]
        print(f"After subject filter '{subject_filter}': {len(seizure_files)} files")
    
    if seizure_filter is not None:
        seizure_pattern = f"_seizure_{seizure_filter:02d}_"
        seizure_files = [f for f in seizure_files if seizure_pattern in f.name]
        print(f"After seizure filter '{seizure_filter}': {len(seizure_files)} files")
    
    return sorted(seizure_files)


def load_custom_configs(config_file: str) -> List[Dict[str, Any]]:
    """Load custom MERLIN configurations from JSON file."""
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


def save_config_template(filename: str = "merlin_seizure_config_template.json"):
    """Save a template configuration file for seizure analysis."""
    template = {
        "configs": [
            {
                "name": "seizure_short",
                "min_length_seconds": 2.0,
                "max_length_seconds": 15.0,
                "description": "Detect short seizure-related anomalies"
            },
            {
                "name": "seizure_medium",
                "min_length_seconds": 15.0,
                "max_length_seconds": 60.0,
                "description": "Detect medium-duration seizure patterns"
            },
            {
                "name": "seizure_long",
                "min_length_seconds": 60.0,
                "max_length_seconds": 300.0,
                "description": "Detect long seizure and recovery patterns"
            }
        ]
    }
    
    with open(filename, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"✓ Seizure analysis configuration template saved to: {filename}")


def main():
    """Main function for seizure-focused MERLIN analysis."""
    parser = argparse.ArgumentParser(description='MERLIN Seizure-Focused Anomaly Detection')
    
    # Data parameters
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--data-path', type=str,
                      help='Path to single seizure preprocessed file')
    group.add_argument('--data-dir', type=str,
                      help='Directory containing seizure preprocessed files')
    
    # Filters for batch processing
    parser.add_argument('--subject', type=str,
                       help='Filter by subject ID (e.g., sub-001)')
    parser.add_argument('--seizure-idx', type=int,
                       help='Filter by seizure index')
    parser.add_argument('--max-files', type=int, default=10,
                       help='Maximum number of files to process (default: 10)')
    
    # MERLIN configuration
    parser.add_argument('--config', choices=['seizure_optimized', 'ultra_fast', 'comprehensive'],
                       default='seizure_optimized',
                       help='Predefined configuration preset')
    parser.add_argument('--config-file', type=str,
                       help='JSON file with custom MERLIN configurations')
    parser.add_argument('--single-config', type=str,
                       help='Run only specific configuration by name')
    
    # Analysis options
    parser.add_argument('--windowed-analysis', action='store_true',
                       help='Also run windowed phase analysis')
    parser.add_argument('--window-size', type=float, default=30.0,
                       help='Window size for windowed analysis (default: 30.0s)')
    
    # Other options
    parser.add_argument('--save-template', action='store_true',
                       help='Save configuration template file and exit')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Save template and exit if requested
    if args.save_template:
        save_config_template()
        return 0
    
    print("MERLIN Seizure-Focused Anomaly Detection")
    print("=" * 50)
    print(f"MERLIN available: {MERLIN_AVAILABLE}")
    
    if not MERLIN_AVAILABLE:
        print("Cannot proceed without MERLIN implementation.")
        return 1
    
    # Load custom configurations if specified
    custom_configs = None
    if args.config_file:
        custom_configs = load_custom_configs(args.config_file)
        if not custom_configs:
            print("Failed to load custom configurations")
            return 1
    
    # Initialize analyzer
    analyzer = SeizureFocusedMerlinAnalyzer(custom_configs=custom_configs)
    
    # Get list of files to process
    if args.data_path:
        seizure_files = [Path(args.data_path)]
    else:
        seizure_files = discover_seizure_files(
            args.data_dir, 
            subject_filter=args.subject,
            seizure_filter=args.seizure_idx
        )
        
        if not seizure_files:
            print("No seizure files found matching criteria")
            return 1
        
        # Limit number of files
        if len(seizure_files) > args.max_files:
            seizure_files = seizure_files[:args.max_files]
            print(f"Processing first {args.max_files} files")
    
    print(f"\nProcessing {len(seizure_files)} seizure files")
    
    # Process each seizure file
    all_continuous_results = []
    all_windowed_results = []
    
    for i, seizure_file in enumerate(seizure_files):
        print(f"\n{'='*80}")
        print(f"PROCESSING FILE {i+1}/{len(seizure_files)}: {seizure_file.name}")
        print(f"{'='*80}")
        
        # Load seizure data
        data = analyzer.load_seizure_data(str(seizure_file))
        if data is None:
            continue
        
        # Prepare MERLIN configurations
        valid_configs = analyzer.prepare_merlin_configs(data)
        if not valid_configs:
            print("No valid MERLIN configurations for this data")
            continue
        
        # Filter to single config if specified
        if args.single_config:
            valid_configs = [c for c in valid_configs if c['name'] == args.single_config]
            if not valid_configs:
                print(f"Configuration '{args.single_config}' not found or invalid")
                continue
        
        # Run continuous analysis
        print(f"\n{'='*60}")
        print(f"CONTINUOUS SEIZURE SEGMENT ANALYSIS")
        print(f"{'='*60}")
        
        for config in valid_configs:
            result = analyzer.run_merlin_on_continuous_data(data, config)
            all_continuous_results.append(result)
            
            if result['success'] and args.verbose:
                # Show sample anomaly regions
                regions = result['anomaly_regions']
                if regions:
                    print(f"  Sample anomaly regions:")
                    for j, region in enumerate(regions[:5]):
                        print(f"    {j+1}. {region['start_time']:.2f}s - "
                              f"{region['end_time']:.2f}s ({region['duration']:.2f}s)")
                    if len(regions) > 5:
                        print(f"    ... and {len(regions) - 5} more regions")
        
        # Run windowed analysis if requested
        if args.windowed_analysis and valid_configs:
            result = analyzer.run_windowed_analysis(
                data, valid_configs[0], args.window_size
            )
            if result:
                all_windowed_results.append(result)
    
    # Analyze and display results
    if all_continuous_results:
        analyzer.analyze_results(all_continuous_results, all_windowed_results)
    else:
        print("\nNo successful analyses to display")
    
    print(f"\n{'='*80}")
    print(f"SEIZURE-FOCUSED MERLIN ANALYSIS COMPLETED")
    print(f"{'='*80}")
    
    return 0


if __name__ == "__main__":
    exit(main())