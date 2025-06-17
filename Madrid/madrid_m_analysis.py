#!/usr/bin/env python3
"""
MADRID m-Parameter Analysis for Seizure Detection

This script analyzes how different subsequence lengths (m) affect sensitivity
in MADRID seizure detection. It tests individual m values and identifies
the optimal subsequence length for maximum sensitivity.

Key features:
- Test individual m values (subsequence lengths) from a defined range
- Calculate sensitivity for each m parameter
- Identify optimal m value that maximizes sensitivity
- Save human-readable analysis results
- Support for different sampling rates and data characteristics

Usage:
    python madrid_m_analysis.py --data-dir DATA_DIR --num-files 10
    python madrid_m_analysis.py --data-dir DATA_DIR --m-range 50 1000 --step 25
    python madrid_m_analysis.py --data-dir DATA_DIR --percentage 25 --output m_analysis.txt


"""

import numpy as np
import pickle
import time
import sys
import os
import argparse
import json
from pathlib import Path
import warnings
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import pandas as pd
from collections import defaultdict
import random

warnings.filterwarnings('ignore')

# Add the models directory to path to import MADRID
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from models.madrid import MADRID
    print("✓ MADRID successfully imported from models/madrid")
    MADRID_AVAILABLE = True
except ImportError as e:
    print("❌ MADRID import failed")
    print(f"Error: {e}")
    MADRID_AVAILABLE = False


class MadridMAnalyzer:
    """
    Analyzer for MADRID m-parameter (subsequence length) optimization.
    
    Tests individual m values to find optimal subsequence lengths for seizure detection.
    """
    
    def __init__(self, use_gpu: bool = True, min_overlap_threshold: float = 0.1):
        """
        Initialize the m-parameter analyzer.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            min_overlap_threshold: Minimum overlap ratio for true positive detection
        """
        self.use_gpu = use_gpu
        self.min_overlap_threshold = min_overlap_threshold
        self.results = []
        self.m_results = {}
        
        print(f"Initialized MadridMAnalyzer:")
        print(f"  GPU acceleration: {use_gpu}")
        print(f"  Minimum overlap threshold: {min_overlap_threshold}")
    
    def generate_m_values(self, fs: int, m_range: Tuple[int, int] = None, 
                         step: int = None, mode: str = 'comprehensive') -> List[int]:
        """
        Generate m values (subsequence lengths) to test.
        
        Args:
            fs: Sampling frequency
            m_range: (min_m, max_m) range in samples
            step: Step size between m values
            mode: 'quick', 'comprehensive', or 'seizure_focused'
            
        Returns:
            List of m values to test
        """
        if m_range is not None:
            min_m, max_m = m_range
        else:
            # Default ranges based on mode and sampling frequency
            if mode == 'quick':
                min_m = int(0.5 * fs)   # 0.5 seconds
                max_m = int(10.0 * fs)  # 10 seconds
                step = step or max(1, fs // 2)  # 0.5 second steps
            elif mode == 'comprehensive':
                min_m = int(0.1 * fs)   # 0.1 seconds
                max_m = int(30.0 * fs)  # 30 seconds
                step = step or max(1, fs // 8)  # 0.125 second steps
            elif mode == 'seizure_focused':
                min_m = int(2 * fs)   # 2 seconds
                max_m = int(15.0 * fs)  # 15 seconds
                step = step or max(1, fs // 5)  # 0.2 second steps
            else:
                min_m = int(1.0 * fs)   # 1 second
                max_m = int(10.0 * fs)  # 10 seconds
                step = step or max(1, fs // 2)  # 0.5 second steps
        
        # Ensure minimum requirements
        min_m = max(2, min_m)  # MADRID requires at least 2 samples
        step = step or max(1, (max_m - min_m) // 50)  # Default to ~50 values
        
        m_values = list(range(min_m, max_m + 1, step))
        
        print(f"Generated {len(m_values)} m values for {mode} analysis:")
        print(f"  Range: {min_m} - {max_m} samples ({min_m/fs:.2f}s - {max_m/fs:.2f}s)")
        print(f"  Step: {step} samples ({step/fs:.3f}s)")
        
        return m_values
    
    def discover_seizure_files(self, data_dir: str, percentage: float = 100.0, 
                              subject_filter: str = None) -> List[Path]:
        """
        Discover and sample seizure files.
        
        Args:
            data_dir: Directory containing seizure files
            percentage: Percentage of files to process (0-100)
            subject_filter: Optional subject filter
            
        Returns:
            List of selected seizure file paths
        """
        data_path = Path(data_dir)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_path}")
        
        # Find all seizure files
        seizure_files = list(data_path.glob("*_seizure_*_preprocessed.pkl"))
        if not seizure_files:
            seizure_files = list(data_path.glob("*seizure*.pkl"))
            if not seizure_files:
                seizure_files = list(data_path.glob("*.pkl"))
        
        print(f"Found {len(seizure_files)} potential seizure files in {data_path}")
        
        # Apply subject filter
        if subject_filter:
            seizure_files = [f for f in seizure_files if subject_filter in f.name]
            print(f"After subject filter '{subject_filter}': {len(seizure_files)} files")
        
        if not seizure_files:
            raise ValueError("No seizure files found matching criteria")
        
        # Calculate number of files to process
        n_files_to_process = max(1, int(len(seizure_files) * percentage / 100))
        
        # Randomly sample files
        random.seed(42)  # For reproducible results
        selected_files = random.sample(seizure_files, n_files_to_process)
        
        print(f"Selected {len(selected_files)} files ({percentage}% of {len(seizure_files)} total)")
        
        return sorted(selected_files)
    
    def load_seizure_data(self, data_path: str) -> Optional[Dict[str, Any]]:
        """
        Load seizure data and extract seizure regions.
        
        Args:
            data_path: Path to seizure segment pickle file
            
        Returns:
            Dictionary with seizure data and true seizure regions
        """
        data_path = Path(data_path)
        
        try:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            
            # Validate data structure
            if 'channels' not in data or not data['channels']:
                return None
            
            channel = data['channels'][0]
            if 'data' not in channel or 'labels' not in channel:
                return None
            
            # Extract metadata
            fs = data.get('sampling_rate', None)
            if fs is None and 'metadata' in data:
                fs = data['metadata'].get('sampling_rate', 125)
            if fs is None:
                fs = 125  # Default fallback
            
            # Find true seizure regions (ictal phases)
            labels = channel['labels']
            seizure_regions = self._find_seizure_regions(labels, fs)
            
            # Add computed information
            data['sampling_rate'] = fs
            data['true_seizure_regions'] = seizure_regions
            data['file_path'] = str(data_path)
            
            return data
            
        except Exception as e:
            print(f"  Warning: Could not load {data_path.name}: {e}")
            return None
    
    def _find_seizure_regions(self, labels: np.ndarray, fs: int) -> List[Dict[str, Any]]:
        """Find continuous seizure (ictal) regions in the data."""
        seizure_regions = []
        
        # Find ictal segments
        ictal_mask = (labels == 'ictal')
        if not np.any(ictal_mask):
            return seizure_regions
        
        # Find continuous ictal regions
        ictal_indices = np.where(ictal_mask)[0]
        
        if len(ictal_indices) == 0:
            return seizure_regions
        
        # Group consecutive indices
        region_start = ictal_indices[0]
        region_end = ictal_indices[0]
        
        for i in range(1, len(ictal_indices)):
            if ictal_indices[i] == region_end + 1:
                region_end = ictal_indices[i]
            else:
                # Add completed region
                seizure_regions.append({
                    'start_sample': region_start,
                    'end_sample': region_end,
                    'start_time': region_start / fs,
                    'end_time': region_end / fs,
                    'duration': (region_end - region_start + 1) / fs,
                    'n_samples': region_end - region_start + 1
                })
                region_start = ictal_indices[i]
                region_end = ictal_indices[i]
        
        # Add final region
        seizure_regions.append({
            'start_sample': region_start,
            'end_sample': region_end,
            'start_time': region_start / fs,
            'end_time': region_end / fs,
            'duration': (region_end - region_start + 1) / fs,
            'n_samples': region_end - region_start + 1
        })
        
        return seizure_regions
    
    def test_single_m_value(self, data: Dict[str, Any], m: int, 
                           threshold_percentile: float = 90) -> Dict[str, Any]:
        """
        Test a single m value (subsequence length) on one data file.
        
        Args:
            data: Seizure data dictionary
            m: Subsequence length to test
            threshold_percentile: Threshold for anomaly detection
            
        Returns:
            Test results for this m value
        """
        channel = data['channels'][0]
        signal_data = channel['data'].astype(np.float64)
        fs = data['sampling_rate']
        true_seizure_regions = data['true_seizure_regions']
        
        # Check if m is valid for this data
        data_length = len(signal_data)
        min_required = m * 3  # Need enough data for training
        
        if data_length < min_required or m < 2:
            return {
                'success': False,
                'error': f'Data too short for m={m}: {data_length} < {min_required}',
                'm': m,
                'm_seconds': m / fs,
                'file_path': data['file_path']
            }
        
        start_time = time.time()
        
        try:
            # Initialize MADRID with single m value
            detector = MADRID(use_gpu=self.use_gpu, enable_output=False)
            
            # Use 1/3 of data for training
            train_test_split = data_length // 3
            
            # Run detection with single m value (min_length = max_length = m)
            multi_length_table, bsf, bsf_loc = detector.fit(
                T=signal_data,
                min_length=m,
                max_length=m,  # Only test this specific m
                step_size=1,   # Doesn't matter since min=max
                train_test_split=train_test_split
            )
            
            # Get anomaly information
            anomaly_info = detector.get_anomaly_scores(
                threshold_percentile=threshold_percentile
            )
            anomalies = anomaly_info['anomalies']
            
            detection_time = time.time() - start_time
            
            # Calculate sensitivity and other metrics
            performance = self._calculate_performance(anomalies, true_seizure_regions, fs)
            
            return {
                'success': True,
                'm': m,
                'm_seconds': m / fs,
                'file_path': data['file_path'],
                'subject_id': data.get('subject_id', 'unknown'),
                'detection_time': detection_time,
                'n_detected_anomalies': len(anomalies),
                'n_true_seizures': len(true_seizure_regions),
                'sensitivity': performance['sensitivity'],
                'precision': performance['precision'],
                'f1_score': performance['f1_score'],
                'true_positives': performance['true_positives'],
                'false_positives': performance['false_positives'],
                'false_negatives': performance['false_negatives'],
                'discord_score': float(bsf[0]) if len(bsf) > 0 and not np.isnan(bsf[0]) else 0.0,
                'threshold_percentile': threshold_percentile,
                'performance': performance
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'm': m,
                'm_seconds': m / fs,
                'file_path': data['file_path'],
                'detection_time': time.time() - start_time
            }
    
    def _calculate_performance(self, detected_anomalies: List[Dict], 
                             true_seizure_regions: List[Dict], fs: int) -> Dict[str, float]:
        """Calculate performance metrics."""
        if not true_seizure_regions:
            return {
                'sensitivity': 0.0,
                'precision': 0.0 if detected_anomalies else 1.0,
                'f1_score': 0.0,
                'true_positives': 0,
                'false_positives': len(detected_anomalies),
                'false_negatives': 0
            }
        
        # Convert to time intervals
        detected_intervals = []
        for anomaly in detected_anomalies:
            if anomaly['location'] is not None:
                start_time = anomaly['location'] / fs
                end_time = start_time + (anomaly['length'] / fs)
                detected_intervals.append({
                    'start_time': start_time,
                    'end_time': end_time
                })
        
        true_intervals = []
        for region in true_seizure_regions:
            true_intervals.append({
                'start_time': region['start_time'],
                'end_time': region['end_time']
            })
        
        # Calculate overlaps
        true_positives = 0
        false_positives = 0
        
        for detected in detected_intervals:
            has_overlap = False
            for true_seizure in true_intervals:
                overlap = self._calculate_overlap(detected, true_seizure)
                if overlap >= self.min_overlap_threshold:
                    has_overlap = True
                    break
            
            if has_overlap:
                true_positives += 1
            else:
                false_positives += 1
        
        # Calculate which seizures were detected
        seizure_detected = [False] * len(true_intervals)
        for i, true_seizure in enumerate(true_intervals):
            for detected in detected_intervals:
                overlap = self._calculate_overlap(detected, true_seizure)
                if overlap >= self.min_overlap_threshold:
                    seizure_detected[i] = True
                    break
        
        false_negatives = sum(1 for detected in seizure_detected if not detected)
        
        # Calculate metrics
        sensitivity = sum(seizure_detected) / len(seizure_detected) if seizure_detected else 0.0
        precision = true_positives / len(detected_intervals) if detected_intervals else 0.0
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
        
        return {
            'sensitivity': sensitivity,
            'precision': precision,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def _calculate_overlap(self, interval1: Dict, interval2: Dict) -> float:
        """Calculate overlap ratio between two time intervals."""
        start1, end1 = interval1['start_time'], interval1['end_time']
        start2, end2 = interval2['start_time'], interval2['end_time']
        
        # Calculate intersection
        intersection_start = max(start1, start2)
        intersection_end = min(end1, end2)
        
        if intersection_start >= intersection_end:
            return 0.0
        
        intersection_duration = intersection_end - intersection_start
        
        # Calculate overlap as fraction of shorter interval
        duration1 = end1 - start1
        duration2 = end2 - start2
        shorter_duration = min(duration1, duration2)
        
        return intersection_duration / shorter_duration if shorter_duration > 0 else 0.0
    
    def run_m_analysis(self, data_files: List[Dict[str, Any]], m_values: List[int],
                      threshold_percentile: float = 90) -> Dict[str, Any]:
        """
        Run m-parameter analysis across all m values and files.
        
        Args:
            data_files: List of loaded seizure data
            m_values: List of m values to test
            threshold_percentile: Threshold for anomaly detection
            
        Returns:
            Analysis results
        """
        print(f"\n{'='*80}")
        print(f"MADRID m-PARAMETER ANALYSIS")
        print(f"{'='*80}")
        print(f"Files: {len(data_files)}")
        print(f"m values: {len(m_values)}")
        print(f"Total tests: {len(data_files) * len(m_values)}")
        print(f"Threshold percentile: {threshold_percentile}%")
        
        all_results = []
        m_stats = defaultdict(list)
        
        # Test each m value
        for i, m in enumerate(m_values):
            fs = data_files[0]['sampling_rate']  # Assume same fs for all files
            print(f"\nTesting m={m} ({m/fs:.3f}s) [{i+1}/{len(m_values)}]")
            
            m_results = []
            successful_tests = 0
            
            # Test on each file
            for data in data_files:
                result = self.test_single_m_value(data, m, threshold_percentile)
                m_results.append(result)
                all_results.append(result)
                
                if result['success']:
                    successful_tests += 1
                    m_stats[m].append(result['sensitivity'])
            
            # Calculate summary for this m value
            if successful_tests > 0:
                sensitivities = [r['sensitivity'] for r in m_results if r['success']]
                mean_sensitivity = np.mean(sensitivities)
                std_sensitivity = np.std(sensitivities)
                max_sensitivity = np.max(sensitivities)
                min_sensitivity = np.min(sensitivities)
                discord_scores = [r['discord_score'] for r in m_results if r['success']]
                mean_discord_score = np.mean(discord_scores)
                
                print(f"  ✓ {successful_tests}/{len(data_files)} successful")
                print(f"  ✓ Sensitivity: {mean_sensitivity:.4f} ± {std_sensitivity:.4f} (range: {min_sensitivity:.3f}-{max_sensitivity:.3f})")
                print(f"  ✓ Mean discord score: {mean_discord_score:.4f}")
            else:
                print(f"  ❌ All tests failed")
        
        # Find best m value
        best_m = None
        best_sensitivity = 0.0
        
        m_summary = {}
        for m, sensitivities in m_stats.items():
            if sensitivities:
                mean_sens = np.mean(sensitivities)
                m_summary[m] = {
                    'mean_sensitivity': mean_sens,
                    'std_sensitivity': np.std(sensitivities),
                    'max_sensitivity': np.max(sensitivities),
                    'min_sensitivity': np.min(sensitivities),
                    'n_tests': len(sensitivities),
                    'sensitivities': sensitivities,
                    'm_seconds': m / data_files[0]['sampling_rate']
                }
                
                if mean_sens > best_sensitivity:
                    best_sensitivity = mean_sens
                    best_m = m
        
        results = {
            'all_results': all_results,
            'm_summary': m_summary,
            'best_m': best_m,
            'best_sensitivity': best_sensitivity,
            'n_files': len(data_files),
            'n_m_values': len(m_values),
            'm_values': m_values,
            'threshold_percentile': threshold_percentile
        }
        
        self.results = all_results
        self.m_results = results
        
        return results
    
    def save_m_analysis(self, output_file: str, include_detailed: bool = True):
        """
        Save m-parameter analysis results in human-readable format.
        
        Args:
            output_file: Output file path
            include_detailed: Whether to include detailed results
        """
        if not self.m_results:
            print("No results to save")
            return
        
        results = self.m_results
        m_summary = results['m_summary']
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # Header
            f.write("="*80 + "\n")
            f.write("MADRID m-PARAMETER ANALYSIS RESULTS\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Files tested: {results['n_files']}\n")
            f.write(f"m values tested: {results['n_m_values']}\n")
            f.write(f"Total tests: {len(results['all_results'])}\n")
            f.write(f"Threshold percentile: {results['threshold_percentile']}%\n")
            f.write(f"Minimum overlap threshold: {self.min_overlap_threshold}\n")
            f.write(f"GPU acceleration: {self.use_gpu}\n\n")
            
            # Optimal m value
            f.write("OPTIMAL m VALUE\n")
            f.write("-"*20 + "\n")
            if results['best_m']:
                best_m_data = m_summary[results['best_m']]
                
                f.write(f"Best m value: {results['best_m']} samples ({best_m_data['m_seconds']:.3f} seconds)\n")
                f.write(f"Mean sensitivity: {best_m_data['mean_sensitivity']:.4f}\n")
                f.write(f"Standard deviation: {best_m_data['std_sensitivity']:.4f}\n")
                f.write(f"Maximum sensitivity achieved: {best_m_data['max_sensitivity']:.4f}\n")
                f.write(f"Minimum sensitivity: {best_m_data['min_sensitivity']:.4f}\n")
                f.write(f"Number of successful tests: {best_m_data['n_tests']}\n\n")
                
                f.write("Recommended MADRID configuration:\n")
                f.write(f"  madrid.fit(T=data, min_length={results['best_m']}, \n")
                f.write(f"             max_length={results['best_m']}, \n")
                f.write(f"             step_size=1, \n")
                f.write(f"             train_test_split=len(data)//3)\n")
                f.write(f"  anomalies = madrid.get_anomaly_scores(threshold_percentile={results['threshold_percentile']})\n\n")
            else:
                f.write("No successful m values found.\n\n")
            
            # Top 10 m values
            f.write("TOP 10 m VALUES\n")
            f.write("-"*20 + "\n")
            f.write(f"{'Rank':<4} {'m (samples)':<12} {'m (seconds)':<12} {'Mean Sens':<10} {'Std':<8} {'Max Sens':<8} {'Tests':<6}\n")
            f.write("-"*70 + "\n")
            
            # Sort by mean sensitivity
            sorted_m = sorted(m_summary.items(), 
                            key=lambda x: x[1]['mean_sensitivity'], reverse=True)
            
            for rank, (m, stats) in enumerate(sorted_m[:10], 1):
                f.write(f"{rank:<4} {m:<12} {stats['m_seconds']:<12.3f} {stats['mean_sensitivity']:<10.4f} "
                       f"{stats['std_sensitivity']:<8.4f} {stats['max_sensitivity']:<8.4f} {stats['n_tests']:<6}\n")
            
            f.write("\n")
            
            # Sensitivity vs m plot data
            f.write("SENSITIVITY vs m ANALYSIS\n")
            f.write("-"*30 + "\n")
            f.write("m_samples | m_seconds | mean_sensitivity | std_sensitivity | max_sensitivity | n_tests\n")
            f.write("-"*80 + "\n")
            
            for m, stats in sorted(m_summary.items()):
                f.write(f"{m:9d} | {stats['m_seconds']:9.3f} | {stats['mean_sensitivity']:15.4f} | "
                       f"{stats['std_sensitivity']:14.4f} | {stats['max_sensitivity']:14.4f} | {stats['n_tests']:7d}\n")
            
            f.write("\n")
            
            # Time scale analysis
            f.write("TIME SCALE ANALYSIS\n")
            f.write("-"*25 + "\n")
            
            # Group by time ranges
            time_ranges = {
                'very_short': (0.0, 0.5),    # < 0.5s
                'short': (0.5, 2.0),         # 0.5-2s
                'medium': (2.0, 5.0),        # 2-5s
                'long': (5.0, 15.0),         # 5-15s
                'very_long': (15.0, 1000.0)  # > 15s
            }
            
            range_analysis = defaultdict(list)
            for m, stats in m_summary.items():
                m_seconds = stats['m_seconds']
                for range_name, (min_time, max_time) in time_ranges.items():
                    if min_time <= m_seconds < max_time:
                        range_analysis[range_name].append(stats['mean_sensitivity'])
                        break
            
            for range_name, (min_time, max_time) in time_ranges.items():
                sensitivities = range_analysis[range_name]
                if sensitivities:
                    mean_sens = np.mean(sensitivities)
                    max_sens = np.max(sensitivities)
                    f.write(f"{range_name:12s} ({min_time:4.1f}-{max_time:4.1f}s): "
                           f"{mean_sens:.4f} mean, {max_sens:.4f} max (n={len(sensitivities)})\n")
                else:
                    f.write(f"{range_name:12s} ({min_time:4.1f}-{max_time:4.1f}s): No data\n")
            
            f.write("\n")
            
            # Detailed results for each m
            if include_detailed:
                f.write("DETAILED m VALUE RESULTS\n")
                f.write("-"*30 + "\n")
                f.write(f"{'m':<6} {'m(s)':<8} {'Files':<6} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}\n")
                f.write("-"*60 + "\n")
                
                for m, stats in sorted(m_summary.items()):
                    f.write(f"{m:<6} {stats['m_seconds']:<8.3f} {stats['n_tests']:<6} "
                           f"{stats['mean_sensitivity']:<8.4f} {stats['std_sensitivity']:<8.4f} "
                           f"{stats['min_sensitivity']:<8.4f} {stats['max_sensitivity']:<8.4f}\n")
                
                f.write("\n")
            
            # Statistics and insights
            f.write("STATISTICAL INSIGHTS\n")
            f.write("-"*25 + "\n")
            
            if m_summary:
                all_sensitivities = [stats['mean_sensitivity'] for stats in m_summary.values()]
                overall_mean = np.mean(all_sensitivities)
                overall_std = np.std(all_sensitivities)
                overall_max = np.max(all_sensitivities)
                
                f.write(f"Overall sensitivity statistics:\n")
                f.write(f"  Mean: {overall_mean:.4f}\n")
                f.write(f"  Standard deviation: {overall_std:.4f}\n")
                f.write(f"  Maximum: {overall_max:.4f}\n")
                f.write(f"  Range: {np.min(all_sensitivities):.4f} - {overall_max:.4f}\n\n")
                
                # Find patterns
                if overall_max > 0.5:
                    f.write("✓ Good sensitivity achieved (>0.5)\n")
                elif overall_max > 0.2:
                    f.write("⚠ Moderate sensitivity achieved (0.2-0.5)\n")
                else:
                    f.write("❌ Low sensitivity achieved (<0.2)\n")
                
                # Find best time scale
                best_time_scale = None
                best_time_sensitivity = 0.0
                for range_name, sensitivities in range_analysis.items():
                    if sensitivities:
                        mean_sens = np.mean(sensitivities)
                        if mean_sens > best_time_sensitivity:
                            best_time_sensitivity = mean_sens
                            best_time_scale = range_name
                
                if best_time_scale:
                    range_bounds = time_ranges[best_time_scale]
                    f.write(f"✓ Best time scale: {best_time_scale} ({range_bounds[0]:.1f}-{range_bounds[1]:.1f}s)\n")
                
                f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-"*20 + "\n")
            
            if results['best_m'] and results['best_sensitivity'] > 0:
                best_m_seconds = m_summary[results['best_m']]['m_seconds']
                
                f.write(f"1. Use m = {results['best_m']} samples ({best_m_seconds:.3f} seconds)\n")
                f.write(f"   Expected sensitivity: {results['best_sensitivity']:.1%}\n\n")
                
                f.write(f"2. MADRID configuration:\n")
                f.write(f"   madrid = MADRID(use_gpu=True)\n")
                f.write(f"   results = madrid.fit(T=signal, min_length={results['best_m']}, max_length={results['best_m']}, \n")
                f.write(f"                        step_size=1, train_test_split=len(signal)//3)\n")
                f.write(f"   anomalies = madrid.get_anomaly_scores(threshold_percentile={results['threshold_percentile']})\n\n")
                
                # Additional recommendations based on results
                if results['best_sensitivity'] < 0.3:
                    f.write(f"3. Performance improvement suggestions:\n")
                    f.write(f"   - Current best sensitivity ({results['best_sensitivity']:.1%}) is low\n")
                    f.write(f"   - Try lower threshold percentiles (80-85%)\n")
                    f.write(f"   - Consider different overlap thresholds\n")
                    f.write(f"   - Test additional m values around {results['best_m']}\n")
                    f.write(f"   - Verify seizure annotations and data quality\n\n")
                
                # Find consistent performers
                consistent_m = []
                for m, stats in m_summary.items():
                    if stats['std_sensitivity'] < 0.1 and stats['mean_sensitivity'] > 0.1:
                        consistent_m.append((m, stats))
                
                if consistent_m:
                    f.write(f"4. Consistent performers (low std, good sensitivity):\n")
                    for m, stats in sorted(consistent_m, key=lambda x: x[1]['mean_sensitivity'], reverse=True)[:3]:
                        f.write(f"   m={m} ({stats['m_seconds']:.3f}s): {stats['mean_sensitivity']:.3f} ± {stats['std_sensitivity']:.3f}\n")
                    f.write("\n")
                
            else:
                f.write("No m values achieved good sensitivity.\n")
                f.write("Consider:\n")
                f.write("1. Expanding m value search range\n")
                f.write("2. Using different threshold percentiles\n")
                f.write("3. Adjusting overlap thresholds\n")
                f.write("4. Checking data preprocessing and seizure annotations\n")
                f.write("5. Testing with different MADRID parameters\n")
        
        print(f"✓ m-parameter analysis saved to: {output_file}")


def main():
    """Main function for m-parameter analysis."""
    parser = argparse.ArgumentParser(description='MADRID m-Parameter Analysis for Seizure Detection')
    
    # Required arguments
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing seizure preprocessed files')
    
    # File selection
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--num-files', type=int, default=10,
                      help='Number of files to test (default: 10)')
    group.add_argument('--percentage', type=float, default=None,
                      help='Percentage of files to test (0-100)')
    
    # m value configuration
    parser.add_argument('--mode', choices=['quick', 'comprehensive', 'seizure_focused'],
                       default='seizure_focused',
                       help='m value generation mode (default: seizure_focused)')
    parser.add_argument('--m-range', type=int, nargs=2,
                       help='Custom m range in samples (min max)')
    parser.add_argument('--step', type=int,
                       help='Step size between m values')
    
    # Analysis options
    parser.add_argument('--threshold', type=float, default=90,
                       help='Threshold percentile for anomaly detection (default: 90)')
    parser.add_argument('--min-overlap', type=float, default=0.1,
                       help='Minimum overlap threshold for true positive (default: 0.1)')
    
    # Filters
    parser.add_argument('--subject', type=str,
                       help='Filter by subject ID (e.g., sub-001)')
    
    # Output options
    parser.add_argument('--output', type=str, default='madrid_m_analysis.txt',
                       help='Output file for results (default: madrid_m_analysis.txt)')
    parser.add_argument('--no-detailed', action='store_true',
                       help='Skip detailed results in output')
    
    # Technical options
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU acceleration')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for file selection (default: 42)')
    
    args = parser.parse_args()
    
    if not MADRID_AVAILABLE:
        print("Cannot proceed without MADRID implementation.")
        return 1
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("MADRID m-Parameter Analysis for Seizure Detection")
    print("=" * 55)
    print(f"Data directory: {args.data_dir}")
    if args.percentage is not None:
        print(f"Processing: {args.percentage}% of files")
    else:
        print(f"Number of test files: {args.num_files}")
    print(f"Mode: {args.mode}")
    print(f"Threshold percentile: {args.threshold}%")
    print(f"Output file: {args.output}")
    print(f"GPU acceleration: {not args.no_gpu}")
    
    try:
        # Initialize analyzer
        analyzer = MadridMAnalyzer(
            use_gpu=not args.no_gpu,
            min_overlap_threshold=args.min_overlap
        )
        
        # Discover and load files
        if args.percentage is not None:
            seizure_files = analyzer.discover_seizure_files(
                data_dir=args.data_dir,
                percentage=args.percentage,
                subject_filter=args.subject
            )
        else:
            seizure_files = analyzer.discover_seizure_files(
                data_dir=args.data_dir,
                percentage=100.0,
                subject_filter=args.subject
            )
            # Limit to num_files
            seizure_files = seizure_files[:args.num_files]
        
        # Load data files
        data_files = []
        for file_path in seizure_files:
            data = analyzer.load_seizure_data(str(file_path))
            if data is not None:
                data_files.append(data)
        
        if not data_files:
            print("No data files loaded")
            return 1
        
        print(f"Successfully loaded {len(data_files)} data files")
        
        # Generate m values to test
        fs = data_files[0]['sampling_rate']  # Use sampling rate from first file
        
        if args.m_range:
            m_values = analyzer.generate_m_values(
                fs=fs, 
                m_range=tuple(args.m_range),
                step=args.step,
                mode=args.mode
            )
        else:
            m_values = analyzer.generate_m_values(
                fs=fs,
                step=args.step,
                mode=args.mode
            )
        
        if not m_values:
            print("No valid m values generated")
            return 1
        
        # Run m analysis
        results = analyzer.run_m_analysis(
            data_files=data_files,
            m_values=m_values,
            threshold_percentile=args.threshold
        )
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"m-PARAMETER ANALYSIS SUMMARY")
        print(f"{'='*80}")
        
        if results['best_m']:
            best_m_data = results['m_summary'][results['best_m']]
            print(f"Best m value: {results['best_m']} samples ({best_m_data['m_seconds']:.3f}s)")
            print(f"Best mean sensitivity: {results['best_sensitivity']:.4f}")
            
            # Show top 5 m values
            m_summary = results['m_summary']
            sorted_m = sorted(m_summary.items(), 
                            key=lambda x: x[1]['mean_sensitivity'], reverse=True)
            
            print(f"\nTop 5 m values:")
            for i, (m, stats) in enumerate(sorted_m[:5], 1):
                print(f"  {i}. m={m} ({stats['m_seconds']:.3f}s): {stats['mean_sensitivity']:.4f} ± {stats['std_sensitivity']:.4f}")
        else:
            print("No successful m values found")
        
        # Save results
        analyzer.save_m_analysis(
            output_file=args.output,
            include_detailed=not args.no_detailed
        )
        
        print(f"\n✓ m-parameter analysis completed successfully!")
        print(f"✓ Results saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during m-parameter analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())