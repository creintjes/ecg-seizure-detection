#!/usr/bin/env python3
"""
MADRID Batch Evaluation for Seizure Detection

This script performs batch evaluation of MADRID on multiple seizure files,
calculating sensitivity metrics and saving human-readable results.

Key features:
- Process a percentage of files from a given directory
- Track whether detected anomalies overlap with true seizure regions
- Calculate sensitivity, specificity, and other performance metrics
- Save detailed results in human-readable format
- Support for different MADRID configurations

Usage:
    python madrid_batch_evaluation.py --data-dir DATA_DIR --percentage 50
    python madrid_batch_evaluation.py --data-dir DATA_DIR --config ictal_focused --output results.txt
    python madrid_batch_evaluation.py --data-dir DATA_DIR --percentage 25 --min-overlap 0.5

Author: Generated for seizure detection evaluation
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


class MadridBatchEvaluator:
    """
    Batch evaluator for MADRID seizure detection performance.
    
    Processes multiple seizure files and calculates aggregated performance metrics.
    """
    
    def __init__(self, use_gpu: bool = True, min_overlap_threshold: float = 0.1):
        """
        Initialize the batch evaluator.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            min_overlap_threshold: Minimum overlap ratio to consider detection as true positive
        """
        self.use_gpu = use_gpu
        self.min_overlap_threshold = min_overlap_threshold
        self.results = []
        self.summary_stats = {}
        
        print(f"Initialized MadridBatchEvaluator:")
        print(f"  GPU acceleration: {use_gpu}")
        print(f"  Minimum overlap threshold: {min_overlap_threshold}")
    
    def get_madrid_configs(self) -> List[Dict[str, Any]]:
        """Get predefined MADRID configurations for seizure detection."""
        return [
            {
                "name": "ultra_sensitive",
                "min_length_seconds": 0.2,
                "max_length_seconds": 2.0,
                "step_size_seconds": 0.1,
                "threshold_percentile": 85,
                "description": "Ultra-sensitive detection for early seizure signs"
            },
            {
                "name": "ictal_focused", 
                "min_length_seconds": 1.0,
                "max_length_seconds": 10.0,
                "step_size_seconds": 0.5,
                "threshold_percentile": 90,
                "description": "Focused on ictal phase detection"
            },
            {
                "name": "robust_detection",
                "min_length_seconds": 0.5,
                "max_length_seconds": 30.0,
                "step_size_seconds": 0.25,
                "threshold_percentile": 95,
                "description": "Robust detection with low false positive rate"
            },
            {
                "name": "comprehensive",
                "min_length_seconds": 0.5,
                "max_length_seconds": 60.0,
                "step_size_seconds": 1.0,
                "threshold_percentile": 92,
                "description": "Comprehensive multi-scale detection"
            }
        ]
    
    def discover_seizure_files(self, data_dir: str, percentage: float, 
                              subject_filter: str = None) -> List[Path]:
        """
        Discover and randomly sample seizure files.
        
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
            # Try alternative patterns
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
                print(f"  ❌ Invalid data structure in {data_path.name}")
                return None
            
            channel = data['channels'][0]
            if 'data' not in channel or 'labels' not in channel:
                print(f"  ❌ Missing data or labels in {data_path.name}")
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
            print(f"  ❌ Error loading {data_path.name}: {e}")
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
    
    def prepare_madrid_config(self, config: Dict[str, Any], fs: int, 
                            data_length: int) -> Optional[Dict[str, Any]]:
        """
        Prepare and validate MADRID configuration for the data.
        
        Args:
            config: Configuration template
            fs: Sampling frequency
            data_length: Length of data in samples
            
        Returns:
            Validated configuration or None if invalid
        """
        # Convert time-based to sample-based parameters
        min_length = int(config['min_length_seconds'] * fs)
        max_length = int(config['max_length_seconds'] * fs)
        step_size = int(config['step_size_seconds'] * fs)
        
        # Ensure minimum requirements
        min_length = max(2, min_length)
        min_required = max_length * 3  # Need enough data for training
        
        if data_length < min_required or min_length >= max_length:
            return None
        
        return {
            'name': config['name'],
            'description': config['description'],
            'min_length': min_length,
            'max_length': max_length,
            'step_size': step_size,
            'threshold_percentile': config['threshold_percentile'],
            'sampling_rate': fs
        }
    
    def run_madrid_detection(self, data: Dict[str, Any], 
                           config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run MADRID detection on a single file.
        
        Args:
            data: Seizure data dictionary
            config: MADRID configuration
            
        Returns:
            Detection results dictionary
        """
        channel = data['channels'][0]
        signal_data = channel['data'].astype(np.float64)
        fs = data['sampling_rate']
        true_seizure_regions = data['true_seizure_regions']
        
        start_time = time.time()
        
        try:
            # Initialize MADRID
            detector = MADRID(use_gpu=self.use_gpu, enable_output=False)
            
            # Use 1/3 of data for training
            train_test_split = len(signal_data) // 3
            
            # Run detection
            multi_length_table, bsf, bsf_loc = detector.fit(
                T=signal_data,
                min_length=config['min_length'],
                max_length=config['max_length'],
                step_size=config['step_size'],
                train_test_split=train_test_split
            )
            
            # Get anomaly information
            anomaly_info = detector.get_anomaly_scores(
                threshold_percentile=config['threshold_percentile']
            )
            anomalies = anomaly_info['anomalies']
            
            detection_time = time.time() - start_time
            
            # Analyze detection performance
            performance = self._analyze_detection_performance(
                anomalies, true_seizure_regions, fs, len(signal_data)
            )
            
            return {
                'success': True,
                'file_path': data['file_path'],
                'subject_id': data.get('subject_id', 'unknown'),
                'run_id': data.get('run_id', 'unknown'),
                'seizure_index': data.get('seizure_index', 0),
                'config_name': config['name'],
                'detection_time': detection_time,
                'n_detected_anomalies': len(anomalies),
                'n_true_seizures': len(true_seizure_regions),
                'true_seizure_regions': true_seizure_regions,
                'detected_anomalies': anomalies,
                'performance': performance,
                'data_info': {
                    'sampling_rate': fs,
                    'duration': len(signal_data) / fs,
                    'n_samples': len(signal_data)
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'file_path': data['file_path'],
                'config_name': config['name'],
                'error': str(e),
                'detection_time': time.time() - start_time
            }
    
    def _analyze_detection_performance(self, detected_anomalies: List[Dict], 
                                     true_seizure_regions: List[Dict], 
                                     fs: int, total_samples: int) -> Dict[str, Any]:
        """
        Analyze detection performance against true seizure regions.
        
        Args:
            detected_anomalies: List of detected anomalies
            true_seizure_regions: List of true seizure regions
            fs: Sampling frequency
            total_samples: Total number of samples
            
        Returns:
            Performance metrics dictionary
        """
        if not true_seizure_regions:
            # No true seizures - all detections are false positives
            return {
                'true_positives': 0,
                'false_positives': len(detected_anomalies),
                'false_negatives': 0,
                'true_negatives': 1,  # The entire signal is true negative
                'sensitivity': 0.0,
                'specificity': 1.0 if len(detected_anomalies) == 0 else 0.0,
                'precision': 0.0,
                'f1_score': 0.0,
                'detected_seizure_overlap': []
            }
        
        # Convert anomalies to time intervals
        detected_intervals = []
        for anomaly in detected_anomalies:
            if anomaly['location'] is not None:
                start_time = anomaly['location'] / fs
                end_time = start_time + (anomaly['length'] / fs)
                detected_intervals.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'score': anomaly['score'],
                    'original_anomaly': anomaly
                })
        
        # Convert true seizures to time intervals
        true_intervals = []
        for region in true_seizure_regions:
            true_intervals.append({
                'start_time': region['start_time'],
                'end_time': region['end_time']
            })
        
        # Calculate overlaps
        detected_overlaps = []
        true_positives = 0
        false_positives = 0
        
        for detected in detected_intervals:
            max_overlap = 0.0
            overlapping_seizure = None
            
            for i, true_seizure in enumerate(true_intervals):
                overlap = self._calculate_overlap(detected, true_seizure)
                if overlap > max_overlap:
                    max_overlap = overlap
                    overlapping_seizure = i
            
            detected_overlaps.append({
                'detected_anomaly': detected,
                'max_overlap': max_overlap,
                'overlapping_seizure_idx': overlapping_seizure,
                'is_true_positive': max_overlap >= self.min_overlap_threshold
            })
            
            if max_overlap >= self.min_overlap_threshold:
                true_positives += 1
            else:
                false_positives += 1
        
        # Calculate false negatives (seizures with no significant detection)
        seizure_detected = [False] * len(true_intervals)
        for overlap_info in detected_overlaps:
            if overlap_info['is_true_positive'] and overlap_info['overlapping_seizure_idx'] is not None:
                seizure_detected[overlap_info['overlapping_seizure_idx']] = True
        
        false_negatives = sum(1 for detected in seizure_detected if not detected)
        
        # Calculate metrics
        sensitivity = true_positives / len(true_intervals) if true_intervals else 0.0
        precision = true_positives / len(detected_intervals) if detected_intervals else 0.0
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
        
        # Specificity is harder to calculate for anomaly detection
        # We approximate it based on the assumption that most of the signal should be normal
        total_seizure_duration = sum(region['duration'] for region in true_seizure_regions)
        total_duration = total_samples / fs
        normal_duration = total_duration - total_seizure_duration
        
        # Estimate true negatives (simplified)
        false_positive_duration = sum(
            (det['end_time'] - det['start_time']) for det in detected_intervals
            if not any(overlap['is_true_positive'] and overlap['detected_anomaly'] == det 
                      for overlap in detected_overlaps)
        )
        
        true_negatives_approx = max(0, normal_duration - false_positive_duration)
        specificity = true_negatives_approx / normal_duration if normal_duration > 0 else 1.0
        
        return {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_negatives_approx': true_negatives_approx,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1_score,
            'detected_seizure_overlap': detected_overlaps,
            'seizure_detection_rate': sum(seizure_detected) / len(seizure_detected) if seizure_detected else 0.0
        }
    
    def _calculate_overlap(self, interval1: Dict, interval2: Dict) -> float:
        """
        Calculate overlap ratio between two time intervals.
        
        Returns overlap as fraction of the shorter interval.
        """
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
    
    def process_batch(self, data_dir: str, percentage: float, 
                     config_names: List[str] = None,
                     subject_filter: str = None) -> List[Dict[str, Any]]:
        """
        Process a batch of seizure files.
        
        Args:
            data_dir: Directory containing seizure files
            percentage: Percentage of files to process
            config_names: List of configuration names to use
            subject_filter: Optional subject filter
            
        Returns:
            List of results for all files and configurations
        """
        print(f"\n{'='*80}")
        print(f"MADRID BATCH EVALUATION")
        print(f"{'='*80}")
        
        # Discover files
        seizure_files = self.discover_seizure_files(data_dir, percentage, subject_filter)
        
        # Get configurations
        all_configs = self.get_madrid_configs()
        if config_names:
            configs = [c for c in all_configs if c['name'] in config_names]
        else:
            configs = all_configs
        
        print(f"Using {len(configs)} MADRID configurations:")
        for config in configs:
            print(f"  - {config['name']}: {config['description']}")
        
        # Process files
        all_results = []
        
        for i, seizure_file in enumerate(seizure_files):
            print(f"\n{'='*60}")
            print(f"PROCESSING FILE {i+1}/{len(seizure_files)}: {seizure_file.name}")
            print(f"{'='*60}")
            
            # Load data
            data = self.load_seizure_data(str(seizure_file))
            if data is None:
                continue
            
            print(f"  ✓ Loaded: {data.get('subject_id', 'unknown')} - "
                  f"{len(data['true_seizure_regions'])} true seizures")
            
            # Process with each configuration
            for config_template in configs:
                print(f"\n  Running {config_template['name']}...")
                
                # Prepare configuration
                config = self.prepare_madrid_config(
                    config_template, 
                    data['sampling_rate'], 
                    len(data['channels'][0]['data'])
                )
                
                if config is None:
                    print(f"    ❌ Invalid configuration for this data")
                    continue
                
                # Run detection
                result = self.run_madrid_detection(data, config)
                all_results.append(result)
                
                if result['success']:
                    perf = result['performance']
                    print(f"    ✓ Detected {result['n_detected_anomalies']} anomalies")
                    print(f"    ✓ Sensitivity: {perf['sensitivity']:.3f}")
                    print(f"    ✓ Precision: {perf['precision']:.3f}")
                    print(f"    ✓ F1-Score: {perf['f1_score']:.3f}")
                else:
                    print(f"    ❌ Failed: {result.get('error', 'Unknown error')}")
        
        self.results = all_results
        return all_results
    
    def calculate_aggregate_metrics(self) -> Dict[str, Any]:
        """Calculate aggregate performance metrics across all results."""
        if not self.results:
            return {}
        
        # Group results by configuration
        config_results = defaultdict(list)
        for result in self.results:
            if result['success']:
                config_results[result['config_name']].append(result)
        
        aggregate_metrics = {}
        
        for config_name, results in config_results.items():
            if not results:
                continue
            
            # Aggregate metrics
            sensitivities = [r['performance']['sensitivity'] for r in results]
            specificities = [r['performance']['specificity'] for r in results]
            precisions = [r['performance']['precision'] for r in results]
            f1_scores = [r['performance']['f1_score'] for r in results]
            
            # Count totals
            total_true_seizures = sum(r['n_true_seizures'] for r in results)
            total_detected_anomalies = sum(r['n_detected_anomalies'] for r in results)
            total_true_positives = sum(r['performance']['true_positives'] for r in results)
            total_false_positives = sum(r['performance']['false_positives'] for r in results)
            total_false_negatives = sum(r['performance']['false_negatives'] for r in results)
            
            # Overall metrics
            overall_sensitivity = total_true_positives / total_true_seizures if total_true_seizures > 0 else 0.0
            overall_precision = total_true_positives / total_detected_anomalies if total_detected_anomalies > 0 else 0.0
            overall_f1 = 2 * (overall_precision * overall_sensitivity) / (overall_precision + overall_sensitivity) if (overall_precision + overall_sensitivity) > 0 else 0.0
            
            # Detection times
            detection_times = [r['detection_time'] for r in results]
            
            aggregate_metrics[config_name] = {
                'n_files': len(results),
                'n_successful': len([r for r in results if r['success']]),
                'total_true_seizures': total_true_seizures,
                'total_detected_anomalies': total_detected_anomalies,
                'total_true_positives': total_true_positives,
                'total_false_positives': total_false_positives,
                'total_false_negatives': total_false_negatives,
                'overall_sensitivity': overall_sensitivity,
                'overall_precision': overall_precision,
                'overall_f1_score': overall_f1,
                'mean_sensitivity': np.mean(sensitivities),
                'std_sensitivity': np.std(sensitivities),
                'mean_specificity': np.mean(specificities),
                'std_specificity': np.std(specificities),
                'mean_precision': np.mean(precisions),
                'std_precision': np.std(precisions),
                'mean_f1_score': np.mean(f1_scores),
                'std_f1_score': np.std(f1_scores),
                'mean_detection_time': np.mean(detection_times),
                'std_detection_time': np.std(detection_times),
                'individual_sensitivities': sensitivities,
                'individual_results': results
            }
        
        self.summary_stats = aggregate_metrics
        return aggregate_metrics
    
    def save_results(self, output_file: str, include_detailed: bool = True):
        """
        Save results in human-readable format.
        
        Args:
            output_file: Output file path
            include_detailed: Whether to include detailed per-file results
        """
        if not self.results:
            print("No results to save")
            return
        
        # Calculate aggregate metrics if not done
        if not self.summary_stats:
            self.calculate_aggregate_metrics()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # Header
            f.write("="*80 + "\n")
            f.write("MADRID BATCH EVALUATION RESULTS\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total files processed: {len(self.results)}\n")
            f.write(f"Successful analyses: {len([r for r in self.results if r['success']])}\n")
            f.write(f"Minimum overlap threshold: {self.min_overlap_threshold}\n")
            f.write(f"GPU acceleration: {self.use_gpu}\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*40 + "\n")
            
            best_config = None
            best_sensitivity = 0.0
            
            for config_name, metrics in self.summary_stats.items():
                if metrics['overall_sensitivity'] > best_sensitivity:
                    best_sensitivity = metrics['overall_sensitivity']
                    best_config = config_name
                
                f.write(f"{config_name:20s}: {metrics['overall_sensitivity']:.3f} sensitivity, "
                       f"{metrics['overall_precision']:.3f} precision, "
                       f"{metrics['overall_f1_score']:.3f} F1\n")
            
            f.write(f"\nBest performing configuration: {best_config} "
                   f"(Sensitivity: {best_sensitivity:.3f})\n\n")
            
            # Detailed Configuration Results
            f.write("DETAILED CONFIGURATION RESULTS\n")
            f.write("-"*50 + "\n\n")
            
            for config_name, metrics in self.summary_stats.items():
                f.write(f"Configuration: {config_name}\n")
                f.write("="*30 + "\n")
                f.write(f"Files processed: {metrics['n_files']}\n")
                f.write(f"Successful analyses: {metrics['n_successful']}\n")
                f.write(f"Total true seizures: {metrics['total_true_seizures']}\n")
                f.write(f"Total detected anomalies: {metrics['total_detected_anomalies']}\n")
                f.write(f"True positives: {metrics['total_true_positives']}\n")
                f.write(f"False positives: {metrics['total_false_positives']}\n")
                f.write(f"False negatives: {metrics['total_false_negatives']}\n\n")
                
                f.write("Overall Performance:\n")
                f.write(f"  Sensitivity: {metrics['overall_sensitivity']:.4f}\n")
                f.write(f"  Precision:   {metrics['overall_precision']:.4f}\n")
                f.write(f"  F1-Score:    {metrics['overall_f1_score']:.4f}\n\n")
                
                f.write("Per-File Statistics:\n")
                f.write(f"  Mean Sensitivity: {metrics['mean_sensitivity']:.4f} ± {metrics['std_sensitivity']:.4f}\n")
                f.write(f"  Mean Specificity: {metrics['mean_specificity']:.4f} ± {metrics['std_specificity']:.4f}\n")
                f.write(f"  Mean Precision:   {metrics['mean_precision']:.4f} ± {metrics['std_precision']:.4f}\n")
                f.write(f"  Mean F1-Score:    {metrics['mean_f1_score']:.4f} ± {metrics['std_f1_score']:.4f}\n\n")
                
                f.write("Performance:\n")
                f.write(f"  Mean detection time: {metrics['mean_detection_time']:.3f}s ± {metrics['std_detection_time']:.3f}s\n\n")
                
                # Sensitivity distribution
                sens_values = metrics['individual_sensitivities']
                if sens_values:
                    f.write("Sensitivity Distribution:\n")
                    f.write(f"  Min:     {min(sens_values):.3f}\n")
                    f.write(f"  25th:    {np.percentile(sens_values, 25):.3f}\n")
                    f.write(f"  Median:  {np.percentile(sens_values, 50):.3f}\n")
                    f.write(f"  75th:    {np.percentile(sens_values, 75):.3f}\n")
                    f.write(f"  Max:     {max(sens_values):.3f}\n")
                    
                    # Count perfect detections
                    perfect_detections = sum(1 for s in sens_values if s >= 0.99)
                    f.write(f"  Perfect detections (≥0.99): {perfect_detections}/{len(sens_values)} "
                           f"({perfect_detections/len(sens_values)*100:.1f}%)\n")
                    
                    # Count good detections
                    good_detections = sum(1 for s in sens_values if s >= 0.8)
                    f.write(f"  Good detections (≥0.80): {good_detections}/{len(sens_values)} "
                           f"({good_detections/len(sens_values)*100:.1f}%)\n")
                
                f.write("\n" + "-"*50 + "\n\n")
            
            # Configuration Comparison
            f.write("CONFIGURATION COMPARISON\n")
            f.write("-"*30 + "\n")
            f.write(f"{'Configuration':<20} {'Sensitivity':<12} {'Precision':<12} {'F1-Score':<10} {'Time(s)':<8}\n")
            f.write("-"*70 + "\n")
            
            for config_name, metrics in sorted(self.summary_stats.items(), 
                                             key=lambda x: x[1]['overall_sensitivity'], reverse=True):
                f.write(f"{config_name:<20} {metrics['overall_sensitivity']:<12.4f} "
                       f"{metrics['overall_precision']:<12.4f} {metrics['overall_f1_score']:<10.4f} "
                       f"{metrics['mean_detection_time']:<8.3f}\n")
            
            f.write("\n")
            
            # Detailed Per-File Results
            if include_detailed:
                f.write("DETAILED PER-FILE RESULTS\n")
                f.write("-"*40 + "\n\n")
                
                successful_results = [r for r in self.results if r['success']]
                
                # Group by configuration
                for config_name in self.summary_stats.keys():
                    config_results = [r for r in successful_results if r['config_name'] == config_name]
                    
                    if not config_results:
                        continue
                    
                    f.write(f"Configuration: {config_name}\n")
                    f.write("="*30 + "\n")
                    f.write(f"{'File':<25} {'Subject':<10} {'Seizures':<8} {'Detected':<9} {'Sens':<6} {'Prec':<6} {'F1':<6}\n")
                    f.write("-"*75 + "\n")
                    
                    for result in config_results:
                        file_name = Path(result['file_path']).name
                        if len(file_name) > 24:
                            file_name = file_name[:21] + "..."
                        
                        perf = result['performance']
                        f.write(f"{file_name:<25} {result['subject_id']:<10} "
                               f"{result['n_true_seizures']:<8} {result['n_detected_anomalies']:<9} "
                               f"{perf['sensitivity']:<6.3f} {perf['precision']:<6.3f} {perf['f1_score']:<6.3f}\n")
                    
                    f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-"*20 + "\n")
            
            if self.summary_stats:
                best_config_data = self.summary_stats[best_config]
                
                f.write(f"1. Best overall configuration: '{best_config}'\n")
                f.write(f"   - Achieves {best_config_data['overall_sensitivity']:.1%} sensitivity\n")
                f.write(f"   - {best_config_data['overall_precision']:.1%} precision\n")
                f.write(f"   - Average detection time: {best_config_data['mean_detection_time']:.2f}s\n\n")
                
                # Find most consistent configuration
                most_consistent = min(self.summary_stats.items(), 
                                    key=lambda x: x[1]['std_sensitivity'])
                f.write(f"2. Most consistent configuration: '{most_consistent[0]}'\n")
                f.write(f"   - Sensitivity std: {most_consistent[1]['std_sensitivity']:.3f}\n")
                f.write(f"   - Mean sensitivity: {most_consistent[1]['mean_sensitivity']:.3f}\n\n")
                
                # Find fastest configuration
                fastest = min(self.summary_stats.items(), 
                            key=lambda x: x[1]['mean_detection_time'])
                f.write(f"3. Fastest configuration: '{fastest[0]}'\n")
                f.write(f"   - Average time: {fastest[1]['mean_detection_time']:.2f}s\n")
                f.write(f"   - Sensitivity: {fastest[1]['overall_sensitivity']:.3f}\n\n")
        
        print(f"✓ Results saved to: {output_file}")


def main():
    """Main function for batch evaluation."""
    parser = argparse.ArgumentParser(description='MADRID Batch Evaluation for Seizure Detection')
    
    # Required arguments
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing seizure preprocessed files')
    parser.add_argument('--percentage', type=float, default=100.0,
                       help='Percentage of files to process (default: 100)')
    
    # Configuration options
    parser.add_argument('--config', nargs='+', 
                       choices=['ultra_sensitive', 'ictal_focused', 'robust_detection', 'comprehensive'],
                       help='Specific configurations to use')
    parser.add_argument('--min-overlap', type=float, default=0.1,
                       help='Minimum overlap threshold for true positive (default: 0.1)')
    
    # Filters
    parser.add_argument('--subject', type=str,
                       help='Filter by subject ID (e.g., sub-001)')
    
    # Output options
    parser.add_argument('--output', type=str, default='madrid_evaluation_results.txt',
                       help='Output file for results (default: madrid_evaluation_results.txt)')
    parser.add_argument('--no-detailed', action='store_true',
                       help='Skip detailed per-file results in output')
    
    # Technical options
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU acceleration')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for file selection (default: 42)')
    
    args = parser.parse_args()
    
    if not MADRID_AVAILABLE:
        print("Cannot proceed without MADRID implementation.")
        return 1
    
    # Validate percentage
    if not 0 < args.percentage <= 100:
        print("Percentage must be between 0 and 100")
        return 1
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("MADRID Batch Evaluation for Seizure Detection")
    print("=" * 50)
    print(f"Data directory: {args.data_dir}")
    print(f"Processing: {args.percentage}% of files")
    print(f"Minimum overlap threshold: {args.min_overlap}")
    print(f"Output file: {args.output}")
    print(f"GPU acceleration: {not args.no_gpu}")
    
    try:
        # Initialize evaluator
        evaluator = MadridBatchEvaluator(
            use_gpu=not args.no_gpu,
            min_overlap_threshold=args.min_overlap
        )
        
        # Process batch
        results = evaluator.process_batch(
            data_dir=args.data_dir,
            percentage=args.percentage,
            config_names=args.config,
            subject_filter=args.subject
        )
        
        if not results:
            print("No results obtained")
            return 1
        
        # Calculate aggregate metrics
        aggregate_metrics = evaluator.calculate_aggregate_metrics()
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"BATCH EVALUATION SUMMARY")
        print(f"{'='*80}")
        
        successful_results = [r for r in results if r['success']]
        print(f"Processed {len(results)} analyses on {len(set(r['file_path'] for r in successful_results if 'file_path' in r))} files")
        print(f"Successful: {len(successful_results)}/{len(results)}")
        
        if aggregate_metrics:
            print(f"\nConfiguration Performance:")
            for config_name, metrics in sorted(aggregate_metrics.items(), 
                                             key=lambda x: x[1]['overall_sensitivity'], reverse=True):
                print(f"  {config_name:20s}: {metrics['overall_sensitivity']:.3f} sensitivity, "
                     f"{metrics['overall_precision']:.3f} precision")
        
        # Save results
        evaluator.save_results(
            output_file=args.output,
            include_detailed=not args.no_detailed
        )
        
        print(f"\n✓ Evaluation completed successfully!")
        print(f"✓ Results saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())