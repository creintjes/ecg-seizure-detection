#!/usr/bin/env python3
"""
MADRID Seizure-Focused Anomaly Detection

This script provides focused MADRID analysis for seizure-only preprocessed ECG data.
It works with the output of preprocess_seizure_only.py and enables efficient testing
of MADRID parameters on seizure segments with context.

Key features:
- Load seizure-only preprocessed data with phase labels
- Analyze seizure phases separately (pre-seizure, ictal, post-seizure)
- Multiple MADRID configurations optimized for seizure detection
- Detailed performance analysis with seizure-specific metrics
- Continuous data analysis with multi-length anomaly detection

Usage:
    python madrid_seizure_analysis.py --data-path DATA_PATH
    python madrid_seizure_analysis.py --data-dir SEIZURE_DATA_DIR --config seizure_optimized
    python madrid_seizure_analysis.py --subject sub-001 --seizure-idx 0

Author: Generated for seizure-focused MADRID analysis
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
import pandas as pd
from collections import defaultdict

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
    print("Please ensure the MADRID implementation is available in models/madrid.py")
    MADRID_AVAILABLE = False


class SeizureFocusedMadridAnalyzer:
    """
    MADRID analyzer specialized for seizure-only preprocessed data.
    
    This analyzer works with continuous ECG segments containing seizures
    and their surrounding context, with detailed phase labeling.
    """
    
    def __init__(self, custom_configs: List[Dict[str, Any]] = None, use_gpu: bool = True):
        """
        Initialize the seizure-focused MADRID analyzer.
        
        Args:
            custom_configs: List of custom MADRID configurations
            use_gpu: Whether to use GPU acceleration
        """
        self.configs = custom_configs if custom_configs else self._get_seizure_optimized_configs()
        self.use_gpu = use_gpu
        
        print(f"Initialized SeizureFocusedMadridAnalyzer:")
        print(f"  Number of configurations: {len(self.configs)}")
        print(f"  GPU acceleration: {use_gpu}")
        for config in self.configs:
            print(f"  - {config['name']}: {config['description']}")
    
    def _get_seizure_optimized_configs(self) -> List[Dict[str, Any]]:
        """Get MADRID configurations optimized for seizure detection."""
        return [
            {
                "name": "ultra_short",
                "min_length_seconds": 0.5,
                "max_length_seconds": 5.0,
                "step_size_seconds": 0.5,
                "description": "Detect very short anomalies (0.5-5s) - cardiac arrhythmias, brief artifacts",
                "adaptive_step": True  # Adapt step size based on sampling rate
            },
            {
                "name": "short_term", 
                "min_length_seconds": 2.0,
                "max_length_seconds": 15.0,
                "step_size_seconds": 1.0,
                "description": "Detect short-term anomalies (2-15s) - brief seizure patterns, autonomic changes",
                "adaptive_step": True
            },
            {
                "name": "medium_term",
                "min_length_seconds": 10.0,
                "max_length_seconds": 60.0,
                "step_size_seconds": 5.0,
                "description": "Detect medium-term anomalies (10s-1min) - seizure onset/offset, autonomic responses",
                "adaptive_step": True
            },
            {
                "name": "long_term",
                "min_length_seconds": 30.0,
                "max_length_seconds": 180.0,
                "step_size_seconds": 10.0,
                "description": "Detect long-term anomalies (30s-3min) - prolonged seizures, post-ictal changes",
                "adaptive_step": True
            },
            {
                "name": "extended_term",
                "min_length_seconds": 120.0,
                "max_length_seconds": 600.0,
                "step_size_seconds": 30.0,
                "description": "Detect extended anomalies (2-10min) - long seizures, recovery patterns",
                "adaptive_step": True
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
            
            # Extract metadata with fallback for sampling rate detection
            fs = data.get('sampling_rate', None)
            
            # If sampling rate not in metadata, try to infer from data
            if fs is None:
                if 'metadata' in data and 'sampling_rate' in data['metadata']:
                    fs = data['metadata']['sampling_rate']
                else:
                    # Try to infer from channel data and duration
                    channel = data['channels'][0]
                    if 'total_duration' in data['metadata']:
                        estimated_fs = len(channel['data']) / data['metadata']['total_duration']
                        fs = int(round(estimated_fs))
                        print(f"  ⚠️  Sampling rate not found, estimated from data: {fs} Hz")
                    else:
                        fs = 125  # Default fallback
                        print(f"  ⚠️  Sampling rate not found, using default: {fs} Hz")
            
            total_duration = data['metadata']['total_duration']
            seizure_duration = data['metadata']['seizure_duration']
            data_len = len(data['channels'][0]['data'])
            labels_len = len(data['channels'][0]['labels'])
            assert data_len == labels_len, f"Mismatch: {data_len} data vs {labels_len} labels"

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
    
    def prepare_madrid_configs(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Prepare MADRID configurations based on data characteristics.
        
        Args:
            data: Seizure data dictionary
            
        Returns:
            List of validated MADRID configurations
        """
        fs = data.get('sampling_rate', 125)
        total_duration = data['metadata']['total_duration']
        seizure_duration = data['metadata']['seizure_duration']
        channel = data['channels'][0]
        n_samples = len(channel['data'])
        
        print(f"\nPreparing MADRID configurations:")
        print(f"  Segment length: {n_samples:,} samples ({total_duration:.1f}s)")
        print(f"  Seizure duration: {seizure_duration:.1f}s")
        print(f"  Sampling rate: {fs} Hz")
        
        valid_configs = []
        
        for config in self.configs:
            # Convert time-based to sample-based parameters
            min_length = int(config['min_length_seconds'] * fs)
            max_length = int(config['max_length_seconds'] * fs)
            
            # Adaptive step size based on sampling rate and configuration
            if config.get('adaptive_step', False):
                # For higher sampling rates, use larger steps to reduce computational load
                if fs >= 250:
                    step_multiplier = max(1, fs // 125)  # Scale with sampling rate
                    step_size = int(config['step_size_seconds'] * fs * step_multiplier)
                else:
                    step_size = int(config['step_size_seconds'] * fs)
                
                # Ensure step size is reasonable (not too large, not too small)
                min_step = max(1, fs // 10)  # At least 0.1 seconds
                max_step = min(step_size, max_length // 4)  # At most 1/4 of max_length
                step_size = max(min_step, min(step_size, max_step))
            else:
                step_size = int(config['step_size_seconds'] * fs)
            
            # Ensure minimum requirements for MADRID
            min_length = max(2, min_length)  # MADRID requires at least 2 samples
            
            # MADRID requires sufficient data for analysis
            # Train portion should be at least 1/3 of data, test portion should have max_length samples
            min_required = max_length * 3
            
            # Additional validation for very high or low sampling rates
            if fs < 50:
                print(f"  ⚠️  Warning: Low sampling rate ({fs} Hz) may affect analysis quality")
            elif fs > 1000:
                print(f"  ⚠️  Warning: High sampling rate ({fs} Hz) will increase computation time")
            
            if n_samples >= min_required and min_length >= 2 and max_length >= min_length:
                valid_config = {
                    'name': config['name'],
                    'description': config['description'],
                    'min_length': min_length,
                    'max_length': max_length,
                    'step_size': step_size,
                    'min_length_seconds': config['min_length_seconds'],
                    'max_length_seconds': config['max_length_seconds'],
                    'step_size_seconds': step_size / fs,  # Actual step size in seconds
                    'original_step_seconds': config['step_size_seconds'],
                    'sampling_rate': fs,
                    'adaptive_step_used': config.get('adaptive_step', False)
                }
                valid_configs.append(valid_config)
                
                if config.get('adaptive_step', False) and step_size != int(config['step_size_seconds'] * fs):
                    print(f"  ✓ {config['name']}: Valid ({min_length}-{max_length} samples, "
                          f"adaptive step {step_size} = {step_size/fs:.2f}s)")
                else:
                    print(f"  ✓ {config['name']}: Valid ({min_length}-{max_length} samples, step {step_size})")
            else:
                print(f"  ❌ {config['name']}: Invalid (requires {min_required}, have {n_samples})")
        
        print(f"\nValid configurations: {len(valid_configs)}/{len(self.configs)}")
        
        # Provide sampling rate specific recommendations
        self._print_sampling_rate_recommendations(fs, valid_configs)
        
        return valid_configs
    
    def _print_sampling_rate_recommendations(self, fs: int, valid_configs: List[Dict[str, Any]]):
        """Print sampling rate specific recommendations."""
        print(f"\n📊 Sampling Rate Analysis ({fs} Hz):")
        
        if fs <= 50:
            print("  ⚠️  Very low sampling rate - may miss short-duration cardiac events")
            print("  💡 Recommendation: Focus on longer-term configurations (medium_term, long_term)")
        elif 50 < fs <= 125:
            print("  ✓ Standard ECG sampling rate - good for seizure detection")
            print("  💡 Recommendation: All configurations should work well")
        elif 125 < fs <= 250:
            print("  ✓ High-quality ECG sampling rate - excellent for detailed analysis")
            print("  💡 Recommendation: Consider ultra_short and short_term for cardiac arrhythmias")
        elif 250 < fs <= 500:
            print("  ⚡ Very high sampling rate - excellent resolution but computationally intensive")
            print("  💡 Recommendation: Use adaptive stepping enabled, consider GPU acceleration")
        else:
            print("  🚀 Extremely high sampling rate - may require downsampling")
            print("  💡 Recommendation: Consider preprocessing with downsampling to 250-500 Hz")
        
        # Performance estimates
        total_samples = sum(len(range(config['min_length'], config['max_length'], config['step_size'])) 
                           for config in valid_configs)
        estimated_time_per_config = total_samples * 0.001  # Rough estimate
        
        print(f"  📈 Estimated analysis time per file: {estimated_time_per_config:.2f}s per configuration")
        
        if fs > 500:
            print(f"  ⚡ Consider using --no-gpu flag if GPU memory is insufficient")
        elif fs > 250:
            print(f"  🎯 GPU acceleration recommended for optimal performance")
    
    def apply_downsampling(self, data: Dict[str, Any], target_fs: int) -> Optional[Dict[str, Any]]:
        """
        Apply downsampling to ECG data.
        
        Args:
            data: Seizure data dictionary
            target_fs: Target sampling rate
            
        Returns:
            Modified data dictionary with downsampled signals
        """
        original_fs = data.get('sampling_rate', 125)
        
        if target_fs >= original_fs:
            print(f"  ⚠️  Target sampling rate ({target_fs} Hz) >= original ({original_fs} Hz), no downsampling needed")
            return data
        
        if original_fs % target_fs != 0:
            print(f"  ⚠️  Warning: {original_fs} Hz is not a multiple of {target_fs} Hz, may cause aliasing")
        
        downsample_factor = original_fs // target_fs
        
        print(f"  🔄 Downsampling from {original_fs} Hz to {target_fs} Hz (factor: {downsample_factor})")
        
        try:
            # Downsample each channel
            for i, channel in enumerate(data['channels']):
                original_data = channel['data']
                original_labels = channel['labels']
                
                # Downsample signal data
                downsampled_data = original_data[::downsample_factor]
                
                # Downsample labels (use most common label in each downsampled segment)
                downsampled_labels = []
                for j in range(0, len(original_labels), downsample_factor):
                    segment_labels = original_labels[j:j+downsample_factor]
                    if len(segment_labels) > 0:
                        # Use most common label in segment
                        unique_labels, counts = np.unique(segment_labels, return_counts=True)
                        most_common_label = unique_labels[np.argmax(counts)]
                        downsampled_labels.append(most_common_label)
                
                downsampled_labels = np.array(downsampled_labels)
                
                # Update channel data
                data['channels'][i]['data'] = downsampled_data
                data['channels'][i]['labels'] = downsampled_labels
                
                print(f"    Channel {i}: {len(original_data):,} → {len(downsampled_data):,} samples")
            
            # Update metadata
            data['sampling_rate'] = target_fs
            data['metadata']['total_duration'] = len(data['channels'][0]['data']) / target_fs
            
            # Recalculate seizure duration if possible
            if 'seizure_duration' in data['metadata']:
                # Seizure duration in seconds should remain the same
                pass
            
            print(f"  ✓ Downsampling completed successfully")
            return data
            
        except Exception as e:
            print(f"  ❌ Downsampling failed: {e}")
            return None
    
    def run_madrid_on_continuous_data(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run MADRID on continuous seizure segment.
        
        Args:
            data: Seizure data dictionary
            config: MADRID configuration
            
        Returns:
            Dictionary with detection results
        """
        channel = data['channels'][0]
        signal_data = channel['data'].astype(np.float64)
        labels = channel['labels']
        fs = data.get('sampling_rate', 125)
        
        print(f"\nRunning MADRID on continuous data:")
        print(f"  Configuration: {config['name']}")
        print(f"  Search range: {config['min_length_seconds']:.1f}s - {config['max_length_seconds']:.1f}s")
        print(f"  Step size: {config['step_size_seconds']:.1f}s")
        print(f"  Data length: {len(signal_data):,} samples ({len(signal_data)/fs:.1f}s)")
        
        start_time = time.time()
        
        try:
            # Initialize MADRID
            detector = MADRID(use_gpu=self.use_gpu, enable_output=False)
            
            # Use 1/3 of data for training, rest for testing
            train_test_split = len(signal_data) // 3
            
            # Run detection
            multi_length_table, bsf, bsf_loc = detector.fit(
                T=signal_data,
                min_length=config['min_length'],
                max_length=config['max_length'],
                step_size=config['step_size'],
                train_test_split=train_test_split
            )
            
            detection_time = time.time() - start_time
            
            # Get anomaly information
            anomaly_info = detector.get_anomaly_scores(threshold_percentile=95)
            anomalies = anomaly_info['anomalies']
            
            print(f"  ✓ Detection completed in {detection_time:.3f}s")
            print(f"  ✓ Found {len(anomalies)} high-confidence anomalies")
            
            # Convert anomalies to binary array for phase analysis
            anomaly_binary = np.zeros(len(signal_data))
            for anomaly in anomalies:
                if anomaly['location'] is not None:
                    start_idx = max(0, anomaly['location'])
                    end_idx = min(len(signal_data), start_idx + anomaly['length'])
                    anomaly_binary[start_idx:end_idx] = 1
            
            n_anomaly_samples = np.sum(anomaly_binary)
            anomaly_rate = n_anomaly_samples / len(signal_data) * 100
            
            # Analyze anomalies by seizure phase
            phase_analysis = self._analyze_anomalies_by_phase(anomaly_binary, labels, fs)
            
            # Find anomaly regions
            anomaly_regions = self._find_anomaly_regions(anomaly_binary, fs)
            
            print(f"  ✓ Found {len(anomaly_regions)} anomaly regions")
            print(f"  ✓ Total anomaly coverage: {anomaly_rate:.2f}%")
            
            return {
                'success': True,
                'multi_length_table': multi_length_table,
                'best_scores': bsf,
                'best_locations': bsf_loc,
                'anomalies': anomalies,
                'anomaly_binary': anomaly_binary,
                'n_anomaly_samples': n_anomaly_samples,
                'anomaly_rate': anomaly_rate,
                'anomaly_regions': anomaly_regions,
                'phase_analysis': phase_analysis,
                'detection_time': detection_time,
                'config': config,
                'detector': detector,
                'data_info': {
                    'subject_id': data['subject_id'],
                    'run_id': data['run_id'],
                    'seizure_index': data['seizure_index'],
                    'total_duration': data['metadata']['total_duration'],
                    'seizure_duration': data['metadata']['seizure_duration'],
                    'train_test_split': train_test_split,
                    'sampling_rate': fs
                }
            }
            
        except Exception as e:
            print(f"  ❌ Detection failed: {str(e)}")
            import traceback
            traceback.print_exc()
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
    
    def run_windowed_analysis(self, data: Dict[str, Any], config: Dict[str, Any], 
                            window_size_seconds: float = 30.0) -> Dict[str, Any]:
        """
        Run MADRID on windowed data for detailed phase analysis.
        
        Args:
            data: Seizure data dictionary
            config: MADRID configuration
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
        
        results = {
            'config': config,
            'window_size_seconds': window_size_seconds,
            'phase_results': {},
            'summary': {}
        }
        
        print(f"\nRunning MADRID on windowed data ({config['name']}):")
        
        # Process each phase
        for phase, windows in phase_windows.items():
            if not windows:
                continue
                
            print(f"\n  Processing {phase} phase ({len(windows)} windows):")
            
            phase_results = []
            successful = 0
            
            for i, window_info in enumerate(windows):
                try:
                    # Initialize MADRID for this window
                    detector = MADRID(use_gpu=self.use_gpu, enable_output=False)
                    
                    start_time = time.time()
                    
                    # Use smaller parameters for windowed analysis
                    window_length = len(window_info['data'])
                    train_split = window_length // 3
                    
                    # Adjust config for window size
                    min_len = min(config['min_length'], window_length // 4)
                    max_len = min(config['max_length'], window_length // 2)
                    step_size = max(1, config['step_size'] // 4)
                    
                    if min_len < 2 or max_len < min_len or train_split < max_len:
                        raise ValueError(f"Window too small for MADRID analysis")
                    
                    multi_length_table, bsf, bsf_loc = detector.fit(
                        T=window_info['data'].astype(np.float64),
                        min_length=min_len,
                        max_length=max_len,
                        step_size=step_size,
                        train_test_split=train_split
                    )
                    
                    detection_time = time.time() - start_time
                    
                    # Get anomaly information
                    anomaly_info = detector.get_anomaly_scores(threshold_percentile=90)
                    n_anomalies = len(anomaly_info['anomalies'])
                    
                    # Calculate anomaly rate based on detected anomalies
                    anomaly_coverage = 0
                    for anomaly in anomaly_info['anomalies']:
                        if anomaly['location'] is not None:
                            anomaly_coverage += anomaly['length']
                    
                    anomaly_rate = anomaly_coverage / len(window_info['data']) * 100
                    
                    phase_results.append({
                        'window_idx': i,
                        'start_time': window_info['start_time'],
                        'end_time': window_info['end_time'],
                        'dominant_label': window_info['dominant_label'],
                        'dominant_ratio': window_info['dominant_ratio'],
                        'n_anomalies': n_anomalies,
                        'anomaly_rate': anomaly_rate,
                        'detection_time': detection_time,
                        'success': True,
                        'best_score': np.max(bsf) if len(bsf) > 0 else 0
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
                avg_best_score = np.mean([r['best_score'] for r in successful_results])
                
                print(f"    ✓ {successful}/{len(windows)} windows successful")
                print(f"    ✓ Avg anomaly rate: {avg_anomaly_rate:.2f}% ± {std_anomaly_rate:.2f}%")
                print(f"    ✓ Avg detection time: {avg_detection_time:.3f}s")
                print(f"    ✓ Avg best score: {avg_best_score:.4f}")
                
                results['summary'][phase] = {
                    'n_windows': len(windows),
                    'n_successful': successful,
                    'avg_anomaly_rate': avg_anomaly_rate,
                    'std_anomaly_rate': std_anomaly_rate,
                    'avg_detection_time': avg_detection_time,
                    'avg_best_score': avg_best_score
                }
            else:
                print(f"    ❌ All windows failed")
                results['summary'][phase] = {
                    'n_windows': len(windows),
                    'n_successful': 0
                }
        
        return results
    
    def plot_results(self, result: Dict[str, Any], data: Dict[str, Any]):
        """
        Plot MADRID analysis results.
        
        Args:
            result: MADRID analysis result
            data: Original seizure data
        """
        if not result['success']:
            print("Cannot plot failed analysis")
            return
        
        channel = data['channels'][0]
        signal_data = channel['data']
        labels = channel['labels']
        fs = data.get('sampling_rate', 125)
        
        # Create time axis
        time_axis = np.arange(len(signal_data)) / fs
        
        # Set up the plot
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        fig.suptitle(f'MADRID Seizure Analysis - {result["config"]["name"]} - '
                    f'{result["data_info"]["subject_id"]} Seizure {result["data_info"]["seizure_index"]}', 
                    fontsize=14)
        
        # Plot 1: Original signal with phase labels
        ax1 = axes[0]
        ax1.plot(time_axis, signal_data, 'b-', alpha=0.7, linewidth=0.8, label='ECG Signal')
        
        # Color-code phases
        phase_colors = {
            'normal': 'green',
            'pre_seizure': 'yellow', 
            'ictal': 'red',
            'post_seizure': 'orange'
        }
        
        # Find phase regions for background coloring
        current_phase = labels[0]
        phase_start = 0
        
        for i in range(1, len(labels)):
            if labels[i] != current_phase:
                if current_phase in phase_colors:
                    ax1.axvspan(time_axis[phase_start], time_axis[i-1], 
                               alpha=0.2, color=phase_colors[current_phase], 
                               label=current_phase if phase_start == 0 else "")
                current_phase = labels[i]
                phase_start = i
        
        # Final phase
        if current_phase in phase_colors:
            ax1.axvspan(time_axis[phase_start], time_axis[-1], 
                       alpha=0.2, color=phase_colors[current_phase])
        
        # Mark train/test split
        train_split = result['data_info']['train_test_split']
        ax1.axvline(x=time_axis[train_split], color='purple', linestyle='--', 
                   linewidth=2, label='Train/Test Split')
        
        ax1.set_title('ECG Signal with Seizure Phases')
        ax1.set_ylabel('ECG Amplitude')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Anomaly detections
        ax2 = axes[1]
        ax2.plot(time_axis, result['anomaly_binary'], 'r-', linewidth=1.5, label='Anomaly Detection')
        
        # Mark individual anomalies
        for i, anomaly in enumerate(result['anomalies'][:10]):  # Show top 10
            if anomaly['location'] is not None:
                anomaly_time = anomaly['location'] / fs
                ax2.axvline(x=anomaly_time, color='red', alpha=0.6, linestyle=':',
                           label='High-confidence Anomaly' if i == 0 else "")
        
        ax2.axvline(x=time_axis[train_split], color='purple', linestyle='--', 
                   linewidth=2, label='Train/Test Split')
        ax2.set_title(f'MADRID Anomaly Detection ({len(result["anomalies"])} high-confidence anomalies)')
        ax2.set_ylabel('Anomaly')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Discord scores by length
        ax3 = axes[2]
        config = result['config']
        lengths_sec = np.arange(config['min_length_seconds'], 
                               config['max_length_seconds'] + config['step_size_seconds'],
                               config['step_size_seconds'])
        
        valid_scores = result['best_scores'][~np.isnan(result['best_scores'])]
        valid_lengths = lengths_sec[:len(valid_scores)]
        
        if len(valid_scores) > 0:
            ax3.plot(valid_lengths, valid_scores, 'bo-', markersize=6, linewidth=2)
            ax3.set_title('Discord Scores by Subsequence Length')
            ax3.set_xlabel('Subsequence Length (seconds)')
            ax3.set_ylabel('Discord Score')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No valid scores to display', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax3.transAxes)
        
        # Plot 4: Phase-wise anomaly rates
        ax4 = axes[3]
        phase_analysis = result['phase_analysis']
        phases = list(phase_analysis.keys())
        anomaly_rates = [phase_analysis[phase]['anomaly_rate'] for phase in phases]
        
        colors = [phase_colors.get(phase, 'gray') for phase in phases]
        bars = ax4.bar(phases, anomaly_rates, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, rate in zip(bars, anomaly_rates):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        ax4.set_title('Anomaly Rate by Seizure Phase')
        ax4.set_ylabel('Anomaly Rate (%)')
        ax4.set_xlabel('Seizure Phase')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print(f"\n{'='*60}")
        print(f"MADRID ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"Configuration: {config['name']}")
        print(f"Detection time: {result['detection_time']:.3f}s")
        print(f"Total anomaly rate: {result['anomaly_rate']:.2f}%")
        print(f"Number of anomaly regions: {len(result['anomaly_regions'])}")
        print(f"\nPhase-wise anomaly rates:")
        for phase, analysis in phase_analysis.items():
            if analysis['n_samples'] > 0:
                print(f"  {phase:12s}: {analysis['anomaly_rate']:5.2f}% "
                      f"({analysis['n_anomalies']:,} / {analysis['n_samples']:,} samples)")
    
    def analyze_results(self, continuous_results: List[Dict], windowed_results: List[Dict] = None):
        """
        Analyze and display comprehensive results.
        
        Args:
            continuous_results: Results from continuous analysis
            windowed_results: Results from windowed analysis (optional)
        """
        print(f"\n{'='*80}")
        print(f"SEIZURE-FOCUSED MADRID ANALYSIS RESULTS")
        print(f"{'='*80}")
        
        if not continuous_results:
            print("No results to analyze")
            return
        
        # Continuous analysis summary
        print(f"\nCONTINUOUS ANALYSIS SUMMARY:")
        print(f"{'Config':<15} | {'Subject':<8} | {'Seizure':<7} | {'Anomaly %':<9} | {'Time (s)':<8} | {'Regions':<7} | {'GPU':<3}")
        print("-" * 85)
        
        successful_results = [r for r in continuous_results if r['success']]
        
        for result in continuous_results:
            if result['success']:
                data_info = result['data_info']
                gpu_used = "Yes" if 'detector' in result and result['detector'].use_gpu else "No"
                print(f"{result['config']['name']:<15} | "
                      f"{data_info['subject_id']:<8} | "
                      f"{data_info['seizure_index']:<7} | "
                      f"{result['anomaly_rate']:<9.2f} | "
                      f"{result['detection_time']:<8.3f} | "
                      f"{len(result['anomaly_regions']):<7} | "
                      f"{gpu_used:<3}")
            else:
                print(f"{result['config']['name']:<15} | FAILED: {result.get('error', 'Unknown')}")
        
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
            
            # Seizure discrimination analysis
            print(f"\nSEIZURE DISCRIMINATION ANALYSIS:")
            print(f"{'Config':<15} | {'Ictal/Normal':<12} | {'Ictal/Pre':<10} | {'Post/Normal':<12}")
            print("-" * 55)
            
            for result in successful_results:
                phase_analysis = result['phase_analysis']
                ictal_rate = phase_analysis['ictal']['anomaly_rate']
                normal_rate = phase_analysis['normal']['anomaly_rate']
                pre_rate = phase_analysis['pre_seizure']['anomaly_rate']
                post_rate = phase_analysis['post_seizure']['anomaly_rate']
                
                ictal_normal_ratio = ictal_rate / max(normal_rate, 0.01)
                ictal_pre_ratio = ictal_rate / max(pre_rate, 0.01)
                post_normal_ratio = post_rate / max(normal_rate, 0.01)
                
                print(f"{result['config']['name']:<15} | "
                      f"{ictal_normal_ratio:<12.2f} | "
                      f"{ictal_pre_ratio:<10.2f} | "
                      f"{post_normal_ratio:<12.2f}")
            
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
            
            # GPU performance analysis
            gpu_results = [r for r in successful_results if 'detector' in r and r['detector'].use_gpu]
            cpu_results = [r for r in successful_results if 'detector' in r and not r['detector'].use_gpu]
            
            if gpu_results and cpu_results:
                gpu_times = [r['detection_time'] for r in gpu_results]
                cpu_times = [r['detection_time'] for r in cpu_results]
                speedup = np.mean(cpu_times) / np.mean(gpu_times)
                print(f"  GPU speedup: {speedup:.2f}x")
        
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
    """Load custom MADRID configurations from JSON file."""
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


def save_config_template(filename: str = "madrid_seizure_config_template.json"):
    """Save a template configuration file for seizure analysis."""
    template = {
        "configs": [
            {
                "name": "seizure_short",
                "min_length_seconds": 1.0,
                "max_length_seconds": 10.0,
                "step_size_seconds": 1.0,
                "description": "Detect short seizure-related anomalies"
            },
            {
                "name": "seizure_medium",
                "min_length_seconds": 5.0,
                "max_length_seconds": 30.0,
                "step_size_seconds": 2.0,
                "description": "Detect medium-duration seizure patterns"
            },
            {
                "name": "seizure_long",
                "min_length_seconds": 20.0,
                "max_length_seconds": 120.0,
                "step_size_seconds": 10.0,
                "description": "Detect long seizure and recovery patterns"
            }
        ]
    }
    
    with open(filename, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"✓ Seizure analysis configuration template saved to: {filename}")


def main():
    """Main function for seizure-focused MADRID analysis."""
    parser = argparse.ArgumentParser(description='MADRID Seizure-Focused Anomaly Detection')
    
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
    
    # MADRID configuration
    parser.add_argument('--config', choices=['seizure_optimized', 'ultra_fast', 'comprehensive'],
                       default='seizure_optimized',
                       help='Predefined configuration preset')
    parser.add_argument('--config-file', type=str,
                       help='JSON file with custom MADRID configurations')
    parser.add_argument('--single-config', type=str,
                       help='Run only specific configuration by name')
    
    # GPU options
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU acceleration')
    
    # Sampling rate options
    parser.add_argument('--override-fs', type=int,
                       help='Override sampling rate detection (Hz)')
    parser.add_argument('--force-downsample', type=int,
                       help='Force downsample to specified rate (Hz)')
    
    # Analysis options
    parser.add_argument('--windowed-analysis', action='store_true',
                       help='Also run windowed phase analysis')
    parser.add_argument('--window-size', type=float, default=30.0,
                       help='Window size for windowed analysis (default: 30.0s)')
    parser.add_argument('--plot-results', action='store_true',
                       help='Plot detailed results for each analysis')
    
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
    
    print("MADRID Seizure-Focused Anomaly Detection")
    print("=" * 50)
    print(f"MADRID available: {MADRID_AVAILABLE}")
    
    if not MADRID_AVAILABLE:
        print("Cannot proceed without MADRID implementation.")
        return 1
    
    # Load custom configurations if specified
    custom_configs = None
    if args.config_file:
        custom_configs = load_custom_configs(args.config_file)
        if not custom_configs:
            print("Failed to load custom configurations")
            return 1
    
    # Initialize analyzer
    use_gpu = not args.no_gpu
    analyzer = SeizureFocusedMadridAnalyzer(custom_configs=custom_configs, use_gpu=use_gpu)
    
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
        
        # Apply sampling rate overrides if specified
        if args.override_fs:
            print(f"  🔄 Overriding sampling rate: {data.get('sampling_rate', 'unknown')} Hz → {args.override_fs} Hz")
            data['sampling_rate'] = args.override_fs
        
        # Apply downsampling if specified
        if args.force_downsample:
            data = analyzer.apply_downsampling(data, args.force_downsample)
            if data is None:
                print("  ❌ Downsampling failed")
                continue
        
        # Prepare MADRID configurations
        valid_configs = analyzer.prepare_madrid_configs(data)
        if not valid_configs:
            print("No valid MADRID configurations for this data")
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
            result = analyzer.run_madrid_on_continuous_data(data, config)
            all_continuous_results.append(result)
            
            if result['success']:
                if args.verbose:
                    # Show sample anomaly regions
                    regions = result['anomaly_regions']
                    if regions:
                        print(f"  Sample anomaly regions:")
                        for j, region in enumerate(regions[:5]):
                            print(f"    {j+1}. {region['start_time']:.2f}s - "
                                  f"{region['end_time']:.2f}s ({region['duration']:.2f}s)")
                        if len(regions) > 5:
                            print(f"    ... and {len(regions) - 5} more regions")
                
                # Plot results if requested
                if args.plot_results:
                    analyzer.plot_results(result, data)
        
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
    print(f"SEIZURE-FOCUSED MADRID ANALYSIS COMPLETED")
    print(f"{'='*80}")
    
    return 0


if __name__ == "__main__":
    exit(main())