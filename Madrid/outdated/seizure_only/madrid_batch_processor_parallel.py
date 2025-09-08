#!/usr/bin/env python3
"""
Parallelized Madrid Batch Processor with JSON Output

Processes multiple preprocessed ECG files with Madrid algorithm in parallel
and saves results in standardized JSON format.

Usage:
    python madrid_batch_processor_parallel.py --data-dir results/preprocessed_all --output-dir madrid_results --max-files 10 --n-workers 4
    python madrid_batch_processor_parallel.py --config-file madrid_config.json --n-workers 8


"""

import os
import sys
import json
import pickle
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import warnings
from multiprocessing import Pool, Manager, cpu_count
from functools import partial
import queue
import threading

warnings.filterwarnings('ignore')

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from models.madrid_v2 import MADRID_V2
    MADRID_AVAILABLE = True
    print("✓ MADRID successfully imported")
except ImportError as e:
    print(f"❌ MADRID import failed: {e}")
    MADRID_AVAILABLE = False
    sys.exit(1)

# Setup logging with thread-safe configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('madrid_batch_processor_parallel.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_single_file(file_path: str, config: Dict[str, Any], output_dir: str) -> Tuple[bool, str, Dict]:
    """
    Process a single file with Madrid analysis
    
    Args:
        file_path: Path to preprocessed file
        config: Configuration dictionary
        output_dir: Output directory for results
        
    Returns:
        Tuple of (success, file_path, result_info)
    """
    try:
        # Create Madrid processor for this worker
        processor = MadridBatchProcessorCore(config)
        
        # Load file
        file_data = processor.load_preprocessed_file(file_path)
        if file_data is None:
            return False, file_path, {"error": "Failed to load file"}
        
        # Run Madrid analysis
        analysis_results, performance_info = processor.analyze_with_madrid(
            file_data['signal'],
            file_data['metadata']['signal_metadata']
        )
        
        # Create JSON output
        json_output = processor.create_json_output(file_data, analysis_results, performance_info)
        
        # Save JSON file
        output_filename = f"madrid_results_{file_data['metadata']['subject_id']}_{file_data['metadata']['run_id']}"
        if file_data['metadata']['seizure_id']:
            output_filename += f"_{file_data['metadata']['seizure_id']}"
        output_filename += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_path = Path(output_dir) / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)
        
        # Return success info
        result_info = {
            "output_file": str(output_path),
            "num_anomalies": len(analysis_results['anomalies']),
            "true_positives": analysis_results['performance_metrics'].get('true_positives', 0),
            "execution_time": performance_info['execution_time_seconds'],
            "seizure_present": file_data['ground_truth']['seizure_present']
        }
        
        return True, file_path, result_info
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return False, file_path, {"error": str(e)}

class ProgressTracker:
    """Thread-safe progress tracker for parallel processing"""
    
    def __init__(self, total_files: int):
        self.total_files = total_files
        self.completed = 0
        self.failed = 0
        self.lock = threading.Lock()
        self.start_time = time.time()
    
    def update(self, success: bool):
        with self.lock:
            if success:
                self.completed += 1
            else:
                self.failed += 1
            
            processed = self.completed + self.failed
            elapsed = time.time() - self.start_time
            
            if processed > 0:
                avg_time = elapsed / processed
                eta = avg_time * (self.total_files - processed)
                
                logger.info(f"Progress: {processed}/{self.total_files} "
                           f"({processed/self.total_files*100:.1f}%) - "
                           f"Success: {self.completed}, Failed: {self.failed} - "
                           f"ETA: {eta/60:.1f}min")

class MadridBatchProcessorCore:
    """
    Core Madrid processing functionality (extracted from original class)
    This will be used in worker processes
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.madrid_detector = MADRID_V2(
            use_gpu=config.get('use_gpu', True),
            enable_output=False  # Disable output in workers to avoid conflicts
        )
        
        self.madrid_params = config.get('madrid_parameters', {
            'm_range': {
                'min_length': 80,
                'max_length': 800,
                'step_size': 80
            },
            'analysis_config': {
                'top_k': 3,
                'train_test_split_ratio': 0.5,
                'train_minutes': 30,
                'threshold_percentile': 95
            },
            'algorithm_settings': {
                'use_gpu': True,
                'downsampling_factor': 1
            }
        })
    
    def load_preprocessed_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load preprocessed ECG file (simplified version)"""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Check if this is seizure-only preprocessed data
            if self.is_seizure_only_format(data):
                return self.load_seizure_only_data(data, file_path)
            else:
                return self.load_windowed_data(data, file_path)
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def is_seizure_only_format(self, data: Dict) -> bool:
        """Check if data is in seizure-only preprocessed format"""
        seizure_only_keys = ['subject_id', 'run_id', 'seizure_index', 'channels']
        
        if all(key in data for key in seizure_only_keys):
            return True
        elif 'channels' in data and isinstance(data['channels'], list):
            if len(data['channels']) > 0 and 'windows' in data['channels'][0]:
                return False
            elif len(data['channels']) > 0 and 'data' in data['channels'][0]:
                return True
        
        return False
    
    def load_seizure_only_data(self, data: Dict, file_path: str) -> Optional[Dict[str, Any]]:
        """Load seizure-only preprocessed data format"""
        try:
            subject_id = data.get('subject_id', 'unknown')
            run_id = data.get('run_id', 'unknown')
            seizure_index = data.get('seizure_index', 0)
            seizure_id = f"seizure_{seizure_index:02d}"
            
            # Get first ECG channel
            if not data.get('channels') or len(data['channels']) == 0:
                return None
            
            channel = data['channels'][0]
            signal_data = channel.get('data', np.array([]))
            
            if len(signal_data) == 0:
                return None
            
            # Extract sampling rate with multiple fallbacks
            sampling_rate = None
            
            if 'sampling_rate' in data:
                sampling_rate = data['sampling_rate']
            elif 'preprocessing_params' in data and 'downsample_freq' in data['preprocessing_params']:
                sampling_rate = data['preprocessing_params']['downsample_freq']
            elif 'downsample_freq' in data:
                sampling_rate = data['downsample_freq']
            else:
                # Infer from filename
                file_path_lower = file_path.lower()
                if '8hz' in file_path_lower:
                    sampling_rate = 8
                elif '32hz' in file_path_lower:
                    sampling_rate = 32
                elif '125hz' in file_path_lower:
                    sampling_rate = 125
                else:
                    sampling_rate = 125  # Default
            
            # Signal metadata
            signal_metadata = {
                'sampling_rate': sampling_rate,
                'original_sampling_rate': 250,
                'signal_length_samples': len(signal_data),
                'signal_duration_seconds': len(signal_data) / sampling_rate,
                'preprocessing_info': {
                    'method': 'seizure_only',
                    'context_minutes': data.get('preprocessing_params', {}).get('context_minutes', 30),
                    'filter_applied': data.get('preprocessing_params', {}).get('filter_params', {}),
                    'downsampling_factor': 250 / sampling_rate
                }
            }
            
            # Extract ground truth
            ground_truth = self.extract_seizure_only_ground_truth(data, sampling_rate)
            
            return {
                'signal': signal_data,
                'metadata': {
                    'subject_id': subject_id,
                    'run_id': run_id,
                    'seizure_id': seizure_id,
                    'source_file': file_path,
                    'signal_metadata': signal_metadata
                },
                'ground_truth': ground_truth
            }
            
        except Exception as e:
            logger.error(f"Error processing seizure-only data {file_path}: {e}")
            return None
    
    def load_windowed_data(self, data: Dict, file_path: str) -> Optional[Dict[str, Any]]:
        """Load windowed preprocessed data format"""
        if not data or "channels" not in data:
            return None
        
        # Extract first channel
        channel_data = data["channels"][0]
        
        # Parse filename for metadata
        filename = Path(file_path).stem
        parts = filename.replace('_preprocessed', '').split('_')
        
        subject_id = parts[0] if len(parts) > 0 else "unknown"
        run_id = parts[1] if len(parts) > 1 else "unknown"
        seizure_id = parts[2] if len(parts) > 2 else None
        
        # Extract signal metadata
        signal_metadata = {
            'sampling_rate': channel_data.get('processed_fs', 125),
            'original_sampling_rate': channel_data.get('original_fs', 250),
            'signal_length_samples': len(channel_data.get('signal', [])),
            'preprocessing_info': data.get('preprocessing_params', {})
        }
        
        sampling_rate = signal_metadata['sampling_rate']
        signal_metadata['signal_duration_seconds'] = signal_metadata['signal_length_samples'] / sampling_rate
        
        # Extract signal
        if 'signal' in channel_data:
            signal = channel_data['signal']
        elif 'windows' in channel_data and len(channel_data['windows']) > 0:
            signal = np.concatenate(channel_data['windows'])
        else:
            return None
        
        # Extract ground truth
        ground_truth = self.extract_ground_truth(data, subject_id, run_id, seizure_id)
        
        return {
            'signal': signal,
            'metadata': {
                'subject_id': subject_id,
                'run_id': run_id,
                'seizure_id': seizure_id,
                'source_file': file_path,
                'signal_metadata': signal_metadata
            },
            'ground_truth': ground_truth
        }
    
    def extract_ground_truth(self, data: Dict, subject_id: str, run_id: str, seizure_id: Optional[str]) -> Dict:
        """Extract ground truth for windowed data"""
        ground_truth = {
            'seizure_present': False,
            'seizure_regions': [],
            'annotation_source': 'preprocessed_data',
            'annotator_id': 'automated'
        }
        
        if seizure_id is not None:
            ground_truth['seizure_present'] = True
            # Add estimated seizure region (simplified)
            signal_length = data['channels'][0].get('signal_length', 0)
            if signal_length > 0:
                onset_sample = int(signal_length * 0.4)
                offset_sample = int(signal_length * 0.6)
                sampling_rate = data['channels'][0].get('processed_fs', 125)
                
                ground_truth['seizure_regions'] = [
                    {
                        'onset_sample': onset_sample,
                        'offset_sample': offset_sample,
                        'onset_time_seconds': onset_sample / sampling_rate,
                        'offset_time_seconds': offset_sample / sampling_rate,
                        'duration_seconds': (offset_sample - onset_sample) / sampling_rate,
                        'seizure_type': 'focal',
                        'confidence': 0.5
                    }
                ]
        
        return ground_truth
    
    def extract_seizure_only_ground_truth(self, data: Dict, sampling_rate: int) -> Dict:
        """Extract ground truth from seizure-only data"""
        ground_truth = {
            'seizure_present': True,
            'seizure_regions': [],
            'annotation_source': 'seizure_only_preprocessing',
            'annotator_id': 'automated'
        }
        
        # Extract seizure timing
        original_seizure_start = data.get('original_seizure_start', 0)
        original_seizure_end = data.get('original_seizure_end', 0)
        extraction_start = data.get('extraction_start', 0)
        extraction_end = data.get('extraction_end', 0)
        
        # Calculate relative position
        if original_seizure_start >= extraction_start and original_seizure_end <= extraction_end:
            relative_start = original_seizure_start - extraction_start
            relative_end = original_seizure_end - extraction_start
            
            onset_sample = int(relative_start * sampling_rate)
            offset_sample = int(relative_end * sampling_rate)
            
            ground_truth['seizure_regions'] = [
                {
                    'onset_sample': onset_sample,
                    'offset_sample': offset_sample,
                    'onset_time_seconds': relative_start,
                    'offset_time_seconds': relative_end,
                    'duration_seconds': relative_end - relative_start,
                    'seizure_type': 'focal',
                    'confidence': 1.0
                }
            ]
        else:
            # Fallback: Use labels
            if 'channels' in data and len(data['channels']) > 0:
                channel = data['channels'][0]
                if 'labels' in channel:
                    labels = channel['labels']
                    ictal_indices = np.where(labels == 'ictal')[0]
                    
                    if len(ictal_indices) > 0:
                        onset_sample = ictal_indices[0]
                        offset_sample = ictal_indices[-1]
                        
                        ground_truth['seizure_regions'] = [
                            {
                                'onset_sample': int(onset_sample),
                                'offset_sample': int(offset_sample),
                                'onset_time_seconds': float(onset_sample / sampling_rate),
                                'offset_time_seconds': float(offset_sample / sampling_rate),
                                'duration_seconds': float((offset_sample - onset_sample) / sampling_rate),
                                'seizure_type': 'focal',
                                'confidence': 0.9
                            }
                        ]
        
        return ground_truth
    
    def analyze_with_madrid(self, signal: np.ndarray, signal_metadata: Dict) -> Tuple[Dict, Dict]:
        """Perform Madrid analysis on signal"""
        start_time = time.time()
        
        # Get adapted parameters
        sampling_rate = signal_metadata['sampling_rate']
        madrid_params = self.adapt_madrid_params_for_sampling_rate(sampling_rate)
        
        # Calculate train_test_split from minutes or ratio
        train_test_split = self.calculate_train_test_split(signal, signal_metadata)
        
        # Remove constant regions before analysis
        cleaned_signal, removal_info = self.remove_constant_regions(signal, madrid_params['min_length'])
        
        # Log constant region removal if any
        if removal_info['regions_removed'] > 0:
            logger.info(f"Removed {removal_info['regions_removed']} constant regions "
                       f"({removal_info['total_samples_removed']} samples, "
                       f"{removal_info['removal_percentage']:.1f}%)")
            
            # Recalculate train_test_split for cleaned signal
            train_test_split = self.calculate_train_test_split_for_cleaned_signal(
                cleaned_signal, signal_metadata, removal_info, train_test_split
            )
        
        try:
            # Run Madrid on cleaned signal
            multi_length_table, bsf, bsf_loc = self.madrid_detector.fit(
                T=cleaned_signal,
                min_length=madrid_params['min_length'],
                max_length=madrid_params['max_length'],
                step_size=madrid_params['step_size'],
                train_test_split=train_test_split,
                factor=1
            )
            
            # Get anomalies
            anomaly_info = self.madrid_detector.get_anomaly_scores(
                threshold_percentile=self.madrid_params['analysis_config']['threshold_percentile']
            )
            anomalies = anomaly_info['anomalies']
            
            execution_time = time.time() - start_time
            
            # Format results
            formatted_anomalies = []
            for i, anomaly in enumerate(anomalies):
                formatted_anomaly = {
                    'rank': i + 1,
                    'm_value': anomaly['length'],
                    'anomaly_score': float(anomaly['score']),
                    'location_sample': int(anomaly['location']) if anomaly['location'] is not None else None,
                    'location_time_seconds': float(anomaly['location'] / sampling_rate) if anomaly['location'] is not None else None,
                    'seizure_hit': False,
                    'normalized_score': float(anomaly['score']),
                    'confidence': min(float(anomaly['score']), 1.0),
                    'anomaly_id': f"m{anomaly['length']}_rank{i+1}_loc{anomaly['location'] if anomaly['location'] is not None else 'unknown'}"
                }
                formatted_anomalies.append(formatted_anomaly)
            
            analysis_results = {
                'anomalies': formatted_anomalies,
                'performance_metrics': {
                    'total_anomalies_detected': len(anomalies),
                    'threshold_used': float(anomaly_info['threshold']),
                    'analysis_successful': True
                },
                'multi_length_matrix': {
                    'description': 'Multi-length discord table with normalized scores',
                    'shape': list(multi_length_table.shape),
                    'm_values': list(range(madrid_params['min_length'], madrid_params['max_length'] + 1, madrid_params['step_size'])),
                    'statistics': {
                        'max_score': float(np.max(bsf)),
                        'min_score': float(np.min(bsf[~np.isnan(bsf)])) if len(bsf[~np.isnan(bsf)]) > 0 else 0.0,
                        'mean_score': float(np.mean(bsf[~np.isnan(bsf)])) if len(bsf[~np.isnan(bsf)]) > 0 else 0.0,
                        'std_score': float(np.std(bsf[~np.isnan(bsf)])) if len(bsf[~np.isnan(bsf)]) > 0 else 0.0
                    }
                },
                'signal_preprocessing': {
                    'original_length': len(signal),
                    'cleaned_length': len(cleaned_signal),
                    'constant_regions_removed': removal_info['regions_removed'],
                    'total_samples_removed': removal_info['total_samples_removed'],
                    'removal_details': removal_info
                }
            }
            
            performance_info = {
                'execution_time_seconds': execution_time,
                'gpu_used': self.madrid_params['algorithm_settings']['use_gpu'],
                'memory_peak_mb': 0,
                'analysis_successful': True
            }
            
            return analysis_results, performance_info
            
        except Exception as e:
            logger.error(f"Madrid analysis failed: {e}")
            
            analysis_results = {
                'anomalies': [],
                'performance_metrics': {
                    'total_anomalies_detected': 0,
                    'analysis_successful': False,
                    'error_message': str(e)
                },
                'multi_length_matrix': {
                    'description': 'Analysis failed',
                    'shape': [0, 0],
                    'm_values': [],
                    'statistics': {}
                }
            }
            
            performance_info = {
                'execution_time_seconds': time.time() - start_time,
                'gpu_used': False,
                'analysis_successful': False,
                'error_message': str(e)
            }
            
            return analysis_results, performance_info
    
    def calculate_train_test_split(self, signal: np.ndarray, signal_metadata: Dict) -> int:
        """Calculate train_test_split from minutes or ratio"""
        sampling_rate = signal_metadata['sampling_rate']
        signal_length = len(signal)
        
        # Check if train_minutes is specified
        if 'train_minutes' in self.madrid_params['analysis_config']:
            train_minutes = self.madrid_params['analysis_config']['train_minutes']
            train_samples = int(train_minutes * 60 * sampling_rate)
            # Ensure we don't exceed signal length
            train_test_split = min(train_samples, signal_length - 1)
        else:
            # Fallback to ratio-based approach
            train_test_split = int(signal_length * self.madrid_params['analysis_config']['train_test_split_ratio'])
        
        return train_test_split
    
    def contains_constant_regions(self, T: np.ndarray, min_length: int) -> bool:
        """
        Check if time series contains constant regions that would make anomaly detection trivial
        Based on madrid_v2.py logic
        
        Args:
            T: Time series
            min_length: Minimum subsequence length
            
        Returns:
            True if constant regions exist
        """
        # Find consecutive identical values
        diff_T = np.diff(T)
        constant_starts = np.where(np.abs(diff_T) < 1e-10)[0]
        
        if len(constant_starts) == 0:
            return False
            
        # Check for regions longer than min_length
        consecutive_groups = np.split(constant_starts, 
                                    np.where(np.diff(constant_starts) != 1)[0] + 1)
        
        max_constant_length = max(len(group) + 1 for group in consecutive_groups)
        
        # Also check overall variance
        if np.var(T) < 0.2 or max_constant_length >= min_length:
            return True
            
        return False
    
    def find_constant_regions(self, T: np.ndarray, min_length: int) -> List[Tuple[int, int]]:
        """
        Find all constant regions in the time series
        
        Args:
            T: Time series
            min_length: Minimum length for a region to be considered constant
            
        Returns:
            List of (start_idx, end_idx) tuples for constant regions
        """
        # Find consecutive identical values
        diff_T = np.diff(T)
        constant_starts = np.where(np.abs(diff_T) < 1e-10)[0]
        
        if len(constant_starts) == 0:
            return []
        
        # Group consecutive indices
        consecutive_groups = np.split(constant_starts, 
                                    np.where(np.diff(constant_starts) != 1)[0] + 1)
        
        constant_regions = []
        for group in consecutive_groups:
            if len(group) + 1 >= min_length:  # +1 because diff reduces length by 1
                start_idx = group[0]
                end_idx = group[-1] + 2  # +2 to include the last constant value
                constant_regions.append((start_idx, min(end_idx, len(T))))
        
        return constant_regions
    
    def remove_constant_regions(self, signal: np.ndarray, min_length: int) -> Tuple[np.ndarray, Dict]:
        """
        Remove constant regions from signal instead of skipping the entire file
        
        Args:
            signal: Input time series
            min_length: Minimum length for Madrid analysis
            
        Returns:
            Tuple of (cleaned_signal, removal_info)
        """
        # Check if signal has constant regions
        if not self.contains_constant_regions(signal, min_length):
            return signal, {
                'regions_removed': 0,
                'total_samples_removed': 0,
                'original_length': len(signal),
                'cleaned_length': len(signal),
                'constant_regions': []
            }
        
        # Find constant regions
        constant_regions = self.find_constant_regions(signal, min_length)
        
        if not constant_regions:
            return signal, {
                'regions_removed': 0,
                'total_samples_removed': 0,
                'original_length': len(signal),
                'cleaned_length': len(signal),
                'constant_regions': []
            }
        
        # Remove constant regions by creating segments
        segments = []
        last_end = 0
        
        for start, end in constant_regions:
            # Add segment before constant region
            if start > last_end:
                segments.append(signal[last_end:start])
            last_end = end
        
        # Add final segment after last constant region
        if last_end < len(signal):
            segments.append(signal[last_end:])
        
        # Concatenate segments
        if segments:
            cleaned_signal = np.concatenate(segments)
        else:
            # If entire signal is constant, create minimal signal with noise
            cleaned_signal = np.random.randn(min_length * 2) * 0.01
        
        # Calculate removal statistics
        total_samples_removed = sum(end - start for start, end in constant_regions)
        
        removal_info = {
            'regions_removed': len(constant_regions),
            'total_samples_removed': total_samples_removed,
            'original_length': len(signal),
            'cleaned_length': len(cleaned_signal),
            'constant_regions': constant_regions,
            'removal_percentage': (total_samples_removed / len(signal)) * 100
        }
        
        return cleaned_signal, removal_info
    
    def calculate_train_test_split_for_cleaned_signal(self, cleaned_signal: np.ndarray, 
                                                     signal_metadata: Dict, 
                                                     removal_info: Dict, 
                                                     original_train_test_split: int) -> int:
        """
        Recalculate train_test_split for cleaned signal, considering removed regions
        
        Args:
            cleaned_signal: Signal after constant region removal
            signal_metadata: Original signal metadata
            removal_info: Information about removed regions
            original_train_test_split: Original train/test split point
            
        Returns:
            Adjusted train_test_split for cleaned signal
        """
        # Calculate proportion of original split point
        original_length = removal_info['original_length']
        split_ratio = original_train_test_split / original_length
        
        # Apply same ratio to cleaned signal
        new_train_test_split = int(len(cleaned_signal) * split_ratio)
        
        # Ensure it's within bounds
        new_train_test_split = max(1, min(new_train_test_split, len(cleaned_signal) - 1))
        
        return new_train_test_split
    
    def adapt_madrid_params_for_sampling_rate(self, sampling_rate: int) -> Dict:
        """Adapt Madrid parameters for sampling rate"""
        if sampling_rate > 0: 
            return {
                'min_length': sampling_rate*10,
                'max_length': sampling_rate*100,
                'step_size': sampling_rate*10
            }
    
    def validate_against_ground_truth(self, anomalies: List[Dict], ground_truth: Dict, sampling_rate: int) -> List[Dict]:
        """Validate anomalies against ground truth"""
        seizure_overlap_info = []
        
        for anomaly in anomalies:
            overlap_info = {
                'anomaly_id': anomaly['anomaly_id'],
                'overlap_with_seizure': False,
                'overlap_sample_start': None,
                'overlap_sample_end': None,
                'overlap_ratio': 0.0,
                'distance_to_onset': None,
                'classification': 'false_positive'
            }
            
            if ground_truth['seizure_present'] and anomaly['location_sample'] is not None:
                anomaly_location = anomaly['location_sample']
                
                for seizure_region in ground_truth['seizure_regions']:
                    onset_sample = seizure_region['onset_sample']
                    offset_sample = seizure_region['offset_sample']
                    
                    tolerance_samples = int(30 * sampling_rate)
                    
                    if onset_sample - tolerance_samples <= anomaly_location <= offset_sample + tolerance_samples:
                        overlap_info['overlap_with_seizure'] = True
                        overlap_info['overlap_sample_start'] = onset_sample
                        overlap_info['overlap_sample_end'] = offset_sample
                        overlap_info['distance_to_onset'] = abs(anomaly_location - onset_sample)
                        overlap_info['classification'] = 'true_positive'
                        overlap_info['overlap_ratio'] = 1.0 if onset_sample <= anomaly_location <= offset_sample else 0.8
                        
                        anomaly['seizure_hit'] = True
                        break
            
            seizure_overlap_info.append(overlap_info)
        
        return seizure_overlap_info
    
    def create_json_output(self, file_data: Dict, analysis_results: Dict, performance_info: Dict) -> Dict:
        """Create JSON output"""
        timestamp = datetime.now().isoformat() + 'Z'
        metadata = file_data['metadata']
        
        analysis_id = f"madrid_{metadata['subject_id']}_{metadata['run_id']}"
        if metadata['seizure_id']:
            analysis_id += f"_{metadata['seizure_id']}"
        analysis_id += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Validate against ground truth
        seizure_overlap_info = self.validate_against_ground_truth(
            analysis_results['anomalies'],
            file_data['ground_truth'],
            metadata['signal_metadata']['sampling_rate']
        )
        
        # Calculate metrics
        true_positives = sum(1 for info in seizure_overlap_info if info['classification'] == 'true_positive')
        false_positives = len(seizure_overlap_info) - true_positives
        
        analysis_results['performance_metrics'].update({
            'true_positives': true_positives,
            'false_positives': false_positives,
            'sensitivity': true_positives / max(1, len(file_data['ground_truth']['seizure_regions'])) if file_data['ground_truth']['seizure_present'] else 0,
            'precision': true_positives / max(1, len(analysis_results['anomalies'])) if analysis_results['anomalies'] else 0
        })
        
        return {
            'analysis_metadata': {
                'analysis_id': analysis_id,
                'timestamp': timestamp,
                'madrid_version': '1.0.0',
                'analysis_type': 'single_file',
                'computation_info': performance_info
            },
            'input_data': {
                'source_file': metadata['source_file'],
                'subject_id': metadata['subject_id'],
                'run_id': metadata['run_id'],
                'seizure_id': metadata['seizure_id'],
                'signal_metadata': metadata['signal_metadata']
            },
            'madrid_parameters': self.madrid_params,
            'analysis_results': analysis_results,
            'validation_data': {
                'ground_truth': file_data['ground_truth'],
                'seizure_overlap_info': seizure_overlap_info
            }
        }

class ParallelMadridBatchProcessor:
    """
    Parallel Madrid Batch Processor
    """
    
    def __init__(self, config: Dict[str, Any], n_workers: int = None):
        self.config = config
        self.n_workers = n_workers or max(1, cpu_count() - 1)
        
        logger.info(f"Parallel Madrid Processor initialized with {self.n_workers} workers")
    
    def process_files(self, data_dir: str, output_dir: str, max_files: int = None) -> List[str]:
        """Process files in parallel"""
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Find files
        pkl_files = list(Path(data_dir).glob("*_preprocessed.pkl"))
        
        if max_files:
            pkl_files = pkl_files[:max_files]
        
        logger.info(f"Found {len(pkl_files)} files to process with {self.n_workers} workers")
        
        if len(pkl_files) == 0:
            logger.warning("No files found to process")
            return []
        
        # Setup progress tracking
        progress_tracker = ProgressTracker(len(pkl_files))
        
        # Create partial function with fixed arguments
        process_func = partial(process_single_file, config=self.config, output_dir=output_dir)
        
        processed_files = []
        failed_files = []
        
        # Process files in parallel
        start_time = time.time()
        
        with Pool(processes=self.n_workers) as pool:
            # Submit all jobs
            file_paths = [str(f) for f in pkl_files]
            results = pool.map(process_func, file_paths)
            
            # Process results
            for success, file_path, result_info in results:
                progress_tracker.update(success)
                
                if success:
                    processed_files.append(file_path)
                    logger.info(f"✓ {Path(file_path).name}: "
                              f"{result_info['num_anomalies']} anomalies, "
                              f"{result_info['true_positives']} TP, "
                              f"{result_info['execution_time']:.1f}s")
                else:
                    failed_files.append(file_path)
                    logger.error(f"❌ {Path(file_path).name}: {result_info.get('error', 'Unknown error')}")
        
        total_time = time.time() - start_time
        
        # Summary
        logger.info(f"\n📊 PARALLEL PROCESSING SUMMARY")
        logger.info(f"Total processing time: {total_time/60:.1f} minutes")
        logger.info(f"Average time per file: {total_time/len(pkl_files):.1f} seconds")
        logger.info(f"Successfully processed: {len(processed_files)}/{len(pkl_files)}")
        logger.info(f"Failed files: {len(failed_files)}")
        logger.info(f"Speedup factor: ~{self.n_workers}x (theoretical)")
        
        return processed_files

def main():
    """Main function for parallel processing"""
    parser = argparse.ArgumentParser(description='Parallel Madrid Batch Processor')
    parser.add_argument('--data-dir', required=True, help='Directory containing preprocessed .pkl files')
    parser.add_argument('--output-dir', required=True, help='Directory to save JSON results')
    parser.add_argument('--max-files', type=int, help='Maximum number of files to process')
    parser.add_argument('--config-file', help='JSON configuration file')
    parser.add_argument('--n-workers', type=int, help=f'Number of worker processes (default: {cpu_count()-1})')
    parser.add_argument('--use-gpu', action='store_true', default=True, help='Use GPU acceleration')
    parser.add_argument('--threshold-percentile', type=float, default=90, help='Anomaly threshold percentile')
    parser.add_argument('--train-minutes', type=float, default=30, help='Minutes of data to use for training')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            'use_gpu': args.use_gpu,
            'enable_output': False,
            'madrid_parameters': {
                'm_range': {
                    'min_length': 80,
                    'max_length': 800,
                    'step_size': 80
                },
                'analysis_config': {
                    'top_k': 5,
                    'train_test_split_ratio': 0.3,
                    'train_minutes': args.train_minutes,
                    'threshold_percentile': args.threshold_percentile
                },
                'algorithm_settings': {
                    'use_gpu': args.use_gpu,
                    'downsampling_factor': 1
                }
            }
        }
    
    # Initialize parallel processor
    processor = ParallelMadridBatchProcessor(config, n_workers=args.n_workers)
    
    # Process files
    processed_files = processor.process_files(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_files=args.max_files
    )
    
    logger.info(f"Parallel processing completed. Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()