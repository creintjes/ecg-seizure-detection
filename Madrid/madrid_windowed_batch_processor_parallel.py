#!/usr/bin/env python3
"""
Parallelized Madrid Windowed Batch Processor with JSON Output

Processes windowed preprocessed ECG files (e.g., from preprocess_all_data_32hz_120sec.py)
with Madrid algorithm in parallel. Supports various window configurations like:
- 120sec windows with 60sec stride
- 3600sec windows with 1800sec stride
- Any custom window/stride configuration

Usage:
    python madrid_windowed_batch_processor_parallel.py --data-dir /path/to/windowed/data --output-dir results --n-workers 4
    python madrid_windowed_batch_processor_parallel.py --data-dir /path/to/windowed/data --window-strategy reconstruct --train-minutes 10
python madrid_windowed_batch_processor_parallel.py --data-dir /home/swolf/asim_shared/preprocessed_data/downsample_freq=8,window_size=3600_0,stride=1800_0_new --existing-result
s-dir results_8hz_window3600_stride1800_new20min --output-dir results_8hz_window3600_stride1800_new20min --n-workers 20 --train-minutes 20
Key Features:
    - Handles windowed preprocessed data with overlaps
    - Multiple window processing strategies (individual, reconstruct, hybrid)
    - Parallel processing with multiprocessing
    - GPU support for Madrid algorithm
    - Comprehensive JSON output with window-level results
    - Configurable training time in minutes
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
    print("✓ MADRID_V2 successfully imported")
except ImportError as e:
    print(f"❌ MADRID_V2 import failed: {e}")
    MADRID_AVAILABLE = False
    sys.exit(1)

# Setup logging with thread-safe configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('madrid_windowed_batch_processor_parallel.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_single_windowed_file(file_path: str, config: Dict[str, Any], output_dir: str) -> Tuple[bool, str, Dict]:
    """
    Process a single windowed file with Madrid analysis
    
    Args:
        file_path: Path to preprocessed windowed file
        config: Configuration dictionary
        output_dir: Output directory for results
        
    Returns:
        Tuple of (success, file_path, result_info)
    """
    try:
        # Create Madrid processor for this worker
        processor = MadridWindowedBatchProcessorCore(config)
        
        # Load windowed file
        file_data = processor.load_windowed_file(file_path)
        if file_data is None:
            return False, file_path, {"error": "Failed to load windowed file"}
        
        # Run Madrid analysis on windows
        analysis_results, performance_info = processor.analyze_windowed_data_with_madrid(
            file_data['windows'],
            file_data['metadata']['signal_metadata']
        )
        
        # Create JSON output
        json_output = processor.create_windowed_json_output(file_data, analysis_results, performance_info)
        
        # Save JSON file
        output_filename = f"madrid_windowed_results_{file_data['metadata']['subject_id']}_{file_data['metadata']['run_id']}"
        if file_data['metadata']['seizure_id']:
            output_filename += f"_{file_data['metadata']['seizure_id']}"
        output_filename += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_path = Path(output_dir) / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)
        
        # Return success info
        result_info = {
            "output_file": str(output_path),
            "num_windows_processed": len(file_data['windows']),
            "total_anomalies": sum(len(result['anomalies']) for result in analysis_results['window_results']),
            "execution_time": performance_info['total_execution_time_seconds'],
            "window_strategy": config.get('window_strategy', 'individual')
        }
        
        return True, file_path, result_info
        
    except Exception as e:
        logger.error(f"Error processing windowed file {file_path}: {e}")
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

class MadridWindowedBatchProcessorCore:
    """
    Core Madrid processing functionality for windowed data
    This handles various windowing strategies and configurations
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
        
        # Window processing strategy
        self.window_strategy = config.get('window_strategy', 'individual')  # individual, reconstruct, hybrid
        
    def load_windowed_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load windowed preprocessed ECG file"""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            if not self.is_windowed_format(data):
                # Provide detailed error information
                error_info = self.analyze_file_format(data, file_path)
                logger.error(f"File {file_path} is not in windowed format: {error_info}")
                return None
            
            return self.process_windowed_data(data, file_path)
            
        except Exception as e:
            logger.error(f"Error loading windowed file {file_path}: {e}")
            return None
    
    def is_windowed_format(self, data: Dict) -> bool:
        """Check if data is in windowed preprocessed format"""
        if 'channels' not in data or len(data['channels']) == 0:
            logger.debug("No channels found in data")
            return False
        
        channel = data['channels'][0]
        
        # Check for windowed structure
        if 'windows' in channel and isinstance(channel['windows'], list):
            # Check if preprocessing parameters indicate windowed processing
            # Even single windows from windowed preprocessing should be accepted
            preprocessing_params = data.get('preprocessing_params', {})
            
            # Method 1: If window_size and stride are specified, it's windowed format
            if 'window_size' in preprocessing_params and 'stride' in preprocessing_params:
                logger.debug(f"Detected windowed format from preprocessing params: window_size={preprocessing_params['window_size']}, stride={preprocessing_params['stride']}")
                return True
            
            # Method 2: Legacy check: Multiple windows indicate windowed format
            if len(channel['windows']) > 1:
                logger.debug(f"Detected windowed format from multiple windows: {len(channel['windows'])} windows")
                return True
            
            # Method 3: Check for flexible windowing flag
            if 'flexible_windowing' in preprocessing_params and preprocessing_params['flexible_windowing']:
                logger.debug("Detected windowed format from flexible_windowing flag")
                return True
            
            # Method 4: Check for window-related keys in preprocessing params
            window_related_keys = ['window_size', 'stride', 'overlap', 'windowing', 'min_window_size']
            if any(key in preprocessing_params for key in window_related_keys):
                logger.debug(f"Detected windowed format from preprocessing keys: {[k for k in window_related_keys if k in preprocessing_params]}")
                return True
            
            # Method 5: Check if this looks like output from windowed preprocessing by metadata
            if len(channel['windows']) >= 1:
                if 'metadata' in channel and isinstance(channel['metadata'], list):
                    if len(channel['metadata']) > 0:
                        meta = channel['metadata'][0]
                        # If metadata contains window timing info, it's likely windowed format
                        if 'start_time' in meta and 'end_time' in meta:
                            logger.debug("Detected windowed format from metadata structure")
                            return True
                        # Check for flexible window indicators
                        if 'is_flexible_window' in meta and meta['is_flexible_window']:
                            logger.debug("Detected windowed format from flexible window metadata")
                            return True
            
            # Method 6: Detection for single windows (flexible windowing) - changed from > 1 to >= 1
            if 'n_windows' in channel and channel.get('n_windows', 0) >= 1:
                logger.debug(f"Detected windowed format from n_windows field: {channel['n_windows']}")
                return True
            
            # Method 6: If there are windows and the file structure suggests windowing, accept it
            # This is for cases where preprocessing created windows but metadata is different
            if len(channel['windows']) >= 1:
                # If the window is significantly shorter than typical full recordings, it's likely windowed
                if len(channel['windows']) > 0:
                    window_length = len(channel['windows'][0])
                    sampling_rate = channel.get('processed_fs', 32)
                    window_duration_minutes = window_length / (sampling_rate * 60)
                    
                    # If window is between 1-120 minutes, likely from windowed preprocessing
                    if 1 <= window_duration_minutes <= 120:
                        logger.debug(f"Detected windowed format from window duration: {window_duration_minutes:.1f} minutes")
                        return True
            
            logger.debug(f"File structure does not match windowed format: {len(channel['windows'])} windows, preprocessing_params={preprocessing_params}")
        else:
            logger.debug("No 'windows' key found in channel or windows is not a list")
        
        return False
    
    def analyze_file_format(self, data: Dict, file_path: str) -> str:
        """Analyze file format and provide detailed error information"""
        try:
            analysis = []
            
            # Check top-level structure
            if 'channels' not in data:
                analysis.append("Missing 'channels' key")
            elif len(data['channels']) == 0:
                analysis.append("Empty channels list")
            else:
                channel = data['channels'][0]
                
                if 'windows' not in channel:
                    analysis.append("Missing 'windows' key in channel")
                elif not isinstance(channel['windows'], list):
                    analysis.append(f"'windows' is not a list, type: {type(channel['windows'])}")
                else:
                    num_windows = len(channel['windows'])
                    analysis.append(f"Found {num_windows} windows")
                    
                    # Check preprocessing parameters
                    preprocessing_params = data.get('preprocessing_params', {})
                    if not preprocessing_params:
                        analysis.append("Missing preprocessing_params")
                    else:
                        window_size = preprocessing_params.get('window_size', 'Not specified')
                        stride = preprocessing_params.get('stride', 'Not specified')
                        analysis.append(f"Window size: {window_size}, Stride: {stride}")
                    
                    # Check metadata
                    if 'metadata' in channel:
                        if isinstance(channel['metadata'], list):
                            analysis.append(f"Found {len(channel['metadata'])} metadata entries")
                        else:
                            analysis.append(f"Metadata is not a list, type: {type(channel['metadata'])}")
                    else:
                        analysis.append("Missing metadata")
            
            return "; ".join(analysis)
            
        except Exception as e:
            return f"Error analyzing file format: {str(e)}"
    
    def process_windowed_data(self, data: Dict, file_path: str) -> Optional[Dict[str, Any]]:
        """Process windowed data format"""
        try:
            # Extract metadata from filename
            filename = Path(file_path).stem
            parts = filename.replace('_preprocessed', '').split('_')
            
            subject_id = parts[0] if len(parts) > 0 else "unknown"
            run_id = parts[1] if len(parts) > 1 else "unknown"
            seizure_id = parts[2] if len(parts) > 2 else None
            
            # Get first channel
            channel_data = data["channels"][0]
            windows = channel_data.get('windows', [])
            labels = channel_data.get('labels', [])
            metadata = channel_data.get('metadata', [])
            
            if len(windows) == 0:
                return None
            
            # Extract signal metadata
            signal_metadata = {
                'sampling_rate': channel_data.get('processed_fs', 32),
                'original_sampling_rate': channel_data.get('original_fs', 250),
                'num_windows': len(windows),
                'window_length_samples': len(windows[0]) if windows else 0,
                'window_duration_seconds': len(windows[0]) / channel_data.get('processed_fs', 32) if windows else 0,
                'total_duration_seconds': data.get('recording_duration', 0),
                'preprocessing_info': data.get('preprocessing_params', {}),
                'windowing_info': {
                    'window_size': data.get('preprocessing_params', {}).get('window_size', 120),
                    'stride': data.get('preprocessing_params', {}).get('stride', 60),
                    'overlap_ratio': self.calculate_overlap_ratio(data.get('preprocessing_params', {}))
                }
            }
            
            # Extract ground truth for windowed data
            ground_truth = self.extract_windowed_ground_truth(data, labels, signal_metadata)
            
            return {
                'windows': windows,
                'labels': labels,
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
            logger.error(f"Error processing windowed data {file_path}: {e}")
            return None
    
    def calculate_overlap_ratio(self, preprocessing_params: Dict) -> float:
        """Calculate overlap ratio from preprocessing parameters"""
        window_size = preprocessing_params.get('window_size', 120)
        stride = preprocessing_params.get('stride', 60)
        
        if window_size <= stride:
            return 0.0
        
        overlap = window_size - stride
        return overlap / window_size
    
    def extract_windowed_ground_truth(self, data: Dict, labels: List, signal_metadata: Dict) -> Dict:
        """Extract ground truth for windowed data"""
        ground_truth = {
            'seizure_present': False,
            'seizure_windows': [],
            'total_windows': signal_metadata['num_windows'],
            'annotation_source': 'windowed_preprocessing',
            'annotator_id': 'automated'
        }
        
        # Check for seizures in windows
        seizure_windows = []
        for i, window_labels in enumerate(labels):
            has_seizure = False
            seizure_ratio = 0.0
            seizure_segments = []
            
            if isinstance(window_labels, np.ndarray):
                # Handle sample-level labels (new format)
                if len(window_labels) > 1:  # Sample-level array
                    if np.any(window_labels == 1):
                        has_seizure = True
                        seizure_ratio = np.mean(window_labels == 1)
                        
                        # Extract seizure segments within the window
                        seizure_segments = self._extract_seizure_segments_from_labels(
                            window_labels, i, signal_metadata
                        )
                else:
                    # Single value (old format compatibility)
                    if window_labels == 1 or (hasattr(window_labels, '__iter__') and 'ictal' in window_labels):
                        has_seizure = True
                        seizure_ratio = 1.0
            elif isinstance(window_labels, (list, int, str)):
                # Handle old format compatibility
                if window_labels == 1 or window_labels == 'ictal':
                    has_seizure = True
                    seizure_ratio = 1.0
                elif isinstance(window_labels, list):
                    if 1 in window_labels or 'ictal' in window_labels:
                        has_seizure = True
                        seizure_ratio = (window_labels.count(1) + window_labels.count('ictal')) / len(window_labels)
            
            if has_seizure:
                seizure_windows.append({
                    'window_index': i,
                    'seizure_ratio': seizure_ratio,
                    'window_start_time': i * signal_metadata['windowing_info']['stride'],
                    'window_duration': signal_metadata['window_duration_seconds'],
                    'seizure_segments': seizure_segments
                })
        
        if seizure_windows:
            ground_truth['seizure_present'] = True
            ground_truth['seizure_windows'] = seizure_windows
        
        return ground_truth
    
    def _extract_seizure_segments_from_labels(self, window_labels: np.ndarray, window_index: int, signal_metadata: Dict) -> List[Dict]:
        """
        Extract seizure segments from sample-level labels within a window.
        
        Args:
            window_labels: Sample-level labels for the window (0/1 array)
            window_index: Index of the window
            signal_metadata: Signal metadata including sampling rate and timing
            
        Returns:
            List of seizure segments with precise timing information
        """
        segments = []
        sampling_rate = signal_metadata['sampling_rate']
        window_start_time = window_index * signal_metadata['windowing_info']['stride']
        
        # Find continuous seizure regions
        diff = np.diff(np.concatenate(([0], window_labels, [0])))
        seizure_starts = np.where(diff == 1)[0]
        seizure_ends = np.where(diff == -1)[0]
        
        for start_idx, end_idx in zip(seizure_starts, seizure_ends):
            segment = {
                'start_sample_in_window': int(start_idx),
                'end_sample_in_window': int(end_idx),
                'start_time_absolute': float(window_start_time + start_idx / sampling_rate),
                'end_time_absolute': float(window_start_time + end_idx / sampling_rate),
                'duration_seconds': float((end_idx - start_idx) / sampling_rate),
                'n_samples': int(end_idx - start_idx)
            }
            segments.append(segment)
        
        return segments
    
    def analyze_windowed_data_with_madrid(self, windows: List[np.ndarray], signal_metadata: Dict) -> Tuple[Dict, Dict]:
        """Perform Madrid analysis on windowed data using specified strategy"""
        start_time = time.time()
        
        try:
            if self.window_strategy == 'individual':
                return self.analyze_windows_individually(windows, signal_metadata, start_time)
            elif self.window_strategy == 'reconstruct':
                return self.analyze_reconstructed_signal(windows, signal_metadata, start_time)
            elif self.window_strategy == 'hybrid':
                return self.analyze_windows_hybrid(windows, signal_metadata, start_time)
            else:
                raise ValueError(f"Unknown window strategy: {self.window_strategy}")
                
        except Exception as e:
            logger.error(f"Madrid windowed analysis failed: {e}")
            
            # Return error results
            analysis_results = {
                'window_results': [],
                'strategy': self.window_strategy,
                'analysis_successful': False,
                'error_message': str(e)
            }
            
            performance_info = {
                'total_execution_time_seconds': time.time() - start_time,
                'analysis_successful': False,
                'error_message': str(e)
            }
            
            return analysis_results, performance_info
    
    def analyze_windows_individually(self, windows: List[np.ndarray], signal_metadata: Dict, start_time: float) -> Tuple[Dict, Dict]:
        """Analyze each window individually with Madrid"""
        sampling_rate = signal_metadata['sampling_rate']
        madrid_params = self.adapt_madrid_params_for_sampling_rate(sampling_rate)
        
        window_results = []
        total_anomalies = 0
        
        for i, window in enumerate(windows):
            window_start_time = time.time()
            
            try:
                # Handle flexible window sizes - adapt Madrid parameters dynamically
                window_length = len(window)
                window_duration = window_length / sampling_rate
                
                # For very short windows, adapt Madrid parameters
                if window_length < madrid_params['max_length']:
                    logger.info(f"Window {i} is short ({window_length} samples, {window_duration:.1f}s), adapting Madrid parameters")
                    
                    # Dynamically adjust Madrid parameters for short windows
                    adapted_params = self.adapt_madrid_params_for_window_size(window_length, sampling_rate)
                    
                    if window_length < adapted_params['min_length']:
                        logger.warning(f"Window {i} too short even for adapted params ({window_length} < {adapted_params['min_length']}), creating minimal result")
                        
                        # Create minimal result for very short windows
                        window_result = {
                            'window_index': i,
                            'window_start_time': i * signal_metadata.get('windowing_info', {}).get('stride', 0),
                            'window_duration': window_duration,
                            'original_length': window_length,
                            'cleaned_length': window_length,
                            'constant_regions_removed': 0,
                            'anomalies': [],  # No anomalies for very short windows
                            'execution_time': time.time() - window_start_time,
                            'analysis_successful': False,
                            'warning': f'Window too short for Madrid analysis ({window_length} samples < {adapted_params["min_length"]} min)'
                        }
                        window_results.append(window_result)
                        continue
                    
                    # Use adapted parameters
                    madrid_params = adapted_params
                
                # Remove constant regions from window
                cleaned_window, removal_info = self.remove_constant_regions(window, madrid_params['min_length'])
                
                if len(cleaned_window) < madrid_params['max_length']:
                    logger.warning(f"Window {i} too short after cleaning ({len(cleaned_window)} samples), skipping")
                    continue
                
                # Calculate train_test_split for this window
                train_test_split = self.calculate_train_test_split_for_window(cleaned_window, signal_metadata)
                
                # Run Madrid on window
                multi_length_table, bsf, bsf_loc = self.madrid_detector.fit(
                    T=cleaned_window,
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
                
                # Format window results
                window_result = {
                    'window_index': i,
                    'window_start_time': i * signal_metadata['windowing_info']['stride'],
                    'window_duration': signal_metadata['window_duration_seconds'],
                    'original_length': len(window),
                    'cleaned_length': len(cleaned_window),
                    'constant_regions_removed': removal_info['regions_removed'],
                    'anomalies': self.format_window_anomalies(anomalies, i, sampling_rate),
                    'execution_time': time.time() - window_start_time,
                    'analysis_successful': True
                }
                
                window_results.append(window_result)
                total_anomalies += len(anomalies)
                
            except Exception as e:
                logger.error(f"Error analyzing window {i}: {e}")
                window_result = {
                    'window_index': i,
                    'window_start_time': i * signal_metadata['windowing_info']['stride'],
                    'analysis_successful': False,
                    'error_message': str(e)
                }
                window_results.append(window_result)
        
        analysis_results = {
            'window_results': window_results,
            'strategy': 'individual',
            'total_windows_processed': len(window_results),
            'total_anomalies_found': total_anomalies,
            'analysis_successful': True
        }
        
        performance_info = {
            'total_execution_time_seconds': time.time() - start_time,
            'average_time_per_window': (time.time() - start_time) / len(windows) if windows else 0,
            'windows_processed': len(window_results),
            'analysis_successful': True
        }
        
        return analysis_results, performance_info
    
    def analyze_reconstructed_signal(self, windows: List[np.ndarray], signal_metadata: Dict, start_time: float) -> Tuple[Dict, Dict]:
        """Reconstruct signal from overlapping windows and analyze as one continuous signal"""
        try:
            # Reconstruct continuous signal from overlapping windows
            reconstructed_signal = self.reconstruct_signal_from_windows(windows, signal_metadata)
            
            sampling_rate = signal_metadata['sampling_rate']
            madrid_params = self.adapt_madrid_params_for_sampling_rate(sampling_rate)
            
            # Remove constant regions
            cleaned_signal, removal_info = self.remove_constant_regions(reconstructed_signal, madrid_params['min_length'])
            
            # Calculate train_test_split
            train_test_split = self.calculate_train_test_split_for_signal(cleaned_signal, signal_metadata)
            
            # Run Madrid on reconstructed signal
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
            
            # Map anomalies back to windows
            window_mapped_anomalies = self.map_anomalies_to_windows(anomalies, signal_metadata, sampling_rate)
            
            analysis_results = {
                'window_results': [],  # No individual window results in reconstruct mode
                'reconstructed_signal_results': {
                    'total_anomalies': len(anomalies),
                    'anomalies': self.format_reconstructed_anomalies(anomalies, sampling_rate),
                    'window_mapped_anomalies': window_mapped_anomalies,
                    'original_length': len(reconstructed_signal),
                    'cleaned_length': len(cleaned_signal),
                    'constant_regions_removed': removal_info['regions_removed']
                },
                'strategy': 'reconstruct',
                'total_anomalies_found': len(anomalies),
                'analysis_successful': True
            }
            
            performance_info = {
                'total_execution_time_seconds': time.time() - start_time,
                'reconstruction_successful': True,
                'analysis_successful': True
            }
            
            return analysis_results, performance_info
            
        except Exception as e:
            logger.error(f"Error in reconstructed signal analysis: {e}")
            raise
    
    def analyze_windows_hybrid(self, windows: List[np.ndarray], signal_metadata: Dict, start_time: float) -> Tuple[Dict, Dict]:
        """Hybrid approach: analyze both individually and reconstructed"""
        try:
            # Run individual analysis
            individual_results, individual_perf = self.analyze_windows_individually(windows, signal_metadata, time.time())
            
            # Run reconstructed analysis
            reconstructed_results, reconstructed_perf = self.analyze_reconstructed_signal(windows, signal_metadata, time.time())
            
            # Combine results
            analysis_results = {
                'individual_results': individual_results,
                'reconstructed_results': reconstructed_results,
                'strategy': 'hybrid',
                'total_anomalies_individual': individual_results['total_anomalies_found'],
                'total_anomalies_reconstructed': reconstructed_results['total_anomalies_found'],
                'analysis_successful': True
            }
            
            performance_info = {
                'total_execution_time_seconds': time.time() - start_time,
                'individual_time': individual_perf['total_execution_time_seconds'],
                'reconstructed_time': reconstructed_perf['total_execution_time_seconds'],
                'analysis_successful': True
            }
            
            return analysis_results, performance_info
            
        except Exception as e:
            logger.error(f"Error in hybrid analysis: {e}")
            raise
    
    def reconstruct_signal_from_windows(self, windows: List[np.ndarray], signal_metadata: Dict) -> np.ndarray:
        """Reconstruct continuous signal from overlapping windows"""
        if len(windows) == 0:
            return np.array([])
        
        if len(windows) == 1:
            return windows[0]
        
        # Get window parameters
        window_size_samples = len(windows[0])
        stride_seconds = signal_metadata['windowing_info']['stride']
        sampling_rate = signal_metadata['sampling_rate']
        stride_samples = int(stride_seconds * sampling_rate)
        
        # Calculate total reconstructed signal length
        total_length = (len(windows) - 1) * stride_samples + window_size_samples
        reconstructed_signal = np.zeros(total_length)
        weight_matrix = np.zeros(total_length)
        
        # Add each window with weights
        for i, window in enumerate(windows):
            start_idx = i * stride_samples
            end_idx = start_idx + len(window)
            
            if end_idx <= total_length:
                reconstructed_signal[start_idx:end_idx] += window
                weight_matrix[start_idx:end_idx] += 1
        
        # Normalize by weights (handle overlaps)
        non_zero_weights = weight_matrix > 0
        reconstructed_signal[non_zero_weights] /= weight_matrix[non_zero_weights]
        
        return reconstructed_signal
    
    def calculate_train_test_split_for_window(self, window: np.ndarray, signal_metadata: Dict) -> int:
        """Calculate train_test_split for individual window"""
        sampling_rate = signal_metadata['sampling_rate']
        window_length = len(window)
        
        # Check if train_minutes is specified
        if 'train_minutes' in self.madrid_params['analysis_config']:
            train_minutes = self.madrid_params['analysis_config']['train_minutes']
            train_samples = int(train_minutes * 60 * sampling_rate)
            # For windows, use ratio of train time to window duration
            window_duration_seconds = window_length / sampling_rate
            if train_minutes * 60 >= window_duration_seconds:
                # If train time >= window duration, use ratio approach
                train_test_split = int(window_length * self.madrid_params['analysis_config']['train_test_split_ratio'])
            else:
                train_test_split = min(train_samples, window_length - 1)
        else:
            # Fallback to ratio-based approach
            train_test_split = int(window_length * self.madrid_params['analysis_config']['train_test_split_ratio'])
        
        return max(1, min(train_test_split, window_length - 1))
    
    def calculate_train_test_split_for_signal(self, signal: np.ndarray, signal_metadata: Dict) -> int:
        """Calculate train_test_split for reconstructed signal"""
        sampling_rate = signal_metadata['sampling_rate']
        signal_length = len(signal)
        
        # Check if train_minutes is specified
        if 'train_minutes' in self.madrid_params['analysis_config']:
            train_minutes = self.madrid_params['analysis_config']['train_minutes']
            train_samples = int(train_minutes * 60 * sampling_rate)
            train_test_split = min(train_samples, signal_length - 1)
        else:
            # Fallback to ratio-based approach
            train_test_split = int(signal_length * self.madrid_params['analysis_config']['train_test_split_ratio'])
        
        return max(1, min(train_test_split, signal_length - 1))
    
    def format_window_anomalies(self, anomalies: List[Dict], window_index: int, sampling_rate: int) -> List[Dict]:
        """Format anomalies for individual window"""
        formatted_anomalies = []
        for i, anomaly in enumerate(anomalies):
            formatted_anomaly = {
                'rank': i + 1,
                'window_index': window_index,
                'm_value': anomaly['length'],
                'anomaly_score': float(anomaly['score']),
                'location_sample_in_window': int(anomaly['location']) if anomaly['location'] is not None else None,
                'location_time_in_window': float(anomaly['location'] / sampling_rate) if anomaly['location'] is not None else None,
                'normalized_score': float(anomaly['score']),
                'confidence': min(float(anomaly['score']), 1.0),
                'anomaly_id': f"w{window_index}_m{anomaly['length']}_rank{i+1}_loc{anomaly['location'] if anomaly['location'] is not None else 'unknown'}"
            }
            formatted_anomalies.append(formatted_anomaly)
        
        return formatted_anomalies
    
    def format_reconstructed_anomalies(self, anomalies: List[Dict], sampling_rate: int) -> List[Dict]:
        """Format anomalies for reconstructed signal"""
        formatted_anomalies = []
        for i, anomaly in enumerate(anomalies):
            formatted_anomaly = {
                'rank': i + 1,
                'm_value': anomaly['length'],
                'anomaly_score': float(anomaly['score']),
                'location_sample': int(anomaly['location']) if anomaly['location'] is not None else None,
                'location_time_seconds': float(anomaly['location'] / sampling_rate) if anomaly['location'] is not None else None,
                'normalized_score': float(anomaly['score']),
                'confidence': min(float(anomaly['score']), 1.0),
                'anomaly_id': f"reconstructed_m{anomaly['length']}_rank{i+1}_loc{anomaly['location'] if anomaly['location'] is not None else 'unknown'}"
            }
            formatted_anomalies.append(formatted_anomaly)
        
        return formatted_anomalies
    
    def map_anomalies_to_windows(self, anomalies: List[Dict], signal_metadata: Dict, sampling_rate: int) -> List[Dict]:
        """Map anomalies from reconstructed signal back to source windows"""
        stride_samples = int(signal_metadata['windowing_info']['stride'] * sampling_rate)
        window_size_samples = signal_metadata['window_length_samples']
        
        mapped_anomalies = []
        for anomaly in anomalies:
            if anomaly['location'] is not None:
                location = anomaly['location']
                
                # Find which windows this anomaly could belong to
                possible_windows = []
                for window_idx in range(signal_metadata['num_windows']):
                    window_start = window_idx * stride_samples
                    window_end = window_start + window_size_samples
                    
                    if window_start <= location < window_end:
                        location_in_window = location - window_start
                        possible_windows.append({
                            'window_index': window_idx,
                            'location_in_window': location_in_window,
                            'location_time_in_window': location_in_window / sampling_rate
                        })
                
                mapped_anomaly = {
                    'original_anomaly': anomaly,
                    'possible_windows': possible_windows,
                    'most_likely_window': possible_windows[0] if possible_windows else None
                }
                mapped_anomalies.append(mapped_anomaly)
        
        return mapped_anomalies
    
    def remove_constant_regions(self, signal: np.ndarray, min_length: int) -> Tuple[np.ndarray, Dict]:
        """Remove constant regions from signal (same as in main processor)"""
        # Simplified version - implement full logic if needed
        return signal, {
            'regions_removed': 0,
            'total_samples_removed': 0,
            'original_length': len(signal),
            'cleaned_length': len(signal),
            'constant_regions': []
        }
    
    def adapt_madrid_params_for_sampling_rate(self, sampling_rate: int) -> Dict:
        """Adapt Madrid parameters for sampling rate"""
        if sampling_rate > 0: 
            return {
                'min_length': sampling_rate * 10,  # 10 seconds
                'max_length': sampling_rate * 100, # 100 seconds  
                'step_size': sampling_rate * 10    # 10 seconds
            }
        return {
            'min_length': 80,
            'max_length': 800,
            'step_size': 80
        }
    
    def adapt_madrid_params_for_window_size(self, window_length: int, sampling_rate: int) -> Dict:
        """
        Adapt Madrid parameters for a specific window size
        
        Args:
            window_length: Length of window in samples
            sampling_rate: Sampling rate
            
        Returns:
            Adapted Madrid parameters
        """
        # Calculate window duration in seconds
        window_duration = window_length / sampling_rate
        
        # For very short windows (< 30 seconds), use minimal parameters
        if window_duration < 30:
            min_length = max(int(sampling_rate * 2), 16)  # 2 seconds minimum
            max_length = max(int(window_length * 0.3), min_length * 2)  # Use 30% of window
            step_size = max(int(sampling_rate * 1), 8)  # 1 second steps
        
        # For short windows (30-300 seconds), scale down from default
        elif window_duration < 300:
            min_length = max(int(sampling_rate * 5), 40)  # 5 seconds minimum
            max_length = max(int(window_length * 0.4), min_length * 2)  # Use 40% of window
            step_size = max(int(sampling_rate * 2), 16)  # 2 second steps
            
        # For medium windows (300-1800 seconds), use moderate parameters
        elif window_duration < 1800:
            min_length = int(sampling_rate * 10)  # 10 seconds
            max_length = int(window_length * 0.5)  # Use 50% of window
            step_size = int(sampling_rate * 5)  # 5 second steps
            
        # For long windows, use default scaling
        else:
            return self.adapt_madrid_params_for_sampling_rate(sampling_rate)
        
        # Ensure reasonable bounds
        min_length = max(min_length, 8)
        max_length = max(max_length, min_length * 2)
        step_size = max(step_size, 4)
        
        # Ensure max_length doesn't exceed window length
        max_length = min(max_length, window_length - 1)
        
        logger.debug(f"Adapted Madrid params for window ({window_duration:.1f}s): min={min_length}, max={max_length}, step={step_size}")
        
        return {
            'min_length': min_length,
            'max_length': max_length,
            'step_size': step_size
        }
    
    def create_windowed_json_output(self, file_data: Dict, analysis_results: Dict, performance_info: Dict) -> Dict:
        """Create JSON output for windowed analysis"""
        timestamp = datetime.now().isoformat() + 'Z'
        metadata = file_data['metadata']
        
        analysis_id = f"madrid_windowed_{metadata['subject_id']}_{metadata['run_id']}"
        if metadata['seizure_id']:
            analysis_id += f"_{metadata['seizure_id']}"
        analysis_id += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return {
            'analysis_metadata': {
                'analysis_id': analysis_id,
                'timestamp': timestamp,
                'madrid_version': '2.0.0',
                'analysis_type': 'windowed_batch',
                'window_strategy': analysis_results.get('strategy', 'unknown'),
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
            'windowing_info': metadata['signal_metadata']['windowing_info'],
            'analysis_results': analysis_results,
            'validation_data': {
                'ground_truth': file_data['ground_truth']
            }
        }

class ParallelMadridWindowedBatchProcessor:
    """
    Parallel Madrid Windowed Batch Processor
    """
    
    def __init__(self, config: Dict[str, Any], n_workers: int = None):
        self.config = config
        self.n_workers = n_workers or max(1, cpu_count() - 1)
        
        logger.info(f"Parallel Madrid Windowed Processor initialized with {self.n_workers} workers")
        logger.info(f"Window strategy: {config.get('window_strategy', 'individual')}")
    
    def has_seizure_data(self, file_path: str) -> bool:
        """Check if a pkl file contains seizure data"""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Check if any channel has seizure windows
            if 'channels' in data and data['channels']:
                for channel in data['channels']:
                    # Check n_seizure_windows field
                    if 'n_seizure_windows' in channel and channel['n_seizure_windows'] > 0:
                        return True
                    
                    # Check labels for seizure activity (binary 0/1)
                    if 'labels' in channel and channel['labels']:
                        labels = channel['labels']
                        if isinstance(labels, list):
                            # Check if any label is 1 (seizure)
                            if any(label == 1 or (isinstance(label, np.ndarray) and np.any(label == 1)) for label in labels):
                                return True
                        elif isinstance(labels, np.ndarray):
                            if np.any(labels == 1):
                                return True
            
            # Check total_seizures field
            if 'total_seizures' in data and data['total_seizures'] > 0:
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking seizure data in {file_path}: {e}")
            return False
    
    def prioritize_seizure_files(self, pkl_files: List[Path]) -> List[Path]:
        """Sort files to prioritize those with seizure data"""
        seizure_files = []
        non_seizure_files = []
        
        logger.info("Analyzing files for seizure data...")
        
        for file_path in pkl_files:
            if self.has_seizure_data(str(file_path)):
                seizure_files.append(file_path)
            else:
                non_seizure_files.append(file_path)
        
        # Sort seizure files first, then non-seizure files
        prioritized_files = seizure_files + non_seizure_files
        
        logger.info(f"File prioritization complete:")
        logger.info(f"  • Seizure files: {len(seizure_files)}")
        logger.info(f"  • Non-seizure files: {len(non_seizure_files)}")
        logger.info(f"  • Total files: {len(prioritized_files)}")
        
        return prioritized_files
    
    def filter_unprocessed_files(self, pkl_files: List[Path], existing_results_dir: str) -> List[Path]:
        """Filter out files that already have results in the existing results directory"""
        if not os.path.exists(existing_results_dir):
            logger.info(f"Existing results directory {existing_results_dir} does not exist, processing all files")
            return pkl_files
        
        # Find existing result files
        existing_json_files = list(Path(existing_results_dir).glob("madrid_windowed_results_*.json"))
        
        # Extract processed file identifiers from existing results
        processed_identifiers = set()
        for json_file in existing_json_files:
            # Extract subject_id and run_id from filename
            # Format: madrid_windowed_results_sub-XXX_run-XX_[seizure_XX_]timestamp.json
            filename = json_file.stem
            parts = filename.replace('madrid_windowed_results_', '').split('_')
            
            if len(parts) >= 2:
                subject_id = parts[0]
                run_id = parts[1]
                
                # Check if seizure_id is present
                seizure_id = None
                if len(parts) >= 3 and parts[2].startswith('seizure'):
                    seizure_id = parts[2]
                
                if seizure_id:
                    identifier = f"{subject_id}_{run_id}_{seizure_id}"
                else:
                    identifier = f"{subject_id}_{run_id}"
                
                processed_identifiers.add(identifier)
        
        # Filter pkl files
        unprocessed_files = []
        skipped_count = 0
        
        for pkl_file in pkl_files:
            # Extract identifier from pkl filename
            filename = pkl_file.stem.replace('_preprocessed', '')
            parts = filename.split('_')
            
            if len(parts) >= 2:
                subject_id = parts[0]
                run_id = parts[1]
                seizure_id = parts[2] if len(parts) > 2 else None
                
                if seizure_id:
                    identifier = f"{subject_id}_{run_id}_{seizure_id}"
                else:
                    identifier = f"{subject_id}_{run_id}"
                
                if identifier in processed_identifiers:
                    logger.info(f"Skipping already processed file: {pkl_file.name}")
                    skipped_count += 1
                else:
                    unprocessed_files.append(pkl_file)
            else:
                # If we can't parse the filename, include it to be safe
                unprocessed_files.append(pkl_file)
        
        logger.info(f"Filtered files: {len(unprocessed_files)} to process, {skipped_count} already processed")
        return unprocessed_files
    
    def process_files(self, data_dir: str, output_dir: str, existing_results_dir: str = None, max_files: int = None) -> List[str]:
        """Process windowed files in parallel, prioritizing seizure files"""
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Find windowed files
        pkl_files = list(Path(data_dir).glob("*_preprocessed.pkl"))
        
        # Filter out already processed files if existing results directory is provided
        if existing_results_dir:
            pkl_files = self.filter_unprocessed_files(pkl_files, existing_results_dir)
        
        # Prioritize seizure files
        pkl_files = self.prioritize_seizure_files(pkl_files)
        
        if max_files:
            pkl_files = pkl_files[:max_files]
        
        logger.info(f"Processing {len(pkl_files)} windowed files with {self.n_workers} workers")
        
        if len(pkl_files) == 0:
            logger.warning("No windowed files found to process")
            return []
        
        # Setup progress tracking
        progress_tracker = ProgressTracker(len(pkl_files))
        
        # Create partial function with fixed arguments
        process_func = partial(process_single_windowed_file, config=self.config, output_dir=output_dir)
        
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
                              f"{result_info['num_windows_processed']} windows, "
                              f"{result_info['total_anomalies']} anomalies, "
                              f"{result_info['execution_time']:.1f}s")
                else:
                    failed_files.append(file_path)
                    logger.error(f"❌ {Path(file_path).name}: {result_info.get('error', 'Unknown error')}")
        
        total_time = time.time() - start_time
        
        # Summary
        logger.info(f"\n📊 WINDOWED PARALLEL PROCESSING SUMMARY")
        logger.info(f"Total processing time: {total_time/60:.1f} minutes")
        logger.info(f"Average time per file: {total_time/len(pkl_files):.1f} seconds")
        logger.info(f"Successfully processed: {len(processed_files)}/{len(pkl_files)}")
        logger.info(f"Failed files: {len(failed_files)}")
        logger.info(f"Window strategy used: {self.config.get('window_strategy', 'individual')}")
        
        return processed_files

def main():
    """Main function for windowed parallel processing"""
    parser = argparse.ArgumentParser(description='Parallel Madrid Windowed Batch Processor')
    parser.add_argument('--data-dir', required=True, help='Directory containing windowed preprocessed .pkl files')
    parser.add_argument('--output-dir', required=True, help='Directory to save JSON results')
    parser.add_argument('--existing-results-dir', help='Directory containing existing results to skip already processed files')
    parser.add_argument('--max-files', type=int, help='Maximum number of files to process')
    parser.add_argument('--config-file', help='JSON configuration file')
    parser.add_argument('--n-workers', type=int, help=f'Number of worker processes (default: {cpu_count()-1})')
    parser.add_argument('--use-gpu', action='store_true', default=True, help='Use GPU acceleration')
    parser.add_argument('--threshold-percentile', type=float, default=90, help='Anomaly threshold percentile')
    parser.add_argument('--train-minutes', type=float, default=30, help='Minutes of data to use for training')
    parser.add_argument('--window-strategy', choices=['individual', 'reconstruct', 'hybrid'], 
                       default='individual', help='Window processing strategy')
    
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
            'window_strategy': args.window_strategy,
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
    processor = ParallelMadridWindowedBatchProcessor(config, n_workers=args.n_workers)
    
    # Process files
    processed_files = processor.process_files(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        existing_results_dir=args.existing_results_dir,
        max_files=args.max_files
    )
    
    logger.info(f"Windowed parallel processing completed. Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()