#!/usr/bin/env python3
"""
Madrid Batch Processor with JSON Output

Processes multiple preprocessed ECG files with Madrid algorithm and saves results 
in standardized JSON format according to specs/DataFormatMadridErgebnisse.md

Usage:
    python madrid_batch_processor.py --data-dir results/preprocessed_all --output-dir madrid_results --max-files 10
    python madrid_batch_processor.py --config-file madrid_config.json

Author: Generated for Madrid batch processing
Date: 2025
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

warnings.filterwarnings('ignore')

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

try:
    from models.madrid import MADRID
    MADRID_AVAILABLE = True
    print("✓ MADRID successfully imported")
except ImportError as e:
    print(f"❌ MADRID import failed: {e}")
    MADRID_AVAILABLE = False
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('madrid_batch_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MadridBatchProcessor:
    """
    Batch processor for Madrid analysis with standardized JSON output
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Madrid Batch Processor
        
        Args:
            config: Configuration dictionary with analysis parameters
        """
        self.config = config
        self.madrid_detector = MADRID(
            use_gpu=config.get('use_gpu', True),
            enable_output=config.get('enable_output', False)
        )
        
        # Analysis parameters
        self.madrid_params = config.get('madrid_parameters', {
            'm_range': {
                'min_length': 80,
                'max_length': 800,
                'step_size': 80
            },
            'analysis_config': {
                'top_k': 3,
                'train_test_split_ratio': 0.5,
                'threshold_percentile': 95
            },
            'algorithm_settings': {
                'use_gpu': True,
                'downsampling_factor': 1
            }
        })
        
        logger.info(f"Madrid Batch Processor initialized with config: {config}")
    
    def load_preprocessed_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load preprocessed ECG file and extract metadata
        Supports both windowed and seizure-only preprocessed data
        
        Args:
            file_path: Path to preprocessed .pkl file
            
        Returns:
            Dictionary with signal data and metadata, or None if loading fails
        """
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
        """
        Check if data is in seizure-only preprocessed format
        
        Args:
            data: Loaded pickle data
            
        Returns:
            True if seizure-only format, False if windowed format
        """
        # Seizure-only format has these keys at top level
        seizure_only_keys = ['subject_id', 'run_id', 'seizure_index', 'channels']
        windowed_keys = ['channels']  # Windowed format has channels with windows/labels
        
        if all(key in data for key in seizure_only_keys):
            return True
        elif 'channels' in data and isinstance(data['channels'], list):
            # Check if channels contain 'windows' (windowed format)
            if len(data['channels']) > 0 and 'windows' in data['channels'][0]:
                return False
            # Check if channels contain 'data' (seizure-only format)
            elif len(data['channels']) > 0 and 'data' in data['channels'][0]:
                return True
        
        return False
    
    def load_seizure_only_data(self, data: Dict, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load seizure-only preprocessed data format
        
        Args:
            data: Loaded seizure-only data
            file_path: Path to source file
            
        Returns:
            Standardized data dictionary
        """
        try:
            subject_id = data.get('subject_id', 'unknown')
            run_id = data.get('run_id', 'unknown')
            seizure_index = data.get('seizure_index', 0)
            seizure_id = f"seizure_{seizure_index:02d}"
            
            # Get first ECG channel
            if not data.get('channels') or len(data['channels']) == 0:
                logger.warning(f"No channels found in seizure-only data: {file_path}")
                return None
            
            channel = data['channels'][0]
            signal_data = channel.get('data', np.array([]))
            
            if len(signal_data) == 0:
                logger.warning(f"Empty signal data in: {file_path}")
                return None
            
            # Extract sampling rate
            sampling_rate = data.get('processing_params', {}).get('downsample_freq', 125)
            original_sampling_rate = 250  # Default for SeizeIT2
            
            # Signal metadata
            signal_metadata = {
                'sampling_rate': sampling_rate,
                'original_sampling_rate': original_sampling_rate,
                'signal_length_samples': len(signal_data),
                'signal_duration_seconds': len(signal_data) / sampling_rate,
                'preprocessing_info': {
                    'method': 'seizure_only',
                    'context_minutes': data.get('processing_params', {}).get('context_minutes', 30),
                    'filter_applied': data.get('processing_params', {}).get('filter_params', {}),
                    'downsampling_factor': original_sampling_rate / sampling_rate
                }
            }
            
            # Extract ground truth from seizure-only data
            ground_truth = self.extract_seizure_only_ground_truth(data, sampling_rate)
            
            logger.info(f"Loaded seizure-only data: {subject_id}_{run_id}_{seizure_id} "
                       f"({len(signal_data)} samples @ {sampling_rate}Hz)")
            
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
        """
        Load windowed preprocessed data format (original implementation)
        
        Args:
            data: Loaded windowed data
            file_path: Path to source file
            
        Returns:
            Standardized data dictionary
        """
        if not data or "channels" not in data:
            logger.warning(f"Invalid windowed data structure in {file_path}")
            return None
        
        # Extract first channel (assuming single ECG channel)
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
        
        # Calculate signal duration
        sampling_rate = signal_metadata['sampling_rate']
        signal_metadata['signal_duration_seconds'] = signal_metadata['signal_length_samples'] / sampling_rate
        
        # Extract continuous signal from windows if available
        if 'signal' in channel_data:
            signal = channel_data['signal']
        elif 'windows' in channel_data and len(channel_data['windows']) > 0:
            # Reconstruct signal from windows (approximate)
            signal = np.concatenate(channel_data['windows'])
            logger.info(f"Reconstructed signal from {len(channel_data['windows'])} windows")
        else:
            logger.error(f"No signal data found in {file_path}")
            return None
        
        # Extract ground truth if available
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
        """
        Extract ground truth seizure information from preprocessed data
        
        Args:
            data: Preprocessed data dictionary
            subject_id: Subject identifier
            run_id: Run identifier
            seizure_id: Seizure identifier (if available)
            
        Returns:
            Ground truth information dictionary
        """
        ground_truth = {
            'seizure_present': False,
            'seizure_regions': [],
            'annotation_source': 'preprocessed_data',
            'annotator_id': 'automated'
        }
        
        # Check if seizure information is available
        if seizure_id is not None:
            ground_truth['seizure_present'] = True
            
            # Try to extract seizure timing from data
            if 'seizure_info' in data:
                seizure_info = data['seizure_info']
                ground_truth['seizure_regions'] = [
                    {
                        'onset_sample': seizure_info.get('onset_sample', 0),
                        'offset_sample': seizure_info.get('offset_sample', 0),
                        'onset_time_seconds': seizure_info.get('onset_time', 0),
                        'offset_time_seconds': seizure_info.get('offset_time', 0),
                        'duration_seconds': seizure_info.get('duration', 0),
                        'seizure_type': seizure_info.get('type', 'unknown'),
                        'confidence': 1.0
                    }
                ]
            else:
                # Estimate seizure region (placeholder)
                signal_length = data['channels'][0].get('signal_length', 0)
                if signal_length > 0:
                    # Assume seizure in middle 20% of signal
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
                            'confidence': 0.5  # Low confidence for estimated regions
                        }
                    ]
        
        return ground_truth
    
    def extract_seizure_only_ground_truth(self, data: Dict, sampling_rate: int) -> Dict:
        """
        Extract ground truth from seizure-only preprocessed data
        
        Args:
            data: Seizure-only preprocessed data
            sampling_rate: Signal sampling rate
            
        Returns:
            Ground truth information dictionary
        """
        ground_truth = {
            'seizure_present': True,  # Always true for seizure-only data
            'seizure_regions': [],
            'annotation_source': 'seizure_only_preprocessing',
            'annotator_id': 'automated'
        }
        
        # Extract seizure timing from seizure-only data
        original_seizure_start = data.get('original_seizure_start', 0)
        original_seizure_end = data.get('original_seizure_end', 0)
        extraction_start = data.get('extraction_start', 0)
        extraction_end = data.get('extraction_end', 0)
        
        # Calculate relative seizure position in extracted segment
        if original_seizure_start >= extraction_start and original_seizure_end <= extraction_end:
            # Seizure is fully contained in extraction
            relative_start = original_seizure_start - extraction_start
            relative_end = original_seizure_end - extraction_start
            
            # Convert to samples
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
            # Fallback: Use labels from channel data if available
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
                                'confidence': 0.9  # High confidence from labels
                            }
                        ]
        
        return ground_truth
    
    def analyze_with_madrid(self, signal: np.ndarray, signal_metadata: Dict) -> Tuple[Dict, Dict]:
        """
        Perform Madrid analysis on signal
        
        Args:
            signal: ECG signal array
            signal_metadata: Signal metadata dictionary
            
        Returns:
            Tuple of (analysis_results, performance_info)
        """
        start_time = time.time()
        
        # Get Madrid parameters adapted for sampling rate
        sampling_rate = signal_metadata['sampling_rate']
        madrid_params = self.adapt_madrid_params_for_sampling_rate(sampling_rate)
        
        try:
            # Run Madrid analysis
            logger.info(f"Running Madrid analysis on signal (length: {len(signal)}, fs: {sampling_rate}Hz)")
            
            multi_length_table, bsf, bsf_loc = self.madrid_detector.fit(
                T=signal,
                min_length=madrid_params['min_length'],
                max_length=madrid_params['max_length'],
                step_size=madrid_params['step_size'],
                train_test_split=int(len(signal) * self.madrid_params['analysis_config']['train_test_split_ratio'])
            )
            
            # Get anomaly information
            anomaly_info = self.madrid_detector.get_anomaly_scores(
                threshold_percentile=self.madrid_params['analysis_config']['threshold_percentile']
            )
            anomalies = anomaly_info['anomalies']
            
            execution_time = time.time() - start_time
            
            # Format anomalies for JSON output
            formatted_anomalies = []
            for i, anomaly in enumerate(anomalies):
                formatted_anomaly = {
                    'rank': i + 1,
                    'm_value': anomaly['length'],
                    'anomaly_score': float(anomaly['score']),
                    'location_sample': int(anomaly['location']) if anomaly['location'] is not None else None,
                    'location_time_seconds': float(anomaly['location'] / sampling_rate) if anomaly['location'] is not None else None,
                    'seizure_hit': False,  # Will be updated with ground truth comparison
                    'normalized_score': float(anomaly['score']),  # Already normalized by Madrid
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
                }
            }
            
            performance_info = {
                'execution_time_seconds': execution_time,
                'gpu_used': self.madrid_params['algorithm_settings']['use_gpu'],
                'memory_peak_mb': 0,  # Placeholder
                'analysis_successful': True
            }
            
            logger.info(f"Madrid analysis completed in {execution_time:.2f}s, found {len(anomalies)} anomalies")
            
            return analysis_results, performance_info
            
        except Exception as e:
            logger.error(f"Madrid analysis failed: {e}")
            
            # Return empty results on failure
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
    
    def adapt_madrid_params_for_sampling_rate(self, sampling_rate: int) -> Dict:
        """
        Adapt Madrid parameters for different sampling rates
        
        Args:
            sampling_rate: Current sampling rate
            
        Returns:
            Adapted Madrid parameters
        """
        base_params = self.madrid_params['m_range']
        
        if sampling_rate >= 100:  # 125Hz
            return {
                'min_length': base_params['min_length'],
                'max_length': base_params['max_length'],
                'step_size': base_params['step_size']
            }
        elif sampling_rate >= 25:  # 32Hz
            factor = 125 / sampling_rate
            return {
                'min_length': max(10, int(base_params['min_length'] / factor)),
                'max_length': max(50, int(base_params['max_length'] / factor)),
                'step_size': max(5, int(base_params['step_size'] / factor))
            }
        else:  # 8Hz
            factor = 125 / sampling_rate
            return {
                'min_length': max(5, int(base_params['min_length'] / factor)),
                'max_length': max(25, int(base_params['max_length'] / factor)),
                'step_size': max(2, int(base_params['step_size'] / factor))
            }
    
    def validate_against_ground_truth(self, anomalies: List[Dict], ground_truth: Dict, sampling_rate: int) -> List[Dict]:
        """
        Validate anomalies against ground truth and calculate overlap information
        
        Args:
            anomalies: List of detected anomalies
            ground_truth: Ground truth seizure information
            sampling_rate: Signal sampling rate
            
        Returns:
            List of seizure overlap information for each anomaly
        """
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
                
                # Check overlap with each seizure region
                for seizure_region in ground_truth['seizure_regions']:
                    onset_sample = seizure_region['onset_sample']
                    offset_sample = seizure_region['offset_sample']
                    
                    # Check if anomaly falls within seizure region (with tolerance)
                    tolerance_samples = int(30 * sampling_rate)  # 30 second tolerance
                    
                    if onset_sample - tolerance_samples <= anomaly_location <= offset_sample + tolerance_samples:
                        overlap_info['overlap_with_seizure'] = True
                        overlap_info['overlap_sample_start'] = onset_sample
                        overlap_info['overlap_sample_end'] = offset_sample
                        overlap_info['distance_to_onset'] = abs(anomaly_location - onset_sample)
                        overlap_info['classification'] = 'true_positive'
                        
                        # Calculate overlap ratio (simplified)
                        overlap_info['overlap_ratio'] = 1.0 if onset_sample <= anomaly_location <= offset_sample else 0.8
                        
                        # Update anomaly seizure_hit status
                        anomaly['seizure_hit'] = True
                        break
            
            seizure_overlap_info.append(overlap_info)
        
        return seizure_overlap_info
    
    def create_json_output(self, file_data: Dict, analysis_results: Dict, performance_info: Dict) -> Dict:
        """
        Create standardized JSON output according to specification
        
        Args:
            file_data: Original file data with metadata
            analysis_results: Madrid analysis results
            performance_info: Performance and timing information
            
        Returns:
            Complete JSON output dictionary
        """
        timestamp = datetime.now().isoformat() + 'Z'
        metadata = file_data['metadata']
        
        # Generate analysis ID
        analysis_id = f"madrid_{metadata['subject_id']}_{metadata['run_id']}"
        if metadata['seizure_id']:
            analysis_id += f"_{metadata['seizure_id']}"
        analysis_id += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Validate anomalies against ground truth
        seizure_overlap_info = self.validate_against_ground_truth(
            analysis_results['anomalies'],
            file_data['ground_truth'],
            metadata['signal_metadata']['sampling_rate']
        )
        
        # Calculate performance metrics
        true_positives = sum(1 for info in seizure_overlap_info if info['classification'] == 'true_positive')
        false_positives = len(seizure_overlap_info) - true_positives
        
        analysis_results['performance_metrics'].update({
            'true_positives': true_positives,
            'false_positives': false_positives,
            'sensitivity': true_positives / max(1, len(file_data['ground_truth']['seizure_regions'])) if file_data['ground_truth']['seizure_present'] else 0,
            'precision': true_positives / max(1, len(analysis_results['anomalies'])) if analysis_results['anomalies'] else 0
        })
        
        # Create complete JSON structure
        json_output = {
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
        
        return json_output
    
    def process_files(self, data_dir: str, output_dir: str, max_files: int = None) -> List[str]:
        """
        Process multiple files in data directory
        
        Args:
            data_dir: Directory containing preprocessed files
            output_dir: Directory to save JSON results
            max_files: Maximum number of files to process (None for all)
            
        Returns:
            List of successfully processed file paths
        """
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Find all preprocessed files
        pkl_files = list(Path(data_dir).glob("*_preprocessed.pkl"))
        
        if max_files:
            pkl_files = pkl_files[:max_files]
        
        logger.info(f"Found {len(pkl_files)} files to process")
        
        processed_files = []
        failed_files = []
        
        for i, file_path in enumerate(pkl_files):
            logger.info(f"Processing file {i+1}/{len(pkl_files)}: {file_path.name}")
            
            try:
                # Load file
                file_data = self.load_preprocessed_file(str(file_path))
                if file_data is None:
                    logger.warning(f"Skipping {file_path.name} - loading failed")
                    failed_files.append(str(file_path))
                    continue
                
                # Run Madrid analysis
                analysis_results, performance_info = self.analyze_with_madrid(
                    file_data['signal'],
                    file_data['metadata']['signal_metadata']
                )
                
                # Create JSON output
                json_output = self.create_json_output(file_data, analysis_results, performance_info)
                
                # Save JSON file
                output_filename = f"madrid_results_{file_data['metadata']['subject_id']}_{file_data['metadata']['run_id']}"
                if file_data['metadata']['seizure_id']:
                    output_filename += f"_{file_data['metadata']['seizure_id']}"
                output_filename += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                output_path = Path(output_dir) / output_filename
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(json_output, f, indent=2, ensure_ascii=False)
                
                logger.info(f"✓ Successfully processed {file_path.name} -> {output_filename}")
                processed_files.append(str(file_path))
                
                # Log summary
                num_anomalies = len(analysis_results['anomalies'])
                num_tp = analysis_results['performance_metrics'].get('true_positives', 0)
                execution_time = performance_info['execution_time_seconds']
                logger.info(f"  -> {num_anomalies} anomalies found, {num_tp} true positives, {execution_time:.1f}s")
                
            except Exception as e:
                logger.error(f"❌ Failed to process {file_path.name}: {e}")
                failed_files.append(str(file_path))
                continue
        
        # Summary
        logger.info(f"\n📊 BATCH PROCESSING SUMMARY")
        logger.info(f"Total files processed: {len(processed_files)}/{len(pkl_files)}")
        logger.info(f"Successfully processed: {len(processed_files)}")
        logger.info(f"Failed files: {len(failed_files)}")
        
        if failed_files:
            logger.warning(f"Failed files: {failed_files}")
        
        return processed_files


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Madrid Batch Processor')
    parser.add_argument('--data-dir', required=True, help='Directory containing preprocessed .pkl files')
    parser.add_argument('--output-dir', required=True, help='Directory to save JSON results')
    parser.add_argument('--max-files', type=int, help='Maximum number of files to process')
    parser.add_argument('--config-file', help='JSON configuration file')
    parser.add_argument('--use-gpu', action='store_true', default=True, help='Use GPU acceleration')
    parser.add_argument('--threshold-percentile', type=float, default=95, help='Anomaly threshold percentile')
    
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
                    'top_k': 3,
                    'train_test_split_ratio': 0.5,
                    'threshold_percentile': args.threshold_percentile
                },
                'algorithm_settings': {
                    'use_gpu': args.use_gpu,
                    'downsampling_factor': 1
                }
            }
        }
    
    # Initialize processor
    processor = MadridBatchProcessor(config)
    
    # Process files
    processed_files = processor.process_files(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_files=args.max_files
    )
    
    logger.info(f"Batch processing completed. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()