#!/usr/bin/env python3
"""
Reprocess files with 0 windows to create flexible-length windows
"""
import os
import sys
import pickle
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import warnings

# Add current directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from preprocessing import ECGPreprocessor
sys.path.append(os.path.join(os.path.dirname(__file__), 'Information', 'Data', 'seizeit2_main'))

from classes.data import Data
from classes.annotation import Annotation

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reprocess_empty_windows.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FlexibleECGPreprocessor(ECGPreprocessor):
    """
    Extended ECG Preprocessor that creates flexible-length windows
    """
    
    def __init__(self, filter_params: Dict, downsample_freq: int, 
                 window_size: float = 3600, stride: float = 1800, 
                 min_window_size: float = 300):
        """
        Initialize with flexible windowing parameters
        
        Args:
            filter_params: Filter parameters
            downsample_freq: Target sampling frequency
            window_size: Preferred window size in seconds
            stride: Window stride in seconds
            min_window_size: Minimum acceptable window size in seconds
        """
        super().__init__(filter_params, downsample_freq, window_size, stride)
        self.min_window_size = min_window_size
        
    def create_flexible_windows(
        self, 
        signal_data: np.ndarray, 
        fs: int,
        annotations: Annotation
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]:
        """
        Create flexible-length windows from signal data.
        If signal is too short for preferred window_size, create smaller windows
        down to min_window_size.
        
        Args:
            signal_data: Preprocessed ECG signal
            fs: Sampling frequency
            annotations: Seizure annotations
            
        Returns:
            Tuple of (windows, labels, metadata)
        """
        signal_length = len(signal_data)
        signal_duration_seconds = signal_length / fs
        
        logger.info(f"Signal length: {signal_length} samples, duration: {signal_duration_seconds:.1f}s")
        
        # Calculate window parameters in samples
        preferred_window_samples = int(self.window_size * fs)
        stride_samples = int(self.stride * fs)
        min_window_samples = int(self.min_window_size * fs)
        
        windows = []
        labels = []
        metadata = []
        
        # Extract seizure events timing
        seizure_events = annotations.events if annotations.events else []
        
        # Create full-signal sample-level labels first
        full_signal_labels = self._create_sample_level_labels(signal_length, int(fs), seizure_events)
        
        # Strategy 1: Try preferred window size
        if signal_length >= preferred_window_samples:
            logger.info(f"Using preferred window size: {self.window_size}s")
            return self.create_windows(signal_data, fs, annotations)
        
        # Strategy 2: Use the entire signal as one window
        else:
            logger.info(f"Signal too short for preferred windows ({self.window_size}s), using entire signal as one window ({signal_duration_seconds:.1f}s)")
            
            # Create single window with entire signal
            windows.append(signal_data)
            
            # Extract sample-level labels for the entire signal
            labels.append(full_signal_labels)
            
            # Calculate summary statistics
            n_seizure_samples = np.sum(full_signal_labels)
            seizure_ratio = n_seizure_samples / len(full_signal_labels) if len(full_signal_labels) > 0 else 0.0
            window_label = 1 if n_seizure_samples > 0 else 0
            
            # Create metadata
            window_metadata = {
                'start_time': 0.0,
                'end_time': signal_duration_seconds,
                'start_idx': 0,
                'end_idx': signal_length,
                'sampling_rate': int(fs),
                'window_label': window_label,
                'n_seizure_samples': int(n_seizure_samples),
                'seizure_ratio': float(seizure_ratio),
                'seizure_segments': self._get_seizure_segments_in_window(full_signal_labels, 0.0, int(fs)),
                'window_type': 'flexible_full_signal',
                'is_flexible_window': True,
                'actual_window_size': signal_duration_seconds
            }
            
            # Add warning if very short
            if signal_duration_seconds < self.min_window_size:
                window_metadata['warning'] = f'Signal shorter than minimum window size ({self.min_window_size}s)'
            
            metadata.append(window_metadata)
        
        logger.info(f"Created {len(windows)} flexible windows")
        return windows, labels, metadata
    
    def preprocess_pipeline_flexible(
        self,
        data_path: str,
        subject_id: str,
        run_id: str,
        output_dir: str = "./results/preprocessed/"
    ) -> Dict:
        """
        Flexible preprocessing pipeline - uses same logic as original but with flexible windowing
        
        Args:
            data_path: Path to SeizeIT2 dataset
            subject_id: Subject identifier
            run_id: Run identifier
            output_dir: Output directory
            
        Returns:
            Processed data dictionary or None if failed
        """
        try:
            logger.info(f"Loading data for {subject_id} {run_id}")
            
            # Use the same data loading logic as the original preprocessing
            from Information.Data.seizeit2_main.classes.data import Data
            from Information.Data.seizeit2_main.classes.annotation import Annotation
            
            recording = [subject_id, run_id]
            data_loader = Data()
            annotations = Annotation.loadAnnotation(data_path, recording)
            
            # Load data
            data_dict = data_loader.loadData(data_path, recording)
            
            if data_dict is None:
                logger.error(f"Failed to load data for {subject_id} {run_id}")
                return None
            
            # Extract recording info
            recording_duration = data_dict.get('recording_duration', 0)
            logger.info(f"Processing {subject_id}_{run_id}: {recording_duration:.1f}s recording")
            
            # Process channels
            processed_channels = []
            total_seizures = 0
            
            for channel_name, channel_data in data_dict['channels'].items():
                logger.info(f"Processing channel: {channel_name}")
                
                fs = channel_data['fs']
                signal_data = channel_data['signal']
                
                if len(signal_data) == 0:
                    logger.warning(f"Empty signal for channel {channel_name}")
                    continue
                
                # Apply preprocessing steps
                filtered_signal = self.apply_bandpass_filter(signal_data, int(fs))
                downsampled_signal = self.downsample(filtered_signal, int(fs))
                downsampled_fs = self.downsample_freq
                
                # Create flexible windows
                windows, window_labels, window_metadata = self.create_flexible_windows(
                    downsampled_signal, downsampled_fs, annotations
                )
                
                # Count seizures
                n_seizure_windows = sum(1 for meta in window_metadata if meta.get('window_label', 0) == 1)
                total_seizures += n_seizure_windows
                
                # Store channel data
                channel_info = {
                    'channel_name': channel_name,
                    'windows': windows,
                    'labels': window_labels,
                    'metadata': window_metadata,
                    'original_fs': int(fs),
                    'processed_fs': downsampled_fs,
                    'n_windows': len(windows),
                    'n_seizure_windows': n_seizure_windows
                }
                processed_channels.append(channel_info)
            
            if not processed_channels:
                logger.error("No channels processed successfully")
                return None
            
            # Prepare output data
            output_data = {
                'subject_id': subject_id,
                'run_id': run_id,
                'channels': processed_channels,
                'preprocessing_params': {
                    'filter_params': self.filter_params,
                    'downsample_freq': self.downsample_freq,
                    'window_size': self.window_size,
                    'stride': self.stride,
                    'min_window_size': self.min_window_size,
                    'flexible_windowing': True
                },
                'recording_duration': recording_duration,
                'total_seizures': total_seizures
            }
            
            # Save processed data
            os.makedirs(output_dir, exist_ok=True)
            
            output_filename = f"{subject_id}_{run_id}_preprocessed.pkl"
            output_path = os.path.join(output_dir, output_filename)
            
            with open(output_path, 'wb') as f:
                pickle.dump(output_data, f)
            
            logger.info(f"Saved preprocessed data to {output_path}")
            logger.info(f"Total windows: {sum(len(ch['windows']) for ch in processed_channels)}")
            logger.info(f"Seizure windows: {total_seizures}")
            
            return output_data
            
        except Exception as e:
            logger.error(f"Error in flexible preprocessing pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None

def find_empty_window_files(preprocessed_dir: str) -> List[str]:
    """
    Find all preprocessed files that have 0 windows
    
    Args:
        preprocessed_dir: Directory containing preprocessed .pkl files
        
    Returns:
        List of file paths with 0 windows
    """
    empty_files = []
    preprocessed_path = Path(preprocessed_dir)
    
    if not preprocessed_path.exists():
        logger.error(f"Directory {preprocessed_dir} does not exist")
        return empty_files
    
    pkl_files = list(preprocessed_path.glob("*_preprocessed.pkl"))
    logger.info(f"Found {len(pkl_files)} preprocessed files to check")
    
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            # Check if file has 0 windows
            if ('channels' in data and len(data['channels']) > 0 and
                'windows' in data['channels'][0] and
                len(data['channels'][0]['windows']) == 0):
                
                empty_files.append(str(pkl_file))
                recording_duration = data.get('recording_duration', 'N/A')
                logger.info(f"Empty file found: {pkl_file.name} (duration: {recording_duration}s)")
                
        except Exception as e:
            logger.warning(f"Error checking {pkl_file.name}: {e}")
    
    logger.info(f"Found {len(empty_files)} files with 0 windows")
    return empty_files

def reprocess_file_with_flexible_windows(
    file_path: str, 
    data_dir: str, 
    output_dir: str,
    preprocessor: FlexibleECGPreprocessor
) -> bool:
    """
    Reprocess a single file with flexible windowing
    
    Args:
        file_path: Path to the empty preprocessed file
        data_dir: Directory containing original SeizeIT2 data
        output_dir: Output directory for reprocessed files
        preprocessor: Flexible preprocessor instance
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Extract subject and run from filename
        filename = Path(file_path).stem.replace('_preprocessed', '')
        parts = filename.split('_')
        
        if len(parts) < 2:
            logger.error(f"Cannot parse filename: {filename}")
            return False
            
        subject_id = parts[0]
        run_id = parts[1]
        
        logger.info(f"Reprocessing {subject_id}_{run_id}")
        
        # Use the flexible preprocessing pipeline (same approach as preprocess_all_data.py)
        result = preprocessor.preprocess_pipeline_flexible(
            data_path=data_dir,  # Pass the SeizeIT2 dataset path
            subject_id=subject_id,
            run_id=run_id,
            output_dir=output_dir
        )
        
        if result:
            logger.info(f"✓ Successfully reprocessed {subject_id}_{run_id}")
            return True
        else:
            logger.error(f"✗ Failed to reprocess {subject_id}_{run_id}")
            return False
            
    except Exception as e:
        logger.error(f"Error reprocessing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Reprocess files with 0 windows using flexible windowing')
    parser.add_argument('--preprocessed-dir', 
                       default='/home/swolf/asim_shared/preprocessed_data/downsample_freq=8,window_size=3600_0,stride=1800_0_new',
                       help='Directory containing preprocessed files with 0 windows')
    parser.add_argument('--original-data-dir', 
                       default='/home/swolf/asim_shared/raw_data/ds005873-1.1.0',
                       help='Directory containing original SeizeIT2 data')
    parser.add_argument('--output-dir',
                       default='/home/swolf/asim_shared/preprocessed_data/downsample_freq=8,window_size=3600_0,stride=1800_0_new_flexible',
                       help='Output directory for reprocessed files')
    parser.add_argument('--window-size', type=float, default=3600.0,
                       help='Preferred window size in seconds (default: 3600)')
    parser.add_argument('--stride', type=float, default=1800.0,
                       help='Window stride in seconds (default: 1800)')
    parser.add_argument('--min-window-size', type=float, default=300.0,
                       help='Minimum acceptable window size in seconds (default: 300)')
    parser.add_argument('--max-files', type=int,
                       help='Maximum number of files to reprocess (for testing)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Only find empty files, don\'t reprocess')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Find files with 0 windows
    logger.info("Finding files with 0 windows...")
    empty_files = find_empty_window_files(args.preprocessed_dir)
    
    if not empty_files:
        logger.info("No files with 0 windows found")
        return
    
    if args.dry_run:
        logger.info("Dry run mode - not reprocessing files")
        logger.info(f"Would reprocess {len(empty_files)} files:")
        for file_path in empty_files:
            logger.info(f"  - {Path(file_path).name}")
        return
    
    # Limit files if requested
    if args.max_files:
        empty_files = empty_files[:args.max_files]
        logger.info(f"Limited to {len(empty_files)} files for processing")
    
    # Initialize flexible preprocessor
    filter_params = {
        'low_freq': 0.5,
        'high_freq': 40.0,
        'order': 4
    }
    
    preprocessor = FlexibleECGPreprocessor(
        filter_params=filter_params,
        downsample_freq=8,
        window_size=args.window_size,
        stride=args.stride,
        min_window_size=args.min_window_size
    )
    
    # Process files
    success_count = 0
    failed_count = 0
    
    for i, file_path in enumerate(empty_files, 1):
        logger.info(f"Processing {i}/{len(empty_files)}: {Path(file_path).name}")
        
        success = reprocess_file_with_flexible_windows(
            file_path=file_path,
            data_dir=args.original_data_dir,
            output_dir=args.output_dir,
            preprocessor=preprocessor
        )
        
        if success:
            success_count += 1
        else:
            failed_count += 1
    
    # Summary
    logger.info(f"\n🎯 REPROCESSING SUMMARY")
    logger.info(f"Files found with 0 windows: {len(empty_files)}")
    logger.info(f"Successfully reprocessed: {success_count}")
    logger.info(f"Failed to reprocess: {failed_count}")
    logger.info(f"Output directory: {args.output_dir}")

if __name__ == "__main__":
    main()