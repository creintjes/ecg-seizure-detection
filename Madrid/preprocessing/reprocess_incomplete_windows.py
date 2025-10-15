#!/usr/bin/env python3
"""
Reprocess files with incomplete trailing windows to capture seizures that may be cut off.

This script identifies files that:
1. Have duration >= window_size (so they have at least 1 window)
2. Have trailing data < window_size that was discarded during preprocessing
3. May have seizure annotations in the trailing segment

It then reprocesses these files to add the trailing data as a flexible-size final window.
"""
import os
import sys
import pickle
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Information', 'Data', 'seizeit2_main'))

from config import RAW_DATA_PATH, PREPROCESSED_DATA_PATH
from preprocessing import ECGPreprocessor
from classes.data import Data
from classes.annotation import Annotation

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reprocess_incomplete_windows.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TrailingWindowPreprocessor(ECGPreprocessor):
    """
    Extended ECG Preprocessor that adds trailing incomplete windows
    """

    def __init__(self, filter_params: Dict, downsample_freq: int,
                 window_size: float = 3600, stride: float = 1800,
                 min_trailing_size: float = 60):
        """
        Initialize with trailing window parameters

        Args:
            filter_params: Filter parameters
            downsample_freq: Target sampling frequency
            window_size: Standard window size in seconds
            stride: Window stride in seconds
            min_trailing_size: Minimum trailing segment size to create a window (seconds)
        """
        super().__init__(filter_params, downsample_freq, window_size, stride)
        self.min_trailing_size = min_trailing_size

    def create_windows_with_trailing(
        self,
        signal_data: np.ndarray,
        fs: int,
        annotations: Annotation
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]:
        """
        Create windows including a trailing window for incomplete data at the end.

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

        # First create standard windows using parent method
        windows, labels, metadata = self.create_windows(signal_data, fs, annotations)

        logger.info(f"Created {len(windows)} standard windows")

        # Calculate where standard windows end
        window_samples = int(self.window_size * fs)
        stride_samples = int(self.stride * fs)

        # Find the last covered sample
        if len(windows) > 0:
            # Last window starts at
            last_window_start = (len(windows) - 1) * stride_samples
            last_covered_sample = last_window_start + window_samples

            # Calculate trailing segment
            trailing_samples = signal_length - last_covered_sample
            trailing_duration = trailing_samples / fs

            logger.info(f"Last standard window covers up to sample {last_covered_sample}/{signal_length}")
            logger.info(f"Trailing segment: {trailing_samples} samples ({trailing_duration:.1f}s)")

            # Check if trailing segment is large enough
            if trailing_duration >= self.min_trailing_size:
                logger.info(f"Adding trailing window of {trailing_duration:.1f}s")

                # Extract trailing segment
                trailing_signal = signal_data[last_covered_sample:]

                # Get seizure events
                seizure_events = annotations.events if annotations.events else []

                # Create sample-level labels for trailing window
                trailing_start_time = last_covered_sample / fs
                trailing_end_time = signal_duration_seconds

                trailing_labels = np.zeros(len(trailing_signal), dtype=int)

                # Mark seizure samples in trailing window
                for seizure_start, seizure_end in seizure_events:
                    # Calculate overlap with trailing window
                    overlap_start = max(trailing_start_time, seizure_start)
                    overlap_end = min(trailing_end_time, seizure_end)

                    if overlap_start < overlap_end:
                        # Convert to sample indices within trailing window
                        sample_start = int((overlap_start - trailing_start_time) * fs)
                        sample_end = int((overlap_end - trailing_start_time) * fs)

                        # Ensure indices are within bounds
                        sample_start = max(0, sample_start)
                        sample_end = min(len(trailing_signal), sample_end)

                        if sample_start < sample_end:
                            trailing_labels[sample_start:sample_end] = 1

                # Calculate statistics
                n_seizure_samples = int(np.sum(trailing_labels))
                seizure_ratio = float(np.mean(trailing_labels))
                window_label = 1 if n_seizure_samples > 0 else 0

                # Extract seizure segments
                seizure_segments = self._get_seizure_segments_in_window(
                    trailing_labels, trailing_start_time, int(fs)
                )

                # Create metadata for trailing window
                trailing_metadata = {
                    'start_time': float(trailing_start_time),
                    'end_time': float(trailing_end_time),
                    'start_idx': int(last_covered_sample),
                    'end_idx': int(signal_length),
                    'sampling_rate': int(fs),
                    'window_label': window_label,
                    'n_seizure_samples': n_seizure_samples,
                    'seizure_ratio': seizure_ratio,
                    'seizure_segments': seizure_segments,
                    'window_type': 'trailing_incomplete',
                    'is_trailing_window': True,
                    'actual_window_size': float(trailing_duration)
                }

                # Add trailing window to results
                windows.append(trailing_signal)
                labels.append(trailing_labels)
                metadata.append(trailing_metadata)

                if window_label == 1:
                    logger.info(f"⚠️  Trailing window contains {n_seizure_samples} seizure samples!")
                    logger.info(f"   Seizure segments: {seizure_segments}")
            else:
                logger.info(f"Trailing segment too small ({trailing_duration:.1f}s < {self.min_trailing_size}s), skipping")
        else:
            logger.warning("No standard windows created, signal may be too short")

        logger.info(f"Final total: {len(windows)} windows (including trailing)")
        return windows, labels, metadata

    def _get_seizure_segments_in_window(
        self,
        window_labels: np.ndarray,
        window_start_time: float,
        sampling_rate: int
    ) -> List[Dict]:
        """
        Extract continuous seizure segments from window labels.

        Args:
            window_labels: Sample-level labels for the window (0/1 array)
            window_start_time: Start time of the window in seconds
            sampling_rate: Sampling frequency

        Returns:
            List of seizure segments with timing information
        """
        segments = []

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

    def preprocess_pipeline_with_trailing(
        self,
        data_path: str,
        subject_id: str,
        run_id: str,
        output_dir: str = "./results/preprocessed/"
    ) -> Dict:
        """
        Preprocessing pipeline with trailing window support

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

            recording = [subject_id, run_id]

            # Load data
            data = Data.loadData(data_path, recording, modalities=['ecg'])
            annotations = Annotation.loadAnnotation(data_path, recording)

            if not data.data:
                logger.error(f"No ECG data found for {subject_id} {run_id}")
                return None

            # Extract recording duration
            if data.data and len(data.data) > 0:
                first_channel_length = len(data.data[0])
                first_channel_fs = data.fs[0]
                recording_duration = first_channel_length / first_channel_fs
            else:
                recording_duration = 0

            logger.info(f"Processing {subject_id}_{run_id}: {recording_duration:.1f}s recording")

            # Process each ECG channel
            processed_channels = []
            total_seizures = 0

            for i, (signal_data, channel_name, fs) in enumerate(
                zip(data.data, data.channels, data.fs)
            ):
                if 'ecg' not in channel_name.lower():
                    continue

                logger.info(f"Processing channel: {channel_name}")

                if len(signal_data) == 0:
                    logger.warning(f"Empty signal for channel {channel_name}")
                    continue

                # Apply preprocessing steps
                filtered_signal = self.apply_bandpass_filter(signal_data, int(fs))
                downsampled_signal = self.downsample(filtered_signal, int(fs))
                downsampled_fs = self.downsample_freq

                # Create windows with trailing
                windows, window_labels, window_metadata = self.create_windows_with_trailing(
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
                    'min_trailing_size': self.min_trailing_size,
                    'trailing_windows_enabled': True
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
            logger.error(f"Error in trailing window preprocessing pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None


def find_files_with_trailing_data(
    preprocessed_dir: str,
    raw_data_dir: str,
    window_size: float = 3600.0,
    stride: float = 1800.0,
    min_trailing_size: float = 60.0
) -> List[Dict]:
    """
    Find all preprocessed files that have trailing data >= min_trailing_size

    Args:
        preprocessed_dir: Directory containing preprocessed .pkl files
        raw_data_dir: Directory containing raw SeizeIT2 data
        window_size: Standard window size in seconds
        stride: Window stride in seconds
        min_trailing_size: Minimum trailing size to consider (seconds)

    Returns:
        List of dicts with file info and trailing segment details
    """
    files_with_trailing = []
    preprocessed_path = Path(preprocessed_dir)

    if not preprocessed_path.exists():
        logger.error(f"Directory {preprocessed_dir} does not exist")
        return files_with_trailing

    pkl_files = list(preprocessed_path.glob("*_preprocessed.pkl"))
    logger.info(f"Found {len(pkl_files)} preprocessed files to check")

    already_processed_count = 0

    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)

            # Check if file was already processed with trailing windows
            preprocessing_params = data.get('preprocessing_params', {})
            if preprocessing_params.get('trailing_windows_enabled', False):
                already_processed_count += 1
                logger.debug(f"Skipping {pkl_file.name}: already has trailing windows")
                continue

            # Get recording duration and number of windows
            recording_duration = data.get('recording_duration', 0)

            if 'channels' not in data or len(data['channels']) == 0:
                continue

            n_windows = data['channels'][0].get('n_windows', 0)

            # Skip if no windows (handled by reprocess_empty_windows.py)
            if n_windows == 0:
                continue

            # Additional check: see if last window has trailing marker
            if 'metadata' in data['channels'][0]:
                last_metadata = data['channels'][0]['metadata'][-1] if data['channels'][0]['metadata'] else {}
                if last_metadata.get('is_trailing_window', False):
                    already_processed_count += 1
                    logger.debug(f"Skipping {pkl_file.name}: already has trailing window marker")
                    continue

            # Calculate expected trailing segment
            # Last window starts at: (n_windows - 1) * stride
            # Last window ends at: (n_windows - 1) * stride + window_size
            last_window_end = (n_windows - 1) * stride + window_size
            trailing_duration = recording_duration - last_window_end

            if trailing_duration >= min_trailing_size:
                # Check if trailing segment might have seizures
                # Extract subject and run ID
                subject_id = data.get('subject_id')
                run_id = data.get('run_id')

                # Load annotations to check for seizures in trailing segment
                has_trailing_seizure = False
                trailing_seizures = []

                try:
                    recording = [subject_id, run_id]
                    annotations = Annotation.loadAnnotation(raw_data_dir, recording)

                    if annotations and annotations.events:
                        for seizure_start, seizure_end in annotations.events:
                            # Check if seizure overlaps with trailing segment
                            if seizure_start < recording_duration and seizure_end > last_window_end:
                                has_trailing_seizure = True
                                overlap_start = max(last_window_end, seizure_start)
                                overlap_end = min(recording_duration, seizure_end)
                                trailing_seizures.append({
                                    'seizure_start': seizure_start,
                                    'seizure_end': seizure_end,
                                    'overlap_in_trailing': (overlap_start, overlap_end),
                                    'overlap_duration': overlap_end - overlap_start
                                })
                except Exception as e:
                    logger.warning(f"Could not load annotations for {subject_id}_{run_id}: {e}")

                file_info = {
                    'file_path': str(pkl_file),
                    'subject_id': subject_id,
                    'run_id': run_id,
                    'recording_duration': recording_duration,
                    'n_windows': n_windows,
                    'last_window_end': last_window_end,
                    'trailing_duration': trailing_duration,
                    'has_trailing_seizure': has_trailing_seizure,
                    'trailing_seizures': trailing_seizures,
                    'n_trailing_seizures': len(trailing_seizures)
                }

                files_with_trailing.append(file_info)

                status = "⚠️ HAS SEIZURES" if has_trailing_seizure else "no seizures"
                logger.info(f"Trailing data found: {pkl_file.name}")
                logger.info(f"  Duration: {recording_duration:.1f}s, Windows: {n_windows}, Trailing: {trailing_duration:.1f}s [{status}]")

        except Exception as e:
            logger.warning(f"Error checking {pkl_file.name}: {e}")

    logger.info(f"\nFound {len(files_with_trailing)} files with trailing data >= {min_trailing_size}s")
    if already_processed_count > 0:
        logger.info(f"Skipped {already_processed_count} files (already processed with trailing windows)")
    files_with_seizures = sum(1 for f in files_with_trailing if f['has_trailing_seizure'])
    logger.info(f"  {files_with_seizures} files have seizures in trailing segment")

    return files_with_trailing


def reprocess_file_with_trailing_windows(
    file_info: Dict,
    raw_data_dir: str,
    output_dir: str,
    preprocessor: TrailingWindowPreprocessor,
    create_backup: bool = False
) -> bool:
    """
    Reprocess a single file to add trailing window

    Args:
        file_info: Dictionary with file information
        raw_data_dir: Directory containing original SeizeIT2 data
        output_dir: Output directory for reprocessed files
        preprocessor: Trailing window preprocessor instance
        create_backup: If True and output_dir == input_dir, create .bak file

    Returns:
        True if successful, False otherwise
    """
    try:
        subject_id = file_info['subject_id']
        run_id = file_info['run_id']
        original_file = Path(file_info['file_path'])

        logger.info(f"Reprocessing {subject_id}_{run_id} (trailing: {file_info['trailing_duration']:.1f}s)")

        if file_info['has_trailing_seizure']:
            logger.info(f"  ⚠️ Contains {file_info['n_trailing_seizures']} seizure(s) in trailing segment!")

        # Create backup if requested and overwriting
        if create_backup and str(original_file.parent) == str(output_dir):
            backup_path = original_file.with_suffix('.pkl.bak')
            if not backup_path.exists():
                import shutil
                shutil.copy2(original_file, backup_path)
                logger.info(f"  Created backup: {backup_path.name}")

        # Reprocess with trailing window support
        result = preprocessor.preprocess_pipeline_with_trailing(
            data_path=raw_data_dir,
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
        logger.error(f"Error reprocessing {file_info['subject_id']}_{file_info['run_id']}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Reprocess files with trailing incomplete windows to capture cut-off seizures'
    )
    parser.add_argument('--preprocessed-dir',
                       default='/home/swolf/asim_shared/preprocessed_data/downsample_freq=8,window_size=3600_0,stride=1800_0_new',
                       help='Directory containing preprocessed files')
    parser.add_argument('--raw-data-dir',
                       default=RAW_DATA_PATH,
                       help='Directory containing original SeizeIT2 data')
    parser.add_argument('--output-dir',
                       default='/home/swolf/asim_shared/preprocessed_data/downsample_freq=8,window_size=3600_0,stride=1800_0_new_with_trailing',
                       help='Output directory for reprocessed files')
    parser.add_argument('--window-size', type=float, default=3600.0,
                       help='Standard window size in seconds (default: 3600)')
    parser.add_argument('--stride', type=float, default=1800.0,
                       help='Window stride in seconds (default: 1800)')
    parser.add_argument('--min-trailing-size', type=float, default=60.0,
                       help='Minimum trailing segment size to reprocess (default: 60s)')
    parser.add_argument('--max-files', type=int,
                       help='Maximum number of files to reprocess (for testing)')
    parser.add_argument('--only-with-seizures', action='store_true',
                       help='Only reprocess files that have seizures in trailing segment')
    parser.add_argument('--create-backup', action='store_true',
                       help='Create .bak files when overwriting (if input == output dir)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Only find files with trailing data, don\'t reprocess')

    args = parser.parse_args()

    # Check if input and output are the same
    input_output_same = str(Path(args.preprocessed_dir).resolve()) == str(Path(args.output_dir).resolve())
    if input_output_same:
        logger.warning("⚠️  Input and output directories are THE SAME!")
        logger.warning("   Files will be OVERWRITTEN with trailing windows added.")
        if args.create_backup:
            logger.info("   Backup files (.pkl.bak) will be created.")
        else:
            logger.warning("   Use --create-backup to create backups before overwriting.")
        if not args.dry_run:
            logger.warning("   Press Ctrl+C within 5 seconds to abort...")
            import time
            time.sleep(5)

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Find files with trailing data
    logger.info("Finding files with trailing data...")
    files_with_trailing = find_files_with_trailing_data(
        preprocessed_dir=args.preprocessed_dir,
        raw_data_dir=args.raw_data_dir,
        window_size=args.window_size,
        stride=args.stride,
        min_trailing_size=args.min_trailing_size
    )

    if not files_with_trailing:
        logger.info("No files with trailing data found")
        return

    # Filter by seizures if requested
    if args.only_with_seizures:
        files_with_trailing = [f for f in files_with_trailing if f['has_trailing_seizure']]
        logger.info(f"Filtered to {len(files_with_trailing)} files with seizures in trailing segment")

    if args.dry_run:
        logger.info("Dry run mode - not reprocessing files")
        logger.info(f"\nWould reprocess {len(files_with_trailing)} files:")

        # Save summary to CSV
        summary_path = Path(args.output_dir) / "trailing_windows_summary.csv"
        df = pd.DataFrame(files_with_trailing)
        df = df[['subject_id', 'run_id', 'recording_duration', 'n_windows',
                 'trailing_duration', 'has_trailing_seizure', 'n_trailing_seizures']]
        df.to_csv(summary_path, index=False)
        logger.info(f"Summary saved to: {summary_path}")

        for f in files_with_trailing[:20]:  # Show first 20
            status = "⚠️ SEIZURES" if f['has_trailing_seizure'] else ""
            logger.info(f"  {f['subject_id']}_{f['run_id']}: {f['trailing_duration']:.1f}s trailing {status}")

        if len(files_with_trailing) > 20:
            logger.info(f"  ... and {len(files_with_trailing) - 20} more")

        return

    # Limit files if requested
    if args.max_files:
        files_with_trailing = files_with_trailing[:args.max_files]
        logger.info(f"Limited to {len(files_with_trailing)} files for processing")

    # Initialize trailing window preprocessor
    filter_params = {
        'low_freq': 0.5,
        'high_freq': 40.0,
        'order': 4
    }

    preprocessor = TrailingWindowPreprocessor(
        filter_params=filter_params,
        downsample_freq=8,
        window_size=args.window_size,
        stride=args.stride,
        min_trailing_size=args.min_trailing_size
    )

    # Process files
    success_count = 0
    failed_count = 0
    recovered_seizures = 0

    for i, file_info in enumerate(files_with_trailing, 1):
        logger.info(f"\nProcessing {i}/{len(files_with_trailing)}: {file_info['subject_id']}_{file_info['run_id']}")

        success = reprocess_file_with_trailing_windows(
            file_info=file_info,
            raw_data_dir=args.raw_data_dir,
            output_dir=args.output_dir,
            preprocessor=preprocessor,
            create_backup=args.create_backup
        )

        if success:
            success_count += 1
            if file_info['has_trailing_seizure']:
                recovered_seizures += file_info['n_trailing_seizures']
        else:
            failed_count += 1

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"🎯 REPROCESSING SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Files found with trailing data:     {len(files_with_trailing)}")
    logger.info(f"Files with seizures in trailing:    {sum(1 for f in files_with_trailing if f['has_trailing_seizure'])}")
    logger.info(f"Successfully reprocessed:            {success_count}")
    logger.info(f"Failed to reprocess:                 {failed_count}")
    logger.info(f"Potential seizures recovered:        {recovered_seizures}")
    logger.info(f"Output directory:                    {args.output_dir}")
    logger.info(f"{'='*80}")

if __name__ == "__main__":
    main()
