#!/usr/bin/env python3
"""
Analyze only trailing windows from reprocessed files and append results to existing JSON files.

This script:
1. Identifies files with trailing windows (is_trailing_window: True)
2. Analyzes ONLY the trailing window with Madrid
3. Finds the corresponding existing JSON result file
4. Appends the trailing window results to the existing JSON

This is much faster than reprocessing entire files!
"""

import os
import sys
import json
import pickle
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
import time

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.madrid_v2 import MADRID_V2

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('madrid_trailing_window_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def has_trailing_window(file_path: str) -> Tuple[bool, Dict]:
    """
    Check if a preprocessed file has a trailing window.

    Returns:
        Tuple of (has_trailing, info_dict)
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # Check preprocessing params
        preprocessing_params = data.get('preprocessing_params', {})
        if not preprocessing_params.get('trailing_windows_enabled', False):
            return False, {}

        # Get channel data
        if 'channels' not in data or len(data['channels']) == 0:
            return False, {}

        channel = data['channels'][0]
        metadata = channel.get('metadata', [])

        if not metadata:
            return False, {}

        # Check last window for trailing marker
        last_meta = metadata[-1]
        if not last_meta.get('is_trailing_window', False):
            return False, {}

        # Extract info
        info = {
            'subject_id': data.get('subject_id'),
            'run_id': data.get('run_id'),
            'trailing_window_index': len(metadata) - 1,
            'trailing_duration': last_meta.get('actual_window_size', 0),
            'has_seizure': last_meta.get('window_label', 0) == 1,
            'n_seizure_samples': last_meta.get('n_seizure_samples', 0),
            'n_windows_total': len(metadata)
        }

        return True, info

    except Exception as e:
        logger.warning(f"Error checking trailing window in {file_path}: {e}")
        return False, {}


def find_existing_json(subject_id: str, run_id: str, results_dir: str) -> Optional[str]:
    """
    Find existing JSON result file for subject and run.

    Returns:
        Path to existing JSON file or None
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        return None

    # Search for matching JSON files
    pattern = f"madrid_windowed_results_{subject_id}_{run_id}_*.json"
    matching_files = list(results_path.glob(pattern))

    if not matching_files:
        return None

    # Return most recent file
    matching_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(matching_files[0])


def analyze_trailing_window_with_madrid(
    window: np.ndarray,
    window_metadata: Dict,
    sampling_rate: int,
    config: Dict
) -> Dict:
    """
    Analyze a single trailing window with Madrid.

    Args:
        window: Trailing window signal data
        window_metadata: Metadata for the trailing window
        sampling_rate: Sampling rate
        config: Madrid configuration

    Returns:
        Analysis results dict
    """
    start_time = time.time()

    try:
        # Initialize Madrid
        madrid = MADRID_V2(
            use_gpu=config.get('use_gpu', True),
            enable_output=False
        )

        # Adapt Madrid parameters for window size and sampling rate
        window_length = len(window)
        window_duration = window_length / sampling_rate

        # Calculate Madrid parameters
        if window_duration < 30:
            min_length = max(int(sampling_rate * 2), 16)
            max_length = max(int(window_length * 0.3), min_length * 2)
            step_size = max(int(sampling_rate * 1), 8)
        elif window_duration < 300:
            min_length = max(int(sampling_rate * 5), 40)
            max_length = max(int(window_length * 0.4), min_length * 2)
            step_size = max(int(sampling_rate * 2), 16)
        else:
            min_length = int(sampling_rate * 10)
            max_length = int(window_length * 0.5)
            step_size = int(sampling_rate * 5)

        # Ensure bounds
        min_length = max(min_length, 8)
        max_length = min(max(max_length, min_length * 2), window_length - 1)
        step_size = max(step_size, 4)

        logger.info(f"Madrid params: min={min_length}, max={max_length}, step={step_size} (window: {window_duration:.1f}s)")

        # Check if window is long enough
        if window_length < min_length:
            logger.warning(f"Window too short ({window_length} < {min_length}), skipping Madrid analysis")
            return {
                'window_index': window_metadata.get('window_index', -1),
                'window_type': 'trailing_incomplete',
                'is_trailing_window': True,
                'window_start_time': window_metadata.get('start_time', 0),
                'window_duration': window_duration,
                'original_length': window_length,
                'cleaned_length': window_length,
                'anomalies': [],
                'execution_time': time.time() - start_time,
                'analysis_successful': False,
                'warning': f'Window too short for Madrid analysis ({window_length} < {min_length})'
            }

        # Calculate train_test_split
        train_minutes = config.get('madrid_parameters', {}).get('analysis_config', {}).get('train_minutes', 20)
        train_samples = int(train_minutes * 60 * sampling_rate)
        train_test_split = min(train_samples, window_length - 1)
        train_test_split = max(1, train_test_split)

        logger.info(f"train_test_split: {train_test_split} (train_minutes: {train_minutes})")

        # Run Madrid
        multi_length_table, bsf, bsf_loc = madrid.fit(
            T=window,
            min_length=min_length,
            max_length=max_length,
            step_size=step_size,
            train_test_split=train_test_split,
            factor=1
        )

        # Get anomalies
        threshold_percentile = config.get('madrid_parameters', {}).get('analysis_config', {}).get('threshold_percentile', 95)
        anomaly_info = madrid.get_anomaly_scores(threshold_percentile=threshold_percentile)
        anomalies = anomaly_info['anomalies']

        # Format anomalies
        formatted_anomalies = []
        for i, anomaly in enumerate(anomalies):
            formatted_anomaly = {
                'rank': i + 1,
                'window_index': window_metadata.get('window_index', -1),
                'm_value': anomaly['length'],
                'anomaly_score': float(anomaly['score']),
                'location_sample_in_window': int(anomaly['location']) if anomaly['location'] is not None else None,
                'location_time_in_window': float(anomaly['location'] / sampling_rate) if anomaly['location'] is not None else None,
                'location_time_absolute': float(window_metadata.get('start_time', 0) + anomaly['location'] / sampling_rate) if anomaly['location'] is not None else None,
                'normalized_score': float(anomaly['score']),
                'confidence': min(float(anomaly['score']), 1.0),
                'anomaly_id': f"trailing_w{window_metadata.get('window_index', -1)}_m{anomaly['length']}_rank{i+1}_loc{anomaly['location'] if anomaly['location'] is not None else 'unknown'}"
            }
            formatted_anomalies.append(formatted_anomaly)

        result = {
            'window_index': window_metadata.get('window_index', -1),
            'window_type': 'trailing_incomplete',
            'is_trailing_window': True,
            'window_start_time': window_metadata.get('start_time', 0),
            'window_end_time': window_metadata.get('end_time', 0),
            'window_duration': window_duration,
            'original_length': window_length,
            'cleaned_length': window_length,
            'constant_regions_removed': 0,
            'anomalies': formatted_anomalies,
            'execution_time': time.time() - start_time,
            'analysis_successful': True,
            'madrid_params_used': {
                'min_length': min_length,
                'max_length': max_length,
                'step_size': step_size,
                'train_test_split': train_test_split,
                'threshold_percentile': threshold_percentile
            }
        }

        logger.info(f"✓ Trailing window analyzed: {len(anomalies)} anomalies found in {time.time() - start_time:.1f}s")

        return result

    except Exception as e:
        logger.error(f"Error analyzing trailing window: {e}")
        import traceback
        traceback.print_exc()

        return {
            'window_index': window_metadata.get('window_index', -1),
            'window_type': 'trailing_incomplete',
            'is_trailing_window': True,
            'window_start_time': window_metadata.get('start_time', 0),
            'window_duration': window_metadata.get('actual_window_size', 0),
            'analysis_successful': False,
            'error_message': str(e),
            'execution_time': time.time() - start_time
        }


def append_trailing_window_to_json(
    json_path: str,
    trailing_window_result: Dict,
    trailing_window_metadata: Dict
) -> bool:
    """
    Append trailing window result to existing JSON file and update ground truth.

    Args:
        json_path: Path to existing JSON result file
        trailing_window_result: Trailing window analysis result
        trailing_window_metadata: Metadata from preprocessed file (contains seizure info)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load existing JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Add trailing window to window_results
        if 'analysis_results' not in data:
            logger.error(f"No 'analysis_results' in {json_path}")
            return False

        if 'window_results' not in data['analysis_results']:
            logger.error(f"No 'window_results' in analysis_results")
            return False

        # Append trailing window result
        data['analysis_results']['window_results'].append(trailing_window_result)

        # Update totals
        if 'total_windows_processed' in data['analysis_results']:
            data['analysis_results']['total_windows_processed'] += 1

        if trailing_window_result.get('analysis_successful', False):
            anomaly_count = len(trailing_window_result.get('anomalies', []))
            if 'total_anomalies_found' in data['analysis_results']:
                data['analysis_results']['total_anomalies_found'] += anomaly_count

        # UPDATE GROUND TRUTH if trailing window has seizure
        has_seizure = trailing_window_metadata.get('window_label', 0) == 1
        if has_seizure:
            logger.info(f"  ⚠️  Trailing window contains seizure - updating ground truth!")

            if 'validation_data' not in data:
                data['validation_data'] = {}

            if 'ground_truth' not in data['validation_data']:
                data['validation_data']['ground_truth'] = {
                    'seizure_present': False,
                    'seizure_windows': [],
                    'total_windows': 0,
                    'annotation_source': 'windowed_preprocessing',
                    'annotator_id': 'automated'
                }

            ground_truth = data['validation_data']['ground_truth']

            # Update seizure_present flag
            ground_truth['seizure_present'] = True

            # Update total_windows count
            if 'total_windows' in ground_truth:
                ground_truth['total_windows'] += 1

            # Extract seizure information from metadata
            seizure_segments = trailing_window_metadata.get('seizure_segments', [])
            n_seizure_samples = trailing_window_metadata.get('n_seizure_samples', 0)
            seizure_ratio = trailing_window_metadata.get('seizure_ratio', 0.0)

            # Add trailing window to seizure_windows list
            window_index = trailing_window_result.get('window_index', -1)

            # Get windowing info from input_data
            signal_metadata = data.get('input_data', {}).get('signal_metadata', {})
            windowing_info = signal_metadata.get('windowing_info', {})
            stride = windowing_info.get('stride', 1800)  # Default stride

            seizure_window_entry = {
                'window_index': window_index,
                'seizure_ratio': float(seizure_ratio),
                'window_start_time': trailing_window_metadata.get('start_time', window_index * stride),
                'window_duration': trailing_window_metadata.get('actual_window_size', 0),
                'seizure_segments': seizure_segments,
                'n_seizure_samples': int(n_seizure_samples),
                'is_trailing_window': True
            }

            if 'seizure_windows' not in ground_truth:
                ground_truth['seizure_windows'] = []

            ground_truth['seizure_windows'].append(seizure_window_entry)

            logger.info(f"  ✓ Ground truth updated: added seizure window {window_index}")
            logger.info(f"    - Seizure ratio: {seizure_ratio:.2%}")
            logger.info(f"    - Seizure samples: {n_seizure_samples}")
            logger.info(f"    - Seizure segments: {len(seizure_segments)}")
        else:
            # No seizure in trailing window - just update total_windows count
            if 'validation_data' in data and 'ground_truth' in data['validation_data']:
                ground_truth = data['validation_data']['ground_truth']
                if 'total_windows' in ground_truth:
                    ground_truth['total_windows'] += 1
                    logger.info(f"  Ground truth updated: total_windows = {ground_truth['total_windows']}")

        # Add metadata about trailing window addition
        if 'analysis_metadata' in data:
            if 'modifications' not in data['analysis_metadata']:
                data['analysis_metadata']['modifications'] = []

            modification_entry = {
                'timestamp': datetime.now().isoformat() + 'Z',
                'modification_type': 'trailing_window_added',
                'description': 'Trailing window analysis appended to existing results',
                'window_index': trailing_window_result.get('window_index'),
                'anomalies_added': len(trailing_window_result.get('anomalies', [])),
                'seizure_in_trailing_window': has_seizure
            }

            if has_seizure:
                modification_entry['ground_truth_updated'] = True
                modification_entry['seizure_info'] = {
                    'n_seizure_samples': trailing_window_metadata.get('n_seizure_samples', 0),
                    'seizure_ratio': trailing_window_metadata.get('seizure_ratio', 0.0),
                    'n_seizure_segments': len(trailing_window_metadata.get('seizure_segments', []))
                }

            data['analysis_metadata']['modifications'].append(modification_entry)

        # Save updated JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Trailing window result appended to {Path(json_path).name}")
        return True

    except Exception as e:
        logger.error(f"Error appending to JSON {json_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_single_file_trailing_window(
    file_path: str,
    results_dir: str,
    config: Dict
) -> Tuple[bool, str, Dict]:
    """
    Process a single file's trailing window.

    Args:
        file_path: Path to preprocessed file
        results_dir: Directory containing existing JSON results
        config: Madrid configuration

    Returns:
        Tuple of (success, file_path, info)
    """
    try:
        # Check if file has trailing window
        has_trailing, info = has_trailing_window(file_path)

        if not has_trailing:
            return False, file_path, {'error': 'No trailing window found'}

        logger.info(f"Processing trailing window: {info['subject_id']}_{info['run_id']}")
        logger.info(f"  Trailing window index: {info['trailing_window_index']}")
        logger.info(f"  Duration: {info['trailing_duration']:.1f}s")
        logger.info(f"  Has seizure: {info['has_seizure']}")

        # Find existing JSON
        json_path = find_existing_json(
            info['subject_id'],
            info['run_id'],
            results_dir
        )

        if not json_path:
            logger.warning(f"No existing JSON found for {info['subject_id']}_{info['run_id']}")
            return False, file_path, {'error': 'No existing JSON result file found'}

        logger.info(f"  Found existing JSON: {Path(json_path).name}")

        # Load file data
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        channel = data['channels'][0]
        windows = channel['windows']
        metadata = channel['metadata']

        trailing_window_index = info['trailing_window_index']
        trailing_window = windows[trailing_window_index]
        trailing_metadata = metadata[trailing_window_index]

        # Add window index to metadata
        trailing_metadata['window_index'] = trailing_window_index

        sampling_rate = channel.get('processed_fs', 8)

        # Analyze trailing window
        logger.info(f"  Analyzing trailing window with Madrid...")
        trailing_result = analyze_trailing_window_with_madrid(
            trailing_window,
            trailing_metadata,
            sampling_rate,
            config
        )

        # Append to existing JSON (with metadata for ground truth update)
        success = append_trailing_window_to_json(
            json_path,
            trailing_result,
            trailing_metadata
        )

        if success:
            result_info = {
                'subject_id': info['subject_id'],
                'run_id': info['run_id'],
                'trailing_window_index': trailing_window_index,
                'anomalies_found': len(trailing_result.get('anomalies', [])),
                'execution_time': trailing_result.get('execution_time', 0),
                'json_file': json_path
            }
            return True, file_path, result_info
        else:
            return False, file_path, {'error': 'Failed to append to JSON'}

    except Exception as e:
        logger.error(f"Error processing trailing window for {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return False, file_path, {'error': str(e)}


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Analyze trailing windows and append results to existing JSON files'
    )
    parser.add_argument('--data-dir', required=True,
                       help='Directory containing reprocessed files with trailing windows')
    parser.add_argument('--results-dir', required=True,
                       help='Directory containing existing Madrid JSON results')
    parser.add_argument('--n-workers', type=int,
                       help=f'Number of worker processes (default: {cpu_count()-1})')
    parser.add_argument('--max-files', type=int,
                       help='Maximum number of files to process (for testing)')
    parser.add_argument('--only-with-seizures', action='store_true',
                       help='Only process trailing windows that contain seizures')
    parser.add_argument('--use-gpu', action='store_true', default=True,
                       help='Use GPU for Madrid')
    parser.add_argument('--threshold-percentile', type=float, default=95,
                       help='Anomaly threshold percentile (default: 95)')
    parser.add_argument('--train-minutes', type=float, default=20,
                       help='Training minutes for Madrid (default: 20)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Only identify files with trailing windows, don\'t process')

    args = parser.parse_args()

    n_workers = args.n_workers or max(1, cpu_count() - 1)

    logger.info("=" * 80)
    logger.info("MADRID TRAILING WINDOW ANALYZER")
    logger.info("=" * 80)
    logger.info(f"Data directory:    {args.data_dir}")
    logger.info(f"Results directory: {args.results_dir}")
    logger.info(f"Workers:           {n_workers}")
    logger.info(f"GPU:               {args.use_gpu}")
    logger.info("")

    # Find files with trailing windows
    logger.info("Step 1: Finding files with trailing windows...")

    data_path = Path(args.data_dir)
    pkl_files = list(data_path.glob("*_preprocessed.pkl"))

    logger.info(f"Found {len(pkl_files)} preprocessed files, checking for trailing windows...")

    files_with_trailing = []
    for pkl_file in pkl_files:
        has_trailing, info = has_trailing_window(str(pkl_file))
        if has_trailing:
            if args.only_with_seizures and not info['has_seizure']:
                continue
            files_with_trailing.append((str(pkl_file), info))

    logger.info(f"\nFound {len(files_with_trailing)} files with trailing windows")

    if args.only_with_seizures:
        with_seizures = sum(1 for _, info in files_with_trailing if info['has_seizure'])
        logger.info(f"  {with_seizures} trailing windows contain seizures")

    if len(files_with_trailing) == 0:
        logger.info("No files with trailing windows found. Exiting.")
        return

    # Show summary
    logger.info("\nTRAILING WINDOW SUMMARY:")
    for i, (file_path, info) in enumerate(files_with_trailing[:10], 1):
        status = "⚠️ SEIZURE" if info['has_seizure'] else "no seizure"
        logger.info(f"  {info['subject_id']}_{info['run_id']}: "
                   f"{info['trailing_duration']:.1f}s trailing [{status}]")

    if len(files_with_trailing) > 10:
        logger.info(f"  ... and {len(files_with_trailing) - 10} more")

    if args.dry_run:
        logger.info("\nDry run mode - exiting without processing")
        return

    # Limit files if requested
    if args.max_files:
        files_with_trailing = files_with_trailing[:args.max_files]
        logger.info(f"\nLimited to {len(files_with_trailing)} files for processing")

    # Setup config
    config = {
        'use_gpu': args.use_gpu,
        'madrid_parameters': {
            'analysis_config': {
                'train_minutes': args.train_minutes,
                'threshold_percentile': args.threshold_percentile
            }
        }
    }

    # Process files
    logger.info(f"\nStep 2: Processing trailing windows with Madrid...")
    logger.info(f"Using {n_workers} workers")
    logger.info("")

    start_time = time.time()
    success_count = 0
    failed_count = 0
    total_anomalies = 0

    # Create process function
    process_func = partial(
        process_single_file_trailing_window,
        results_dir=args.results_dir,
        config=config
    )

    # Process in parallel
    file_paths = [fp for fp, _ in files_with_trailing]

    with Pool(processes=n_workers) as pool:
        results = pool.map(process_func, file_paths)

        for success, file_path, result_info in results:
            if success:
                success_count += 1
                anomalies = result_info.get('anomalies_found', 0)
                total_anomalies += anomalies
                exec_time = result_info.get('execution_time', 0)
                logger.info(f"✓ {result_info['subject_id']}_{result_info['run_id']}: "
                           f"{anomalies} anomalies, {exec_time:.1f}s")
            else:
                failed_count += 1
                logger.error(f"❌ {Path(file_path).name}: {result_info.get('error', 'Unknown error')}")

    total_time = time.time() - start_time

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("🎯 TRAILING WINDOW ANALYSIS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Files with trailing windows:    {len(files_with_trailing)}")
    logger.info(f"Successfully processed:          {success_count}")
    logger.info(f"Failed:                          {failed_count}")
    logger.info(f"Total anomalies found:           {total_anomalies}")
    logger.info(f"Total execution time:            {total_time/60:.1f} minutes")
    logger.info(f"Average time per file:           {total_time/len(files_with_trailing):.1f} seconds")
    logger.info("=" * 80)

    logger.info(f"\n✓ Results appended to existing JSON files in: {args.results_dir}")


if __name__ == "__main__":
    main()
