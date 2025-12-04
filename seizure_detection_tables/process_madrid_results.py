#!/usr/bin/env python3
"""
Process Madrid raw results and create unified detection tables.

This script:
1. Reads raw Madrid JSON results (per run)
2. Extracts ground truth seizures from validation_data
3. Tests different thresholds and clustering strategies
4. Finds best configurations for: Sensitivity, FAR, HMS
5. Outputs unified CSV files with: subject, run, seizure_idx, detected, config, window_type

Usage:
    python process_madrid_results.py --raw-data-dir /path/to/madrid/raw/results
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import pandas as pd
import numpy as np
from datetime import datetime


class MadridResultsProcessor:
    """Process Madrid raw results and generate unified detection tables."""

    def __init__(self, raw_data_dir: Path,
                 pre_seizure_window: float = 5.0,
                 post_seizure_window: float = 3.0):
        """
        Initialize the processor.

        Args:
            raw_data_dir: Directory containing madrid_windowed_results_*.json files
            pre_seizure_window: Minutes before seizure for extended window (default: 5.0)
            post_seizure_window: Minutes after seizure for extended window (default: 3.0)
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.pre_seizure_seconds = pre_seizure_window * 60.0
        self.post_seizure_seconds = post_seizure_window * 60.0

        # Training set subjects (sub-001 to sub-096)
        self.training_subjects = [f"sub-{i:03d}" for i in range(1, 97)]

        # Test set subjects (sub-097 to sub-125)
        self.test_subjects = [f"sub-{i:03d}" for i in range(97, 126)]

        # Saturated test runs to exclude
        self.saturated_test_runs = {
            ("sub-099", "run-01"), ("sub-114", "run-03"), ("sub-115", "run-11"),
            ("sub-115", "run-32"), ("sub-117", "run-13"), ("sub-118", "run-07"),
            ("sub-119", "run-24"), ("sub-119", "run-36"), ("sub-123", "run-22"),
            ("sub-124", "run-19"), ("sub-124", "run-43"), ("sub-124", "run-63"),
            ("sub-125", "run-36"), ("sub-125", "run-67")
        }

    def load_raw_results(self, subset: str = 'test', exclude_saturated: bool = False) -> List[Dict]:
        """
        Load raw Madrid JSON result files.

        Args:
            subset: Which subset to load - 'training', 'test', or 'both'
            exclude_saturated: If True, exclude saturated test runs

        Returns:
            List of result dictionaries
        """
        json_files = list(self.raw_data_dir.glob("madrid_windowed_results_*.json"))

        print(f"Found {len(json_files)} raw result files")

        results = []
        skipped_saturated = 0

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                # Extract subject and run from input_data
                subject_id = data['input_data']['subject_id']
                run_id = data['input_data']['run_id']

                # Filter by subset
                if subset == 'training':
                    if subject_id not in self.training_subjects:
                        continue
                elif subset == 'test':
                    if subject_id not in self.test_subjects:
                        continue
                elif subset == 'both':
                    # Include all subjects
                    if subject_id not in self.training_subjects and subject_id not in self.test_subjects:
                        continue
                else:
                    raise ValueError(f"Invalid subset: {subset}. Must be 'training', 'test', or 'both'")

                # Exclude saturated runs if requested
                if exclude_saturated and (subject_id, run_id) in self.saturated_test_runs:
                    skipped_saturated += 1
                    continue

                results.append({
                    'file': json_file,
                    'subject': subject_id,
                    'run': run_id,
                    'data': data
                })

            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue

        print(f"Loaded {len(results)} files from {subset} set")
        if exclude_saturated and skipped_saturated > 0:
            print(f"Excluded {skipped_saturated} saturated runs")

        return results

    def extract_seizures(self, result: Dict) -> List[Dict]:
        """
        Extract seizure information from validation_data.

        Returns:
            List of seizures with their time ranges
        """
        data = result['data']
        validation_data = data.get('validation_data', {})
        ground_truth = validation_data.get('ground_truth', {})

        if not ground_truth.get('seizure_present', False):
            return []

        seizures = []
        seizure_windows = ground_truth.get('seizure_windows', [])

        # Extract unique seizures (may span multiple windows)
        seizure_segments_by_time = {}

        for sz_window in seizure_windows:
            for segment in sz_window.get('seizure_segments', []):
                start_time = segment['start_time_absolute']
                end_time = segment['end_time_absolute']

                # Use start time as unique identifier
                if start_time not in seizure_segments_by_time:
                    seizure_segments_by_time[start_time] = {
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': segment['duration_seconds']
                    }

        # Convert to list and sort by start time
        seizures = list(seizure_segments_by_time.values())
        seizures.sort(key=lambda x: x['start_time'])

        # Add seizure index (0-based per run)
        for idx, seizure in enumerate(seizures):
            seizure['seizure_idx'] = idx

        return seizures

    def extract_anomalies(self, result: Dict) -> List[Dict]:
        """Extract all anomalies from window results."""
        data = result['data']
        window_results = data.get('analysis_results', {}).get('window_results', [])

        anomalies = []
        for window in window_results:
            window_start = window['window_start_time']

            for anomaly in window.get('anomalies', []):
                # Calculate absolute time of anomaly
                absolute_time = window_start + anomaly['location_time_in_window']

                anomalies.append({
                    'time': absolute_time,
                    'score': anomaly['anomaly_score'],
                    'window_index': window['window_index']
                })

        return anomalies

    def cluster_anomalies(self, anomalies: List[Dict], time_threshold: float) -> List[List[Dict]]:
        """
        Cluster anomalies based on time proximity.

        Args:
            anomalies: List of anomaly dictionaries
            time_threshold: Maximum time gap in seconds between anomalies in same cluster

        Returns:
            List of clusters (each cluster is a list of anomalies)
        """
        if not anomalies:
            return []

        # Sort by time
        sorted_anomalies = sorted(anomalies, key=lambda x: x['time'])

        clusters = []
        current_cluster = [sorted_anomalies[0]]

        for anomaly in sorted_anomalies[1:]:
            # Check if anomaly is close enough to last anomaly in current cluster
            time_gap = anomaly['time'] - current_cluster[-1]['time']

            if time_gap <= time_threshold:
                current_cluster.append(anomaly)
            else:
                # Start new cluster
                clusters.append(current_cluster)
                current_cluster = [anomaly]

        # Add last cluster
        clusters.append(current_cluster)

        return clusters

    def check_seizure_detection(self, seizure: Dict, clusters: List[List[Dict]],
                                use_extended_window: bool = False) -> bool:
        """
        Check if a seizure was detected by any cluster.

        Args:
            seizure: Seizure dictionary with start_time and end_time
            clusters: List of anomaly clusters
            use_extended_window: If True, use extended detection window

        Returns:
            True if seizure was detected, False otherwise
        """
        seizure_start = seizure['start_time']
        seizure_end = seizure['end_time']

        # Define detection window
        if use_extended_window:
            detection_start = seizure_start - self.pre_seizure_seconds
            detection_end = seizure_end + self.post_seizure_seconds
        else:
            detection_start = seizure_start
            detection_end = seizure_end

        # Check if any cluster overlaps with detection window
        for cluster in clusters:
            # Get time range of cluster
            cluster_start = min(a['time'] for a in cluster)
            cluster_end = max(a['time'] for a in cluster)

            # Check for overlap
            if cluster_start <= detection_end and cluster_end >= detection_start:
                return True

        return False

    def evaluate_threshold(self, results: List[Dict], threshold: float,
                          clustering_time: float, use_extended_window: bool = False) -> Dict:
        """
        Evaluate detection performance for a specific threshold and clustering strategy.

        Args:
            results: List of raw result dictionaries
            threshold: Anomaly score threshold
            clustering_time: Time threshold for clustering in seconds
            use_extended_window: Use extended detection window

        Returns:
            Dictionary with performance metrics and per-seizure detections
        """
        all_detections = []
        total_seizures = 0
        detected_seizures = 0
        total_false_positives = 0
        total_duration = 0.0

        for result in results:
            subject = result['subject']
            run = result['run']

            # Get recording duration
            duration = result['data']['input_data']['signal_metadata']['total_duration_seconds']
            total_duration += duration

            # Extract seizures and anomalies
            seizures = self.extract_seizures(result)
            all_anomalies = self.extract_anomalies(result)

            # Filter anomalies by threshold
            filtered_anomalies = [a for a in all_anomalies if a['score'] >= threshold]

            # Cluster anomalies
            clusters = self.cluster_anomalies(filtered_anomalies, clustering_time)

            # Check each seizure
            for seizure in seizures:
                total_seizures += 1
                detected = self.check_seizure_detection(seizure, clusters, use_extended_window)

                all_detections.append({
                    'subject': subject,
                    'run': run,
                    'seizure_idx': seizure['seizure_idx'],
                    'detected': detected
                })

                if detected:
                    detected_seizures += 1

            # Count false positives (clusters not matching any seizure)
            for cluster in clusters:
                is_false_positive = True
                for seizure in seizures:
                    if self.check_seizure_detection(seizure, [cluster], use_extended_window):
                        is_false_positive = False
                        break

                if is_false_positive:
                    total_false_positives += 1

        # Calculate metrics
        sensitivity = detected_seizures / total_seizures if total_seizures > 0 else 0.0
        total_duration_hours = total_duration / 3600.0
        far = total_false_positives / total_duration_hours if total_duration_hours > 0 else 0.0

        # Challenge Score = Sensitivity (%) - 0.4 * FAR
        challenge_score = (sensitivity * 100) - (0.4 * far)

        return {
            'threshold': threshold,
            'clustering_time': clustering_time,
            'sensitivity': sensitivity,
            'far': far,
            'challenge_score': challenge_score,
            'detected_seizures': detected_seizures,
            'total_seizures': total_seizures,
            'total_false_positives': total_false_positives,
            'detections': all_detections
        }

    def find_best_configurations(self, results: List[Dict],
                                use_extended_window: bool = False,
                                n_thresholds: int = 100) -> Dict[str, Dict]:
        """
        Find best configurations for Sensitivity, FAR, and HMS.

        Args:
            results: List of raw result dictionaries
            use_extended_window: Use extended detection window
            n_thresholds: Number of thresholds to test

        Returns:
            Dictionary with best configs: {'Sensitivity': {...}, 'FAR': {...}, 'HMS': {...}}
        """
        window_type = "window" if use_extended_window else "no_window"

        # Determine clustering strategy based on window type
        # From the scripts: withSDW uses time_180s, withoutSDW uses time_600s
        clustering_time = 180.0 if use_extended_window else 600.0

        print(f"\n{'='*80}")
        print(f"Finding best configurations for {window_type}")
        print(f"Clustering time: {clustering_time}s")
        print(f"Testing {n_thresholds} thresholds...")
        print(f"{'='*80}")

        # Test thresholds from 0.0 to 1.0
        thresholds = np.linspace(0.0, 1.0, n_thresholds)

        evaluations = []
        for i, threshold in enumerate(thresholds):
            if (i + 1) % 10 == 0:
                print(f"Progress: {i+1}/{n_thresholds} thresholds tested")

            eval_result = self.evaluate_threshold(
                results, threshold, clustering_time, use_extended_window
            )
            evaluations.append(eval_result)

        # Find best configurations
        best_configs = {}

        # Best Sensitivity
        best_sensitivity = max(evaluations, key=lambda x: x['sensitivity'])
        best_configs['Sensitivity'] = best_sensitivity

        # Best FAR (lowest FAR with sensitivity > 0)
        valid_evals = [e for e in evaluations if e['sensitivity'] > 0]
        if valid_evals:
            best_far = min(valid_evals, key=lambda x: x['far'])
            best_configs['FAR'] = best_far

        # Best HMS (Challenge Score)
        best_hms = max(evaluations, key=lambda x: x['challenge_score'])
        best_configs['HMS'] = best_hms

        print(f"\nBest Configurations for {window_type}:")
        print(f"  Sensitivity: {best_configs['Sensitivity']['sensitivity']:.4f} "
              f"(FAR: {best_configs['Sensitivity']['far']:.4f})")
        if 'FAR' in best_configs:
            print(f"  FAR: {best_configs['FAR']['far']:.4f} "
                  f"(Sensitivity: {best_configs['FAR']['sensitivity']:.4f})")
        print(f"  HMS: Challenge Score {best_configs['HMS']['challenge_score']:.4f} "
              f"(Sensitivity: {best_configs['HMS']['sensitivity']:.4f}, FAR: {best_configs['HMS']['far']:.4f})")

        return best_configs

    def create_unified_tables(self, output_dir: Path = None):
        """
        Create unified detection tables for all configurations.

        Strategy:
        1. Load training data (sub-001 to sub-096)
        2. Find best thresholds using training data
        3. Load test data (sub-097 to sub-125, excluding saturated runs)
        4. Apply best thresholds to test data
        5. Output detections on test data only

        Output: CSV files with columns: subject, run, seizure_idx, detected, config, window_type
        """
        if output_dir is None:
            output_dir = Path('.')

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        print(f"\n{'='*80}")
        print(f"STEP 1: LOAD TRAINING DATA")
        print(f"{'='*80}")

        # Load training data for threshold selection
        training_results = self.load_raw_results(subset='training', exclude_saturated=False)

        if not training_results:
            print("ERROR: No training data found!")
            return

        print(f"\n{'='*80}")
        print(f"STEP 2: LOAD TEST DATA (excluding saturated runs)")
        print(f"{'='*80}")

        # Load test data (excluding saturated runs) for final evaluation
        test_results = self.load_raw_results(subset='test', exclude_saturated=True)

        if not test_results:
            print("ERROR: No test data found!")
            return

        all_detections = []

        # Process both window types
        for use_extended_window in [True, False]:
            window_type = "window" if use_extended_window else "no_window"

            print(f"\n{'='*80}")
            print(f"STEP 3: FIND BEST THRESHOLDS ON TRAINING DATA ({window_type})")
            print(f"{'='*80}")

            # Find best configurations using TRAINING data
            best_configs_training = self.find_best_configurations(training_results, use_extended_window)

            print(f"\n{'='*80}")
            print(f"STEP 4: APPLY THRESHOLDS TO TEST DATA ({window_type})")
            print(f"{'='*80}")

            # Apply best thresholds to TEST data
            clustering_time = 180.0 if use_extended_window else 600.0

            for config_name, config_result in best_configs_training.items():
                best_threshold = config_result['threshold']

                print(f"\nApplying {config_name} threshold ({best_threshold:.4f}) to test data...")

                # Evaluate on test data with the best threshold from training
                test_eval = self.evaluate_threshold(
                    test_results, best_threshold, clustering_time, use_extended_window
                )

                print(f"  Test sensitivity: {test_eval['sensitivity']:.4f} "
                      f"({test_eval['detected_seizures']}/{test_eval['total_seizures']} seizures)")
                print(f"  Test FAR: {test_eval['far']:.4f} events/hour")
                print(f"  Test HMS: {test_eval['challenge_score']:.4f}")

                # Add detections from test data
                for detection in test_eval['detections']:
                    all_detections.append({
                        'subject': detection['subject'],
                        'run': detection['run'],
                        'seizure_idx': detection['seizure_idx'],
                        'detected': detection['detected'],
                        'config': config_name,
                        'window_type': window_type
                    })

        # Create DataFrame
        df = pd.DataFrame(all_detections)

        # Save unified table
        output_file = output_dir / 'madrid_unified.csv'
        df.to_csv(output_file, index=False)

        print(f"\n{'='*80}")
        print(f"UNIFIED TABLE CREATED")
        print(f"{'='*80}")
        print(f"Saved to: {output_file}")
        print(f"Total records: {len(df)}")
        print(f"Configurations: {df['config'].unique().tolist()}")
        print(f"Window types: {df['window_type'].unique().tolist()}")
        print(f"Test subjects only (saturated runs excluded)")
        print(f"{'='*80}")

        return df


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Process Madrid raw results and create unified detection tables'
    )
    parser.add_argument(
        '--raw-data-dir',
        type=str,
        default='./madrid_raw_results',  # PLACEHOLDER - user will update
        help='Directory containing madrid_windowed_results_*.json files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Output directory for unified tables'
    )
    parser.add_argument(
        '--pre-seizure-window',
        type=float,
        default=5.0,
        help='Minutes before seizure for extended window (default: 5.0)'
    )
    parser.add_argument(
        '--post-seizure-window',
        type=float,
        default=3.0,
        help='Minutes after seizure for extended window (default: 3.0)'
    )
    parser.add_argument(
        '--n-thresholds',
        type=int,
        default=100,
        help='Number of thresholds to test (default: 100)'
    )

    args = parser.parse_args()

    # Check if raw data directory exists
    raw_data_dir = Path(args.raw_data_dir)
    if not raw_data_dir.exists():
        print(f"ERROR: Raw data directory not found: {raw_data_dir}")
        print(f"\nPlease update --raw-data-dir to point to your Madrid raw results")
        print(f"The directory should contain files like: madrid_windowed_results_sub-*.json")
        return

    # Initialize processor
    processor = MadridResultsProcessor(
        raw_data_dir,
        pre_seizure_window=args.pre_seizure_window,
        post_seizure_window=args.post_seizure_window
    )

    # Create unified tables
    processor.create_unified_tables(Path(args.output_dir))


if __name__ == '__main__':
    main()
