#!/usr/bin/env python3
"""
Create subject-level aggregated metrics from Madrid raw results.

Similar to process_madrid_results.py but outputs subject-level metrics
in the same format as confidence/MP/merged_mp_results.csv:
- subject
- sensitivity (aggregated per subject)
- false_alarms_per_hour (aggregated per subject)
- HMS (calculated from sensitivity and FAR)
- config (Sensitivity, FAR, HMS)
- window_type (window, no_window)

Usage:
    python create_madrid_subject_level_metrics.py --raw-data-dir /path/to/madrid/results --output madrid_subject_metrics.csv
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import pandas as pd
import numpy as np
from datetime import datetime


class MadridSubjectMetricsCreator:
    """Create subject-level metrics from Madrid raw results."""

    def __init__(self, raw_data_dir: Path,
                 pre_seizure_window: float = 5.0,
                 post_seizure_window: float = 3.0):
        """
        Initialize the creator.

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
        """Extract seizure information from validation_data."""
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
        """Cluster anomalies based on time proximity."""
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
        """Check if a seizure was detected by any cluster."""
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

    def evaluate_threshold_per_subject(self, results: List[Dict], threshold: float,
                                       clustering_time: float,
                                       use_extended_window: bool = False) -> Dict[str, Dict]:
        """
        Evaluate detection performance per subject for a specific threshold.

        Returns:
            Dictionary mapping subject_id to metrics
        """
        subject_metrics = defaultdict(lambda: {
            'total_seizures': 0,
            'detected_seizures': 0,
            'false_positives': 0,
            'total_duration': 0.0
        })

        for result in results:
            subject = result['subject']

            # Get recording duration
            duration = result['data']['input_data']['signal_metadata']['total_duration_seconds']
            subject_metrics[subject]['total_duration'] += duration

            # Extract seizures and anomalies
            seizures = self.extract_seizures(result)
            all_anomalies = self.extract_anomalies(result)

            # Filter anomalies by threshold
            filtered_anomalies = [a for a in all_anomalies if a['score'] >= threshold]

            # Cluster anomalies
            clusters = self.cluster_anomalies(filtered_anomalies, clustering_time)

            # Check each seizure
            for seizure in seizures:
                subject_metrics[subject]['total_seizures'] += 1
                detected = self.check_seizure_detection(seizure, clusters, use_extended_window)

                if detected:
                    subject_metrics[subject]['detected_seizures'] += 1

            # Count false positives (clusters not matching any seizure)
            for cluster in clusters:
                is_false_positive = True
                for seizure in seizures:
                    if self.check_seizure_detection(seizure, [cluster], use_extended_window):
                        is_false_positive = False
                        break

                if is_false_positive:
                    subject_metrics[subject]['false_positives'] += 1

        # Calculate per-subject metrics
        subject_results = {}
        for subject, data in subject_metrics.items():
            total_seizures = data['total_seizures']
            detected_seizures = data['detected_seizures']
            false_positives = data['false_positives']
            duration_hours = data['total_duration'] / 3600.0

            sensitivity = detected_seizures / total_seizures if total_seizures > 0 else 0.0
            far = false_positives / duration_hours if duration_hours > 0 else 0.0
            hms = (sensitivity * 100) - (0.4 * far)

            subject_results[subject] = {
                'sensitivity': sensitivity,
                'false_alarms_per_hour': far,
                'HMS': hms,
                'total_seizures': total_seizures,
                'detected_seizures': detected_seizures,
                'false_positives': false_positives,
                'duration_hours': duration_hours
            }

        return subject_results

    def evaluate_threshold_global(self, results: List[Dict], threshold: float,
                                  clustering_time: float,
                                  use_extended_window: bool = False) -> Dict:
        """
        Evaluate detection performance globally (file-level) for a specific threshold.
        Same as process_madrid_results.py approach.

        Returns:
            Dictionary with global performance metrics
        """
        total_seizures = 0
        detected_seizures = 0
        total_false_positives = 0
        total_duration = 0.0

        for result in results:
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

        # Calculate global metrics
        sensitivity = detected_seizures / total_seizures if total_seizures > 0 else 0.0
        total_duration_hours = total_duration / 3600.0
        far = total_false_positives / total_duration_hours if total_duration_hours > 0 else 0.0
        hms = (sensitivity * 100) - (0.4 * far)

        return {
            'threshold': threshold,
            'sensitivity': sensitivity,
            'far': far,
            'hms': hms,
            'detected_seizures': detected_seizures,
            'total_seizures': total_seizures,
            'total_false_positives': total_false_positives
        }

    def find_best_threshold(self, training_results: List[Dict], test_results: List[Dict],
                           use_extended_window: bool = False,
                           optimization_target: str = 'sensitivity',
                           n_thresholds: int = 100) -> Tuple[float, Dict[str, Dict], Dict]:
        """
        Find best threshold based on GLOBAL metrics on training data.
        Then calculate per-subject metrics on test data with that threshold.

        Args:
            training_results: Training data for threshold selection
            test_results: Test data for final evaluation
            use_extended_window: Use extended detection window
            optimization_target: 'sensitivity', 'far', or 'hms'
            n_thresholds: Number of thresholds to test

        Returns:
            Tuple of (best_threshold, test_subject_metrics_dict, global_test_metrics_dict)
        """
        # Determine clustering strategy based on window type
        clustering_time = 180.0 if use_extended_window else 600.0

        # Test thresholds from 0.0 to 1.0 on TRAINING data
        thresholds = np.linspace(0.0, 1.0, n_thresholds)

        evaluations = []
        for threshold in thresholds:
            eval_result = self.evaluate_threshold_global(
                training_results, threshold, clustering_time, use_extended_window
            )
            evaluations.append(eval_result)

        # Find best threshold based on global training metrics
        if optimization_target == 'sensitivity':
            # Maximize global sensitivity
            best_eval = max(evaluations, key=lambda x: x['sensitivity'])
        elif optimization_target == 'far':
            # Minimize global FAR (only if sensitivity > 0)
            valid_evals = [e for e in evaluations if e['sensitivity'] > 0]
            if valid_evals:
                best_eval = min(valid_evals, key=lambda x: x['far'])
            else:
                # Fallback to highest sensitivity if no valid FAR
                best_eval = max(evaluations, key=lambda x: x['sensitivity'])
        elif optimization_target == 'hms':
            # Maximize global HMS
            best_eval = max(evaluations, key=lambda x: x['hms'])
        else:
            raise ValueError(f"Invalid optimization_target: {optimization_target}")

        best_threshold = best_eval['threshold']

        # Now calculate per-subject metrics on TEST data with the best threshold
        test_subject_metrics = self.evaluate_threshold_per_subject(
            test_results, best_threshold, clustering_time, use_extended_window
        )

        # Also calculate global test metrics for reporting
        global_test_metrics = self.evaluate_threshold_global(
            test_results, best_threshold, clustering_time, use_extended_window
        )

        return best_threshold, test_subject_metrics, global_test_metrics

    def create_subject_metrics_csv(self, output_file: Path, n_thresholds: int = 100):
        """
        Create subject-level metrics CSV in the same format as merged_mp_results.csv

        Strategy:
        1. Load training data (sub-001 to sub-096)
        2. Load test data (sub-097 to sub-125, excluding saturated runs)
        3. Find best thresholds using training data
        4. Apply best thresholds to test data
        5. Output per-subject metrics for test data only

        Output columns: subject, sensitivity, false_alarms_per_hour, HMS, config, window_type
        """
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

        all_metrics = []

        # Process both window types
        for use_extended_window in [True, False]:
            window_type = "window" if use_extended_window else "no_window"
            clustering_time = 180.0 if use_extended_window else 600.0

            print(f"\n{'='*80}")
            print(f"STEP 3: FIND BEST THRESHOLDS ON TRAINING DATA ({window_type})")
            print(f"{'='*80}")

            # Find best configurations for each optimization target
            for config_name in ['Sensitivity', 'FAR', 'HMS']:
                print(f"\nFinding best threshold for {config_name} using TRAINING data...")

                best_threshold, test_subject_metrics, global_test_metrics = self.find_best_threshold(
                    training_results,
                    test_results,
                    use_extended_window=use_extended_window,
                    optimization_target=config_name.lower(),
                    n_thresholds=n_thresholds
                )

                print(f"  Best threshold (from training): {best_threshold:.4f}")
                print(f"  Test global sensitivity: {global_test_metrics['sensitivity']:.4f} "
                      f"({global_test_metrics['detected_seizures']}/{global_test_metrics['total_seizures']} seizures)")
                print(f"  Test global FAR: {global_test_metrics['far']:.4f} events/hour")
                print(f"  Test global HMS: {global_test_metrics['hms']:.4f}")
                print(f"  Test per-subject avg sensitivity: {np.mean([m['sensitivity'] for m in test_subject_metrics.values()]):.4f}")
                print(f"  Test per-subject avg FAR: {np.mean([m['false_alarms_per_hour'] for m in test_subject_metrics.values()]):.4f}")

                # Add to results (TEST subject metrics only)
                for subject, metrics in test_subject_metrics.items():
                    all_metrics.append({
                        'subject': subject,
                        'sensitivity': metrics['sensitivity'],
                        'false_alarms_per_hour': metrics['false_alarms_per_hour'],
                        'HMS': metrics['HMS'],
                        'config': config_name,
                        'window_type': window_type
                    })

        # Create DataFrame
        df = pd.DataFrame(all_metrics)

        # Sort by subject, config, window_type
        df = df.sort_values(['subject', 'config', 'window_type'])

        # Save to CSV
        df.to_csv(output_file, index=False)

        print(f"\n{'='*80}")
        print(f"SUBJECT-LEVEL METRICS CREATED")
        print(f"{'='*80}")
        print(f"Saved to: {output_file}")
        print(f"Total rows: {len(df)}")
        print(f"Unique subjects: {df['subject'].nunique()}")
        print(f"Configurations: {df['config'].unique().tolist()}")
        print(f"Window types: {df['window_type'].unique().tolist()}")
        print(f"Test subjects only (saturated runs excluded)")
        print(f"\nFormat matches: confidence/MP/merged_mp_results.csv")
        print(f"{'='*80}")

        return df


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Create subject-level metrics from Madrid raw results'
    )
    parser.add_argument(
        '--raw-data-dir',
        type=str,
        required=True,
        help='Directory containing madrid_windowed_results_*.json files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='madrid_subject_metrics.csv',
        help='Output CSV file (default: madrid_subject_metrics.csv)'
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
        print(f"\nThe directory should contain files like: madrid_windowed_results_sub-*.json")
        return

    # Initialize creator
    creator = MadridSubjectMetricsCreator(
        raw_data_dir,
        pre_seizure_window=args.pre_seizure_window,
        post_seizure_window=args.post_seizure_window
    )

    # Create subject-level metrics CSV
    output_file = Path(args.output)
    creator.create_subject_metrics_csv(output_file, n_thresholds=args.n_thresholds)


if __name__ == '__main__':
    main()
