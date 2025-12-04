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

        # Test set subjects (sub-097 to sub-125)
        self.test_subjects = [f"sub-{i:03d}" for i in range(97, 126)]

    def load_raw_results(self) -> List[Dict]:
        """Load all raw Madrid JSON result files."""
        json_files = list(self.raw_data_dir.glob("madrid_windowed_results_*.json"))

        print(f"Found {len(json_files)} raw result files")

        results = []
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                # Extract subject and run from input_data
                subject_id = data['input_data']['subject_id']
                run_id = data['input_data']['run_id']

                # Only process test set
                if subject_id not in self.test_subjects:
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

        print(f"Loaded {len(results)} test set files")
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

    def find_best_threshold(self, results: List[Dict],
                           use_extended_window: bool = False,
                           optimization_target: str = 'sensitivity',
                           n_thresholds: int = 100) -> Tuple[float, Dict[str, Dict]]:
        """
        Find best threshold for a given optimization target.

        Args:
            results: List of raw result dictionaries
            use_extended_window: Use extended detection window
            optimization_target: 'sensitivity', 'far', or 'hms'
            n_thresholds: Number of thresholds to test

        Returns:
            Tuple of (best_threshold, subject_metrics_dict)
        """
        # Determine clustering strategy based on window type
        clustering_time = 180.0 if use_extended_window else 600.0

        # Test thresholds from 0.0 to 1.0
        thresholds = np.linspace(0.0, 1.0, n_thresholds)

        best_threshold = None
        best_score = -float('inf') if optimization_target != 'far' else float('inf')
        best_subject_metrics = None

        for threshold in thresholds:
            subject_metrics = self.evaluate_threshold_per_subject(
                results, threshold, clustering_time, use_extended_window
            )

            # Calculate aggregate metric for optimization
            if optimization_target == 'sensitivity':
                # Maximize average sensitivity
                avg_sensitivity = np.mean([m['sensitivity'] for m in subject_metrics.values()])
                score = avg_sensitivity
                is_better = score > best_score

            elif optimization_target == 'far':
                # Minimize average FAR (only if sensitivity > 0)
                subjects_with_detection = [m for m in subject_metrics.values() if m['sensitivity'] > 0]
                if subjects_with_detection:
                    avg_far = np.mean([m['false_alarms_per_hour'] for m in subjects_with_detection])
                    score = avg_far
                    is_better = score < best_score
                else:
                    is_better = False

            elif optimization_target == 'hms':
                # Maximize average HMS
                avg_hms = np.mean([m['HMS'] for m in subject_metrics.values()])
                score = avg_hms
                is_better = score > best_score

            else:
                raise ValueError(f"Invalid optimization_target: {optimization_target}")

            if is_better:
                best_score = score
                best_threshold = threshold
                best_subject_metrics = subject_metrics

        return best_threshold, best_subject_metrics

    def create_subject_metrics_csv(self, output_file: Path, n_thresholds: int = 100):
        """
        Create subject-level metrics CSV in the same format as merged_mp_results.csv

        Output columns: subject, sensitivity, false_alarms_per_hour, HMS, config, window_type
        """
        # Load raw results
        results = self.load_raw_results()

        if not results:
            print("No raw results found!")
            return

        all_metrics = []

        # Process both window types
        for use_extended_window in [True, False]:
            window_type = "window" if use_extended_window else "no_window"
            clustering_time = 180.0 if use_extended_window else 600.0

            print(f"\n{'='*80}")
            print(f"Processing {window_type} (clustering: {clustering_time}s)")
            print(f"{'='*80}")

            # Find best configurations for each optimization target
            for config_name in ['Sensitivity', 'FAR', 'HMS']:
                print(f"\nFinding best threshold for {config_name}...")

                best_threshold, subject_metrics = self.find_best_threshold(
                    results,
                    use_extended_window=use_extended_window,
                    optimization_target=config_name.lower(),
                    n_thresholds=n_thresholds
                )

                print(f"  Best threshold: {best_threshold:.4f}")
                print(f"  Avg sensitivity: {np.mean([m['sensitivity'] for m in subject_metrics.values()]):.4f}")
                print(f"  Avg FAR: {np.mean([m['false_alarms_per_hour'] for m in subject_metrics.values()]):.4f}")
                print(f"  Avg HMS: {np.mean([m['HMS'] for m in subject_metrics.values()]):.4f}")

                # Add to results
                for subject, metrics in subject_metrics.items():
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
