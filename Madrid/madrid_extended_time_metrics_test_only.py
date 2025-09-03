#!/usr/bin/env python3
"""
Madrid Extended Time Window Metrics Calculator - TEST SET ONLY
Calculates sensitivity and false alarms per hour at file level with extended detection window.
A seizure is considered detected if anomalies occur within 5 minutes before or 3 minutes after the seizure.
Only evaluates on test set: sub097-sub125.
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import argparse


class MadridExtendedTimeMetricsTestOnly:
    def __init__(self, results_dir: str, output_dir: str = None, threshold: float = None, 
                 pre_seizure_minutes: float = 5.0, post_seizure_minutes: float = 3.0):
        """
        Initialize the extended time window metrics calculator for test set only.
        
        Args:
            results_dir: Directory containing Madrid windowed results JSON files
            output_dir: Directory to save metrics results (default: same as results_dir)
            threshold: Anomaly score threshold for detection (if None, uses top-ranked anomalies)
            pre_seizure_minutes: Minutes before seizure start to consider as detection window
            post_seizure_minutes: Minutes after seizure end to consider as detection window
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir) if output_dir else self.results_dir / "test_only_extended_metrics"
        self.threshold = threshold
        self.pre_seizure_seconds = pre_seizure_minutes * 60.0  # Convert to seconds
        self.post_seizure_seconds = post_seizure_minutes * 60.0  # Convert to seconds
        self.output_dir.mkdir(exist_ok=True)
    
    def is_test_file(self, filename: str) -> bool:
        """
        Determine if a file belongs to the test set (sub097-sub125).
        
        Args:
            filename: Name of the file (e.g., "madrid_windowed_results_sub-077_run-04_20250730_040717.json")
        
        Returns:
            True if file is in test set, False otherwise
        """
        # Extract subject ID from filename
        match = re.search(r'sub-(\d{3})', filename)
        if match:
            subject_num = int(match.group(1))
            return 97 <= subject_num <= 125
        return False
        
    def load_result_file(self, filepath: Path) -> Dict[str, Any]:
        """Load and parse a Madrid results JSON file."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def extract_anomalies_for_window(self, window_data: Dict[str, Any], 
                                   use_threshold: bool = False) -> List[Dict[str, Any]]:
        """
        Extract anomalies from a window based on threshold or top-ranked.
        
        Args:
            window_data: Window result data
            use_threshold: If True, use threshold; if False, use top-ranked anomaly
        """
        anomalies = window_data.get('anomalies', [])
        
        if not anomalies:
            return []
        
        if use_threshold or self.threshold is not None:
            # Filter by threshold
            return [a for a in anomalies if a.get('anomaly_score', 0) >= self.threshold]
        else:
            # Use top-ranked anomaly only
            return [anomalies[0]] if anomalies else []
    
    def group_seizures_by_time(self, seizure_windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Group seizure windows into individual seizures based on start_time_absolute and end_time_absolute.
        Windows with identical time intervals belong to the same seizure.
        """
        if not seizure_windows:
            return []
        
        # Collect all seizure segments from all windows
        all_segments = []
        for window in seizure_windows:
            window_index = window.get('window_index')
            seizure_segments = window.get('seizure_segments', [])
            
            for segment in seizure_segments:
                segment_with_window = {
                    **segment,
                    'window_index': window_index,
                    'window': window
                }
                all_segments.append(segment_with_window)
        
        # Group segments by their absolute time intervals
        seizure_groups = {}
        for segment in all_segments:
            # Create a key based on start and end time (rounded to handle floating point precision)
            start_time = round(segment.get('start_time_absolute', 0), 1)
            end_time = round(segment.get('end_time_absolute', 0), 1)
            time_key = (start_time, end_time)
            
            if time_key not in seizure_groups:
                seizure_groups[time_key] = {
                    'start_time_absolute': start_time,
                    'end_time_absolute': end_time,
                    'duration_seconds': end_time - start_time,
                    'windows': []
                }
            
            seizure_groups[time_key]['windows'].append(segment['window'])
        
        # Convert to list and sort by start time
        seizures = list(seizure_groups.values())
        seizures.sort(key=lambda x: x['start_time_absolute'])
        
        # Add seizure IDs, extended time windows, and remove duplicates from windows
        for i, seizure in enumerate(seizures):
            seizure['seizure_id'] = f"seizure_{i+1}"
            # Remove duplicate windows (same seizure might span multiple overlapping windows)
            unique_windows = {w.get('window_index'): w for w in seizure['windows']}.values()
            seizure['windows'] = list(unique_windows)
            seizure['num_windows'] = len(seizure['windows'])
            
            # Add extended time window for detection
            seizure['extended_start_time'] = seizure['start_time_absolute'] - self.pre_seizure_seconds
            seizure['extended_end_time'] = seizure['end_time_absolute'] + self.post_seizure_seconds
            seizure['extended_duration_seconds'] = seizure['extended_end_time'] - seizure['extended_start_time']
        
        return seizures
    
    def calculate_file_level_metrics(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate file-level metrics by aggregating all window results."""
        
        # Extract basic info
        metadata = result_data.get('analysis_metadata', {})
        input_data = result_data.get('input_data', {})
        validation_data = result_data.get('validation_data', {})
        analysis_results = result_data.get('analysis_results', {})
        window_results = analysis_results.get('window_results', [])
        
        subject_id = input_data.get('subject_id', 'unknown')
        run_id = input_data.get('run_id', 'unknown')
        
        # Get ground truth and group seizures
        ground_truth = validation_data.get('ground_truth', {})
        seizure_present = ground_truth.get('seizure_present', False)
        seizure_windows = ground_truth.get('seizure_windows', [])
        seizure_window_indices = set(sw.get('window_index', -1) for sw in seizure_windows)
        
        # Group seizure windows into individual seizures
        individual_seizures = self.group_seizures_by_time(seizure_windows)
        num_seizures = len(individual_seizures)
        
        # Get signal info
        signal_metadata = input_data.get('signal_metadata', {})
        total_duration_hours = signal_metadata.get('total_duration_seconds', 0) / 3600.0
        window_duration_seconds = signal_metadata.get('window_duration_seconds', 3600.0)
        
        # Aggregate all detections across windows
        all_detections = []
        seizure_detections = []
        non_seizure_detections = []
        
        for window in window_results:
            window_index = window.get('window_index')
            detected_anomalies = self.extract_anomalies_for_window(window)
            
            for anomaly in detected_anomalies:
                # Add window context to anomaly
                anomaly_with_context = {
                    **anomaly,
                    'window_index': window_index,
                    'window_start_time': window.get('window_start_time', 0),
                    'is_in_seizure_window': window_index in seizure_window_indices
                }
                
                all_detections.append(anomaly_with_context)
                
                if window_index in seizure_window_indices:
                    seizure_detections.append(anomaly_with_context)
                else:
                    non_seizure_detections.append(anomaly_with_context)
        
        # Calculate file-level metrics
        total_detections = len(all_detections)
        true_positive_detections = len(seizure_detections)
        false_positive_detections = len(non_seizure_detections)
        
        # Calculate which seizures were detected using extended time-based overlap
        detected_seizures = []
        for seizure in individual_seizures:
            seizure_start = seizure['start_time_absolute']
            seizure_end = seizure['end_time_absolute']
            extended_start = seizure['extended_start_time']
            extended_end = seizure['extended_end_time']
            
            # Check if any detection temporally overlaps with this seizure's extended window
            seizure_detected = False
            seizure_detections_count = 0
            overlapping_detections = []
            
            for detection in all_detections:
                # Calculate absolute time of the anomaly
                window_start_time = detection.get('window_start_time', 0)
                location_time_in_window = detection.get('location_time_in_window', 0)
                anomaly_absolute_time = window_start_time + location_time_in_window
                
                # Check if anomaly time overlaps with extended seizure time period
                if extended_start <= anomaly_absolute_time <= extended_end:
                    seizure_detected = True
                    seizure_detections_count += 1
                    
                    # Determine detection type
                    if anomaly_absolute_time < seizure_start:
                        detection_type = "pre_seizure"
                        time_offset = seizure_start - anomaly_absolute_time
                    elif anomaly_absolute_time > seizure_end:
                        detection_type = "post_seizure"
                        time_offset = anomaly_absolute_time - seizure_end
                    else:
                        detection_type = "during_seizure"
                        time_offset = anomaly_absolute_time - seizure_start
                    
                    overlapping_detections.append({
                        **detection,
                        'anomaly_absolute_time': anomaly_absolute_time,
                        'detection_type': detection_type,
                        'time_offset_seconds': time_offset,
                        'time_from_seizure_start': anomaly_absolute_time - seizure_start
                    })
            
            if seizure_detected:
                detected_seizures.append({
                    'seizure_id': seizure['seizure_id'],
                    'start_time': seizure_start,
                    'end_time': seizure_end,
                    'extended_start_time': extended_start,
                    'extended_end_time': extended_end,
                    'detection_count': seizure_detections_count,
                    'overlapping_detections': overlapping_detections
                })
        
        # File-level sensitivity: detected seizures / total seizures
        if num_seizures > 0:
            file_level_sensitivity = len(detected_seizures) / num_seizures
        else:
            # No seizures present, sensitivity is not applicable
            file_level_sensitivity = None
        
        # Recalculate true/false positives based on extended time-based overlap
        # True positives: detections that temporally overlap with any extended seizure window
        # False positives: detections that don't temporally overlap with any extended seizure window
        extended_time_based_true_positives = sum(len(ds['overlapping_detections']) for ds in detected_seizures)
        extended_time_based_false_positives = total_detections - extended_time_based_true_positives
        
        # False alarms per hour (file level)
        false_alarms_per_hour = extended_time_based_false_positives / total_duration_hours if total_duration_hours > 0 else 0.0
        
        # Additional metrics (updated for extended time-based logic)
        seizure_detection_density = extended_time_based_true_positives / len(seizure_window_indices) if seizure_window_indices else 0.0
        detection_precision = extended_time_based_true_positives / total_detections if total_detections > 0 else 0.0
        
        return {
            'file_info': {
                'subject_id': subject_id,
                'run_id': run_id,
                'analysis_id': metadata.get('analysis_id', 'unknown'),
                'timestamp': metadata.get('timestamp', 'unknown')
            },
            'extended_time_settings': {
                'pre_seizure_minutes': self.pre_seizure_seconds / 60.0,
                'post_seizure_minutes': self.post_seizure_seconds / 60.0,
                'pre_seizure_seconds': self.pre_seizure_seconds,
                'post_seizure_seconds': self.post_seizure_seconds
            },
            'file_summary': {
                'total_duration_hours': round(total_duration_hours, 2),
                'total_windows': len(window_results),
                'seizure_present': seizure_present,
                'num_seizures': num_seizures,
                'num_seizure_windows': len(seizure_window_indices),
                'individual_seizures': individual_seizures
            },
            'file_level_detections': {
                'total_detections': total_detections,
                'extended_time_true_positives': extended_time_based_true_positives,
                'extended_time_false_positives': extended_time_based_false_positives,
                'exact_time_true_positives': 0,  # Will be calculated for comparison if needed
                'exact_time_false_positives': 0,  # Will be calculated for comparison if needed
                'window_based_true_positives': true_positive_detections,  # Keep for comparison
                'window_based_false_positives': false_positive_detections,  # Keep for comparison
                'seizure_detection_density': round(seizure_detection_density, 4),
                'detection_precision': round(detection_precision, 4)
            },
            'file_level_metrics': {
                'sensitivity': file_level_sensitivity,
                'false_alarms_per_hour': round(false_alarms_per_hour, 4)
            },
            'detection_details': {
                'detected_seizures': detected_seizures,
                'seizure_detections': seizure_detections,
                'non_seizure_detections': non_seizure_detections[:10]  # Limit to first 10 for readability
            }
        }
    
    def process_all_files(self) -> Dict[str, Any]:
        """Process all Madrid result files in the directory - TEST SET ONLY."""
        all_json_files = list(self.results_dir.glob("madrid_windowed_results_*.json"))
        
        # Filter for test set files only
        json_files = [f for f in all_json_files if self.is_test_file(f.name)]
        
        if not json_files:
            print(f"No test set Madrid result files found in {self.results_dir}")
            return None
        
        print(f"Found {len(all_json_files)} total Madrid result files")
        print(f"Processing {len(json_files)} test set files (sub097-sub125)")
        
        all_results = {}
        summary_stats = {
            'total_files_processed': 0,
            'files_with_seizures': 0,
            'files_without_seizures': 0,
            'files_with_detections': 0,
            'total_duration_hours': 0.0,
            'total_detections': 0,
            'total_seizure_detections': 0,
            'total_false_alarms': 0,
            'skipped_files': len(all_json_files) - len(json_files)
        }
        
        for json_file in sorted(json_files):
            print(f"Processing {json_file.name}...")
            
            result_data = self.load_result_file(json_file)
            if result_data is None:
                continue
            
            file_metrics = self.calculate_file_level_metrics(result_data)
            all_results[json_file.name] = file_metrics
            
            # Update summary stats
            summary_stats['total_files_processed'] += 1
            summary_stats['total_duration_hours'] += file_metrics['file_summary']['total_duration_hours']
            summary_stats['total_detections'] += file_metrics['file_level_detections']['total_detections']
            summary_stats['total_seizure_detections'] += file_metrics['file_level_detections']['extended_time_true_positives']
            summary_stats['total_false_alarms'] += file_metrics['file_level_detections']['extended_time_false_positives']
            
            if file_metrics['file_summary']['seizure_present']:
                summary_stats['files_with_seizures'] += 1
            else:
                summary_stats['files_without_seizures'] += 1
            
            if file_metrics['file_level_detections']['total_detections'] > 0:
                summary_stats['files_with_detections'] += 1
        
        # Calculate overall file-level performance
        total_seizures = sum(r['file_summary']['num_seizures'] for r in all_results.values())
        total_detected_seizures = sum(len(r['detection_details']['detected_seizures']) for r in all_results.values())
        
        seizure_files_detected = sum(1 for r in all_results.values() 
                                   if r['file_level_metrics']['sensitivity'] == 1.0)
        
        # Calculate both file-level and seizure-level sensitivity
        file_level_sensitivity = seizure_files_detected / summary_stats['files_with_seizures'] if summary_stats['files_with_seizures'] > 0 else 0.0
        seizure_level_sensitivity = total_detected_seizures / total_seizures if total_seizures > 0 else 0.0
        overall_false_alarms_per_hour = summary_stats['total_false_alarms'] / summary_stats['total_duration_hours'] if summary_stats['total_duration_hours'] > 0 else 0.0
        
        return {
            'analysis_metadata': {
                'calculation_timestamp': datetime.now().isoformat(),
                'results_directory': str(self.results_dir),
                'threshold_used': self.threshold,
                'detection_strategy': 'threshold' if self.threshold else 'top_ranked',
                'metric_level': 'file_level_extended_time',
                'dataset': 'TEST_SET_ONLY (sub097-sub125)',
                'pre_seizure_minutes': self.pre_seizure_seconds / 60.0,
                'post_seizure_minutes': self.post_seizure_seconds / 60.0
            },
            'summary_statistics': {
                **summary_stats,
                'total_seizures': total_seizures,
                'total_detected_seizures': total_detected_seizures,
                'file_level_sensitivity': round(file_level_sensitivity, 4),
                'seizure_level_sensitivity': round(seizure_level_sensitivity, 4),
                'overall_false_alarms_per_hour': round(overall_false_alarms_per_hour, 4),
                'seizure_files_detected': seizure_files_detected,
                'detection_rate': round(summary_stats['files_with_detections'] / summary_stats['total_files_processed'], 4) if summary_stats['total_files_processed'] > 0 else 0.0
            },
            'individual_results': all_results
        }
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save results in both JSON and human-readable formats."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"madrid_extended_time_metrics_test_only_{timestamp}"
        
        # Save JSON version
        json_path = self.output_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to: {json_path}")
        
        # Save human-readable version
        txt_path = self.output_dir / f"{filename}.txt"
        self.save_human_readable_report(results, txt_path)
        print(f"Human-readable report saved to: {txt_path}")
    
    def save_human_readable_report(self, results: Dict[str, Any], filepath: Path):
        """Save results in human-readable format."""
        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MADRID EXTENDED TIME WINDOW SEIZURE DETECTION METRICS REPORT - TEST SET ONLY\n")
            f.write("=" * 80 + "\n\n")
            
            # Analysis metadata
            metadata = results['analysis_metadata']
            f.write("ANALYSIS INFORMATION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Calculation Time: {metadata['calculation_timestamp']}\n")
            f.write(f"Results Directory: {metadata['results_directory']}\n")
            f.write(f"Dataset: {metadata['dataset']}\n")
            f.write(f"Detection Strategy: {metadata['detection_strategy']}\n")
            f.write(f"Metric Level: {metadata['metric_level']}\n")
            f.write(f"Extended Time Window: -{metadata['pre_seizure_minutes']:.1f} min to +{metadata['post_seizure_minutes']:.1f} min\n")
            if metadata['threshold_used']:
                f.write(f"Threshold Used: {metadata['threshold_used']}\n")
            f.write("\n")
            
            # Summary statistics
            summary = results['summary_statistics']
            f.write("SUMMARY STATISTICS (TEST SET ONLY):\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Test Files Processed: {summary['total_files_processed']}\n")
            f.write(f"Files Skipped (Training Set): {summary['skipped_files']}\n")
            f.write(f"Files with Seizures: {summary['files_with_seizures']}\n")
            f.write(f"Files without Seizures: {summary['files_without_seizures']}\n")
            f.write(f"Files with Any Detections: {summary['files_with_detections']}\n")
            f.write(f"Total Recording Duration: {summary['total_duration_hours']:.2f} hours\n")
            f.write(f"Total Detections: {summary['total_detections']}\n")
            f.write(f"Total Seizure Detections (Extended Window): {summary['total_seizure_detections']}\n")
            f.write(f"Total False Alarms (Extended Window): {summary['total_false_alarms']}\n")
            f.write("\n")
            
            f.write("OVERALL TEST SET PERFORMANCE (Extended Time Window):\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Seizures Across Test Files: {summary['total_seizures']}\n")
            f.write(f"Total Detected Seizures: {summary['total_detected_seizures']}\n")
            f.write(f"File-Level Sensitivity: {summary['file_level_sensitivity']:.4f} ({summary['file_level_sensitivity']*100:.2f}%)\n")
            f.write(f"  (Files with All Seizures Detected: {summary['seizure_files_detected']}/{summary['files_with_seizures']})\n")
            f.write(f"Seizure-Level Sensitivity: {summary['seizure_level_sensitivity']:.4f} ({summary['seizure_level_sensitivity']*100:.2f}%)\n")
            f.write(f"  (Individual Seizures Detected: {summary['total_detected_seizures']}/{summary['total_seizures']})\n")
            f.write(f"Overall False Alarms per Hour: {summary['overall_false_alarms_per_hour']:.4f}\n")
            f.write(f"Detection Rate: {summary['detection_rate']:.4f} ({summary['detection_rate']*100:.2f}%)\n")
            f.write(f"  (Files with Any Detections: {summary['files_with_detections']}/{summary['total_files_processed']})\n")
            f.write("\n")
            
            # Individual results
            f.write("INDIVIDUAL TEST FILE RESULTS:\n")
            f.write("=" * 80 + "\n")
            
            for filename, file_results in sorted(results['individual_results'].items()):
                f.write(f"\nFile: {filename}\n")
                f.write("-" * len(filename) + "-----\n")
                
                info = file_results['file_info']
                summary_info = file_results['file_summary']
                detections = file_results['file_level_detections']
                metrics = file_results['file_level_metrics']
                time_settings = file_results['extended_time_settings']
                
                f.write(f"Subject: {info['subject_id']}, Run: {info['run_id']}\n")
                f.write(f"Duration: {summary_info['total_duration_hours']:.2f} hours ({summary_info['total_windows']} windows)\n")
                f.write(f"Extended Time Window: -{time_settings['pre_seizure_minutes']:.1f} min to +{time_settings['post_seizure_minutes']:.1f} min\n")
                f.write(f"Seizure Present: {'Yes' if summary_info['seizure_present'] else 'No'}")
                if summary_info['seizure_present']:
                    f.write(f" ({summary_info['num_seizures']} seizures, {summary_info['num_seizure_windows']} seizure windows)")
                f.write("\n")
                
                # Show individual seizures with extended windows
                if summary_info['num_seizures'] > 0:
                    f.write(f"Individual Seizures (with Extended Windows):\n")
                    for seizure in summary_info['individual_seizures']:
                        f.write(f"  {seizure['seizure_id']}: {seizure['start_time_absolute']:.1f}s - {seizure['end_time_absolute']:.1f}s ({seizure['duration_seconds']:.1f}s)\n")
                        f.write(f"    Extended Window: {seizure['extended_start_time']:.1f}s - {seizure['extended_end_time']:.1f}s ({seizure['extended_duration_seconds']:.1f}s)\n")
                
                f.write(f"\nFILE-LEVEL DETECTIONS (Extended Time-Based Overlap):\n")
                f.write(f"  Total Detections: {detections['total_detections']}\n")
                f.write(f"  Extended Time True Positives: {detections['extended_time_true_positives']}\n")
                f.write(f"  Extended Time False Positives: {detections['extended_time_false_positives']}\n")
                f.write(f"  Window-Based TP (for comparison): {detections['window_based_true_positives']}\n")
                f.write(f"  Window-Based FP (for comparison): {detections['window_based_false_positives']}\n")
                f.write(f"  Seizure Detection Density: {detections['seizure_detection_density']:.4f}\n")
                f.write(f"  Detection Precision: {detections['detection_precision']:.4f}\n")
                
                f.write(f"\nFILE-LEVEL METRICS:\n")
                if metrics['sensitivity'] is not None:
                    f.write(f"  Sensitivity: {metrics['sensitivity']:.4f} ({metrics['sensitivity']*100:.1f}%)\n")
                    detected_seizures = file_results['detection_details']['detected_seizures']
                    if detected_seizures:
                        f.write(f"  Detected Seizures: {len(detected_seizures)}/{summary_info['num_seizures']}\n")
                        for det_seizure in detected_seizures:
                            f.write(f"    {det_seizure['seizure_id']}: {det_seizure['detection_count']} extended-time detections\n")
                            # Show first few overlapping detections with timing and type
                            overlapping = det_seizure['overlapping_detections'][:3]  # Limit to first 3
                            for overlap in overlapping:
                                abs_time = overlap['anomaly_absolute_time']
                                detection_type = overlap['detection_type']
                                offset = overlap['time_offset_seconds']
                                if detection_type == "pre_seizure":
                                    f.write(f"      Detection at {abs_time:.1f}s ({offset:.1f}s BEFORE seizure start)\n")
                                elif detection_type == "post_seizure":
                                    f.write(f"      Detection at {abs_time:.1f}s ({offset:.1f}s AFTER seizure end)\n")
                                else:
                                    f.write(f"      Detection at {abs_time:.1f}s (DURING seizure, {offset:.1f}s from start)\n")
                            if len(det_seizure['overlapping_detections']) > 3:
                                f.write(f"      ... and {len(det_seizure['overlapping_detections'])-3} more\n")
                    else:
                        f.write(f"  No seizures detected (no extended temporal overlap)\n")
                else:
                    f.write(f"  Sensitivity: N/A (no seizure present)\n")
                f.write(f"  False Alarms/Hour: {metrics['false_alarms_per_hour']:.4f}\n")
                
                f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate file-level sensitivity with extended time windows - TEST SET ONLY (sub097-sub125)"
    )
    parser.add_argument(
        "results_dir", 
        help="Directory containing Madrid windowed results JSON files"
    )
    parser.add_argument(
        "-o", "--output-dir", 
        help="Output directory for metrics results (default: results_dir/test_only_extended_metrics)"
    )
    parser.add_argument(
        "-t", "--threshold", 
        type=float,
        help="Anomaly score threshold for detection (default: use top-ranked anomaly per window)"
    )
    parser.add_argument(
        "--pre-seizure-minutes", 
        type=float,
        default=5.0,
        help="Minutes before seizure start to consider as detection window (default: 5.0)"
    )
    parser.add_argument(
        "--post-seizure-minutes", 
        type=float,
        default=3.0,
        help="Minutes after seizure end to consider as detection window (default: 3.0)"
    )
    parser.add_argument(
        "--filename", 
        help="Output filename prefix (default: auto-generated with timestamp)"
    )
    
    args = parser.parse_args()
    
    # Initialize calculator
    calculator = MadridExtendedTimeMetricsTestOnly(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
        pre_seizure_minutes=args.pre_seizure_minutes,
        post_seizure_minutes=args.post_seizure_minutes
    )
    
    # Process all files
    results = calculator.process_all_files()
    
    if results is None:
        print("No results to process.")
        return
    
    # Save results
    calculator.save_results(results, args.filename)
    
    # Print summary to console
    summary = results['summary_statistics']
    print(f"\n{'='*50}")
    print("EXTENDED TIME WINDOW SUMMARY - TEST SET ONLY:")
    print(f"{'='*50}")
    print(f"Dataset: TEST SET (sub097-sub125)")
    print(f"Extended Window: -{args.pre_seizure_minutes:.1f} min to +{args.post_seizure_minutes:.1f} min")
    print(f"Test files processed: {summary['total_files_processed']}")
    print(f"Training files skipped: {summary['skipped_files']}")
    print(f"Total duration: {summary['total_duration_hours']:.2f} hours")
    print(f"Total seizures: {summary['total_seizures']}")
    print(f"File-Level Sensitivity: {summary['file_level_sensitivity']:.4f} ({summary['file_level_sensitivity']*100:.2f}%)")
    print(f"  Files with all seizures detected: {summary['seizure_files_detected']}/{summary['files_with_seizures']}")
    print(f"Seizure-Level Sensitivity: {summary['seizure_level_sensitivity']:.4f} ({summary['seizure_level_sensitivity']*100:.2f}%)")
    print(f"  Individual seizures detected: {summary['total_detected_seizures']}/{summary['total_seizures']}")
    print(f"Overall False Alarms/Hour: {summary['overall_false_alarms_per_hour']:.4f}")


if __name__ == "__main__":
    main()