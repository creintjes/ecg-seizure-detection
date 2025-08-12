#!/usr/bin/env python3
"""
Madrid File-Level Metrics Calculator
Calculates sensitivity and false alarms per hour at file level (not window level).
Merges all window information into aggregated file-level metrics.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import argparse


class MadridFileLevelMetrics:
    def __init__(self, results_dir: str, output_dir: str = None, threshold: float = None):
        """
        Initialize the file-level metrics calculator.
        
        Args:
            results_dir: Directory containing Madrid windowed results JSON files
            output_dir: Directory to save metrics results (default: same as results_dir)
            threshold: Anomaly score threshold for detection (if None, uses top-ranked anomalies)
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir) if output_dir else self.results_dir
        self.threshold = threshold
        self.output_dir.mkdir(exist_ok=True)
        
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
        
        # Add seizure IDs and remove duplicates from windows
        for i, seizure in enumerate(seizures):
            seizure['seizure_id'] = f"seizure_{i+1}"
            # Remove duplicate windows (same seizure might span multiple overlapping windows)
            unique_windows = {w.get('window_index'): w for w in seizure['windows']}.values()
            seizure['windows'] = list(unique_windows)
            seizure['num_windows'] = len(seizure['windows'])
        
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
        
        # Calculate which seizures were detected
        detected_seizures = []
        for seizure in individual_seizures:
            seizure_window_indices_for_this_seizure = set(w.get('window_index') for w in seizure['windows'])
            
            # Check if any detection occurred in windows belonging to this seizure
            seizure_detected = False
            seizure_detections_count = 0
            
            for detection in seizure_detections:
                if detection['window_index'] in seizure_window_indices_for_this_seizure:
                    seizure_detected = True
                    seizure_detections_count += 1
            
            if seizure_detected:
                detected_seizures.append({
                    'seizure_id': seizure['seizure_id'],
                    'start_time': seizure['start_time_absolute'],
                    'end_time': seizure['end_time_absolute'],
                    'detection_count': seizure_detections_count
                })
        
        # File-level sensitivity: detected seizures / total seizures
        if num_seizures > 0:
            file_level_sensitivity = len(detected_seizures) / num_seizures
        else:
            # No seizures present, sensitivity is not applicable
            file_level_sensitivity = None
        
        # False alarms per hour (file level)
        false_alarms_per_hour = false_positive_detections / total_duration_hours if total_duration_hours > 0 else 0.0
        
        # Additional metrics
        seizure_detection_density = true_positive_detections / len(seizure_window_indices) if seizure_window_indices else 0.0
        detection_precision = true_positive_detections / total_detections if total_detections > 0 else 0.0
        
        return {
            'file_info': {
                'subject_id': subject_id,
                'run_id': run_id,
                'analysis_id': metadata.get('analysis_id', 'unknown'),
                'timestamp': metadata.get('timestamp', 'unknown')
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
                'true_positive_detections': true_positive_detections,
                'false_positive_detections': false_positive_detections,
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
        """Process all Madrid result files in the directory."""
        json_files = list(self.results_dir.glob("madrid_windowed_results_*.json"))
        
        if not json_files:
            print(f"No Madrid result files found in {self.results_dir}")
            return None
        
        print(f"Found {len(json_files)} Madrid result files")
        
        all_results = {}
        summary_stats = {
            'total_files_processed': 0,
            'files_with_seizures': 0,
            'files_without_seizures': 0,
            'files_with_detections': 0,
            'total_duration_hours': 0.0,
            'total_detections': 0,
            'total_seizure_detections': 0,
            'total_false_alarms': 0
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
            summary_stats['total_seizure_detections'] += file_metrics['file_level_detections']['true_positive_detections']
            summary_stats['total_false_alarms'] += file_metrics['file_level_detections']['false_positive_detections']
            
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
                'metric_level': 'file_level'
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
            filename = f"madrid_file_level_metrics_{timestamp}"
        
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
            f.write("MADRID FILE-LEVEL SEIZURE DETECTION METRICS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Analysis metadata
            metadata = results['analysis_metadata']
            f.write("ANALYSIS INFORMATION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Calculation Time: {metadata['calculation_timestamp']}\n")
            f.write(f"Results Directory: {metadata['results_directory']}\n")
            f.write(f"Detection Strategy: {metadata['detection_strategy']}\n")
            f.write(f"Metric Level: {metadata['metric_level']}\n")
            if metadata['threshold_used']:
                f.write(f"Threshold Used: {metadata['threshold_used']}\n")
            f.write("\n")
            
            # Summary statistics
            summary = results['summary_statistics']
            f.write("SUMMARY STATISTICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Files Processed: {summary['total_files_processed']}\n")
            f.write(f"Files with Seizures: {summary['files_with_seizures']}\n")
            f.write(f"Files without Seizures: {summary['files_without_seizures']}\n")
            f.write(f"Files with Any Detections: {summary['files_with_detections']}\n")
            f.write(f"Total Recording Duration: {summary['total_duration_hours']:.2f} hours\n")
            f.write(f"Total Detections: {summary['total_detections']}\n")
            f.write(f"Total Seizure Detections: {summary['total_seizure_detections']}\n")
            f.write(f"Total False Alarms: {summary['total_false_alarms']}\n")
            f.write("\n")
            
            f.write("OVERALL PERFORMANCE:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Seizures Across All Files: {summary['total_seizures']}\n")
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
            f.write("INDIVIDUAL FILE RESULTS:\n")
            f.write("=" * 80 + "\n")
            
            for filename, file_results in results['individual_results'].items():
                f.write(f"\nFile: {filename}\n")
                f.write("-" * len(filename) + "-----\n")
                
                info = file_results['file_info']
                summary_info = file_results['file_summary']
                detections = file_results['file_level_detections']
                metrics = file_results['file_level_metrics']
                
                f.write(f"Subject: {info['subject_id']}, Run: {info['run_id']}\n")
                f.write(f"Duration: {summary_info['total_duration_hours']:.2f} hours ({summary_info['total_windows']} windows)\n")
                f.write(f"Seizure Present: {'Yes' if summary_info['seizure_present'] else 'No'}")
                if summary_info['seizure_present']:
                    f.write(f" ({summary_info['num_seizures']} seizures, {summary_info['num_seizure_windows']} seizure windows)")
                f.write("\n")
                
                # Show individual seizures
                if summary_info['num_seizures'] > 0:
                    f.write(f"Individual Seizures:\n")
                    for seizure in summary_info['individual_seizures']:
                        f.write(f"  {seizure['seizure_id']}: {seizure['start_time_absolute']:.1f}s - {seizure['end_time_absolute']:.1f}s ({seizure['duration_seconds']:.1f}s, {seizure['num_windows']} windows)\n")
                
                f.write(f"\nFILE-LEVEL DETECTIONS:\n")
                f.write(f"  Total Detections: {detections['total_detections']}\n")
                f.write(f"  Seizure Detections: {detections['true_positive_detections']}\n")
                f.write(f"  False Alarms: {detections['false_positive_detections']}\n")
                f.write(f"  Seizure Detection Density: {detections['seizure_detection_density']:.4f}\n")
                f.write(f"  Detection Precision: {detections['detection_precision']:.4f}\n")
                
                f.write(f"\nFILE-LEVEL METRICS:\n")
                if metrics['sensitivity'] is not None:
                    f.write(f"  Sensitivity: {metrics['sensitivity']:.4f} ({metrics['sensitivity']*100:.1f}%)\n")
                    detected_seizures = file_results['detection_details']['detected_seizures']
                    if detected_seizures:
                        f.write(f"  Detected Seizures: {len(detected_seizures)}/{summary_info['num_seizures']}\n")
                        for det_seizure in detected_seizures:
                            f.write(f"    {det_seizure['seizure_id']}: {det_seizure['detection_count']} detections\n")
                    else:
                        f.write(f"  No seizures detected\n")
                else:
                    f.write(f"  Sensitivity: N/A (no seizure present)\n")
                f.write(f"  False Alarms/Hour: {metrics['false_alarms_per_hour']:.4f}\n")
                
                f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate file-level sensitivity and false alarms per hour from Madrid windowed results"
    )
    parser.add_argument(
        "results_dir", 
        help="Directory containing Madrid windowed results JSON files"
    )
    parser.add_argument(
        "-o", "--output-dir", 
        help="Output directory for metrics results (default: same as results_dir)"
    )
    parser.add_argument(
        "-t", "--threshold", 
        type=float,
        help="Anomaly score threshold for detection (default: use top-ranked anomaly per window)"
    )
    parser.add_argument(
        "--filename", 
        help="Output filename prefix (default: auto-generated with timestamp)"
    )
    
    args = parser.parse_args()
    
    # Initialize calculator
    calculator = MadridFileLevelMetrics(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        threshold=args.threshold
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
    print("MULTI-SEIZURE FILE-LEVEL SUMMARY:")
    print(f"{'='*50}")
    print(f"Files processed: {summary['total_files_processed']}")
    print(f"Total duration: {summary['total_duration_hours']:.2f} hours")
    print(f"Total seizures: {summary['total_seizures']}")
    print(f"File-Level Sensitivity: {summary['file_level_sensitivity']:.4f} ({summary['file_level_sensitivity']*100:.2f}%)")
    print(f"  Files with all seizures detected: {summary['seizure_files_detected']}/{summary['files_with_seizures']}")
    print(f"Seizure-Level Sensitivity: {summary['seizure_level_sensitivity']:.4f} ({summary['seizure_level_sensitivity']*100:.2f}%)")
    print(f"  Individual seizures detected: {summary['total_detected_seizures']}/{summary['total_seizures']}")
    print(f"Overall False Alarms/Hour: {summary['overall_false_alarms_per_hour']:.4f}")


if __name__ == "__main__":
    main()