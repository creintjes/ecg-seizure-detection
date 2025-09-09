#!/usr/bin/env python3
"""
Madrid Metrics Calculator
Calculates sensitivity and false alarms per hour from Madrid windowed batch results.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import argparse


class MadridMetricsCalculator:
    def __init__(self, results_dir: str, output_dir: str = None, threshold: float = None):
        """
        Initialize the metrics calculator.
        
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
    
    def calculate_metrics_for_file(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics for a single result file."""
        
        # Extract basic info
        metadata = result_data.get('analysis_metadata', {})
        input_data = result_data.get('input_data', {})
        validation_data = result_data.get('validation_data', {})
        analysis_results = result_data.get('analysis_results', {})
        window_results = analysis_results.get('window_results', [])
        
        subject_id = input_data.get('subject_id', 'unknown')
        run_id = input_data.get('run_id', 'unknown')
        
        # Get ground truth
        ground_truth = validation_data.get('ground_truth', {})
        seizure_present = ground_truth.get('seizure_present', False)
        seizure_windows = ground_truth.get('seizure_windows', [])
        seizure_window_indices = set(sw.get('window_index', -1) for sw in seizure_windows)
        
        # Get signal info
        signal_metadata = input_data.get('signal_metadata', {})
        total_duration_hours = signal_metadata.get('total_duration_seconds', 0) / 3600.0
        window_duration_seconds = signal_metadata.get('window_duration_seconds', 3600.0)
        window_duration_hours = window_duration_seconds / 3600.0
        
        # Calculate detections (anomaly-based)
        total_detections = 0
        true_positive_detections = 0
        false_positive_detections = 0
        true_positive_windows = set()
        false_positive_windows = set()
        
        for window in window_results:
            window_index = window.get('window_index')
            
            # Extract anomalies for this window
            detected_anomalies = self.extract_anomalies_for_window(window)
            
            if detected_anomalies:  # If any anomalies detected in this window
                num_anomalies_in_window = len(detected_anomalies)
                total_detections += num_anomalies_in_window
                
                if window_index in seizure_window_indices:
                    # All anomalies in seizure window are true positives
                    true_positive_detections += num_anomalies_in_window
                    true_positive_windows.add(window_index)
                else:
                    # All anomalies in non-seizure window are false positives
                    false_positive_detections += num_anomalies_in_window
                    false_positive_windows.add(window_index)
        
        # Calculate metrics
        num_seizure_windows = len(seizure_window_indices)
        num_true_positive_windows = len(true_positive_windows)
        num_false_positive_windows = len(false_positive_windows)
        
        # Sensitivity = TP detections / total seizure windows (assuming 1 seizure per seizure window)
        sensitivity = (true_positive_detections / num_seizure_windows) if num_seizure_windows > 0 else 0.0
        
        # False alarms per hour = FP detections / total hours
        false_alarms_per_hour = false_positive_detections
        if total_duration_hours > 0:
            false_alarms_per_hour = false_alarms_per_hour / total_duration_hours
        else:
            false_alarms_per_hour = 0.0
        
        return {
            'file_info': {
                'subject_id': subject_id,
                'run_id': run_id,
                'analysis_id': metadata.get('analysis_id', 'unknown'),
                'timestamp': metadata.get('timestamp', 'unknown')
            },
            'data_info': {
                'total_duration_hours': round(total_duration_hours, 2),
                'total_windows': len(window_results),
                'window_duration_hours': round(window_duration_hours, 2),
                'seizure_present': seizure_present,
                'num_seizure_windows': num_seizure_windows
            },
            'detection_summary': {
                'total_detections': total_detections,
                'true_positive_detections': true_positive_detections,
                'false_positive_detections': false_positive_detections,
                'true_positive_windows': num_true_positive_windows,
                'false_positive_windows': num_false_positive_windows,
                'detected_seizure_windows': sorted(list(true_positive_windows)),
                'false_positive_window_indices': sorted(list(false_positive_windows))
            },
            'metrics': {
                'sensitivity': round(sensitivity, 4),
                'false_alarms_per_hour': round(false_alarms_per_hour, 4)
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
            'total_sensitivity_sum': 0.0,
            'total_false_alarms_sum': 0.0,
            'total_duration_hours': 0.0
        }
        
        for json_file in sorted(json_files):
            print(f"Processing {json_file.name}...")
            
            result_data = self.load_result_file(json_file)
            if result_data is None:
                continue
            
            metrics = self.calculate_metrics_for_file(result_data)
            all_results[json_file.name] = metrics
            
            # Update summary stats
            summary_stats['total_files_processed'] += 1
            summary_stats['total_duration_hours'] += metrics['data_info']['total_duration_hours']
            
            if metrics['data_info']['seizure_present']:
                summary_stats['files_with_seizures'] += 1
                summary_stats['total_sensitivity_sum'] += metrics['metrics']['sensitivity']
            else:
                summary_stats['files_without_seizures'] += 1
            
            summary_stats['total_false_alarms_sum'] += metrics['metrics']['false_alarms_per_hour']
        
        # Calculate overall averages
        if summary_stats['files_with_seizures'] > 0:
            avg_sensitivity = summary_stats['total_sensitivity_sum'] / summary_stats['files_with_seizures']
        else:
            avg_sensitivity = 0.0
        
        if summary_stats['total_files_processed'] > 0:
            avg_false_alarms = summary_stats['total_false_alarms_sum'] / summary_stats['total_files_processed']
        else:
            avg_false_alarms = 0.0
        
        return {
            'analysis_metadata': {
                'calculation_timestamp': datetime.now().isoformat(),
                'results_directory': str(self.results_dir),
                'threshold_used': self.threshold,
                'detection_strategy': 'threshold' if self.threshold else 'top_ranked'
            },
            'summary_statistics': {
                **summary_stats,
                'average_sensitivity': round(avg_sensitivity, 4),
                'average_false_alarms_per_hour': round(avg_false_alarms, 4),
                'total_duration_hours': round(summary_stats['total_duration_hours'], 2)
            },
            'individual_results': all_results
        }
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save results in both JSON and human-readable formats."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"madrid_metrics_results_{timestamp}"
        
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
            f.write("MADRID SEIZURE DETECTION METRICS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Analysis metadata
            metadata = results['analysis_metadata']
            f.write("ANALYSIS INFORMATION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Calculation Time: {metadata['calculation_timestamp']}\n")
            f.write(f"Results Directory: {metadata['results_directory']}\n")
            f.write(f"Detection Strategy: {metadata['detection_strategy']}\n")
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
            f.write(f"Total Recording Duration: {summary['total_duration_hours']:.2f} hours\n")
            f.write("\n")
            
            f.write("OVERALL PERFORMANCE:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Average Sensitivity: {summary['average_sensitivity']:.4f} ({summary['average_sensitivity']*100:.2f}%)\n")
            f.write(f"Average False Alarms per Hour: {summary['average_false_alarms_per_hour']:.4f}\n")
            f.write("\n")
            
            # Individual results
            f.write("INDIVIDUAL FILE RESULTS:\n")
            f.write("=" * 80 + "\n")
            
            for filename, file_results in results['individual_results'].items():
                f.write(f"\nFile: {filename}\n")
                f.write("-" * len(filename) + "-----\n")
                
                info = file_results['file_info']
                data = file_results['data_info']
                detection = file_results['detection_summary']
                metrics = file_results['metrics']
                
                f.write(f"Subject: {info['subject_id']}, Run: {info['run_id']}\n")
                f.write(f"Duration: {data['total_duration_hours']:.2f} hours ({data['total_windows']} windows)\n")
                f.write(f"Seizure Present: {'Yes' if data['seizure_present'] else 'No'}")
                if data['seizure_present']:
                    f.write(f" ({data['num_seizure_windows']} seizure windows)")
                f.write("\n")
                
                f.write(f"Total Detections: {detection['total_detections']}\n")
                f.write(f"True Positive Detections: {detection['true_positive_detections']}\n")
                f.write(f"False Positive Detections: {detection['false_positive_detections']}\n")
                f.write(f"True Positive Windows: {detection['true_positive_windows']}\n")
                f.write(f"False Positive Windows: {detection['false_positive_windows']}\n")
                
                f.write(f"\nMETRICS:\n")
                f.write(f"  Sensitivity: {metrics['sensitivity']:.4f} ({metrics['sensitivity']*100:.2f}%)\n")
                f.write(f"  False Alarms/Hour: {metrics['false_alarms_per_hour']:.4f}\n")
                
                if detection['detected_seizure_windows']:
                    f.write(f"  Detected Seizure Windows: {detection['detected_seizure_windows']}\n")
                
                if detection['false_positive_window_indices']:
                    fp_windows = detection['false_positive_window_indices']
                    if len(fp_windows) <= 10:
                        f.write(f"  False Positive Windows: {fp_windows}\n")
                    else:
                        f.write(f"  False Positive Windows: {fp_windows[:10]}... (and {len(fp_windows)-10} more)\n")
                
                f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate sensitivity and false alarms per hour from Madrid windowed results"
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
    calculator = MadridMetricsCalculator(
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
    print("SUMMARY:")
    print(f"{'='*50}")
    print(f"Files processed: {summary['total_files_processed']}")
    print(f"Total duration: {summary['total_duration_hours']:.2f} hours")
    print(f"Average Sensitivity: {summary['average_sensitivity']:.4f} ({summary['average_sensitivity']*100:.2f}%)")
    print(f"Average False Alarms/Hour: {summary['average_false_alarms_per_hour']:.4f}")


if __name__ == "__main__":
    main()