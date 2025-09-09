#!/usr/bin/env python3
"""
Simple analysis script to evaluate Madrid metrics by eventType using only standard library.

This script:
1. Loads Madrid clustering results from tolerance_adjusted_smart_clustered
2. Cross-references with original SeizeIT2 annotations to get eventType information
3. Calculates sensitivity, precision, and FAR metrics per eventType
4. Generates text-based report
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

class SimpleMadridEventTypeAnalyzer:
    """Simple analyzer for Madrid results by seizure eventType."""
    
    def __init__(self, madrid_results_dir: str, seizeit2_data_path: str):
        """
        Initialize the analyzer.
        
        Args:
            madrid_results_dir: Path to Madrid results directory
            seizeit2_data_path: Path to SeizeIT2 dataset for annotations
        """
        self.madrid_results_dir = Path(madrid_results_dir)
        self.seizeit2_data_path = Path(seizeit2_data_path)
        
    def load_madrid_results(self) -> Dict[str, Any]:
        """Load Madrid results from JSON files."""
        results = {}
        
        # Load individual seizure results
        individual_results_dir = self.madrid_results_dir / "madrid_dir_400_examples"
        if individual_results_dir.exists():
            print(f"Loading results from: {individual_results_dir}")
            for json_file in individual_results_dir.glob("madrid_results_*.json"):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        # Extract subject, run, seizure info from filename or data
                        subject_id = data['input_data']['subject_id']
                        run_id = data['input_data']['run_id']
                        seizure_id = data['input_data']['seizure_id']
                        
                        key = f"{subject_id}_{run_id}_{seizure_id}"
                        results[key] = data
                        
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
        else:
            print(f"Directory not found: {individual_results_dir}")
                    
        return results
    
    def get_seizure_eventtype(self, subject_id: str, run_id: str, seizure_id: str) -> str:
        """
        Get eventType for a specific seizure from SeizeIT2 annotations.
        
        Args:
            subject_id: Subject identifier (e.g., 'sub-001')
            run_id: Run identifier (e.g., 'run-03') 
            seizure_id: Seizure identifier (e.g., 'seizure_00')
            
        Returns:
            eventType string or 'unknown' if not found
        """
        try:
            # Construct path to TSV file
            tsv_file = self.seizeit2_data_path / subject_id / 'ses-01' / 'eeg' / f"{subject_id}_ses-01_task-szMonitoring_{run_id}_events.tsv"
            
            if not tsv_file.exists():
                print(f"TSV file not found: {tsv_file}")
                return 'unknown'
            
            # Parse TSV file manually (since pandas is not available)
            with open(tsv_file, 'r') as f:
                lines = f.readlines()
                if len(lines) < 2:  # Need header + at least one data line
                    return 'unknown'
                
                # Parse header
                header = lines[0].strip().split('\t')
                eventtype_col = None
                for i, col in enumerate(header):
                    if col == 'eventType':
                        eventtype_col = i
                        break
                
                if eventtype_col is None:
                    print(f"eventType column not found in {tsv_file}")
                    return 'unknown'
                
                # Parse data lines and find seizure events
                seizure_events = []
                for line in lines[1:]:
                    parts = line.strip().split('\t')
                    if len(parts) > eventtype_col:
                        event_type = parts[eventtype_col]
                        if event_type not in ['bckg', 'impd']:  # Skip background and impedance events
                            seizure_events.append(event_type)
                
                # Extract seizure index from seizure_id
                seizure_idx = int(seizure_id.split('_')[-1])
                
                if seizure_idx < len(seizure_events):
                    return seizure_events[seizure_idx]
                else:
                    return 'unknown'
                    
        except Exception as e:
            print(f"Error getting eventType for {subject_id}_{run_id}_{seizure_id}: {e}")
            return 'unknown'
    
    def calculate_metrics_by_eventtype(self, madrid_results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics (sensitivity, precision, FAR) by eventType.
        
        Args:
            madrid_results: Dictionary of Madrid results
            
        Returns:
            Dictionary with metrics per eventType
        """
        eventtype_data = defaultdict(lambda: {
            'total_seizures': 0,
            'detected_seizures': 0,
            'true_positives': 0,
            'false_positives': 0,
            'total_anomalies': 0
        })
        
        print(f"Processing {len(madrid_results)} Madrid results...")
        
        for key, result in madrid_results.items():
            # Parse subject, run, seizure from key
            parts = key.split('_')
            subject_id = parts[0]
            run_id = parts[1] 
            seizure_id = '_'.join(parts[2:])
            
            # Get eventType
            eventtype = self.get_seizure_eventtype(subject_id, run_id, seizure_id)
            print(f"Processing {key}: eventType = {eventtype}")
            
            # Extract metrics from Madrid results
            if 'analysis_results' in result:
                analysis = result['analysis_results']
                
                eventtype_data[eventtype]['total_seizures'] += 1
                
                # Check if seizure was detected (has any true positives)
                has_detection = False
                tp_count = 0
                fp_count = 0
                total_anomalies = len(analysis.get('ranked_anomalies', []))
                
                for anomaly in analysis.get('ranked_anomalies', []):
                    if anomaly.get('seizure_hit', False):
                        tp_count += 1
                        has_detection = True
                    else:
                        fp_count += 1
                
                if has_detection:
                    eventtype_data[eventtype]['detected_seizures'] += 1
                
                eventtype_data[eventtype]['true_positives'] += tp_count
                eventtype_data[eventtype]['false_positives'] += fp_count
                eventtype_data[eventtype]['total_anomalies'] += total_anomalies
        
        # Calculate final metrics
        metrics_by_eventtype = {}
        for eventtype, data in eventtype_data.items():
            metrics = {}
            
            # Sensitivity (recall) = detected_seizures / total_seizures
            metrics['sensitivity'] = data['detected_seizures'] / data['total_seizures'] if data['total_seizures'] > 0 else 0
            
            # Precision = TP / (TP + FP)
            total_predictions = data['true_positives'] + data['false_positives']
            metrics['precision'] = data['true_positives'] / total_predictions if total_predictions > 0 else 0
            
            # False Alarm Rate = FP / (TP + FP)
            metrics['false_alarm_rate'] = data['false_positives'] / total_predictions if total_predictions > 0 else 0
            
            # Additional metrics
            metrics['total_seizures'] = data['total_seizures']
            metrics['detected_seizures'] = data['detected_seizures']
            metrics['true_positives'] = data['true_positives']
            metrics['false_positives'] = data['false_positives']
            metrics['total_anomalies'] = data['total_anomalies']
            
            metrics_by_eventtype[eventtype] = metrics
            
        return metrics_by_eventtype
    
    def generate_report(self, metrics_by_eventtype: Dict[str, Dict[str, float]], output_dir: str):
        """Generate detailed text report of metrics by eventType.""" 
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        report_file = output_path / 'madrid_eventtype_analysis_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("Madrid Seizure Detection: Analysis by EventType\n")
            f.write("=" * 60 + "\n\n")
            
            # Sort by total seizures for better readability
            sorted_eventtypes = sorted(metrics_by_eventtype.items(), 
                                     key=lambda x: x[1]['total_seizures'], reverse=True)
            
            f.write("DETAILED RESULTS BY EVENTTYPE:\n")
            f.write("-" * 40 + "\n")
            
            for eventtype, metrics in sorted_eventtypes:
                f.write(f"\nEventType: {eventtype}\n")
                f.write(f"  Total Seizures: {metrics['total_seizures']}\n")
                f.write(f"  Detected Seizures: {metrics['detected_seizures']}\n")
                f.write(f"  Sensitivity: {metrics['sensitivity']:.3f} ({metrics['sensitivity']*100:.1f}%)\n")
                f.write(f"  Precision: {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)\n")
                f.write(f"  False Alarm Rate: {metrics['false_alarm_rate']:.3f} ({metrics['false_alarm_rate']*100:.1f}%)\n")
                f.write(f"  True Positives: {metrics['true_positives']}\n")
                f.write(f"  False Positives: {metrics['false_positives']}\n")
                f.write(f"  Total Anomalies: {metrics['total_anomalies']}\n")
            
            # Overall statistics
            f.write(f"\n\nOVERALL STATISTICS:\n")
            f.write("-" * 25 + "\n")
            
            total_seizures = sum(m['total_seizures'] for m in metrics_by_eventtype.values())
            total_detected = sum(m['detected_seizures'] for m in metrics_by_eventtype.values())
            total_tp = sum(m['true_positives'] for m in metrics_by_eventtype.values())
            total_fp = sum(m['false_positives'] for m in metrics_by_eventtype.values())
            
            overall_sensitivity = total_detected / total_seizures if total_seizures > 0 else 0
            overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            overall_far = total_fp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            
            f.write(f"Total Seizures Analyzed: {total_seizures}\n")
            f.write(f"Total EventTypes Found: {len(metrics_by_eventtype)}\n")
            f.write(f"Overall Sensitivity: {overall_sensitivity:.3f} ({overall_sensitivity*100:.1f}%)\n")
            f.write(f"Overall Precision: {overall_precision:.3f} ({overall_precision*100:.1f}%)\n")
            f.write(f"Overall False Alarm Rate: {overall_far:.3f} ({overall_far*100:.1f}%)\n")
            
            # Summary table
            f.write(f"\n\nSUMMARY TABLE:\n")
            f.write("-" * 20 + "\n")
            f.write(f"{'EventType':<15} {'N':<5} {'Sens%':<7} {'Prec%':<7} {'FAR%':<7}\n")
            f.write("-" * 50 + "\n")
            
            for eventtype, metrics in sorted_eventtypes:
                f.write(f"{eventtype:<15} {metrics['total_seizures']:<5} "
                       f"{metrics['sensitivity']*100:<7.1f} {metrics['precision']*100:<7.1f} "
                       f"{metrics['false_alarm_rate']*100:<7.1f}\n")
            
        print(f"Report saved to {report_file}")
        
        # Also print summary to console
        print("\n" + "="*60)
        print("MADRID EVENTTYPE ANALYSIS SUMMARY")
        print("="*60)
        print(f"{'EventType':<15} {'N':<5} {'Sens%':<7} {'Prec%':<7} {'FAR%':<7}")
        print("-" * 50)
        
        for eventtype, metrics in sorted_eventtypes:
            print(f"{eventtype:<15} {metrics['total_seizures']:<5} "
                  f"{metrics['sensitivity']*100:<7.1f} {metrics['precision']*100:<7.1f} "
                  f"{metrics['false_alarm_rate']*100:<7.1f}")
        
        print(f"\nTotal Seizures: {total_seizures}")
        print(f"Total EventTypes: {len(metrics_by_eventtype)}")
        print(f"Overall Sensitivity: {overall_sensitivity*100:.1f}%")
        print(f"Overall Precision: {overall_precision*100:.1f}%")
        print(f"Overall False Alarm Rate: {overall_far*100:.1f}%")
    
    def run_analysis(self, output_dir: str = "madrid_eventtype_analysis"):
        """Run complete analysis pipeline."""
        print("Loading Madrid results...")
        madrid_results = self.load_madrid_results()
        print(f"Loaded {len(madrid_results)} Madrid result files")
        
        if len(madrid_results) == 0:
            print("No Madrid results found. Check the directory path.")
            return None
        
        print("Calculating metrics by eventType...")
        metrics_by_eventtype = self.calculate_metrics_by_eventtype(madrid_results)
        print(f"Found {len(metrics_by_eventtype)} different eventTypes")
        
        print("Generating report...")
        self.generate_report(metrics_by_eventtype, output_dir)
        
        return metrics_by_eventtype

def main():
    """Main execution function."""
    # Configuration - adjust these paths as needed
    madrid_results_dir = "madrid_results copy"
    seizeit2_data_path = "../Information/Data/seizeit2-main"
    output_dir = "madrid_eventtype_analysis"
    
    # Initialize analyzer
    analyzer = SimpleMadridEventTypeAnalyzer(madrid_results_dir, seizeit2_data_path)
    
    # Run analysis
    try:
        results = analyzer.run_analysis(output_dir)
        if results:
            print("\nAnalysis completed successfully!")
            print(f"Results saved to: {output_dir}/")
        else:
            print("Analysis failed - no results generated.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()