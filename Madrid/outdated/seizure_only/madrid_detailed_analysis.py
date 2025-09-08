#!/usr/bin/env python3
"""
Detailed analysis of Madrid results by individual seizures.

This script analyzes the available Madrid results and provides detailed metrics
per seizure and overall performance statistics.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

class MadridDetailedAnalyzer:
    """Detailed analyzer for Madrid results."""
    
    def __init__(self, madrid_results_dir: str):
        self.madrid_results_dir = Path(madrid_results_dir)
        
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
    
    def analyze_individual_seizures(self, madrid_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze each seizure individually."""
        seizure_analyses = []
        
        for key, result in madrid_results.items():
            analysis = {
                'seizure_id': key,
                'subject_id': result['input_data']['subject_id'],
                'run_id': result['input_data']['run_id'],
                'seizure_idx': result['input_data']['seizure_id'],
                'signal_duration': result['input_data']['signal_metadata']['signal_duration_seconds'],
                'sampling_rate': result['input_data']['signal_metadata']['sampling_rate']
            }
            
            # Extract ground truth information
            if 'validation_data' in result and 'ground_truth' in result['validation_data']:
                gt = result['validation_data']['ground_truth']
                analysis['seizure_present'] = gt.get('seizure_present', False)
                
                if 'seizure_regions' in gt and len(gt['seizure_regions']) > 0:
                    seizure_region = gt['seizure_regions'][0]  # Take first seizure
                    analysis['seizure_onset_time'] = seizure_region.get('onset_time_seconds', 0)
                    analysis['seizure_duration'] = seizure_region.get('duration_seconds', 0)
                    analysis['seizure_type'] = seizure_region.get('seizure_type', 'unknown')
                else:
                    analysis['seizure_onset_time'] = 0
                    analysis['seizure_duration'] = 0
                    analysis['seizure_type'] = 'unknown'
            else:
                analysis['seizure_present'] = False
                analysis['seizure_onset_time'] = 0
                analysis['seizure_duration'] = 0
                analysis['seizure_type'] = 'unknown'
            
            # Extract performance metrics
            if 'analysis_results' in result:
                ar = result['analysis_results']
                total_anomalies = len(ar.get('anomalies', []))
                
                # Count true positives and false positives
                tp_count = 0
                fp_count = 0
                
                for anomaly in ar.get('anomalies', []):
                    if anomaly.get('seizure_hit', False):
                        tp_count += 1
                    else:
                        fp_count += 1
                
                analysis['total_anomalies'] = total_anomalies
                analysis['true_positives'] = tp_count
                analysis['false_positives'] = fp_count
                analysis['seizure_detected'] = tp_count > 0
                
                # Calculate metrics
                if total_anomalies > 0:
                    analysis['precision'] = tp_count / total_anomalies
                    analysis['false_alarm_rate'] = fp_count / total_anomalies
                else:
                    analysis['precision'] = 0.0
                    analysis['false_alarm_rate'] = 0.0
                
                # Sensitivity is binary: 1 if seizure detected, 0 if not
                analysis['sensitivity'] = 1.0 if analysis['seizure_detected'] else 0.0
                
            else:
                analysis['total_anomalies'] = 0
                analysis['true_positives'] = 0
                analysis['false_positives'] = 0
                analysis['seizure_detected'] = False
                analysis['precision'] = 0.0
                analysis['false_alarm_rate'] = 0.0
                analysis['sensitivity'] = 0.0
            
            seizure_analyses.append(analysis)
        
        return seizure_analyses
    
    def calculate_overall_metrics(self, seizure_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall performance metrics."""
        total_seizures = len(seizure_analyses)
        detected_seizures = sum(1 for s in seizure_analyses if s['seizure_detected'])
        total_tp = sum(s['true_positives'] for s in seizure_analyses)
        total_fp = sum(s['false_positives'] for s in seizure_analyses)
        total_anomalies = sum(s['total_anomalies'] for s in seizure_analyses)
        
        # Overall metrics
        overall_metrics = {
            'total_seizures': total_seizures,
            'detected_seizures': detected_seizures,
            'total_anomalies': total_anomalies,
            'total_true_positives': total_tp,
            'total_false_positives': total_fp,
            'overall_sensitivity': detected_seizures / total_seizures if total_seizures > 0 else 0,
            'overall_precision': total_tp / total_anomalies if total_anomalies > 0 else 0,
            'overall_false_alarm_rate': total_fp / total_anomalies if total_anomalies > 0 else 0
        }
        
        return overall_metrics
    
    def analyze_by_patient(self, seizure_analyses: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Analyze performance by patient (subject)."""
        patient_data = defaultdict(lambda: {
            'seizures': [],
            'total_seizures': 0,
            'detected_seizures': 0,
            'total_tp': 0,
            'total_fp': 0,
            'total_anomalies': 0
        })
        
        # Group by patient
        for seizure in seizure_analyses:
            subject_id = seizure['subject_id']
            patient_data[subject_id]['seizures'].append(seizure)
            patient_data[subject_id]['total_seizures'] += 1
            patient_data[subject_id]['detected_seizures'] += 1 if seizure['seizure_detected'] else 0
            patient_data[subject_id]['total_tp'] += seizure['true_positives']
            patient_data[subject_id]['total_fp'] += seizure['false_positives']
            patient_data[subject_id]['total_anomalies'] += seizure['total_anomalies']
        
        # Calculate metrics per patient
        patient_metrics = {}
        for subject_id, data in patient_data.items():
            metrics = {
                'total_seizures': data['total_seizures'],
                'detected_seizures': data['detected_seizures'],
                'total_anomalies': data['total_anomalies'],
                'total_tp': data['total_tp'],
                'total_fp': data['total_fp'],
                'sensitivity': data['detected_seizures'] / data['total_seizures'] if data['total_seizures'] > 0 else 0,
                'precision': data['total_tp'] / data['total_anomalies'] if data['total_anomalies'] > 0 else 0,
                'false_alarm_rate': data['total_fp'] / data['total_anomalies'] if data['total_anomalies'] > 0 else 0
            }
            patient_metrics[subject_id] = metrics
        
        return patient_metrics
    
    def generate_detailed_report(self, seizure_analyses: List[Dict[str, Any]], 
                               overall_metrics: Dict[str, Any],
                               patient_metrics: Dict[str, Dict[str, Any]], 
                               output_dir: str):
        """Generate detailed analysis report."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        report_file = output_path / 'madrid_detailed_analysis_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("Madrid Seizure Detection: Detailed Analysis Report\n")
            f.write("=" * 70 + "\n\n")
            
            # Overall summary
            f.write("OVERALL PERFORMANCE SUMMARY:\n")
            f.write("-" * 35 + "\n")
            f.write(f"Total Seizures Analyzed: {overall_metrics['total_seizures']}\n")
            f.write(f"Seizures Successfully Detected: {overall_metrics['detected_seizures']}\n")
            f.write(f"Overall Sensitivity: {overall_metrics['overall_sensitivity']:.3f} ({overall_metrics['overall_sensitivity']*100:.1f}%)\n")
            f.write(f"Overall Precision: {overall_metrics['overall_precision']:.3f} ({overall_metrics['overall_precision']*100:.1f}%)\n")
            f.write(f"Overall False Alarm Rate: {overall_metrics['overall_false_alarm_rate']:.3f} ({overall_metrics['overall_false_alarm_rate']*100:.1f}%)\n")
            f.write(f"Total Anomalies Generated: {overall_metrics['total_anomalies']}\n")
            f.write(f"Total True Positives: {overall_metrics['total_true_positives']}\n")
            f.write(f"Total False Positives: {overall_metrics['total_false_positives']}\n")
            
            # Patient-level analysis
            f.write(f"\n\nPERFORMANCE BY PATIENT:\n")
            f.write("-" * 25 + "\n")
            f.write(f"{'Patient':<12} {'N':<3} {'Det':<3} {'Sens%':<7} {'Prec%':<7} {'FAR%':<7} {'TP':<4} {'FP':<4}\n")
            f.write("-" * 70 + "\n")
            
            for subject_id in sorted(patient_metrics.keys()):
                metrics = patient_metrics[subject_id]
                f.write(f"{subject_id:<12} {metrics['total_seizures']:<3} {metrics['detected_seizures']:<3} "
                       f"{metrics['sensitivity']*100:<7.1f} {metrics['precision']*100:<7.1f} "
                       f"{metrics['false_alarm_rate']*100:<7.1f} {metrics['total_tp']:<4} {metrics['total_fp']:<4}\n")
            
            # Individual seizure details
            f.write(f"\n\nINDIVIDUAL SEIZURE ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"{'Seizure ID':<25} {'Duration':<8} {'Detected':<8} {'TP':<3} {'FP':<3} {'Prec%':<7}\n")
            f.write("-" * 70 + "\n")
            
            for seizure in sorted(seizure_analyses, key=lambda x: x['seizure_id']):
                detected_str = "YES" if seizure['seizure_detected'] else "NO"
                f.write(f"{seizure['seizure_id']:<25} {seizure['seizure_duration']:<8.1f} "
                       f"{detected_str:<8} {seizure['true_positives']:<3} {seizure['false_positives']:<3} "
                       f"{seizure['precision']*100:<7.1f}\n")
            
            # Statistics
            f.write(f"\n\nSTATISTICAL SUMMARY:\n")
            f.write("-" * 20 + "\n")
            
            # Seizure durations
            durations = [s['seizure_duration'] for s in seizure_analyses]
            if durations:
                avg_duration = sum(durations) / len(durations)
                min_duration = min(durations)
                max_duration = max(durations)
                f.write(f"Seizure Duration Statistics:\n")
                f.write(f"  Average: {avg_duration:.1f} seconds\n")
                f.write(f"  Range: {min_duration:.1f} - {max_duration:.1f} seconds\n")
            
            # Signal characteristics
            signal_durations = [s['signal_duration'] for s in seizure_analyses]
            if signal_durations:
                avg_signal = sum(signal_durations) / len(signal_durations)
                f.write(f"\nSignal Duration Statistics:\n")
                f.write(f"  Average Signal Length: {avg_signal:.1f} seconds ({avg_signal/60:.1f} minutes)\n")
            
            # Detection performance breakdown
            detected_count = sum(1 for s in seizure_analyses if s['seizure_detected'])
            undetected_count = len(seizure_analyses) - detected_count
            f.write(f"\nDetection Breakdown:\n")
            f.write(f"  Successfully Detected: {detected_count} ({detected_count/len(seizure_analyses)*100:.1f}%)\n")
            f.write(f"  Missed (False Negatives): {undetected_count} ({undetected_count/len(seizure_analyses)*100:.1f}%)\n")
        
        print(f"Detailed report saved to {report_file}")
        
        # Console summary
        print("\n" + "="*70)
        print("MADRID DETAILED ANALYSIS SUMMARY")
        print("="*70)
        print(f"Total Seizures: {overall_metrics['total_seizures']}")
        print(f"Detected Seizures: {overall_metrics['detected_seizures']}")
        print(f"Overall Sensitivity: {overall_metrics['overall_sensitivity']*100:.1f}%")
        print(f"Overall Precision: {overall_metrics['overall_precision']*100:.1f}%")
        print(f"Overall False Alarm Rate: {overall_metrics['overall_false_alarm_rate']*100:.1f}%")
        
        print(f"\nPer-Patient Summary:")
        print(f"{'Patient':<12} {'Seizures':<8} {'Detected':<8} {'Sensitivity':<12}")
        print("-" * 45)
        for subject_id in sorted(patient_metrics.keys()):
            metrics = patient_metrics[subject_id]
            print(f"{subject_id:<12} {metrics['total_seizures']:<8} {metrics['detected_seizures']:<8} "
                  f"{metrics['sensitivity']*100:.1f}%")
    
    def run_analysis(self, output_dir: str = "madrid_detailed_analysis"):
        """Run complete detailed analysis."""
        print("Loading Madrid results...")
        madrid_results = self.load_madrid_results()
        print(f"Loaded {len(madrid_results)} Madrid result files")
        
        if len(madrid_results) == 0:
            print("No Madrid results found. Check the directory path.")
            return None
        
        print("Analyzing individual seizures...")
        seizure_analyses = self.analyze_individual_seizures(madrid_results)
        
        print("Calculating overall metrics...")
        overall_metrics = self.calculate_overall_metrics(seizure_analyses)
        
        print("Analyzing by patient...")
        patient_metrics = self.analyze_by_patient(seizure_analyses)
        
        print("Generating detailed report...")
        self.generate_detailed_report(seizure_analyses, overall_metrics, patient_metrics, output_dir)
        
        return {
            'seizure_analyses': seizure_analyses,
            'overall_metrics': overall_metrics,
            'patient_metrics': patient_metrics
        }

def main():
    """Main execution function."""
    # Configuration
    madrid_results_dir = "madrid_results copy"
    output_dir = "madrid_detailed_analysis"
    
    # Initialize analyzer
    analyzer = MadridDetailedAnalyzer(madrid_results_dir)
    
    # Run analysis
    try:
        results = analyzer.run_analysis(output_dir)
        if results:
            print("\nDetailed analysis completed successfully!")
            print(f"Results saved to: {output_dir}/")
        else:
            print("Analysis failed - no results generated.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()