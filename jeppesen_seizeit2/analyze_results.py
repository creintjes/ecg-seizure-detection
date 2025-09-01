#!/usr/bin/env python3
"""
Script to analyze Jeppesen seizure detection results from CSV file.
Calculates overall sensitivity and false alarms per hour.
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys
from pathlib import Path

# Add seizeit2 data classes to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Information', 'Data', 'seizeit2_main'))

def load_recording_durations_from_raw_data(raw_data_path="/home/swolf/asim_shared/raw_data/ds005873-1.1.0"):
    """
    Load recording durations from raw SeizeIT2 dataset annotation files.
    
    Parameters:
    raw_data_path (str): Path to raw SeizeIT2 dataset
    
    Returns:
    dict: Dictionary mapping subject_id to total recording duration in hours
    """
    try:
        from classes.annotation import Annotation
    except ImportError:
        print("Warning: Could not import Annotation class. Make sure seizeit2_main is in the path.")
        return {}
    
    durations = {}
    
    if not Path(raw_data_path).exists():
        print(f"Warning: Raw data path {raw_data_path} not found.")
        return durations
    
    raw_data_path = Path(raw_data_path)
    
    # Find all subjects
    subjects = [x for x in raw_data_path.glob("sub-*") if x.is_dir()]
    print(f"Found {len(subjects)} subjects in raw data")
    
    total_hours = 0
    processed_recordings = 0
    
    for subject_dir in subjects:
        subject_id = subject_dir.name
        
        # Look for EEG sessions (where annotation files are stored)
        eeg_dir = subject_dir / 'ses-01' / 'eeg'
        if eeg_dir.exists():
            # Find all runs for this subject
            event_files = list(eeg_dir.glob("*_events.tsv"))
            
            subject_total = 0
            
            for event_file in event_files:
                try:
                    # Extract run ID from filename
                    # Format: sub-XXX_ses-01_task-szMonitoring_run-XX_events.tsv
                    parts = event_file.stem.split('_')
                    run_part = [p for p in parts if p.startswith('run-')]
                    
                    if run_part:
                        run_id = run_part[0]
                        
                        # Load annotation to get recording duration
                        recording = [subject_id, run_id]
                        annotation = Annotation.loadAnnotation(str(raw_data_path), recording)
                        
                        duration_seconds = annotation.rec_duration
                        duration_hours = duration_seconds / 3600
                        
                        subject_total += duration_hours
                        total_hours += duration_hours
                        processed_recordings += 1
                        
                        print(f"  {subject_id} {run_id}: {duration_hours:.2f}h")
                        
                except Exception as e:
                    print(f"Warning: Could not load duration for {subject_id} {run_id}: {e}")
                    continue
            
            if subject_total > 0:
                durations[subject_id] = subject_total
    
    print(f"\nTotal recording time: {total_hours:.1f} hours ({processed_recordings} recordings)")
    return durations, total_hours

def calculate_overall_metrics(csv_file, raw_data_path="/home/swolf/asim_shared/raw_data/ds005873-1.1.0"):
    """
    Calculate overall sensitivity and false alarms per hour from results CSV.
    Uses actual recording durations from raw data.
    
    Parameters:
    csv_file (str): Path to the CSV file with results
    raw_data_path (str): Path to raw SeizeIT2 dataset
    
    Returns:
    dict: Dictionary with calculated metrics
    """
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Load actual recording durations from raw data
    print("Loading recording durations from raw data...")
    subject_durations, total_dataset_hours = load_recording_durations_from_raw_data(raw_data_path)
    
    # Group by parameter to get overall metrics for each method
    results = {}
    
    # Get unique parameters and subjects
    parameters = df['parameter'].unique()
    subjects_in_csv = df['subject'].unique()
    
    print(f"\nAnalyzing {len(df)} records from {len(subjects_in_csv)} subjects")
    print(f"Found {len(parameters)} different parameters/methods")
    print("\n" + "="*80)
    
    for param in parameters:
        param_data = df[df['parameter'] == param]
        
        # Calculate overall sensitivity
        total_correct_pred = param_data['num_correct_pred_seizures'].sum()
        total_seizures = param_data['num_seizures'].sum()
        overall_sensitivity = total_correct_pred / total_seizures if total_seizures > 0 else 0
        
        # Calculate false alarms per hour using actual recording times
        total_false_alarms = param_data['num_pred_seizures'].sum() - total_correct_pred
        
        # Calculate total recording hours for subjects in this parameter
        total_record_hours = 0
        subjects_with_duration = 0
        
        for subject_id in param_data['subject'].unique():
            if subject_id in subject_durations:
                total_record_hours += subject_durations[subject_id]
                subjects_with_duration += 1
        
        # Calculate overall false alarms per hour
        overall_fad = total_false_alarms / total_record_hours if total_record_hours > 0 else 0
        
        # Also keep the average FAD from individual subjects for comparison
        avg_fad = param_data['FAD'].mean()
        
        # Calculate additional metrics
        total_predictions = param_data['num_pred_seizures'].sum()
        overall_precision = total_correct_pred / total_predictions if total_predictions > 0 else 0
        
        # Store results
        results[param] = {
            'overall_sensitivity': overall_sensitivity,
            'total_correct_predictions': total_correct_pred,
            'total_seizures': total_seizures,
            'total_false_alarms': total_false_alarms,
            'total_predictions': total_predictions,
            'overall_precision': overall_precision,
            'overall_fad': overall_fad,
            'avg_fad': avg_fad,
            'total_record_hours': total_record_hours,
            'subjects_with_duration': subjects_with_duration,
            'num_subjects': len(param_data)
        }
        
        print(f"Parameter: {param}")
        print(f"  Subjects analyzed: {len(param_data)}")
        print(f"  Subjects with duration info: {subjects_with_duration}/{len(param_data['subject'].unique())}")
        print(f"  Overall Sensitivity: {overall_sensitivity:.4f} ({total_correct_pred}/{total_seizures})")
        print(f"  Overall Precision: {overall_precision:.4f} ({total_correct_pred}/{total_predictions})")
        print(f"  Total False Alarms: {total_false_alarms}")
        print(f"  Total Recording Hours (from raw data): {total_record_hours:.1f}h")
        print(f"  Overall FAD (False Alarms per Hour): {overall_fad:.4f}")
        print(f"  Average individual FAD (from CSV): {avg_fad:.4f}")
        print("-" * 60)
    
    return results

def find_best_parameters(results):
    """
    Find parameters with best sensitivity and best precision.
    """
    print("\n" + "="*80)
    print("BEST PERFORMING PARAMETERS:")
    print("="*80)
    
    # Best sensitivity
    best_sensitivity = max(results.items(), key=lambda x: x[1]['overall_sensitivity'])
    print(f"Best Sensitivity: {best_sensitivity[0]}")
    print(f"  Sensitivity: {best_sensitivity[1]['overall_sensitivity']:.4f}")
    print(f"  Overall FAD: {best_sensitivity[1]['overall_fad']:.4f}")
    print(f"  Average FAD: {best_sensitivity[1]['avg_fad']:.4f}")
    print(f"  Precision: {best_sensitivity[1]['overall_precision']:.4f}")
    
    # Best precision
    best_precision = max(results.items(), key=lambda x: x[1]['overall_precision'])
    print(f"\nBest Precision: {best_precision[0]}")
    print(f"  Precision: {best_precision[1]['overall_precision']:.4f}")
    print(f"  Sensitivity: {best_precision[1]['overall_sensitivity']:.4f}")
    print(f"  Overall FAD: {best_precision[1]['overall_fad']:.4f}")
    print(f"  Average FAD: {best_precision[1]['avg_fad']:.4f}")
    
    # Best F1-score (harmonic mean of sensitivity and precision)
    f1_scores = {}
    for param, metrics in results.items():
        sens = metrics['overall_sensitivity']
        prec = metrics['overall_precision']
        if sens + prec > 0:
            f1_scores[param] = 2 * (sens * prec) / (sens + prec)
        else:
            f1_scores[param] = 0
    
    best_f1 = max(f1_scores.items(), key=lambda x: x[1])
    print(f"\nBest F1-Score: {best_f1[0]}")
    print(f"  F1-Score: {best_f1[1]:.4f}")
    print(f"  Sensitivity: {results[best_f1[0]]['overall_sensitivity']:.4f}")
    print(f"  Precision: {results[best_f1[0]]['overall_precision']:.4f}")
    print(f"  Overall FAD: {results[best_f1[0]]['overall_fad']:.4f}")
    print(f"  Average FAD: {results[best_f1[0]]['avg_fad']:.4f}")

def save_summary(results, output_file):
    """
    Save summary to a CSV file.
    """
    summary_data = []
    for param, metrics in results.items():
        summary_data.append({
            'parameter': param,
            'overall_sensitivity': metrics['overall_sensitivity'],
            'overall_precision': metrics['overall_precision'],
            'overall_fad': metrics['overall_fad'],
            'avg_fad': metrics['avg_fad'],
            'total_correct_predictions': metrics['total_correct_predictions'],
            'total_seizures': metrics['total_seizures'],
            'total_false_alarms': metrics['total_false_alarms'],
            'total_record_hours': metrics['total_record_hours'],
            'num_subjects': metrics['num_subjects']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('overall_sensitivity', ascending=False)
    summary_df.to_csv(output_file, index=False)
    print(f"\nSummary saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze Jeppesen seizure detection results')
    parser.add_argument('csv_file', help='Path to the results CSV file')
    parser.add_argument('--output', '-o', help='Output file for summary (optional)')
    parser.add_argument('--raw-data-path', default='/home/swolf/asim_shared/raw_data/ds005873-1.1.0',
                       help='Path to raw SeizeIT2 dataset (default: /home/swolf/asim_shared/raw_data/ds005873-1.1.0)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: File {args.csv_file} not found!")
        return
    
    # Calculate metrics using actual recording durations
    results = calculate_overall_metrics(args.csv_file, args.raw_data_path)
    
    # Find best parameters
    find_best_parameters(results)
    
    # Save summary if output file specified
    if args.output:
        save_summary(results, args.output)
    
    print(f"\nAnalysis complete! Analyzed {len(results)} different parameters.")

if __name__ == "__main__":
    main()