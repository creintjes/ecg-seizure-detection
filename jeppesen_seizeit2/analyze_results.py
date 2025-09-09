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

def calculate_responder_metrics(param_data, subject_durations):
    """
    Calculate responder metrics for patients where at least 2/3 of seizures are detected.
    
    Parameters:
    param_data (DataFrame): Parameter-specific data from CSV
    subject_durations (dict): Dictionary mapping subject_id to recording duration in hours
    
    Returns:
    dict: Dictionary with responder analysis results
    """
    responders = []
    non_responders = []
    patients_with_seizures = []
    
    # Group by subject to calculate per-patient detection rates
    for subject_id in param_data['subject'].unique():
        subject_data = param_data[param_data['subject'] == subject_id]
        
        # Sum across all recordings for this subject
        total_seizures = subject_data['num_seizures'].sum()
        detected_seizures = subject_data['num_correct_pred_seizures'].sum()
        total_predictions = subject_data['num_pred_seizures'].sum()
        false_alarms = total_predictions - detected_seizures
        
        # Get recording duration for this subject
        subject_duration_hours = subject_durations.get(subject_id, 0)
        subject_fad = false_alarms / subject_duration_hours if subject_duration_hours > 0 else 0
        
        if total_seizures > 0:
            patients_with_seizures.append(subject_id)
            detection_rate = detected_seizures / total_seizures
            
            patient_info = {
                'patient_id': subject_id,
                'total_seizures': total_seizures,
                'detected_seizures': detected_seizures,
                'detection_rate': detection_rate,
                'false_alarms': false_alarms,
                'duration_hours': subject_duration_hours,
                'fad': subject_fad
            }
            
            # Check if patient is a responder (≥2/3 seizures detected)
            if detection_rate >= 2/3:
                responders.append(patient_info)
            else:
                non_responders.append(patient_info)
    
    # Calculate responder-specific metrics
    total_seizures_responders = sum(p['total_seizures'] for p in responders)
    detected_seizures_responders = sum(p['detected_seizures'] for p in responders)
    total_false_alarms_responders = sum(p['false_alarms'] for p in responders)
    total_duration_responders = sum(p['duration_hours'] for p in responders)
    
    responder_sensitivity = (detected_seizures_responders / total_seizures_responders 
                            if total_seizures_responders > 0 else None)
    responder_fad = (total_false_alarms_responders / total_duration_responders 
                     if total_duration_responders > 0 else None)
    
    # Calculate non-responder metrics
    total_seizures_non_responders = sum(p['total_seizures'] for p in non_responders)
    detected_seizures_non_responders = sum(p['detected_seizures'] for p in non_responders)
    total_false_alarms_non_responders = sum(p['false_alarms'] for p in non_responders)
    total_duration_non_responders = sum(p['duration_hours'] for p in non_responders)
    
    non_responder_sensitivity = (detected_seizures_non_responders / total_seizures_non_responders 
                                if total_seizures_non_responders > 0 else None)
    non_responder_fad = (total_false_alarms_non_responders / total_duration_non_responders 
                         if total_duration_non_responders > 0 else None)
    
    return {
        'total_patients_with_seizures': len(patients_with_seizures),
        'num_responders': len(responders),
        'num_non_responders': len(non_responders),
        'responder_rate': len(responders) / len(patients_with_seizures) if patients_with_seizures else 0,
        'responder_sensitivity': responder_sensitivity,
        'responder_fad': responder_fad,
        'non_responder_sensitivity': non_responder_sensitivity,
        'non_responder_fad': non_responder_fad,
        'responders': responders,
        'non_responders': non_responders,
        'total_seizures_responders': total_seizures_responders,
        'detected_seizures_responders': detected_seizures_responders,
        'total_false_alarms_responders': total_false_alarms_responders,
        'total_duration_responders': total_duration_responders,
        'total_seizures_non_responders': total_seizures_non_responders,
        'detected_seizures_non_responders': detected_seizures_non_responders,
        'total_false_alarms_non_responders': total_false_alarms_non_responders,
        'total_duration_non_responders': total_duration_non_responders
    }

def is_test_subject(subject_id):
    """
    Determine if a subject belongs to the test set (sub097-sub125).
    
    Parameters:
    subject_id (str): Subject ID (e.g., "sub-077" or "sub-123")
    
    Returns:
    bool: True if subject is in test set, False otherwise
    """
    import re
    # Extract subject number from subject_id
    match = re.search(r'sub-?(\d{3})', subject_id)
    if match:
        subject_num = int(match.group(1))
        return 97 <= subject_num <= 125
    return False

def is_train_subject(subject_id):
    """
    Determine if a subject belongs to the training set (sub001-sub096).
    
    Parameters:
    subject_id (str): Subject ID (e.g., "sub-077" or "sub-123")
    
    Returns:
    bool: True if subject is in training set, False otherwise
    """
    import re
    # Extract subject number from subject_id
    match = re.search(r'sub-?(\d{3})', subject_id)
    if match:
        subject_num = int(match.group(1))
        return 1 <= subject_num <= 96
    return False

def calculate_metrics_for_dataset(df_subset, subject_durations, dataset_name):
    """
    Calculate metrics for a specific dataset (training or test).
    
    Parameters:
    df_subset (DataFrame): Subset of data (training or test)
    subject_durations (dict): Dictionary mapping subject_id to recording duration
    dataset_name (str): Name of the dataset ("Training" or "Test")
    
    Returns:
    dict: Dictionary with calculated metrics for each parameter
    """
    return calculate_metrics_for_dataset_with_output(df_subset, subject_durations, dataset_name, print)

def calculate_metrics_for_dataset_with_output(df_subset, subject_durations, dataset_name, output_func):
    """
    Calculate metrics for a specific dataset (training or test) with custom output function.
    
    Parameters:
    df_subset (DataFrame): Subset of data (training or test)
    subject_durations (dict): Dictionary mapping subject_id to recording duration
    dataset_name (str): Name of the dataset ("Training" or "Test")
    output_func (function): Function to use for output (print or custom)
    
    Returns:
    dict: Dictionary with calculated metrics for each parameter
    """
    results = {}
    parameters = df_subset['parameter'].unique()
    
    output_func(f"\n{dataset_name} Set Analysis:")
    output_func(f"Analyzing {len(df_subset)} records from {len(df_subset['subject'].unique())} subjects")
    output_func(f"Found {len(parameters)} different parameters/methods")
    output_func("-" * 60)
    
    for param in parameters:
        param_data = df_subset[df_subset['parameter'] == param]
        
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
        
        # Calculate additional metrics
        total_predictions = param_data['num_pred_seizures'].sum()
        overall_precision = total_correct_pred / total_predictions if total_predictions > 0 else 0
        
        # Calculate responder metrics
        responder_analysis = calculate_responder_metrics(param_data, subject_durations)
        
        # Store results
        results[param] = {
            'overall_sensitivity': overall_sensitivity,
            'total_correct_predictions': total_correct_pred,
            'total_seizures': total_seizures,
            'total_false_alarms': total_false_alarms,
            'total_predictions': total_predictions,
            'overall_precision': overall_precision,
            'overall_fad': overall_fad,
            'total_record_hours': total_record_hours,
            'subjects_with_duration': subjects_with_duration,
            'num_subjects': len(param_data['subject'].unique()),
            'responder_analysis': responder_analysis
        }
    
    return results

def calculate_overall_metrics_with_train_test_split(csv_file, raw_data_path="/home/swolf/asim_shared/raw_data/ds005873-1.1.0", output_txt_file=None):
    """
    Calculate metrics using train/test split approach.
    Select best parameters on training data (sub001-sub096) and evaluate on test data (sub097-sub125).
    
    Parameters:
    csv_file (str): Path to the CSV file with results
    raw_data_path (str): Path to raw SeizeIT2 dataset
    output_txt_file (str, optional): Path to save detailed analysis report as text file
    
    Returns:
    dict: Dictionary with train/test results and best parameter evaluation
    """
    
    # Prepare output capture
    output_lines = []
    def tee_print(text):
        """Print to console and capture for file output"""
        print(text)
        output_lines.append(text)
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Split into training and test sets
    train_subjects = [subj for subj in df['subject'].unique() if is_train_subject(subj)]
    test_subjects = [subj for subj in df['subject'].unique() if is_test_subject(subj)]
    
    df_train = df[df['subject'].isin(train_subjects)]
    df_test = df[df['subject'].isin(test_subjects)]
    
    tee_print(f"Original dataset: {len(df)} records from {len(df['subject'].unique())} subjects")
    tee_print(f"Training set: {len(df_train)} records from {len(train_subjects)} subjects (sub001-sub096)")
    tee_print(f"Test set: {len(df_test)} records from {len(test_subjects)} subjects (sub097-sub125)")
    
    if len(df_train) == 0:
        tee_print("WARNING: No training set subjects found in the CSV file!")
        return {}
    if len(df_test) == 0:
        tee_print("WARNING: No test set subjects found in the CSV file!")
        return {}
    
    # Load actual recording durations from raw data
    tee_print("Loading recording durations from raw data...")
    subject_durations, total_dataset_hours = load_recording_durations_from_raw_data(raw_data_path)
    
    # Phase 1: Calculate metrics on training set
    tee_print("\n" + "="*80)
    tee_print("PHASE 1: PARAMETER SELECTION ON TRAINING SET")
    tee_print("="*80)
    
    train_results = calculate_metrics_for_dataset_with_output(df_train, subject_durations, "Training", tee_print)
    
    # Phase 2: Select best parameters based on training set
    tee_print("\n" + "="*80)
    tee_print("PHASE 2: BEST PARAMETER SELECTION FROM TRAINING SET")
    tee_print("="*80)
    
    if not train_results:
        tee_print("No training results available!")
        return {}
    
    # Select best parameters from training set
    best_sensitivity_param = max(train_results.items(), key=lambda x: x[1]['overall_sensitivity'])
    best_fad_param = min(train_results.items(), key=lambda x: x[1]['overall_fad'])
    
    tee_print(f"Best Sensitivity on Training Set: {best_sensitivity_param[0]}")
    tee_print(f"  Training Sensitivity: {best_sensitivity_param[1]['overall_sensitivity']:.4f}")
    tee_print(f"  Training FAD: {best_sensitivity_param[1]['overall_fad']:.4f}")
    
    tee_print(f"\nBest FAD on Training Set: {best_fad_param[0]}")
    tee_print(f"  Training FAD: {best_fad_param[1]['overall_fad']:.4f}")
    tee_print(f"  Training Sensitivity: {best_fad_param[1]['overall_sensitivity']:.4f}")
    
    # Phase 3: Evaluate selected parameters on test set
    tee_print("\n" + "="*80)
    tee_print("PHASE 3: EVALUATION OF SELECTED PARAMETERS ON TEST SET")
    tee_print("="*80)
    
    test_results = calculate_metrics_for_dataset_with_output(df_test, subject_durations, "Test", tee_print)
    
    # Extract test performance for selected parameters
    selected_params = {best_sensitivity_param[0], best_fad_param[0]}
    
    test_evaluation = {}
    for param in selected_params:
        if param in test_results:
            test_evaluation[param] = test_results[param]
            
            tee_print(f"\nTest Set Performance for {param}:")
            tee_print(f"  Test Sensitivity: {test_results[param]['overall_sensitivity']:.4f}")
            tee_print(f"  Test FAD: {test_results[param]['overall_fad']:.4f}")
            
            resp_analysis = test_results[param]['responder_analysis']
            tee_print(f"  Test Responder Analysis:")
            tee_print(f"    Patients with seizures: {resp_analysis['total_patients_with_seizures']}")
            tee_print(f"    Responders (≥2/3 seizures detected): {resp_analysis['num_responders']}")
            tee_print(f"    Responder rate: {resp_analysis['responder_rate']:.4f} ({resp_analysis['responder_rate']*100:.2f}%)")
            if resp_analysis['responder_sensitivity'] is not None:
                tee_print(f"    Responder Sensitivity: {resp_analysis['responder_sensitivity']:.4f} ({resp_analysis['responder_sensitivity']*100:.2f}%)")
            if resp_analysis['responder_fad'] is not None:
                tee_print(f"    Responder FAD: {resp_analysis['responder_fad']:.4f}")
            if resp_analysis['non_responder_sensitivity'] is not None:
                tee_print(f"    Non-responder Sensitivity: {resp_analysis['non_responder_sensitivity']:.4f} ({resp_analysis['non_responder_sensitivity']*100:.2f}%)")
            if resp_analysis['non_responder_fad'] is not None:
                tee_print(f"    Non-responder FAD: {resp_analysis['non_responder_fad']:.4f}")
        else:
            tee_print(f"\nWARNING: Parameter {param} not found in test set!")
    
    # Save output to text file if specified
    if output_txt_file:
        from datetime import datetime
        import os
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_txt_file)
        if output_dir:  # Only create directory if there is a directory path
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_txt_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("JEPPESEN SEIZURE DETECTION ANALYSIS REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input CSV file: {csv_file}\n")
            f.write(f"Raw data path: {raw_data_path}\n")
            f.write("="*80 + "\n\n")
            
            for line in output_lines:
                f.write(line + "\n")
        
        tee_print(f"\nDetailed analysis report saved to: {output_txt_file}")
    
    return {
        'train_results': train_results,
        'test_results': test_results,
        'best_sensitivity_param': best_sensitivity_param[0],
        'best_fad_param': best_fad_param[0],
        'train_best_sensitivity': best_sensitivity_param[1],
        'train_best_fad': best_fad_param[1],
        'test_evaluation': test_evaluation,
        'train_subjects': train_subjects,
        'test_subjects': test_subjects
    }

def find_best_parameters_legacy(results):
    """
    Legacy function for backwards compatibility - not used with train/test split.
    """
    pass

def save_train_test_summary(train_test_results, output_file):
    """
    Save train/test split results to a CSV file.
    """
    summary_data = []
    
    # Add training results
    for param, metrics in train_test_results['train_results'].items():
        resp_analysis = metrics['responder_analysis']
        summary_data.append({
            'dataset': 'training',
            'parameter': param,
            'overall_sensitivity': metrics['overall_sensitivity'],
            'overall_precision': metrics['overall_precision'],
            'overall_fad': metrics['overall_fad'],
            'total_correct_predictions': metrics['total_correct_predictions'],
            'total_seizures': metrics['total_seizures'],
            'total_false_alarms': metrics['total_false_alarms'],
            'total_record_hours': metrics['total_record_hours'],
            'num_subjects': metrics['num_subjects'],
            'num_responders': resp_analysis['num_responders'],
            'responder_rate': resp_analysis['responder_rate'],
            'responder_sensitivity': resp_analysis['responder_sensitivity'],
            'responder_fad': resp_analysis['responder_fad'],
            'non_responder_sensitivity': resp_analysis['non_responder_sensitivity'],
            'non_responder_fad': resp_analysis['non_responder_fad']
        })
    
    # Add test results for selected parameters
    for param, metrics in train_test_results['test_evaluation'].items():
        resp_analysis = metrics['responder_analysis']
        summary_data.append({
            'dataset': 'test',
            'parameter': param,
            'overall_sensitivity': metrics['overall_sensitivity'],
            'overall_precision': metrics['overall_precision'],
            'overall_fad': metrics['overall_fad'],
            'total_correct_predictions': metrics['total_correct_predictions'],
            'total_seizures': metrics['total_seizures'],
            'total_false_alarms': metrics['total_false_alarms'],
            'total_record_hours': metrics['total_record_hours'],
            'num_subjects': metrics['num_subjects'],
            'num_responders': resp_analysis['num_responders'],
            'responder_rate': resp_analysis['responder_rate'],
            'responder_sensitivity': resp_analysis['responder_sensitivity'],
            'responder_fad': resp_analysis['responder_fad'],
            'non_responder_sensitivity': resp_analysis['non_responder_sensitivity'],
            'non_responder_fad': resp_analysis['non_responder_fad']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_file, index=False)
    print(f"\nTrain/Test summary saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze Jeppesen seizure detection results with train/test split')
    parser.add_argument('csv_file', help='Path to the results CSV file')
    parser.add_argument('--output', '-o', help='Output CSV file for summary (optional)')
    parser.add_argument('--output-txt', help='Output TXT file for detailed analysis report (optional)')
    parser.add_argument('--raw-data-path', default='/home/swolf/asim_shared/raw_data/ds005873-1.1.0',
                       help='Path to raw SeizeIT2 dataset (default: /home/swolf/asim_shared/raw_data/ds005873-1.1.0)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: File {args.csv_file} not found!")
        return
    
    # Generate default output file names if not specified
    csv_base = os.path.splitext(args.csv_file)[0]
    
    # Set default output files based on input filename
    csv_output = args.output if args.output else f"{csv_base}_train_test_summary.csv"
    txt_output = args.output_txt if args.output_txt else f"{csv_base}_analysis_report.txt"
    
    # Calculate metrics using train/test split approach
    results = calculate_overall_metrics_with_train_test_split(
        args.csv_file, 
        args.raw_data_path, 
        output_txt_file=txt_output
    )
    
    if not results:
        print("No results to analyze.")
        return
    
    # Save CSV summary 
    save_train_test_summary(results, csv_output)
    
    print(f"\nAnalysis complete! ")
    print(f"Training parameters analyzed: {len(results['train_results'])}")
    print(f"Selected parameters evaluated on test set: {len(results['test_evaluation'])}")
    print(f"Best sensitivity parameter: {results['best_sensitivity_param']}")
    print(f"Best FAD parameter: {results['best_fad_param']}")
    print(f"\nOutput files:")
    print(f"  CSV summary: {csv_output}")
    print(f"  TXT report: {txt_output}")

if __name__ == "__main__":
    main()