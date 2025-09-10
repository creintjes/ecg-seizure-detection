#!/usr/bin/env python3
"""
Batch evaluation script for multiple joint anomaly score files.
Runs evaluation across different datasets and threshold scales.
"""

import os
import re
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import glob
import argparse
import pyarrow.feather as feather

# Import the evaluation functions
from eval import evaluate_joint_anomaly_score, intervals_from_mask, event_metrics_for_clusters
from clustering import analyze_detection_mask


def extract_dataset_id_from_filename(filename: str) -> Optional[int]:
    """Extract dataset ID from filename like '098_no_window-joint_anomaly_score.pkl' or '098_window-joint_anomaly_score.pkl'"""
    match = re.search(r'(\d+)_(?:no_)?window-joint_anomaly_score\.pkl', filename)
    if match:
        return int(match.group(1))
    return None


def find_joint_anomaly_files(results_dir: str, windowed: Optional[bool] = None) -> List[str]:
    """Find all joint anomaly score pickle files in the results directory.
    
    Args:
        results_dir: Directory to search in
        windowed: If True, only windowed files; if False, only no_window files; if None, all files
    """
    if windowed is True:
        pattern = os.path.join(results_dir, "*_window-joint_anomaly_score.pkl")
    elif windowed is False:
        pattern = os.path.join(results_dir, "*_no_window-joint_anomaly_score.pkl")
    else:
        # Find both types
        pattern1 = os.path.join(results_dir, "*_window-joint_anomaly_score.pkl")
        pattern2 = os.path.join(results_dir, "*_no_window-joint_anomaly_score.pkl")
        files = glob.glob(pattern1) + glob.glob(pattern2)
        return sorted(files)
    
    files = glob.glob(pattern)
    return sorted(files)


def load_existing_results(feather_path: str) -> Optional[pd.DataFrame]:
    """Load existing results from feather file if it exists."""
    if os.path.exists(feather_path):
        try:
            df = feather.read_feather(feather_path)
            print(f"Loaded existing results: {len(df)} evaluations from {feather_path}")
            return df
        except Exception as e:
            print(f"Warning: Could not load existing results from {feather_path}: {e}")
            return None
    return None


def get_completed_datasets(df: Optional[pd.DataFrame], threshold_scales: List[float]) -> Set[int]:
    """Get set of dataset IDs that have been completed for all threshold scales."""
    if df is None or len(df) == 0:
        return set()
    
    # Group by dataset_id and check if all threshold scales are present
    completed = set()
    for dataset_id in df['dataset_id'].unique():
        dataset_scales = set(df[df['dataset_id'] == dataset_id]['threshold_scale'].unique())
        if set(threshold_scales).issubset(dataset_scales):
            completed.add(dataset_id)
    
    return completed


def append_to_feather(new_results: List[Dict], feather_path: str):
    """Append new results to feather file."""
    if not new_results:
        return
    
    new_df = pd.DataFrame(new_results)
    
    # Load existing data if file exists
    if os.path.exists(feather_path):
        try:
            existing_df = feather.read_feather(feather_path)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        except Exception as e:
            print(f"Warning: Could not load existing feather file: {e}")
            combined_df = new_df
    else:
        combined_df = new_df
    
    # Save to feather file
    try:
        feather.write_feather(combined_df, feather_path)
        print(f"Appended {len(new_results)} new results to {feather_path}")
    except Exception as e:
        print(f"Error saving to feather file: {e}")
        # Fallback to CSV
        csv_path = feather_path.replace('.feather', '.csv')
        combined_df.to_csv(csv_path, index=False)
        print(f"Saved to CSV instead: {csv_path}")


def save_results(results: Dict[str, Any], output_path: str):
    """Save results to pickle file."""
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to: {output_path}")


def batch_evaluate_joint_anomaly_scores(
    results_dir: str = "/home/mballo_sw/Repositories/ecg-seizure-detection/TimeVQVAE-AD/evaluation/results/final/window/train",
    threshold_scales: List[float] = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2],
    sampling_rate: float = 8.0,
    output_dir: str = "batch_results",
    run_clustering: bool = False,
    windowed: Optional[bool] = None,
    verbose: bool = False,
    fixed_strategy_config: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run batch evaluation on all joint anomaly score files.
    Uses incremental storage with feather files for resumability.
    
    Args:
        results_dir: Directory containing joint anomaly score files
        threshold_scales: List of threshold scale factors to test
        sampling_rate: Sampling rate in Hz
        output_dir: Directory to save batch results
        run_clustering: Whether to run clustering analysis
        windowed: If True, only windowed files; if False, only no_window files; if None, all files
        verbose: Whether to print detailed output for each evaluation
        fixed_strategy_config: Path to JSON file with fixed clustering strategies per threshold
        
    Returns:
        Dictionary containing all evaluation results
    """
    # Load fixed strategy configuration if provided
    fixed_strategies = None
    if fixed_strategy_config:
        import json
        print(f"Loading fixed strategy configuration from: {fixed_strategy_config}")
        try:
            with open(fixed_strategy_config, 'r') as f:
                config_data = json.load(f)
                fixed_strategies = config_data.get('strategies', {})
                print(f"Loaded fixed strategies for {len(fixed_strategies)} thresholds")
                if verbose:
                    for thresh, strategy in fixed_strategies.items():
                        print(f"  Threshold {thresh}: {strategy}")
        except Exception as e:
            print(f"Warning: Could not load fixed strategy config: {e}")
            fixed_strategies = None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up feather file path for incremental storage
    feather_path = os.path.join(output_dir, "evaluation_results.feather")
    
    # Load existing results if available
    existing_df = load_existing_results(feather_path)
    completed_datasets = get_completed_datasets(existing_df, threshold_scales)
    
    if completed_datasets:
        print(f"Found {len(completed_datasets)} already completed datasets: {sorted(completed_datasets)}")
    
    # Find all joint anomaly score files
    joint_files = find_joint_anomaly_files(results_dir, windowed=windowed)
    if not joint_files:
        raise ValueError(f"No joint anomaly score files found in {results_dir}")
    
    print(f"Found {len(joint_files)} joint anomaly score files:")
    for f in joint_files:
        dataset_id = extract_dataset_id_from_filename(os.path.basename(f))
        status = "COMPLETED" if dataset_id in completed_datasets else "PENDING"
        print(f"  {os.path.basename(f)} - {status}")
    
    # Filter out already completed files
    pending_files = []
    for joint_file in joint_files:
        dataset_id = extract_dataset_id_from_filename(os.path.basename(joint_file))
        if dataset_id not in completed_datasets:
            pending_files.append(joint_file)
    
    if not pending_files:
        print("All datasets have been completed!")
        # Return existing results
        if existing_df is not None:
            return {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'results_dir': results_dir,
                    'threshold_scales': threshold_scales,
                    'sampling_rate': sampling_rate,
                    'windowed': windowed,
                    'files_processed': [],
                    'run_clustering': run_clustering,
                    'resume_mode': True
                },
                'summary_metrics': existing_df.to_dict('records'),
                'evaluations': {},
                'clustering_results': {} if run_clustering else None
            }
    
    print(f"\nProcessing {len(pending_files)} pending datasets...")
    
    # Initialize results storage
    batch_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'results_dir': results_dir,
            'threshold_scales': threshold_scales,
            'sampling_rate': sampling_rate,
            'windowed': windowed,
            'files_processed': [],
            'run_clustering': run_clustering,
            'resume_mode': len(completed_datasets) > 0
        },
        'evaluations': {},
        'summary_metrics': [],
        'clustering_results': {} if run_clustering else None
    }
    
    # Global efficiency tracking
    total_possible_evaluations = 0
    total_actual_evaluations = 0
    total_skipped_evaluations = 0
    
    # Process each pending file
    for joint_file in pending_files:
        filename = os.path.basename(joint_file)
        dataset_id = extract_dataset_id_from_filename(filename)
        
        if dataset_id is None:
            print(f"Warning: Could not extract dataset ID from {filename}, skipping...")
            continue
            
        print(f"\n{'='*80}")
        print(f"Processing Dataset {dataset_id:03d} ({filename})")
        print(f"{'='*80}")
        
        batch_results['metadata']['files_processed'].append(filename)
        batch_results['evaluations'][dataset_id] = {}
        
        if run_clustering:
            batch_results['clustering_results'][dataset_id] = {}
        
        # Store results for this dataset to batch append later
        dataset_results = []
        
        # Test each threshold scale
        skipped_thresholds = 0
        total_thresholds = len(threshold_scales)
        total_possible_evaluations += total_thresholds
        
        for scale in threshold_scales:
            print(f"\n--- Threshold Scale: {scale} ---")
            
            try:
                # Run evaluation
                results, strategy_data, truth = evaluate_joint_anomaly_score(
                    dataset_index=dataset_id,
                    sampling_rate=sampling_rate,
                    results_path=joint_file,
                    best_freq_threshold_scale=scale,
                    verbose=verbose
                )
                # Store basic results
                batch_results['evaluations'][dataset_id][scale] = {
                    'basic_metrics': results,
                    'n_samples': len(truth),
                    'n_true_anomalies': int(truth.sum()),
                    'anomaly_percentage': (truth.sum() / len(truth)) * 100,
                    'threshold_scale': scale
                }
                
                # Calculate responder status based on sensitivity >2/3 (for datasets with seizures)
                has_seizures = int(truth.sum()) > 0
                is_responder_pre = has_seizures and results.get('sensitivity', 0) > (2/3)
                
                # Create summary row for this evaluation
                summary_row = {
                    'dataset_id': dataset_id,
                    'threshold_scale': round(scale, 2),  # Round to 2 decimal places
                    'n_samples': len(truth),
                    'n_true_anomalies': int(truth.sum()),
                    'anomaly_percentage': (truth.sum() / len(truth)) * 100,
                    'has_seizures': has_seizures,
                    'is_responder_pre': is_responder_pre,
                    'is_responder': is_responder_pre,  # Default to pre-clustering for analysis compatibility
                    'tp_events': results.get('TP_events', 0),
                    'fn_events': results.get('FN_events', 0),
                    'fp_events': results.get('FP_events', 0),
                    'event_recall': results.get('sensitivity', 0),
                    'event_sensitivity': results.get('sensitivity', 0),  # Same as recall for events
                    'event_iou': results.get('event_iou', 0),
                    'false_alarm_rate_per_hour': results.get('false_alarm_rate_per_hour', 0),
                    'predicted_samples': int(strategy_data['best_freq']['mask'].sum()),
                    'predicted_percentage': (strategy_data['best_freq']['mask'].sum() / len(truth)) * 100,
                    'processing_timestamp': datetime.now().isoformat()
                }
                
                dataset_results.append(summary_row)
                batch_results['summary_metrics'].append(summary_row)
                
                # Check for early termination optimization
                # If no anomalies are predicted, higher thresholds will also have 0 predictions
                n_predicted = int(strategy_data['best_freq']['mask'].sum())
                if n_predicted == 0:
                    print(f"  ⚡ OPTIMIZATION: 0 anomalies predicted at scale {scale}. Skipping higher thresholds (they will also be 0).")
                    
                    # Fill in remaining thresholds with zeros for completeness
                    remaining_scales = [s for s in threshold_scales if s > scale]
                    for remaining_scale in remaining_scales:
                        zero_summary_row = {
                            'dataset_id': dataset_id,
                            'threshold_scale': round(remaining_scale, 2),
                            'n_samples': len(truth),
                            'n_true_anomalies': int(truth.sum()),
                            'anomaly_percentage': (truth.sum() / len(truth)) * 100,
                            'has_seizures': has_seizures,
                            'is_responder_pre': False,  # Can't be responder with 0 predictions
                            'is_responder': False,
                            'tp_events': 0,
                            'fn_events': len(results.get('truth_events', [])),  # All real events are missed
                            'fp_events': 0,
                            'event_recall': 0.0,
                            'event_sensitivity': 0.0,
                            'event_iou': 0.0,
                            'false_alarm_rate_per_hour': 0.0,
                            'predicted_samples': 0,
                            'predicted_percentage': 0.0,
                            'processing_timestamp': datetime.now().isoformat()
                        }
                        
                        # Add zero results to storage
                        dataset_results.append(zero_summary_row)
                        batch_results['summary_metrics'].append(zero_summary_row)
                        
                        # Add basic results storage for consistency
                        batch_results['evaluations'][dataset_id][remaining_scale] = {
                            'basic_metrics': {
                                'truth_events': results.get('truth_events', []),
                                'pred_events': [],
                                'TP_events': 0,
                                'FN_events': len(results.get('truth_events', [])),
                                'FP_events': 0,
                                'sensitivity': 0.0,
                                'false_alarm_rate_per_hour': 0.0
                            },
                            'n_samples': len(truth),
                            'n_true_anomalies': int(truth.sum()),
                            'anomaly_percentage': (truth.sum() / len(truth)) * 100,
                            'threshold_scale': remaining_scale
                        }
                    
                    print(f"  📈 Filled {len(remaining_scales)} higher thresholds with zero predictions.")
                    skipped_thresholds = len(remaining_scales)
                    break  # Exit the threshold loop early
                
                # Handle clustering if requested (keeping existing logic)
                if run_clustering and np.any(strategy_data['best_freq']['mask']):
                    print(f"Running clustering for dataset {dataset_id}, scale {scale}...")
                    
                    mask = strategy_data['best_freq']['mask']
                    scores = strategy_data['best_freq']['scores']
                    gt_intervals = intervals_from_mask(truth, sampling_rate)
                    
                    # Calculate ground truth statistics for clustering
                    true_anomaly_samples = int(truth.sum())
                    true_anomaly_events = len(gt_intervals)
                    
                    try:
                        # Check if we should use fixed strategy mode
                        if fixed_strategies and str(scale) in fixed_strategies:
                            # Use fixed strategy mode
                            fixed_strategy_name = fixed_strategies[str(scale)]
                            print(f"  Using fixed strategy: {fixed_strategy_name}")
                            
                            clustering_results = analyze_detection_mask(
                                mask=mask,
                                fs=sampling_rate,
                                file_id=f"rec_{dataset_id:03d}",
                                scores=scores,
                                subject_id=f"subj_{dataset_id:03d}",
                                gt_intervals=gt_intervals,
                                fixed_strategy=fixed_strategy_name,
                                true_anomaly_samples=true_anomaly_samples,
                                true_anomaly_events=true_anomaly_events
                                # output_folder=os.path.join(output_dir, f"rec_{dataset_id:03d}_scale_{scale}_fixed_clustered")  # Disabled to avoid file pollution
                            )
                        else:
                            # Use adaptive strategy mode (original behavior)
                            clustering_results = analyze_detection_mask(
                                mask=mask,
                                fs=sampling_rate,
                                file_id=f"rec_{dataset_id:03d}",
                                scores=scores,
                                subject_id=f"subj_{dataset_id:03d}",
                                gt_intervals=gt_intervals,
                                true_anomaly_samples=true_anomaly_samples,
                                true_anomaly_events=true_anomaly_events
                                # output_folder=os.path.join(output_dir, f"rec_{dataset_id:03d}_scale_{scale}_clustered")  # Disabled to avoid file pollution
                            )
                        
                        # Extract key clustering metrics
                        best = clustering_results["best_results"]
                        
                        # Event-level metrics on clustered output
                        ev = event_metrics_for_clusters(
                            best["representatives"], gt_intervals, len(truth), sampling_rate
                        )
                        
                        batch_results['clustering_results'][dataset_id][scale] = {
                            'best_strategy': clustering_results["best_strategy"],
                            'n_clusters': best["clusters"],
                            'n_representatives': len(best["representatives"]),
                            'event_recall': ev['event_recall'],
                            'event_iou': ev['event_iou'],
                            'event_far_per_hour': ev['event_far_per_hour'],
                            'event_tp': ev['TP_events'],
                            'event_fn': ev['FN_events'],
                            'event_fp': ev['FP_events'],
                            'n_pred_events': ev['n_pred_events']
                        }
                        
                        # Calculate post-clustering anomaly coverage percentage
                        clustered_anomaly_duration = 0.0
                        for rep in best["representatives"]:
                            cluster_start = rep.get('cluster_start_seconds', rep['location_time_seconds'])
                            cluster_end = rep.get('cluster_end_seconds', rep['location_time_seconds'])
                            clustered_anomaly_duration += (cluster_end - cluster_start)
                        
                        clustered_anomaly_samples = clustered_anomaly_duration * sampling_rate
                        clustered_anomaly_percentage = (clustered_anomaly_samples / len(truth)) * 100
                        
                        # Calculate post-clustering responder status
                        is_responder_post = has_seizures and ev['event_recall'] > (2/3)
                        
                        # Add clustering metrics to summary
                        summary_row.update({
                            'clustering_strategy': clustering_results["best_strategy"],
                            'n_clusters': best["clusters"],
                            'n_representatives': len(best["representatives"]),
                            'cluster_event_recall': ev['event_recall'],
                            'cluster_event_iou': ev['event_iou'],
                            'clustered_samples': int(clustered_anomaly_samples),
                            'clustered_percentage': clustered_anomaly_percentage,
                            'cluster_event_far_per_hour': ev['event_far_per_hour'],
                            'is_responder_post': is_responder_post,
                            'is_responder': is_responder_post  # Update to post-clustering for analysis
                        })
                        
                        if verbose:
                            print(f"  Clustering: {best['clusters']} clusters, {len(best['representatives'])} representatives")
                            print(f"  Event-level: Recall={ev['event_recall']:.3f}, IoU={ev['event_iou']:.3f}, FAR/hr={ev['event_far_per_hour']:.2f}")
                        
                        # Always print key clustering info for debugging
                        n_original_events = results.get('TP_events', 0) + results.get('FP_events', 0)
                        n_clustered_events = ev['TP_events'] + ev['FP_events']
                        print(f"  CLUSTERING DEBUG: {n_original_events} events → {n_clustered_events} events (reduction: {n_original_events - n_clustered_events})")
                        print(f"  Strategy: {clustering_results['best_strategy']}")
                        
                    except Exception as e:
                        print(f"  Warning: Clustering failed for dataset {dataset_id}, scale {scale}: {e}")
                        batch_results['clustering_results'][dataset_id][scale] = {'error': str(e)}
                        
                elif run_clustering:
                    print(f"  Skipping clustering (no positive predictions)")
                    batch_results['clustering_results'][dataset_id][scale] = {'skipped': 'no_positive_predictions'}
                
                # Print metrics comparison
                print(f"  Basic metrics (BEFORE clustering): Recall={results.get('sensitivity', 0):.3f}, "
                      f"IoU={results.get('event_iou', 0):.3f}, "
                      f"FAR/hr={results.get('false_alarm_rate_per_hour', 0):.2f}")
                
                # Print clustered metrics if clustering was run
                if run_clustering and np.any(strategy_data['best_freq']['mask']) and 'error' not in batch_results['clustering_results'][dataset_id][scale]:
                    ev = batch_results['clustering_results'][dataset_id][scale]
                    print(f"  Clustered metrics (AFTER clustering): Recall={ev['event_recall']:.3f}, "
                          f"IoU={ev['event_iou']:.3f}, "
                          f"FAR/hr={ev['event_far_per_hour']:.2f}")
                    
                    # Detailed comparison
                    far_reduction = results.get('false_alarm_rate_per_hour', 0) - ev['event_far_per_hour']
                    far_reduction_pct = (far_reduction / results.get('false_alarm_rate_per_hour', 0)) * 100 if results.get('false_alarm_rate_per_hour', 0) > 0 else 0
                    print(f"  FAR reduction: {far_reduction:.2f}/hr ({far_reduction_pct:.1f}%)")
                
            except Exception as e:
                print(f"Error processing dataset {dataset_id} with scale {scale}: {e}")
                batch_results['evaluations'][dataset_id][scale] = {'error': str(e)}
                continue
        
        # Print efficiency summary for this dataset
        evaluated_thresholds = total_thresholds - skipped_thresholds
        efficiency_pct = (skipped_thresholds / total_thresholds) * 100 if total_thresholds > 0 else 0
        
        # Update global counters
        total_actual_evaluations += evaluated_thresholds
        total_skipped_evaluations += skipped_thresholds
        
        print(f"\n🚀 DATASET {dataset_id:03d} EFFICIENCY SUMMARY:")
        print(f"   Total thresholds: {total_thresholds}")
        print(f"   Evaluated: {evaluated_thresholds}")
        print(f"   Skipped (optimization): {skipped_thresholds}")
        print(f"   Efficiency gain: {efficiency_pct:.1f}% time saved")
        
        # Append this dataset's results to feather file
        if dataset_results:
            append_to_feather(dataset_results, feather_path)
            print(f"Dataset {dataset_id:03d} completed and saved to feather file")
    
    # Print global efficiency summary
    global_efficiency_pct = (total_skipped_evaluations / total_possible_evaluations) * 100 if total_possible_evaluations > 0 else 0
    print(f"\n{'='*80}")
    print(f"🎯 GLOBAL BATCH EFFICIENCY SUMMARY")
    print(f"{'='*80}")
    print(f"Total datasets processed: {len(pending_files)}")
    print(f"Total possible threshold evaluations: {total_possible_evaluations:,}")
    print(f"Actual evaluations performed: {total_actual_evaluations:,}")
    print(f"Evaluations skipped (optimization): {total_skipped_evaluations:,}")
    print(f"Overall efficiency gain: {global_efficiency_pct:.1f}% computation time saved")
    print(f"Evaluation compression ratio: {(total_actual_evaluations / total_possible_evaluations):.3f}")
    
    # Add efficiency metrics to metadata
    batch_results['metadata'].update({
        'efficiency_stats': {
            'total_possible_evaluations': total_possible_evaluations,
            'total_actual_evaluations': total_actual_evaluations,
            'total_skipped_evaluations': total_skipped_evaluations,
            'efficiency_percentage': global_efficiency_pct,
            'compression_ratio': total_actual_evaluations / total_possible_evaluations if total_possible_evaluations > 0 else 0
        }
    })
    
    return batch_results


def save_summary_csv(batch_results: Dict[str, Any], output_path: str, feather_path: str = None):
    """Save summary metrics as CSV for easy analysis."""
    # Try to get the most complete dataset from feather file first
    df = None
    if feather_path and os.path.exists(feather_path):
        try:
            df = feather.read_feather(feather_path)
            print(f"Using complete results from feather file: {len(df)} evaluations")
        except Exception as e:
            print(f"Warning: Could not load feather file: {e}")
    
    # Fallback to batch_results if feather file not available
    if df is None and 'summary_metrics' in batch_results and batch_results['summary_metrics']:
        df = pd.DataFrame(batch_results['summary_metrics'])
    
    if df is not None and len(df) > 0:
        df.to_csv(output_path, index=False)
        print(f"Summary CSV saved to: {output_path}")
        
        # Print basic statistics
        print(f"\nSummary Statistics:")
        print(f"Datasets processed: {df['dataset_id'].nunique()}")
        print(f"Threshold scales tested: {df['threshold_scale'].nunique()}")
        print(f"Total evaluations: {len(df)}")
        
        # Responder analysis
        responders = df[df['is_responder'] == True]['dataset_id'].nunique()
        non_responders = df[df['is_responder'] == False]['dataset_id'].nunique()
        total_datasets = df['dataset_id'].nunique()
        responder_ratio = responders / total_datasets if total_datasets > 0 else 0
        
        print(f"\nResponder Analysis:")
        print(f"Total datasets: {total_datasets}")
        print(f"Responders (with seizures): {responders}")
        print(f"Non-responders (no seizures): {non_responders}")
        print(f"Responder ratio: {responder_ratio:.3f}")
        
        # Best results per dataset
        print(f"\nBest recall per dataset:")
        best_recall = df.loc[df.groupby('dataset_id')['event_recall'].idxmax()]
        for _, row in best_recall.iterrows():
            responder_status = "responder" if row['is_responder'] else "non-responder"
            print(f"  Dataset {row['dataset_id']:03d} ({responder_status}): Recall={row['event_recall']:.3f} (scale={row['threshold_scale']})")
    else:
        print("No summary metrics available to save as CSV")


def compute_threshold_summary_stats(batch_results: Dict[str, Any], output_dir: str, feather_path: str = None):
    """Compute and save summary statistics grouped by threshold scale."""
    # Try to get the most complete dataset from feather file first
    df = None
    if feather_path and os.path.exists(feather_path):
        try:
            df = feather.read_feather(feather_path)
            print(f"Using complete results from feather file for threshold analysis: {len(df)} evaluations")
        except Exception as e:
            print(f"Warning: Could not load feather file for threshold analysis: {e}")
    
    # Fallback to batch_results if feather file not available
    if df is None and 'summary_metrics' in batch_results and batch_results['summary_metrics']:
        df = pd.DataFrame(batch_results['summary_metrics'])
    
    if df is None or len(df) == 0:
        print("No summary metrics available for threshold analysis")
        return
    
    # Group by threshold scale and compute statistics
    threshold_stats = []
    
    for scale in sorted(df['threshold_scale'].unique()):
        scale_data = df[df['threshold_scale'] == scale]
        responder_data = scale_data[scale_data['is_responder'] == True]
        non_responder_data = scale_data[scale_data['is_responder'] == False]
        
        # Overall statistics
        overall_stats = {
            'threshold_scale': scale,
            'n_datasets': len(scale_data),
            'n_responders': len(responder_data),
            'n_non_responders': len(non_responder_data),
            'responder_ratio': len(responder_data) / len(scale_data) if len(scale_data) > 0 else 0,
            
            # Anomaly percentage metrics
            'mean_real_anomaly_percentage': scale_data['anomaly_percentage'].mean(),
            'std_real_anomaly_percentage': scale_data['anomaly_percentage'].std(),
            'mean_predicted_percentage_before': scale_data['predicted_percentage'].mean(),
            'std_predicted_percentage_before': scale_data['predicted_percentage'].std(),
            
            # Post-clustering percentages (if clustering was performed)
            'mean_predicted_percentage_after': scale_data['clustered_percentage'].mean() if 'clustered_percentage' in scale_data.columns else None,
            'std_predicted_percentage_after': scale_data['clustered_percentage'].std() if 'clustered_percentage' in scale_data.columns else None,
            
            # Overall metrics
            'mean_sensitivity_all': scale_data['event_sensitivity'].mean(),
            'std_sensitivity_all': scale_data['event_sensitivity'].std(),
            'mean_far_per_hour_all': scale_data['false_alarm_rate_per_hour'].mean(),
            'std_far_per_hour_all': scale_data['false_alarm_rate_per_hour'].std(),
            
            # Responder-only metrics
            'mean_sensitivity_responders': responder_data['event_sensitivity'].mean() if len(responder_data) > 0 else 0,
            'std_sensitivity_responders': responder_data['event_sensitivity'].std() if len(responder_data) > 0 else 0,
            'mean_far_per_hour_responders': responder_data['false_alarm_rate_per_hour'].mean() if len(responder_data) > 0 else 0,
            'std_far_per_hour_responders': responder_data['false_alarm_rate_per_hour'].std() if len(responder_data) > 0 else 0,
            
            # Responder-only anomaly percentages
            'mean_predicted_percentage_before_responders': responder_data['predicted_percentage'].mean() if len(responder_data) > 0 else 0,
            'mean_predicted_percentage_after_responders': responder_data['clustered_percentage'].mean() if len(responder_data) > 0 and 'clustered_percentage' in responder_data.columns else None,
            
            # Non-responder metrics (only FAR and percentages make sense)
            'mean_far_per_hour_non_responders': non_responder_data['false_alarm_rate_per_hour'].mean() if len(non_responder_data) > 0 else 0,
            'std_far_per_hour_non_responders': non_responder_data['false_alarm_rate_per_hour'].std() if len(non_responder_data) > 0 else 0,
            'mean_predicted_percentage_before_non_responders': non_responder_data['predicted_percentage'].mean() if len(non_responder_data) > 0 else 0,
            'mean_predicted_percentage_after_non_responders': non_responder_data['clustered_percentage'].mean() if len(non_responder_data) > 0 and 'clustered_percentage' in non_responder_data.columns else None,
        }
        threshold_stats.append(overall_stats)
    
    # Save threshold summary
    threshold_df = pd.DataFrame(threshold_stats)
    threshold_csv_path = os.path.join(output_dir, "threshold_summary_stats.csv")
    threshold_df.to_csv(threshold_csv_path, index=False)
    print(f"Threshold summary statistics saved to: {threshold_csv_path}")
    
    # Print summary table
    print(f"\n{'='*160}")
    print(f"THRESHOLD SUMMARY STATISTICS")
    print(f"{'='*160}")
    print(f"{'Scale':<6} {'N_Data':<7} {'N_Resp':<7} {'Resp%':<6} {'Real%':<6} {'Pred%_Before':<12} {'Pred%_After':<11} {'Sens_All':<9} {'FAR_All':<8} {'Sens_Resp':<10} {'FAR_Resp':<9}")
    print(f"{'-'*160}")
    
    for _, row in threshold_df.iterrows():
        # Handle None values for clustering percentages
        pred_after_str = f"{row['mean_predicted_percentage_after']:.2f}" if row['mean_predicted_percentage_after'] is not None else "N/A"
        
        print(f"{row['threshold_scale']:<6.2f} {row['n_datasets']:<7.0f} {row['n_responders']:<7.0f} "
              f"{row['responder_ratio']*100:<6.1f} {row['mean_real_anomaly_percentage']:<6.2f} "
              f"{row['mean_predicted_percentage_before']:<12.2f} {pred_after_str:<11} "
              f"{row['mean_sensitivity_all']:<9.3f} {row['mean_far_per_hour_all']:<8.2f} "
              f"{row['mean_sensitivity_responders']:<10.3f} {row['mean_far_per_hour_responders']:<9.2f}")
    
    # Print additional detailed breakdown if clustering data is available
    if threshold_df['mean_predicted_percentage_after'].notna().any():
        print(f"\n{'='*140}")
        print(f"ANOMALY COVERAGE BREAKDOWN BY THRESHOLD")
        print(f"{'='*140}")
        print(f"{'Scale':<6} {'Real%':<8} {'Pred%_Before':<13} {'Pred%_After':<12} {'Reduction%':<11} {'Resp_Before%':<12} {'Resp_After%':<11} {'NonResp_Before%':<15}")
        print(f"{'-'*140}")
        
        for _, row in threshold_df.iterrows():
            # Calculate reduction percentage
            before_pct = row['mean_predicted_percentage_before']
            after_pct = row['mean_predicted_percentage_after']
            reduction_pct = ((before_pct - after_pct) / before_pct * 100) if (after_pct is not None and before_pct > 0) else 0
            
            # Handle None values
            after_str = f"{after_pct:.2f}" if after_pct is not None else "N/A"
            reduction_str = f"{reduction_pct:.1f}" if after_pct is not None else "N/A"
            resp_after_str = f"{row['mean_predicted_percentage_after_responders']:.2f}" if row['mean_predicted_percentage_after_responders'] is not None else "N/A"
            
            print(f"{row['threshold_scale']:<6.2f} {row['mean_real_anomaly_percentage']:<8.2f} "
                  f"{row['mean_predicted_percentage_before']:<13.2f} {after_str:<12} "
                  f"{reduction_str:<11} {row['mean_predicted_percentage_before_responders']:<12.2f} "
                  f"{resp_after_str:<11} {row['mean_predicted_percentage_before_non_responders']:<15.2f}")
        
        print(f"\n{'='*80}")
        print(f"LEGEND:")
        print(f"{'='*80}")
        print(f"Real%          : Actual anomaly percentage in ground truth")
        print(f"Pred%_Before   : Percentage detected as anomalies (before clustering)")  
        print(f"Pred%_After    : Percentage detected as anomalies (after clustering)")
        print(f"Reduction%     : Percentage reduction in anomaly coverage after clustering")
        print(f"Resp_Before%   : Predicted percentage for responder datasets (before clustering)")
        print(f"Resp_After%    : Predicted percentage for responder datasets (after clustering)")
        print(f"NonResp_Before%: Predicted percentage for non-responder datasets (before clustering)")
        print(f"Sens_All       : Overall event sensitivity")
        print(f"FAR_All        : Overall false alarm rate per hour")
        print(f"Sens_Resp      : Event sensitivity for responder datasets only")
        print(f"FAR_Resp       : False alarm rate per hour for responder datasets only")
    
    return threshold_df


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation of joint anomaly scores")
    parser.add_argument("--results-dir", default="/home/mballo_sw/Repositories/ecg-seizure-detection/TimeVQVAE-AD/evaluation/results/final/window/test",
                       help="Directory containing joint anomaly score files")
    parser.add_argument("--output-dir", default="batch_results",
                       help="Directory to save batch results (relative to results-dir if not absolute path)")
    parser.add_argument("--sampling-rate", type=float, default=8.0,
                       help="Sampling rate in Hz")
    parser.add_argument("--windowed", action="store_true",
                       help="Process only windowed model results")
    parser.add_argument("--no-window", action="store_true",
                       help="Process only non-windowed model results") 
    parser.add_argument("--no-clustering", action="store_true",
                       help="Skip clustering analysis")
    parser.add_argument("--fixed-strategy-config", type=str, default=None,
                       help="Path to JSON file with fixed clustering strategies per threshold (for second evaluation run)")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed output for each evaluation")
    
    args = parser.parse_args()
    
    # Handle output directory - make it relative to results_dir if not absolute
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(args.results_dir, args.output_dir)
    
    # Determine windowed parameter
    if args.windowed and args.no_window:
        raise ValueError("Cannot specify both --windowed and --no-window")
    elif args.windowed:
        windowed = True
    elif args.no_window:
        windowed = False
    else:
        windowed = None  # Process all files
    threshold_scales = [0.7, 0.75, 0.8, 0.85, 0.9]#[round(0.3 + (x * 0.01), 2) for x in range(0, 100, 5)]  # 0.1 to 0.39 in 0.01 increments, rounded to 2 decimals

    print("Starting batch evaluation...")
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Threshold scales: {threshold_scales}")
    print(f"Sampling rate: {args.sampling_rate} Hz")
    print(f"Windowed filter: {windowed}")
    print(f"Run clustering: {not args.no_clustering}")
    if args.fixed_strategy_config:
        print(f"Fixed strategy config: {args.fixed_strategy_config}")
        print("🔧 Running in FIXED STRATEGY mode")
    else:
        print("🔄 Running in ADAPTIVE STRATEGY mode")
    
    # Run batch evaluation
    batch_results = batch_evaluate_joint_anomaly_scores(
        results_dir=args.results_dir,
        threshold_scales=threshold_scales,
        sampling_rate=args.sampling_rate,
        output_dir=args.output_dir,
        run_clustering=not args.no_clustering,
        windowed=windowed,
        verbose=args.verbose,
        fixed_strategy_config=args.fixed_strategy_config
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    feather_path = os.path.join(args.output_dir, "evaluation_results.feather")
    
    # Save full results as pickle (only if new results were generated)
    if batch_results['summary_metrics']:
        pickle_path = os.path.join(args.output_dir, f"batch_evaluation_results_{timestamp}.pkl")
        save_results(batch_results, pickle_path)
    
    # Save summary as CSV (using feather file as primary source)
    csv_path = os.path.join(args.output_dir, f"batch_evaluation_summary_{timestamp}.csv")
    save_summary_csv(batch_results, csv_path, feather_path)
    
    # Compute and save threshold summary statistics (using feather file as primary source)
    compute_threshold_summary_stats(batch_results, args.output_dir, feather_path)
    
    print(f"\nBatch evaluation completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Check 'threshold_summary_stats.csv' for aggregated statistics by threshold scale")


if __name__ == "__main__":
    main()
