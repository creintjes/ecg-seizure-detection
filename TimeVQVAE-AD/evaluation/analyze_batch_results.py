#!/usr/bin/env python3
"""
Analysis script for batch evaluation results.
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional
import argparse


def load_batch_results(pickle_path: str) -> Dict[str, Any]:
    """Load batch results from pickle file."""
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)





def analyze_clustering_vs_basic(df: pd.DataFrame) -> None:
    """Compare clustering results vs basic detection."""
    if 'cluster_event_recall' not in df.columns:
        print("No clustering results found in data.")
        return
        
    print(f"\n=== Clustering vs Basic Detection Analysis ===")
    
    # Filter out rows without clustering results
    clustering_df = df.dropna(subset=['cluster_event_recall'])
    
    if len(clustering_df) == 0:
        print("No valid clustering results found.")
        return
    
    print(f"Comparing {len(clustering_df)} evaluations with clustering results...")
    
    # Show absolute values first
    print("\nAbsolute performance values:")
    print("BASIC DETECTION:")
    print(f"  Mean Event Recall: {clustering_df['event_recall'].mean():.3f} ± {clustering_df['event_recall'].std():.3f}")
    print(f"  Mean FAR/hour: {clustering_df['false_alarm_rate_per_hour'].mean():.2f} ± {clustering_df['false_alarm_rate_per_hour'].std():.2f}")
    print(f"  Mean Event IoU: {clustering_df['event_iou'].mean():.3f} ± {clustering_df['event_iou'].std():.3f}")
    
    print("\nCLUSTERED DETECTION:")
    print(f"  Mean Event Recall: {clustering_df['cluster_event_recall'].mean():.3f} ± {clustering_df['cluster_event_recall'].std():.3f}")
    print(f"  Mean FAR/hour: {clustering_df['cluster_event_far_per_hour'].mean():.2f} ± {clustering_df['cluster_event_far_per_hour'].std():.2f}")
    print(f"  Mean Event IoU: {clustering_df['cluster_event_iou'].mean():.3f} ± {clustering_df['cluster_event_iou'].std():.3f}")
    
    # Responder analysis (basic detection sensitivity > 2/3)
    basic_responders = clustering_df[clustering_df['event_recall'] > 2/3]
    basic_non_responders = clustering_df[clustering_df['event_recall'] <= 2/3]
    
    print(f"\nRESPONDER ANALYSIS (Basic Detection Sensitivity > 2/3):")
    print(f"Responders: {len(basic_responders)}/{len(clustering_df)} ({len(basic_responders)/len(clustering_df)*100:.1f}%)")
    print(f"Non-responders: {len(basic_non_responders)}/{len(clustering_df)} ({len(basic_non_responders)/len(clustering_df)*100:.1f}%)")
    
    if len(basic_responders) > 0:
        print("\nRESPONDERS - BASIC DETECTION:")
        print(f"  Mean Event Recall: {basic_responders['event_recall'].mean():.3f} ± {basic_responders['event_recall'].std():.3f}")
        print(f"  Mean FAR/hour: {basic_responders['false_alarm_rate_per_hour'].mean():.2f} ± {basic_responders['false_alarm_rate_per_hour'].std():.2f}")
        print(f"  Mean Event IoU: {basic_responders['event_iou'].mean():.3f} ± {basic_responders['event_iou'].std():.3f}")
        
        print("RESPONDERS - CLUSTERED DETECTION:")
        print(f"  Mean Event Recall: {basic_responders['cluster_event_recall'].mean():.3f} ± {basic_responders['cluster_event_recall'].std():.3f}")
        print(f"  Mean FAR/hour: {basic_responders['cluster_event_far_per_hour'].mean():.2f} ± {basic_responders['cluster_event_far_per_hour'].std():.2f}")
        print(f"  Mean Event IoU: {basic_responders['cluster_event_iou'].mean():.3f} ± {basic_responders['cluster_event_iou'].std():.3f}")
    
    if len(basic_non_responders) > 0:
        print("\nNON-RESPONDERS - BASIC DETECTION:")
        print(f"  Mean Event Recall: {basic_non_responders['event_recall'].mean():.3f} ± {basic_non_responders['event_recall'].std():.3f}")
        print(f"  Mean FAR/hour: {basic_non_responders['false_alarm_rate_per_hour'].mean():.2f} ± {basic_non_responders['false_alarm_rate_per_hour'].std():.2f}")
        print(f"  Mean Event IoU: {basic_non_responders['event_iou'].mean():.3f} ± {basic_non_responders['event_iou'].std():.3f}")
        
        print("NON-RESPONDERS - CLUSTERED DETECTION:")
        print(f"  Mean Event Recall: {basic_non_responders['cluster_event_recall'].mean():.3f} ± {basic_non_responders['cluster_event_recall'].std():.3f}")
        print(f"  Mean FAR/hour: {basic_non_responders['cluster_event_far_per_hour'].mean():.2f} ± {basic_non_responders['cluster_event_far_per_hour'].std():.2f}")
        print(f"  Mean Event IoU: {basic_non_responders['cluster_event_iou'].mean():.3f} ± {basic_non_responders['cluster_event_iou'].std():.3f}")
    
    # Compare metrics
    metrics_comparison = {
        'Basic Event Recall': clustering_df['event_recall'],
        'Clustered Event Recall': clustering_df['cluster_event_recall'],
        'Basic Event IoU': clustering_df['event_iou'],
        'Clustered Event IoU': clustering_df['cluster_event_iou'],
        'Basic FAR/hour': clustering_df['false_alarm_rate_per_hour'],
        'Clustered FAR/hour': clustering_df['cluster_event_far_per_hour']
    }
    
    print("\nMetric improvements (Clustered - Basic):")
    for basic_col, clustered_col in [
        ('event_recall', 'cluster_event_recall'),
        ('event_iou', 'cluster_event_iou'),
        ('false_alarm_rate_per_hour', 'cluster_event_far_per_hour')
    ]:
        improvement = clustering_df[clustered_col] - clustering_df[basic_col]
        print(f"  {basic_col}: mean={improvement.mean():.3f}, std={improvement.std():.3f}")
        print(f"    Improved: {(improvement > 0).sum()}/{len(improvement)} cases")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Recall comparison
    axes[0, 0].scatter(clustering_df['event_recall'], clustering_df['cluster_event_recall'], alpha=0.6)
    axes[0, 0].plot([0, 1], [0, 1], 'r--', alpha=0.8)
    axes[0, 0].set_xlabel('Basic Event Recall')
    axes[0, 0].set_ylabel('Clustered Event Recall')
    axes[0, 0].set_title('Event Recall: Clustering vs Basic')
    axes[0, 0].grid(True, alpha=0.3)
    
    # IoU comparison
    axes[0, 1].scatter(clustering_df['event_iou'], clustering_df['cluster_event_iou'], alpha=0.6)
    axes[0, 1].plot([0, 1], [0, 1], 'r--', alpha=0.8)
    axes[0, 1].set_xlabel('Basic Event IoU')
    axes[0, 1].set_ylabel('Clustered Event IoU')
    axes[0, 1].set_title('Event IoU: Clustering vs Basic')
    axes[0, 1].grid(True, alpha=0.3)
    
    # FAR comparison (log scale due to potentially large range)
    axes[1, 0].scatter(clustering_df['false_alarm_rate_per_hour'], 
                      clustering_df['cluster_event_far_per_hour'], alpha=0.6)
    max_far = max(clustering_df['false_alarm_rate_per_hour'].max(), 
                  clustering_df['cluster_event_far_per_hour'].max())
    axes[1, 0].plot([0, max_far], [0, max_far], 'r--', alpha=0.8)
    axes[1, 0].set_xlabel('Basic FAR/hour')
    axes[1, 0].set_ylabel('Clustered FAR/hour')
    axes[1, 0].set_title('False Alarm Rate: Clustering vs Basic')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Number of clusters vs performance
    axes[1, 1].scatter(clustering_df['n_clusters'], clustering_df['cluster_event_recall'], alpha=0.6)
    axes[1, 1].set_xlabel('Number of Clusters')
    axes[1, 1].set_ylabel('Clustered Event Recall')
    axes[1, 1].set_title('Event Recall vs Number of Clusters')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def dataset_summary(df: pd.DataFrame) -> None:
    """Print summary statistics per dataset."""
    print(f"\n=== Dataset Summary ===")
    
    dataset_summary = df.groupby('dataset_id').agg({
        'n_true_anomalies': 'first',
        'anomaly_percentage': 'first',
        'event_recall': ['max', 'mean'],
        'event_iou': ['max', 'mean'],
        'false_alarm_rate_per_hour': ['min', 'mean']
    }).round(3)
    
    # Flatten column names
    dataset_summary.columns = ['_'.join(col).strip() for col in dataset_summary.columns.values]
    
    print("Dataset characteristics and best performance:")
    print(dataset_summary)
    
    # Find optimal threshold scales
    print(f"\nOptimal threshold scales per dataset (based on event_recall):")
    optimal_scales = df.loc[df.groupby('dataset_id')['event_recall'].idxmax()]
    for _, row in optimal_scales.iterrows():
        print(f"  Dataset {int(row['dataset_id']):03d}: scale={row['threshold_scale']}, "
              f"recall={row['event_recall']:.3f}, iou={row['event_iou']:.3f}")


def create_performance_heatmap(df: pd.DataFrame, metric: str = 'event_recall') -> None:
    """Create heatmap of performance across datasets and threshold scales."""
    print(f"\n=== Performance Heatmap ({metric}) ===")
    
    # Pivot data for heatmap
    pivot_data = df.pivot(index='dataset_id', columns='threshold_scale', values=metric)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', 
                cbar_kws={'label': metric})
    plt.title(f'{metric} across Datasets and Threshold Scales')
    plt.xlabel('Threshold Scale')
    plt.ylabel('Dataset ID')
    plt.tight_layout()
    plt.show()


def create_threshold_visualization(results_df: pd.DataFrame) -> None:
    """Create visualization of threshold performance."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Get data for each group
    all_data = results_df[results_df['group'] == 'All']
    responder_data = results_df[results_df['group'] == 'Responders']
    non_responder_data = results_df[results_df['group'] == 'Non-Responders']
    
    # Plot 1: Sensitivity by threshold
    axes[0, 0].errorbar(all_data['threshold_scale'], all_data['mean_sensitivity'], 
                       yerr=all_data['std_sensitivity'], marker='o', label='All', capsize=5)
    if len(responder_data) > 0:
        axes[0, 0].errorbar(responder_data['threshold_scale'], responder_data['mean_sensitivity'], 
                           yerr=responder_data['std_sensitivity'], marker='s', label='Responders', capsize=5)
    if len(non_responder_data) > 0:
        axes[0, 0].errorbar(non_responder_data['threshold_scale'], non_responder_data['mean_sensitivity'], 
                           yerr=non_responder_data['std_sensitivity'], marker='^', label='Non-Responders', capsize=5)
    axes[0, 0].set_xlabel('Threshold Scale')
    axes[0, 0].set_ylabel('Mean Sensitivity')
    axes[0, 0].set_title('Sensitivity vs Threshold Scale')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: FAR by threshold
    axes[0, 1].errorbar(all_data['threshold_scale'], all_data['mean_far_per_hour'], 
                       yerr=all_data['std_far_per_hour'], marker='o', label='All', capsize=5)
    if len(responder_data) > 0:
        axes[0, 1].errorbar(responder_data['threshold_scale'], responder_data['mean_far_per_hour'], 
                           yerr=responder_data['std_far_per_hour'], marker='s', label='Responders', capsize=5)
    if len(non_responder_data) > 0:
        axes[0, 1].errorbar(non_responder_data['threshold_scale'], non_responder_data['mean_far_per_hour'], 
                           yerr=non_responder_data['std_far_per_hour'], marker='^', label='Non-Responders', capsize=5)
    axes[0, 1].set_xlabel('Threshold Scale')
    axes[0, 1].set_ylabel('Mean FAR/hour')
    axes[0, 1].set_title('FAR vs Threshold Scale')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Responder rate by threshold
    axes[1, 0].plot(all_data['threshold_scale'], all_data['responder_rate'], 
                   marker='o', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Threshold Scale')
    axes[1, 0].set_ylabel('Responder Rate')
    axes[1, 0].set_title('Responder Rate vs Threshold Scale')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)
    
    # Plot 4: Number of subjects by threshold and group
    threshold_scales = all_data['threshold_scale'].values
    all_counts = all_data['n_subjects'].values
    responder_counts = responder_data['n_subjects'].values if len(responder_data) > 0 else np.zeros_like(threshold_scales)
    non_responder_counts = non_responder_data['n_subjects'].values if len(non_responder_data) > 0 else np.zeros_like(threshold_scales)
    
    width = 0.25
    x = np.arange(len(threshold_scales))
    
    axes[1, 1].bar(x - width, all_counts, width, label='All', alpha=0.8)
    axes[1, 1].bar(x, responder_counts, width, label='Responders', alpha=0.8)
    axes[1, 1].bar(x + width, non_responder_counts, width, label='Non-Responders', alpha=0.8)
    
    axes[1, 1].set_xlabel('Threshold Scale')
    axes[1, 1].set_ylabel('Number of Subjects')
    axes[1, 1].set_title('Subject Count by Group')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(threshold_scales)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def analyze_threshold_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze performance across different thresholds, split by responder status.
    Returns a comprehensive table with mean FAR and sensitivity per threshold.
    """
    print(f"\n=== THRESHOLD PERFORMANCE ANALYSIS ===")
    
    # Create responder column for all data
    df_analysis = df.copy()
    df_analysis['responder'] = df_analysis['event_recall'] > 2/3
    
    results = []
    
    for threshold in sorted(df_analysis['threshold_scale'].unique()):
        threshold_data = df_analysis[df_analysis['threshold_scale'] == threshold]
        
        # Overall statistics
        overall_stats = {
            'threshold_scale': threshold,
            'group': 'All',
            'n_subjects': len(threshold_data),
            'mean_sensitivity': threshold_data['event_recall'].mean(),
            'std_sensitivity': threshold_data['event_recall'].std(),
            'mean_far_per_hour': threshold_data['false_alarm_rate_per_hour'].mean(),
            'std_far_per_hour': threshold_data['false_alarm_rate_per_hour'].std(),
            'mean_iou': threshold_data['event_iou'].mean(),
            'std_iou': threshold_data['event_iou'].std(),
            'responder_rate': (threshold_data['responder']).mean()
        }
        results.append(overall_stats)
        
        # Responder statistics
        responders = threshold_data[threshold_data['responder']]
        if len(responders) > 0:
            responder_stats = {
                'threshold_scale': threshold,
                'group': 'Responders',
                'n_subjects': len(responders),
                'mean_sensitivity': responders['event_recall'].mean(),
                'std_sensitivity': responders['event_recall'].std(),
                'mean_far_per_hour': responders['false_alarm_rate_per_hour'].mean(),
                'std_far_per_hour': responders['false_alarm_rate_per_hour'].std(),
                'mean_iou': responders['event_iou'].mean(),
                'std_iou': responders['event_iou'].std(),
                'responder_rate': 1.0
            }
            results.append(responder_stats)
        
        # Non-responder statistics
        non_responders = threshold_data[~threshold_data['responder']]
        if len(non_responders) > 0:
            non_responder_stats = {
                'threshold_scale': threshold,
                'group': 'Non-Responders',
                'n_subjects': len(non_responders),
                'mean_sensitivity': non_responders['event_recall'].mean(),
                'std_sensitivity': non_responders['event_recall'].std(),
                'mean_far_per_hour': non_responders['false_alarm_rate_per_hour'].mean(),
                'std_far_per_hour': non_responders['false_alarm_rate_per_hour'].std(),
                'mean_iou': non_responders['event_iou'].mean(),
                'std_iou': non_responders['event_iou'].std(),
                'responder_rate': 0.0
            }
            results.append(non_responder_stats)
    
    # Convert to DataFrame and round
    results_df = pd.DataFrame(results)
    numeric_cols = ['mean_sensitivity', 'std_sensitivity', 'mean_far_per_hour', 'std_far_per_hour', 
                   'mean_iou', 'std_iou', 'responder_rate']
    results_df[numeric_cols] = results_df[numeric_cols].round(3)
    
    return results_df


def create_summary_table(optimal_data: pd.DataFrame, optimal_threshold: float) -> pd.DataFrame:
    """Create a summary table for the optimal threshold results."""
    summary_table = optimal_data[['dataset_id', 'event_recall', 'false_alarm_rate_per_hour', 'event_iou']].copy()
    summary_table['responder'] = summary_table['event_recall'] > 2/3
    summary_table['threshold_scale'] = optimal_threshold
    
    # Rename columns for clarity
    summary_table = summary_table.rename(columns={
        'dataset_id': 'Dataset_ID',
        'event_recall': 'Sensitivity',
        'false_alarm_rate_per_hour': 'FAR_per_hour',
        'event_iou': 'IoU',
        'responder': 'Responder_2_3',
        'threshold_scale': 'Threshold_Scale'
    })
    
    return summary_table


def main():
    parser = argparse.ArgumentParser(description="Analyze batch evaluation results - threshold performance")
    parser.add_argument("results_file", help="Path to batch evaluation results pickle file")
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.results_file}")
    batch_results = load_batch_results(args.results_file)
    
    # Convert to DataFrame
    if 'summary_metrics' not in batch_results or not batch_results['summary_metrics']:
        print("No summary metrics found in results file.")
        return
    
    df = pd.DataFrame(batch_results['summary_metrics'])
    
    print(f"Loaded data: {len(df)} evaluations across {df['dataset_id'].nunique()} datasets")
    print(f"Threshold scales tested: {sorted(df['threshold_scale'].unique())}")
    
    # Main analysis: threshold performance
    threshold_results = analyze_threshold_performance(df)
    
    # Display results
    print("\nThreshold Performance Summary:")
    print("=" * 100)
    print(threshold_results.to_string(index=False))
    
    # Save results table
    output_dir = os.path.dirname(args.results_file)
    threshold_csv_path = os.path.join(output_dir, "threshold_performance_analysis.csv")
    threshold_results.to_csv(threshold_csv_path, index=False)
    print(f"\nThreshold performance table saved to: {threshold_csv_path}")
    
    # Create visualization
    create_threshold_visualization(threshold_results)
    
    # Print metadata
    print(f"\n=== Batch Run Metadata ===")
    metadata = batch_results['metadata']
    print(f"Timestamp: {metadata['timestamp']}")
    print(f"Files processed: {len(metadata['files_processed'])}")
    print(f"Sampling rate: {metadata['sampling_rate']} Hz")
    print(f"Clustering enabled: {metadata['run_clustering']}")


if __name__ == "__main__":
    main()
