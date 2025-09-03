#!/usr/bin/env python3
"""
Madrid Clustered Threshold Trade-off Analysis - TEST SET ONLY
Tests multiple thresholds with time_180s clustering and plots Sensitivity vs False Alarms per Hour trade-off
using extended time windows (5 minutes before, 3 minutes after seizures).
Only evaluates on test set: sub097-sub125.
"""

import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import argparse
from collections import defaultdict


class MadridClusteredThresholdTradeoffAnalyzerTestOnly:
    def __init__(self, results_dir: str, output_dir: str = None,
                 pre_seizure_minutes: float = 5.0, post_seizure_minutes: float = 3.0):
        """
        Initialize the clustered threshold trade-off analyzer for test set only.
        
        Args:
            results_dir: Directory containing Madrid windowed results JSON files
            output_dir: Directory to save plots and results (default: same as results_dir)
            pre_seizure_minutes: Minutes before seizure start to consider as detection window
            post_seizure_minutes: Minutes after seizure end to consider as detection window
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir) if output_dir else self.results_dir / "test_only_clustered_tradeoff"
        self.pre_seizure_seconds = pre_seizure_minutes * 60.0
        self.post_seizure_seconds = post_seizure_minutes * 60.0
        self.clustering_time_threshold = 180  # Fixed to time_180s strategy
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
    
    def extract_all_anomaly_scores(self) -> List[float]:
        """Extract all anomaly scores from TEST SET files to determine threshold range."""
        all_json_files = list(self.results_dir.glob("madrid_windowed_results_*.json"))
        
        # Filter for test set files only
        json_files = [f for f in all_json_files if self.is_test_file(f.name)]
        
        all_scores = []
        
        print(f"Extracting anomaly scores from {len(json_files)} test set files (sub097-sub125)...")
        print(f"Skipping {len(all_json_files) - len(json_files)} training set files")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                window_results = data.get('analysis_results', {}).get('window_results', [])
                for window in window_results:
                    anomalies = window.get('anomalies', [])
                    for anomaly in anomalies:
                        score = anomaly.get('anomaly_score', 0)
                        if score > 0:  # Only include positive scores
                            all_scores.append(score)
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue
        
        all_scores.sort()
        print(f"Extracted {len(all_scores)} anomaly scores from test set")
        if all_scores:
            print(f"Score range: {min(all_scores):.6f} to {max(all_scores):.6f}")
        
        return all_scores
    
    def generate_threshold_range(self, all_scores: List[float], num_thresholds: int = 50) -> List[float]:
        """Generate a range of thresholds to test based on anomaly score distribution."""
        if not all_scores:
            return []
        
        # Use percentiles to generate thresholds
        percentiles = np.linspace(0, 99.9, num_thresholds)
        thresholds = [np.percentile(all_scores, p) for p in percentiles]
        
        # Remove duplicates and sort
        thresholds = sorted(list(set(thresholds)))
        
        print(f"Generated {len(thresholds)} unique thresholds")
        print(f"Threshold range: {min(thresholds):.6f} to {max(thresholds):.6f}")
        
        return thresholds
    
    def extract_file_level_anomalies(self, result_data: Dict[str, Any], threshold: float) -> List[Dict[str, Any]]:
        """
        Extract all anomalies from a file and convert to file-level format.
        Uses threshold to filter anomalies.
        """
        analysis_results = result_data.get('analysis_results', {})
        window_results = analysis_results.get('window_results', [])
        
        all_anomalies = []
        
        for window in window_results:
            window_index = window.get('window_index')
            window_start_time = window.get('window_start_time', 0)
            anomalies = window.get('anomalies', [])
            
            if not anomalies:
                continue
            
            # Filter by threshold - include all anomalies >= threshold
            selected_anomalies = [a for a in anomalies if a.get('anomaly_score', 0) >= threshold]
            
            # Convert selected anomalies to file-level representation
            for anomaly in selected_anomalies:
                location_time_in_window = anomaly.get('location_time_in_window', 0)
                absolute_time = window_start_time + location_time_in_window
                
                file_level_anomaly = {
                    'absolute_time': absolute_time,
                    'anomaly_score': anomaly.get('anomaly_score', 0),
                    'original_window_index': window_index,
                    'original_location_time_in_window': location_time_in_window,
                    'window_start_time': window_start_time
                }
                
                all_anomalies.append(file_level_anomaly)
        
        # Sort by absolute time
        all_anomalies.sort(key=lambda x: x['absolute_time'])
        
        return all_anomalies
    
    def group_seizures_by_time(self, seizure_windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group seizure windows into individual seizures with extended time windows."""
        if not seizure_windows:
            return []
        
        # Collect all seizure segments
        all_segments = []
        for window in seizure_windows:
            seizure_segments = window.get('seizure_segments', [])
            for segment in seizure_segments:
                all_segments.append(segment)
        
        # Group segments by their absolute time intervals
        seizure_groups = {}
        for segment in all_segments:
            start_time = round(segment.get('start_time_absolute', 0), 1)
            end_time = round(segment.get('end_time_absolute', 0), 1)
            time_key = (start_time, end_time)
            
            if time_key not in seizure_groups:
                seizure_groups[time_key] = {
                    'start_time_absolute': start_time,
                    'end_time_absolute': end_time,
                    'duration_seconds': end_time - start_time,
                    'extended_start_time': start_time - self.pre_seizure_seconds,
                    'extended_end_time': end_time + self.post_seizure_seconds
                }
        
        # Convert to list and sort by start time
        seizures = list(seizure_groups.values())
        seizures.sort(key=lambda x: x['start_time_absolute'])
        
        return seizures
    
    def time_based_clustering(self, anomalies: List[Dict[str, Any]], 
                            time_threshold_seconds: float) -> List[List[Dict[str, Any]]]:
        """
        Perform time-based clustering with fixed time windows.
        
        Args:
            anomalies: List of anomalies sorted by absolute_time
            time_threshold_seconds: Maximum time distance for clustering (180 seconds)
        
        Returns:
            List of clusters, where each cluster is a list of anomalies
        """
        if not anomalies:
            return []
        
        clusters = []
        current_cluster = [anomalies[0]]
        
        for i in range(1, len(anomalies)):
            time_diff = anomalies[i]['absolute_time'] - current_cluster[-1]['absolute_time']
            
            if time_diff <= time_threshold_seconds:
                # Add to current cluster
                current_cluster.append(anomalies[i])
            else:
                # Start new cluster
                clusters.append(current_cluster)
                current_cluster = [anomalies[i]]
        
        # Add the last cluster
        if current_cluster:
            clusters.append(current_cluster)
        
        return clusters
    
    def select_cluster_representative(self, cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select representative from cluster based on minimal mean time distance.
        
        Args:
            cluster: List of anomalies in the cluster
        
        Returns:
            Representative anomaly with cluster metadata
        """
        if len(cluster) == 1:
            representative = cluster[0].copy()
            representative['cluster_metadata'] = {
                'cluster_size': 1,
                'min_score': representative['anomaly_score'],
                'max_score': representative['anomaly_score'],
                'mean_time_distance': 0.0
            }
            return representative
        
        # Calculate mean time distance for each anomaly to all others in cluster
        times = [a['absolute_time'] for a in cluster]
        best_idx = 0
        min_mean_distance = float('inf')
        
        for i, anomaly in enumerate(cluster):
            mean_distance = np.mean([abs(anomaly['absolute_time'] - other_time) 
                                   for other_time in times if other_time != anomaly['absolute_time']])
            
            if mean_distance < min_mean_distance:
                min_mean_distance = mean_distance
                best_idx = i
        
        # Create representative with cluster metadata
        representative = cluster[best_idx].copy()
        scores = [a['anomaly_score'] for a in cluster]
        
        representative['cluster_metadata'] = {
            'cluster_size': len(cluster),
            'min_score': min(scores),
            'max_score': max(scores),
            'mean_time_distance': min_mean_distance,
            'cluster_time_span': max(times) - min(times),
            'original_cluster_anomalies': [a['absolute_time'] for a in cluster]
        }
        
        return representative
    
    def calculate_clustered_metrics_for_threshold(self, threshold: float) -> Dict[str, Any]:
        """Calculate clustered metrics for a single threshold using time_180s strategy - TEST SET ONLY."""
        print(f"Processing threshold {threshold:.6f} with time_180s clustering on test set...")
        
        all_json_files = list(self.results_dir.glob("madrid_windowed_results_*.json"))
        
        # Filter for test set files only
        json_files = [f for f in all_json_files if self.is_test_file(f.name)]
        
        # Overall statistics
        overall_stats = {
            'files_processed': 0,
            'files_skipped': len(all_json_files) - len(json_files),
            'total_duration_hours': 0.0,
            'total_seizures': 0,
            'detected_seizures': 0,
            'total_anomalies_before_clustering': 0,
            'total_anomalies_after_clustering': 0,
            'total_true_positives': 0,
            'total_false_positives': 0
        }
        
        for json_file in json_files:
            try:
                result_data = self.load_result_file(json_file)
                if result_data is None:
                    continue
                
                # Extract basic info
                input_data = result_data.get('input_data', {})
                validation_data = result_data.get('validation_data', {})
                
                # Get signal duration
                signal_metadata = input_data.get('signal_metadata', {})
                file_duration_hours = signal_metadata.get('total_duration_seconds', 0) / 3600.0
                overall_stats['total_duration_hours'] += file_duration_hours
                
                # Get anomalies at file level (filtered by threshold)
                all_anomalies = self.extract_file_level_anomalies(result_data, threshold)
                overall_stats['total_anomalies_before_clustering'] += len(all_anomalies)
                
                if len(all_anomalies) == 0:
                    overall_stats['files_processed'] += 1
                    continue
                
                # Get seizures
                ground_truth = validation_data.get('ground_truth', {})
                seizure_windows = ground_truth.get('seizure_windows', [])
                individual_seizures = self.group_seizures_by_time(seizure_windows)
                overall_stats['total_seizures'] += len(individual_seizures)
                
                # Apply time_180s clustering
                clusters = self.time_based_clustering(all_anomalies, self.clustering_time_threshold)
                representatives = [self.select_cluster_representative(cluster) for cluster in clusters]
                overall_stats['total_anomalies_after_clustering'] += len(representatives)
                
                # Calculate TP/FP based on extended time overlap for representatives
                file_detected_seizures = set()
                file_true_positives = 0
                file_false_positives = 0
                
                for rep in representatives:
                    rep_time = rep['absolute_time']
                    is_true_positive = False
                    
                    # Check if representative overlaps with any extended seizure window
                    for i, seizure in enumerate(individual_seizures):
                        if seizure['extended_start_time'] <= rep_time <= seizure['extended_end_time']:
                            is_true_positive = True
                            file_detected_seizures.add(i)
                            break
                    
                    if is_true_positive:
                        file_true_positives += 1
                    else:
                        file_false_positives += 1
                
                overall_stats['detected_seizures'] += len(file_detected_seizures)
                overall_stats['total_true_positives'] += file_true_positives
                overall_stats['total_false_positives'] += file_false_positives
                overall_stats['files_processed'] += 1
                
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue
        
        # Calculate final metrics
        if overall_stats['total_seizures'] > 0:
            sensitivity = overall_stats['detected_seizures'] / overall_stats['total_seizures']
        else:
            sensitivity = None
        
        if overall_stats['total_duration_hours'] > 0:
            false_alarms_per_hour = overall_stats['total_false_positives'] / overall_stats['total_duration_hours']
        else:
            false_alarms_per_hour = 0.0
        
        return {
            'threshold': threshold,
            'clustering_strategy': 'time_180s',
            'dataset': 'TEST_SET_ONLY (sub097-sub125)',
            'sensitivity': sensitivity,
            'false_alarms_per_hour': false_alarms_per_hour,
            'total_detections_before_clustering': overall_stats['total_anomalies_before_clustering'],
            'total_detections_after_clustering': overall_stats['total_anomalies_after_clustering'],
            'total_true_positives': overall_stats['total_true_positives'],
            'total_false_positives': overall_stats['total_false_positives'],
            'total_seizures': overall_stats['total_seizures'],
            'detected_seizures': overall_stats['detected_seizures'],
            'files_processed': overall_stats['files_processed'],
            'files_skipped': overall_stats['files_skipped'],
            'total_duration_hours': overall_stats['total_duration_hours'],
            'anomaly_reduction': ((overall_stats['total_anomalies_before_clustering'] - overall_stats['total_anomalies_after_clustering']) / 
                                overall_stats['total_anomalies_before_clustering'] if overall_stats['total_anomalies_before_clustering'] > 0 else 0)
        }
    
    def run_threshold_analysis(self, thresholds: List[float]) -> List[Dict[str, Any]]:
        """Run clustered metrics calculation for each threshold on TEST SET."""
        results = []
        
        print(f"Running clustered analysis for {len(thresholds)} thresholds on TEST SET...")
        print(f"Using clustering strategy: time_{self.clustering_time_threshold}s")
        print(f"Extended window: -{self.pre_seizure_seconds/60:.1f} min to +{self.post_seizure_seconds/60:.1f} min")
        print(f"Dataset: TEST SET ONLY (sub097-sub125)")
        
        for i, threshold in enumerate(thresholds):
            print(f"Processing threshold {i+1}/{len(thresholds)}: {threshold:.6f}")
            
            try:
                threshold_result = self.calculate_clustered_metrics_for_threshold(threshold)
                results.append(threshold_result)
            except Exception as e:
                print(f"Error processing threshold {threshold}: {e}")
                continue
        
        return results
    
    def create_tradeoff_plot(self, results: List[Dict[str, Any]], output_filename: str):
        """Create and save the sensitivity vs false alarms per hour trade-off plot."""
        
        # Extract data for plotting
        thresholds = [r['threshold'] for r in results]
        sensitivities = [r['sensitivity'] if r['sensitivity'] is not None else 0.0 for r in results]
        false_alarms_per_hour = [r['false_alarms_per_hour'] for r in results]
        total_detections_before = [r['total_detections_before_clustering'] for r in results]
        total_detections_after = [r['total_detections_after_clustering'] for r in results]
        anomaly_reductions = [r['anomaly_reduction'] for r in results]
        
        # Create the main trade-off plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Sensitivity vs False Alarms per Hour
        ax1.plot(false_alarms_per_hour, sensitivities, 'b.-', linewidth=2, markersize=4)
        ax1.set_xlabel('False Alarms per Hour')
        ax1.set_ylabel('Sensitivity')
        ax1.set_title(f'Sensitivity vs False Alarms per Hour - TEST SET ONLY\n(Madrid Clustered: time_{self.clustering_time_threshold}s, Extended Time: -{self.pre_seizure_seconds/60:.1f}min to +{self.post_seizure_seconds/60:.1f}min)')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(left=0)
        ax1.set_ylim(0, 1)
        
        
        # Plot 2: Total detections before vs after clustering
        ax2.plot(thresholds, total_detections_before, 'r.-', label='Before Clustering', linewidth=2, markersize=4)
        ax2.plot(thresholds, total_detections_after, 'g.-', label='After Clustering', linewidth=2, markersize=4)
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Total Detections')
        ax2.set_title(f'Total Detections vs Threshold - TEST SET ONLY\n(time_{self.clustering_time_threshold}s Clustering)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Plot 3: Anomaly reduction vs threshold
        ax3.plot(thresholds, [ar * 100 for ar in anomaly_reductions], 'purple', linewidth=2, marker='o', markersize=4)
        ax3.set_xlabel('Threshold')
        ax3.set_ylabel('Anomaly Reduction (%)')
        ax3.set_title(f'Anomaly Reduction vs Threshold - TEST SET ONLY\n(time_{self.clustering_time_threshold}s Clustering)')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)
        
        # Plot 4: Sensitivity vs Anomaly Reduction
        ax4.scatter([ar * 100 for ar in anomaly_reductions], [s * 100 for s in sensitivities], 
                   c=thresholds, cmap='viridis', alpha=0.7, s=50)
        ax4.set_xlabel('Anomaly Reduction (%)')
        ax4.set_ylabel('Sensitivity (%)')
        ax4.set_title(f'Sensitivity vs Anomaly Reduction - TEST SET ONLY\n(time_{self.clustering_time_threshold}s Clustering)')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, 100)
        ax4.set_ylim(0, 100)
        cbar = plt.colorbar(ax4.collections[0], ax=ax4)
        cbar.set_label('Threshold')
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = self.output_dir / output_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Clustered trade-off plot saved to: {plot_path}")
        
        # Also save as PDF for high quality
        pdf_path = self.output_dir / output_filename.replace('.png', '.pdf')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Clustered trade-off plot (PDF) saved to: {pdf_path}")
        
        plt.close()
    
    def save_results(self, results: List[Dict[str, Any]], output_filename: str):
        """Save detailed results to JSON and CSV files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_path = self.output_dir / f"{output_filename}_results.json"
        with open(json_path, 'w') as f:
            json.dump({
                'analysis_metadata': {
                    'analysis_type': 'threshold_tradeoff_clustered_time_180s_test_only',
                    'timestamp': timestamp,
                    'results_directory': str(self.results_dir),
                    'dataset': 'TEST_SET_ONLY (sub097-sub125)',
                    'num_thresholds_tested': len(results),
                    'clustering_strategy': f'time_{self.clustering_time_threshold}s',
                    'pre_seizure_minutes': self.pre_seizure_seconds / 60.0,
                    'post_seizure_minutes': self.post_seizure_seconds / 60.0
                },
                'results': results
            }, f, indent=2)
        print(f"Detailed results saved to: {json_path}")
        
        # Save CSV for easy analysis
        csv_path = self.output_dir / f"{output_filename}_results.csv"
        with open(csv_path, 'w') as f:
            f.write("threshold,clustering_strategy,dataset,sensitivity,false_alarms_per_hour,total_detections_before_clustering,total_detections_after_clustering,total_true_positives,total_false_positives,total_seizures,detected_seizures,files_processed,files_skipped,total_duration_hours,anomaly_reduction\n")
            for r in results:
                f.write(f"{r['threshold']},{r['clustering_strategy']},{r['dataset']},{r['sensitivity']},{r['false_alarms_per_hour']},{r['total_detections_before_clustering']},{r['total_detections_after_clustering']},{r['total_true_positives']},{r['total_false_positives']},{r['total_seizures']},{r['detected_seizures']},{r['files_processed']},{r['files_skipped']},{r['total_duration_hours']},{r['anomaly_reduction']}\n")
        print(f"CSV results saved to: {csv_path}")
    
    def run_full_analysis(self, num_thresholds: int = 50):
        """Run the complete clustered threshold trade-off analysis on TEST SET."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("Starting Madrid Clustered Threshold Trade-off Analysis - TEST SET ONLY...")
        print(f"Results directory: {self.results_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Dataset: TEST SET ONLY (sub097-sub125)")
        print(f"Clustering strategy: time_{self.clustering_time_threshold}s")
        print(f"Extended time window: -{self.pre_seizure_seconds/60:.1f} min to +{self.post_seizure_seconds/60:.1f} min")
        
        # Step 1: Extract all anomaly scores from TEST SET
        all_scores = self.extract_all_anomaly_scores()
        if not all_scores:
            print("No anomaly scores found in test set!")
            return None
        
        # Step 2: Generate threshold range
        thresholds = self.generate_threshold_range(all_scores, num_thresholds)
        if not thresholds:
            print("Could not generate thresholds!")
            return None
        
        # Step 3: Run analysis for each threshold on TEST SET
        results = self.run_threshold_analysis(thresholds)
        if not results:
            print("No results generated!")
            return None
        
        # Step 4: Create plots
        plot_filename = f"madrid_clustered_time180s_threshold_tradeoff_test_only_{timestamp}.png"
        self.create_tradeoff_plot(results, plot_filename)
        
        # Step 5: Save results
        results_filename = f"madrid_clustered_time180s_threshold_tradeoff_test_only_{timestamp}"
        self.save_results(results, results_filename)
        
        # Step 6: Print summary
        print(f"\n{'='*60}")
        print("CLUSTERED THRESHOLD TRADE-OFF ANALYSIS SUMMARY - TEST SET ONLY")
        print(f"{'='*60}")
        print(f"Dataset: TEST SET (sub097-sub125)")
        print(f"Clustering strategy: time_{self.clustering_time_threshold}s")
        print(f"Extended window: -{self.pre_seizure_seconds/60:.1f} min to +{self.post_seizure_seconds/60:.1f} min")
        print(f"Total thresholds tested: {len(results)}")
        
        if results and len(results) > 0:
            print(f"Files processed: {results[0]['files_processed']}")
            print(f"Training files skipped: {results[0]['files_skipped']}")
            
            best_sens = max(results, key=lambda x: x['sensitivity'] or 0)
            min_fa = min([r for r in results if r['false_alarms_per_hour'] > 0], 
                        key=lambda x: x['false_alarms_per_hour'], default=results[0])
            best_reduction = max(results, key=lambda x: x['anomaly_reduction'])
            
            print(f"\nBest Sensitivity on Test Set: {best_sens['sensitivity']:.4f}")
            print(f"  Threshold: {best_sens['threshold']:.6f}")
            print(f"  False Alarms/Hour: {best_sens['false_alarms_per_hour']:.4f}")
            print(f"  Anomaly Reduction: {best_sens['anomaly_reduction']:.4f} ({best_sens['anomaly_reduction']*100:.1f}%)")
            
            print(f"\nMinimum False Alarms/Hour on Test Set: {min_fa['false_alarms_per_hour']:.4f}")
            print(f"  Threshold: {min_fa['threshold']:.6f}")
            print(f"  Sensitivity: {min_fa['sensitivity']:.4f}")
            print(f"  Anomaly Reduction: {min_fa['anomaly_reduction']:.4f} ({min_fa['anomaly_reduction']*100:.1f}%)")
            
            print(f"\nBest Anomaly Reduction on Test Set: {best_reduction['anomaly_reduction']:.4f} ({best_reduction['anomaly_reduction']*100:.1f}%)")
            print(f"  Threshold: {best_reduction['threshold']:.6f}")
            print(f"  Sensitivity: {best_reduction['sensitivity']:.4f}")
            print(f"  False Alarms/Hour: {best_reduction['false_alarms_per_hour']:.4f}")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate threshold trade-off analysis with time_180s clustering - TEST SET ONLY (sub097-sub125)"
    )
    parser.add_argument(
        "results_dir", 
        help="Directory containing Madrid windowed results JSON files"
    )
    parser.add_argument(
        "-o", "--output-dir", 
        help="Output directory for plots and results (default: results_dir/test_only_clustered_tradeoff)"
    )
    parser.add_argument(
        "-n", "--num-thresholds", 
        type=int,
        default=50,
        help="Number of thresholds to test (default: 50)"
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
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = MadridClusteredThresholdTradeoffAnalyzerTestOnly(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        pre_seizure_minutes=args.pre_seizure_minutes,
        post_seizure_minutes=args.post_seizure_minutes
    )
    
    # Run analysis
    results = analyzer.run_full_analysis(args.num_thresholds)
    
    if results is None:
        print("Analysis failed.")
        return 1
    
    print(f"\nAnalysis completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())