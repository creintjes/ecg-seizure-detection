#!/usr/bin/env python3
"""
Madrid Clustering Analyzer
Reduces false alarms by clustering temporally close anomalies and selecting representatives.
Maintains sensitivity while reducing false positive rate through intelligent clustering.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import argparse
import numpy as np
from collections import defaultdict


class MadridClusteringAnalyzer:
    def __init__(self, results_dir: str, output_dir: str = None, threshold: float = None):
        """
        Initialize the clustering analyzer.
        
        Args:
            results_dir: Directory containing Madrid windowed results JSON files
            output_dir: Directory to save clustering results (default: same as results_dir)
            threshold: Anomaly score threshold for detection (if None, uses top-ranked anomalies)
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir) if output_dir else self.results_dir
        self.threshold = threshold
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for organized output
        (self.output_dir / "metrics_before").mkdir(exist_ok=True)
        (self.output_dir / "strategy_comparison").mkdir(exist_ok=True) 
        (self.output_dir / "clusters").mkdir(exist_ok=True)
        (self.output_dir / "metrics_after").mkdir(exist_ok=True)
        
        # Time-based clustering thresholds in seconds
        self.time_thresholds = [2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1200, 1800]
        
    def load_result_file(self, filepath: Path) -> Dict[str, Any]:
        """Load and parse a Madrid results JSON file."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def extract_file_level_anomalies(self, result_data: Dict[str, Any]) -> Tuple[List[Dict], Dict]:
        """
        Extract anomalies at file level (removing window information) and ground truth.
        
        Returns:
            Tuple of (anomalies_list, file_metadata)
        """
        # Extract metadata
        metadata = result_data.get('analysis_metadata', {})
        input_data = result_data.get('input_data', {})
        validation_data = result_data.get('validation_data', {})
        analysis_results = result_data.get('analysis_results', {})
        window_results = analysis_results.get('window_results', [])
        
        # Get file-level information
        subject_id = input_data.get('subject_id', 'unknown')
        run_id = input_data.get('run_id', 'unknown')
        signal_metadata = input_data.get('signal_metadata', {})
        total_duration_hours = signal_metadata.get('total_duration_seconds', 0) / 3600.0
        window_duration_seconds = signal_metadata.get('window_duration_seconds', 3600.0)
        
        # Get ground truth information
        ground_truth = validation_data.get('ground_truth', {})
        seizure_present = ground_truth.get('seizure_present', False)
        seizure_windows = ground_truth.get('seizure_windows', [])
        seizure_window_indices = set(sw.get('window_index', -1) for sw in seizure_windows)
        
        # Extract all anomalies with file-level timestamps and labels
        file_level_anomalies = []
        
        for window in window_results:
            window_index = window.get('window_index', 0)
            window_start_time = window_index * window_duration_seconds
            
            # Extract anomalies from this window
            anomalies = window.get('anomalies', [])
            
            # Filter anomalies based on threshold or use top-ranked
            if self.threshold is not None:
                selected_anomalies = [a for a in anomalies if a.get('anomaly_score', 0) >= self.threshold]
            else:
                selected_anomalies = anomalies[:1] if anomalies else []  # Top-ranked only
            
            # Convert to file-level anomalies with absolute timestamps
            for anomaly in selected_anomalies:
                file_level_anomaly = {
                    'anomaly_id': f"{subject_id}_{run_id}_w{window_index}_a{anomaly.get('sequence_index', 0)}",
                    'absolute_time': window_start_time + anomaly.get('start_time', 0),
                    'anomaly_score': anomaly.get('anomaly_score', 0),
                    'window_index': window_index,
                    'start_time_in_window': anomaly.get('start_time', 0),
                    'duration': anomaly.get('duration', anomaly.get('end_time', 0) - anomaly.get('start_time', 0)),
                    'is_true_positive': window_index in seizure_window_indices
                }
                file_level_anomalies.append(file_level_anomaly)
        
        # Sort anomalies by absolute time
        file_level_anomalies.sort(key=lambda x: x['absolute_time'])
        
        file_metadata = {
            'subject_id': subject_id,
            'run_id': run_id,
            'total_duration_hours': total_duration_hours,
            'window_duration_seconds': window_duration_seconds,
            'seizure_present': seizure_present,
            'num_seizure_windows': len(seizure_window_indices),
            'seizure_window_indices': list(seizure_window_indices)
        }
        
        return file_level_anomalies, file_metadata
    
    def calculate_base_metrics(self, anomalies: List[Dict], metadata: Dict) -> Dict[str, Any]:
        """Calculate baseline metrics before clustering."""
        total_anomalies = len(anomalies)
        true_positives = sum(1 for a in anomalies if a['is_true_positive'])
        false_positives = total_anomalies - true_positives
        
        # Calculate sensitivity
        num_seizure_windows = metadata['num_seizure_windows']
        sensitivity = (true_positives / num_seizure_windows) if num_seizure_windows > 0 else 0.0
        
        # Calculate false alarms per hour
        total_duration_hours = metadata['total_duration_hours']
        false_alarms_per_hour = (false_positives / total_duration_hours) if total_duration_hours > 0 else 0.0
        
        return {
            'total_anomalies': total_anomalies,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'sensitivity': sensitivity,
            'false_alarms_per_hour': false_alarms_per_hour,
            'num_seizure_windows': num_seizure_windows,
            'total_duration_hours': total_duration_hours
        }
    
    def time_based_clustering(self, anomalies: List[Dict], time_threshold: float) -> List[List[Dict]]:
        """
        Cluster anomalies based on temporal proximity.
        
        Args:
            anomalies: List of anomaly dictionaries sorted by absolute_time
            time_threshold: Maximum time gap in seconds to consider anomalies in same cluster
            
        Returns:
            List of clusters, each cluster is a list of anomaly dictionaries
        """
        if not anomalies:
            return []
        
        clusters = []
        current_cluster = [anomalies[0]]
        
        for i in range(1, len(anomalies)):
            time_gap = anomalies[i]['absolute_time'] - anomalies[i-1]['absolute_time']
            
            if time_gap <= time_threshold:
                current_cluster.append(anomalies[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [anomalies[i]]
        
        # Don't forget the last cluster
        clusters.append(current_cluster)
        
        return clusters
    
    def select_cluster_representative(self, cluster: List[Dict]) -> Dict[str, Any]:
        """
        Select representative anomaly from cluster using minimal mean time distance.
        
        Args:
            cluster: List of anomaly dictionaries
            
        Returns:
            Representative anomaly with cluster metadata
        """
        if len(cluster) == 1:
            representative = cluster[0].copy()
        else:
            # Calculate mean time distance for each anomaly to all others in cluster
            min_mean_distance = float('inf')
            representative_idx = 0
            
            for i, anomaly in enumerate(cluster):
                distances = [abs(anomaly['absolute_time'] - other['absolute_time']) 
                           for j, other in enumerate(cluster) if i != j]
                mean_distance = np.mean(distances) if distances else 0
                
                if mean_distance < min_mean_distance:
                    min_mean_distance = mean_distance
                    representative_idx = i
            
            representative = cluster[representative_idx].copy()
        
        # Add cluster metadata
        cluster_true_positives = sum(1 for a in cluster if a['is_true_positive'])
        anomaly_scores = [a['anomaly_score'] for a in cluster]
        times = [a['absolute_time'] for a in cluster]
        
        representative.update({
            'cluster_id': f"cluster_{hash(str(sorted(times)))}_size{len(cluster)}",
            'cluster_size': len(cluster),
            'cluster_true_positives': cluster_true_positives,
            'cluster_min_score': min(anomaly_scores),
            'cluster_max_score': max(anomaly_scores),
            'cluster_mean_score': np.mean(anomaly_scores),
            'cluster_time_span': max(times) - min(times),
            'mean_time_distance_to_cluster': min_mean_distance if len(cluster) > 1 else 0.0
        })
        
        return representative
    
    def evaluate_clustering_strategy(self, base_metrics: Dict, representatives: List[Dict], 
                                   metadata: Dict, strategy_name: str) -> Dict[str, Any]:
        """
        Evaluate a clustering strategy and calculate performance metrics.
        
        Args:
            base_metrics: Baseline metrics before clustering
            representatives: List of cluster representatives
            metadata: File metadata
            strategy_name: Name of the clustering strategy
            
        Returns:
            Strategy evaluation results
        """
        # Calculate metrics after clustering
        total_representatives = len(representatives)
        rep_true_positives = sum(1 for r in representatives if r['is_true_positive'])
        rep_false_positives = total_representatives - rep_true_positives
        
        # Calculate new sensitivity and false alarms per hour
        num_seizure_windows = metadata['num_seizure_windows']
        new_sensitivity = (rep_true_positives / num_seizure_windows) if num_seizure_windows > 0 else 0.0
        
        total_duration_hours = metadata['total_duration_hours']
        new_false_alarms_per_hour = (rep_false_positives / total_duration_hours) if total_duration_hours > 0 else 0.0
        
        # Calculate reductions
        anomaly_reduction = 1.0 - (total_representatives / base_metrics['total_anomalies']) if base_metrics['total_anomalies'] > 0 else 0.0
        fp_reduction = 1.0 - (rep_false_positives / base_metrics['false_positives']) if base_metrics['false_positives'] > 0 else 0.0
        
        # Calculate score function: 0.6*fp_reduction + 0.3*anomaly_reduction - 2.0*max(0, base_sens - sens)
        sensitivity_penalty = max(0, base_metrics['sensitivity'] - new_sensitivity)
        score = 0.6 * fp_reduction + 0.3 * anomaly_reduction - 2.0 * sensitivity_penalty
        
        return {
            'strategy_name': strategy_name,
            'metrics_after_clustering': {
                'total_representatives': total_representatives,
                'true_positives': rep_true_positives,
                'false_positives': rep_false_positives,
                'sensitivity': new_sensitivity,
                'false_alarms_per_hour': new_false_alarms_per_hour
            },
            'improvements': {
                'anomaly_reduction': anomaly_reduction,
                'fp_reduction': fp_reduction,
                'sensitivity_change': new_sensitivity - base_metrics['sensitivity'],
                'false_alarms_reduction': base_metrics['false_alarms_per_hour'] - new_false_alarms_per_hour
            },
            'score': score,
            'cluster_statistics': {
                'num_clusters': len(representatives),
                'avg_cluster_size': np.mean([r['cluster_size'] for r in representatives]) if representatives else 0,
                'max_cluster_size': max([r['cluster_size'] for r in representatives]) if representatives else 0,
                'single_anomaly_clusters': sum(1 for r in representatives if r['cluster_size'] == 1)
            }
        }
    
    def process_file(self, filepath: Path) -> Dict[str, Any]:
        """Process a single Madrid results file with clustering analysis."""
        print(f"Processing {filepath.name}...")
        
        # Load result data
        result_data = self.load_result_file(filepath)
        if result_data is None:
            return None
        
        # Extract file-level anomalies
        anomalies, metadata = self.extract_file_level_anomalies(result_data)
        
        if not anomalies:
            print(f"  No anomalies found in {filepath.name}")
            return None
        
        # Calculate base metrics
        base_metrics = self.calculate_base_metrics(anomalies, metadata)
        
        # Test all clustering strategies
        strategy_results = []
        best_score = float('-inf')
        best_strategy_name = None
        best_representatives = None
        
        for time_threshold in self.time_thresholds:
            strategy_name = f"time_based_{time_threshold}s"
            
            # Perform clustering
            clusters = self.time_based_clustering(anomalies, time_threshold)
            
            # Select representatives
            representatives = [self.select_cluster_representative(cluster) for cluster in clusters]
            
            # Evaluate strategy
            evaluation = self.evaluate_clustering_strategy(
                base_metrics, representatives, metadata, strategy_name
            )
            
            strategy_results.append(evaluation)
            
            # Track best strategy
            if evaluation['score'] > best_score:
                best_score = evaluation['score']
                best_strategy_name = strategy_name
                best_representatives = representatives
        
        return {
            'file_name': filepath.name,
            'metadata': metadata,
            'base_metrics': base_metrics,
            'strategy_results': strategy_results,
            'best_strategy': {
                'name': best_strategy_name,
                'score': best_score,
                'representatives': best_representatives
            }
        }
    
    def process_all_files(self) -> Dict[str, Any]:
        """Process all Madrid result files with clustering analysis."""
        json_files = list(self.results_dir.glob("madrid_windowed_results_*.json"))
        
        if not json_files:
            print(f"No Madrid result files found in {self.results_dir}")
            return None
        
        print(f"Found {len(json_files)} Madrid result files")
        
        all_file_results = {}
        overall_base_metrics = {
            'total_files': 0,
            'total_anomalies': 0,
            'total_true_positives': 0,
            'total_false_positives': 0,
            'total_seizure_windows': 0,
            'total_duration_hours': 0.0
        }
        
        # Process each file
        for json_file in sorted(json_files):
            file_result = self.process_file(json_file)
            if file_result is None:
                continue
            
            all_file_results[json_file.name] = file_result
            
            # Accumulate overall metrics
            base_metrics = file_result['base_metrics']
            overall_base_metrics['total_files'] += 1
            overall_base_metrics['total_anomalies'] += base_metrics['total_anomalies']
            overall_base_metrics['total_true_positives'] += base_metrics['true_positives']
            overall_base_metrics['total_false_positives'] += base_metrics['false_positives']
            overall_base_metrics['total_seizure_windows'] += base_metrics['num_seizure_windows']
            overall_base_metrics['total_duration_hours'] += base_metrics['total_duration_hours']
        
        # Calculate overall base metrics
        overall_sensitivity = (overall_base_metrics['total_true_positives'] / 
                             overall_base_metrics['total_seizure_windows']) if overall_base_metrics['total_seizure_windows'] > 0 else 0.0
        overall_false_alarms_per_hour = (overall_base_metrics['total_false_positives'] / 
                                       overall_base_metrics['total_duration_hours']) if overall_base_metrics['total_duration_hours'] > 0 else 0.0
        
        overall_base_metrics.update({
            'overall_sensitivity': overall_sensitivity,
            'overall_false_alarms_per_hour': overall_false_alarms_per_hour
        })
        
        return {
            'analysis_metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'results_directory': str(self.results_dir),
                'output_directory': str(self.output_dir),
                'threshold_used': self.threshold,
                'time_thresholds_tested': self.time_thresholds,
                'detection_strategy': 'threshold' if self.threshold else 'top_ranked'
            },
            'overall_base_metrics': overall_base_metrics,
            'file_results': all_file_results
        }
    
    def save_base_metrics(self, results: Dict[str, Any]):
        """Save base metrics before clustering."""
        base_metrics_path = self.output_dir / "metrics_before" / "metrics_summary.json"
        
        base_metrics_summary = {
            'analysis_metadata': results['analysis_metadata'],
            'overall_metrics': results['overall_base_metrics'],
            'individual_file_metrics': {
                filename: file_result['base_metrics'] 
                for filename, file_result in results['file_results'].items()
            }
        }
        
        with open(base_metrics_path, 'w') as f:
            json.dump(base_metrics_summary, f, indent=2)
        
        print(f"Base metrics saved to: {base_metrics_path}")
    
    def save_strategy_comparison(self, results: Dict[str, Any]):
        """Save strategy comparison results."""
        comparison_path = self.output_dir / "strategy_comparison" / "comparison.json"
        
        # Aggregate strategy results across all files
        strategy_aggregates = defaultdict(lambda: {
            'total_score': 0.0,
            'total_files': 0,
            'total_anomaly_reduction': 0.0,
            'total_fp_reduction': 0.0,
            'total_sensitivity_change': 0.0,
            'file_count': 0
        })
        
        comparison_data = {
            'analysis_metadata': results['analysis_metadata'],
            'strategy_aggregates': {},
            'individual_file_strategies': {}
        }
        
        # Process each file's strategy results
        for filename, file_result in results['file_results'].items():
            comparison_data['individual_file_strategies'][filename] = file_result['strategy_results']
            
            # Aggregate across strategies
            for strategy_result in file_result['strategy_results']:
                strategy_name = strategy_result['strategy_name']
                agg = strategy_aggregates[strategy_name]
                
                agg['total_score'] += strategy_result['score']
                agg['total_files'] += 1
                agg['total_anomaly_reduction'] += strategy_result['improvements']['anomaly_reduction']
                agg['total_fp_reduction'] += strategy_result['improvements']['fp_reduction']
                agg['total_sensitivity_change'] += strategy_result['improvements']['sensitivity_change']
                agg['file_count'] += 1
        
        # Calculate averages
        for strategy_name, agg in strategy_aggregates.items():
            if agg['file_count'] > 0:
                comparison_data['strategy_aggregates'][strategy_name] = {
                    'avg_score': agg['total_score'] / agg['file_count'],
                    'avg_anomaly_reduction': agg['total_anomaly_reduction'] / agg['file_count'],
                    'avg_fp_reduction': agg['total_fp_reduction'] / agg['file_count'],
                    'avg_sensitivity_change': agg['total_sensitivity_change'] / agg['file_count'],
                    'files_processed': agg['file_count']
                }
        
        with open(comparison_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        print(f"Strategy comparison saved to: {comparison_path}")
    
    def save_best_representatives(self, results: Dict[str, Any]):
        """Save best representatives for each file."""
        representatives_path = self.output_dir / "clusters" / "best_representatives.json"
        
        best_representatives_data = {
            'analysis_metadata': results['analysis_metadata'],
            'file_representatives': {}
        }
        
        for filename, file_result in results['file_results'].items():
            best_strategy = file_result['best_strategy']
            best_representatives_data['file_representatives'][filename] = {
                'strategy_name': best_strategy['name'],
                'strategy_score': best_strategy['score'],
                'representatives': best_strategy['representatives'],
                'base_metrics': file_result['base_metrics'],
                'metadata': file_result['metadata']
            }
        
        with open(representatives_path, 'w') as f:
            json.dump(best_representatives_data, f, indent=2)
        
        print(f"Best representatives saved to: {representatives_path}")
    
    def calculate_final_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate final metrics after applying best clustering strategies."""
        overall_after_metrics = {
            'total_representatives': 0,
            'total_true_positives': 0,
            'total_false_positives': 0,
            'total_seizure_windows': 0,
            'total_duration_hours': 0.0
        }
        
        # Accumulate metrics from best strategies
        for filename, file_result in results['file_results'].items():
            best_strategy = file_result['best_strategy']
            representatives = best_strategy['representatives']
            metadata = file_result['metadata']
            
            overall_after_metrics['total_representatives'] += len(representatives)
            overall_after_metrics['total_true_positives'] += sum(1 for r in representatives if r['is_true_positive'])
            overall_after_metrics['total_false_positives'] += sum(1 for r in representatives if not r['is_true_positive'])
            overall_after_metrics['total_seizure_windows'] += metadata['num_seizure_windows']
            overall_after_metrics['total_duration_hours'] += metadata['total_duration_hours']
        
        # Calculate final overall metrics
        final_sensitivity = (overall_after_metrics['total_true_positives'] / 
                           overall_after_metrics['total_seizure_windows']) if overall_after_metrics['total_seizure_windows'] > 0 else 0.0
        final_false_alarms_per_hour = (overall_after_metrics['total_false_positives'] / 
                                     overall_after_metrics['total_duration_hours']) if overall_after_metrics['total_duration_hours'] > 0 else 0.0
        
        overall_after_metrics.update({
            'final_sensitivity': final_sensitivity,
            'final_false_alarms_per_hour': final_false_alarms_per_hour
        })
        
        # Calculate overall improvements
        base_metrics = results['overall_base_metrics']
        overall_improvements = {
            'anomaly_reduction': 1.0 - (overall_after_metrics['total_representatives'] / base_metrics['total_anomalies']) if base_metrics['total_anomalies'] > 0 else 0.0,
            'fp_reduction': 1.0 - (overall_after_metrics['total_false_positives'] / base_metrics['total_false_positives']) if base_metrics['total_false_positives'] > 0 else 0.0,
            'sensitivity_change': final_sensitivity - base_metrics['overall_sensitivity'],
            'false_alarms_reduction': base_metrics['overall_false_alarms_per_hour'] - final_false_alarms_per_hour
        }
        
        return {
            'before_clustering': base_metrics,
            'after_clustering': overall_after_metrics,
            'overall_improvements': overall_improvements
        }
    
    def save_final_metrics(self, results: Dict[str, Any]):
        """Save final metrics after clustering."""
        final_metrics = self.calculate_final_metrics(results)
        final_metrics_path = self.output_dir / "metrics_after" / "final_metrics.json"
        
        final_data = {
            'analysis_metadata': results['analysis_metadata'],
            **final_metrics
        }
        
        with open(final_metrics_path, 'w') as f:
            json.dump(final_data, f, indent=2)
        
        print(f"Final metrics saved to: {final_metrics_path}")
        
        return final_metrics
    
    def print_console_summary(self, final_metrics: Dict[str, Any]):
        """Print console summary of clustering results."""
        before = final_metrics['before_clustering']
        after = final_metrics['after_clustering']
        improvements = final_metrics['overall_improvements']
        
        print(f"\n{'='*80}")
        print("MADRID CLUSTERING ANALYSIS SUMMARY")
        print(f"{'='*80}")
        
        print(f"\nBASELINE METRICS (Before Clustering):")
        print(f"  Total Anomalies: {before['total_anomalies']}")
        print(f"  True Positives: {before['total_true_positives']}")
        print(f"  False Positives: {before['total_false_positives']}")
        print(f"  Sensitivity: {before['overall_sensitivity']:.4f} ({before['overall_sensitivity']*100:.2f}%)")
        print(f"  False Alarms/Hour: {before['overall_false_alarms_per_hour']:.4f}")
        
        print(f"\nMETRICS AFTER CLUSTERING:")
        print(f"  Total Representatives: {after['total_representatives']}")
        print(f"  True Positives: {after['total_true_positives']}")
        print(f"  False Positives: {after['total_false_positives']}")
        print(f"  Sensitivity: {after['final_sensitivity']:.4f} ({after['final_sensitivity']*100:.2f}%)")
        print(f"  False Alarms/Hour: {after['final_false_alarms_per_hour']:.4f}")
        
        print(f"\nIMPROVEMENTS:")
        print(f"  Anomaly Reduction: {improvements['anomaly_reduction']:.2%}")
        print(f"  False Positive Reduction: {improvements['fp_reduction']:.2%}")
        print(f"  Sensitivity Change: {improvements['sensitivity_change']:+.4f}")
        print(f"  False Alarms/Hour Reduction: {improvements['false_alarms_reduction']:+.4f}")
        
        print(f"\n{'='*80}")
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run complete clustering analysis pipeline."""
        print("Starting Madrid clustering analysis...")
        
        # Process all files
        results = self.process_all_files()
        if results is None:
            print("No results to process.")
            return None
        
        # Save base metrics
        self.save_base_metrics(results)
        
        # Save strategy comparison
        self.save_strategy_comparison(results)
        
        # Save best representatives
        self.save_best_representatives(results)
        
        # Calculate and save final metrics
        final_metrics = self.save_final_metrics(results)
        
        # Print summary
        self.print_console_summary(final_metrics)
        
        print(f"\nAll results saved to: {self.output_dir}")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Reduce false alarms through clustering analysis of Madrid windowed results"
    )
    parser.add_argument(
        "results_dir", 
        help="Directory containing Madrid windowed results JSON files"
    )
    parser.add_argument(
        "-o", "--output-dir", 
        help="Output directory for clustering results (default: same as results_dir)"
    )
    parser.add_argument(
        "-t", "--threshold", 
        type=float,
        help="Anomaly score threshold for detection (default: use top-ranked anomaly per window)"
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = MadridClusteringAnalyzer(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        threshold=args.threshold
    )
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    if results is None:
        print("Analysis failed.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())