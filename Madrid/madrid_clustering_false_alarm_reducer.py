#!/usr/bin/env python3
"""
Madrid Clustering-based False Alarm Reducer
Reduces false alarms per hour while maintaining sensitivity using temporal clustering.
Works at file level, removing window information before clustering.
"""

import json
import os
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import argparse
from collections import defaultdict


class MadridClusteringFalseAlarmReducer:
    def __init__(self, results_dir: str, output_dir: str = None, threshold: float = None,
                 pre_seizure_minutes: float = 5.0, post_seizure_minutes: float = 3.0):
        """
        Initialize the clustering-based false alarm reducer.
        
        Args:
            results_dir: Directory containing Madrid windowed results JSON files
            output_dir: Directory to save clustering results (default: same as results_dir)
            threshold: Anomaly score threshold for detection (if None, uses top-ranked anomaly per window)
            pre_seizure_minutes: Minutes before seizure start to consider as detection window
            post_seizure_minutes: Minutes after seizure end to consider as detection window
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir) if output_dir else self.results_dir / "clustered_results"
        self.threshold = threshold
        self.pre_seizure_seconds = pre_seizure_minutes * 60.0
        self.post_seizure_seconds = post_seizure_minutes * 60.0
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for organized output
        (self.output_dir / "metrics_before").mkdir(exist_ok=True)
        (self.output_dir / "strategy_comparison").mkdir(exist_ok=True)
        (self.output_dir / "clusters").mkdir(exist_ok=True)
        (self.output_dir / "metrics_after").mkdir(exist_ok=True)
        
        # Define time-based clustering thresholds (2s to 1800s)
        self.time_thresholds = [
            2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 300, 
            420, 600, 900, 1200, 1500, 1800
        ]
    
    def load_result_file(self, filepath: Path) -> Dict[str, Any]:
        """Load and parse a Madrid results JSON file."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def extract_file_level_anomalies(self, result_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract all anomalies from a file and convert to file-level format.
        Removes window information and works with absolute timestamps.
        Uses threshold if provided, otherwise uses top-ranked anomaly per window.
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
            
            # Select anomalies based on threshold or top-ranked strategy
            selected_anomalies = []
            
            if self.threshold is not None:
                # Filter by threshold - include all anomalies >= threshold
                selected_anomalies = [a for a in anomalies if a.get('anomaly_score', 0) >= self.threshold]
            else:
                # Use top-ranked anomaly only (same as extended time metrics)
                selected_anomalies = [anomalies[0]]
            
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
        """Group seizure windows into individual seizures (same as extended time metrics)."""
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
    
    def calculate_base_metrics(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate base metrics before clustering (TP, FP, sensitivity, false alarms/h)."""
        # Extract basic info
        input_data = result_data.get('input_data', {})
        validation_data = result_data.get('validation_data', {})
        
        subject_id = input_data.get('subject_id', 'unknown')
        run_id = input_data.get('run_id', 'unknown')
        
        # Get signal duration
        signal_metadata = input_data.get('signal_metadata', {})
        total_duration_hours = signal_metadata.get('total_duration_seconds', 0) / 3600.0
        
        # Get anomalies at file level
        all_anomalies = self.extract_file_level_anomalies(result_data)
        
        # Get seizures
        ground_truth = validation_data.get('ground_truth', {})
        seizure_windows = ground_truth.get('seizure_windows', [])
        individual_seizures = self.group_seizures_by_time(seizure_windows)
        
        # Calculate TP/FP based on extended time overlap
        true_positives = 0
        false_positives = 0
        detected_seizures = set()
        
        for anomaly in all_anomalies:
            anomaly_time = anomaly['absolute_time']
            is_true_positive = False
            
            # Check if anomaly overlaps with any extended seizure window
            for i, seizure in enumerate(individual_seizures):
                if seizure['extended_start_time'] <= anomaly_time <= seizure['extended_end_time']:
                    is_true_positive = True
                    detected_seizures.add(i)
                    break
            
            if is_true_positive:
                true_positives += 1
            else:
                false_positives += 1
        
        # Calculate sensitivity
        total_seizures = len(individual_seizures)
        sensitivity = len(detected_seizures) / total_seizures if total_seizures > 0 else None
        
        # Calculate false alarms per hour
        false_alarms_per_hour = false_positives / total_duration_hours if total_duration_hours > 0 else 0
        
        return {
            'file_info': {
                'subject_id': subject_id,
                'run_id': run_id,
                'total_duration_hours': total_duration_hours
            },
            'anomalies': all_anomalies,
            'seizures': individual_seizures,
            'metrics': {
                'total_anomalies': len(all_anomalies),
                'true_positives': true_positives,
                'false_positives': false_positives,
                'total_seizures': total_seizures,
                'detected_seizures': len(detected_seizures),
                'sensitivity': sensitivity,
                'false_alarms_per_hour': false_alarms_per_hour
            }
        }
    
    def time_based_clustering(self, anomalies: List[Dict[str, Any]], 
                            time_threshold_seconds: float) -> List[List[Dict[str, Any]]]:
        """
        Perform time-based clustering with fixed time windows.
        
        Args:
            anomalies: List of anomalies sorted by absolute_time
            time_threshold_seconds: Maximum time distance for clustering
        
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
                'tp_count': 0,  # Will be calculated later
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
            'tp_count': 0,  # Will be calculated later
            'min_score': min(scores),
            'max_score': max(scores),
            'mean_time_distance': min_mean_distance,
            'cluster_time_span': max(times) - min(times),
            'original_cluster_anomalies': [a['absolute_time'] for a in cluster]
        }
        
        return representative
    
    def evaluate_clustering_strategy(self, base_metrics: Dict[str, Any], 
                                   representatives: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate a clustering strategy by calculating metrics on representatives.
        
        Args:
            base_metrics: Original metrics before clustering
            representatives: List of cluster representatives
        
        Returns:
            Evaluation results with metrics and improvements
        """
        if not representatives:
            return {
                'metrics': {
                    'total_anomalies': 0,
                    'true_positives': 0,
                    'false_positives': 0,
                    'sensitivity': 0.0,
                    'false_alarms_per_hour': 0.0
                },
                'improvements': {
                    'anomaly_reduction': 1.0,
                    'fp_reduction': 1.0,
                    'sensitivity_change': -base_metrics['metrics']['sensitivity'] if base_metrics['metrics']['sensitivity'] else 0.0
                }
            }
        
        # Calculate TP/FP for representatives
        seizures = base_metrics['seizures']
        total_duration_hours = base_metrics['file_info']['total_duration_hours']
        
        true_positives = 0
        false_positives = 0
        detected_seizures = set()
        
        for rep in representatives:
            rep_time = rep['absolute_time']
            is_true_positive = False
            
            # Check if representative overlaps with any extended seizure window
            for i, seizure in enumerate(seizures):
                if seizure['extended_start_time'] <= rep_time <= seizure['extended_end_time']:
                    is_true_positive = True
                    detected_seizures.add(i)
                    rep['cluster_metadata']['tp_count'] = 1
                    break
            
            if is_true_positive:
                true_positives += 1
            else:
                false_positives += 1
        
        # Calculate new metrics
        total_seizures = base_metrics['metrics']['total_seizures']
        new_sensitivity = len(detected_seizures) / total_seizures if total_seizures > 0 else None
        new_false_alarms_per_hour = false_positives / total_duration_hours if total_duration_hours > 0 else 0
        
        # Calculate improvements
        base_total = base_metrics['metrics']['total_anomalies']
        base_fp = base_metrics['metrics']['false_positives']
        base_sensitivity = base_metrics['metrics']['sensitivity']
        
        anomaly_reduction = (base_total - len(representatives)) / base_total if base_total > 0 else 0
        fp_reduction = (base_fp - false_positives) / base_fp if base_fp > 0 else 0
        sensitivity_change = (new_sensitivity - base_sensitivity) if (new_sensitivity is not None and base_sensitivity is not None) else 0
        
        # Calculate score function: 0.6*fp_reduction + 0.3*anomaly_reduction - 2.0*max(0, base_sens - sens)
        sensitivity_penalty = max(0, base_sensitivity - new_sensitivity) if (base_sensitivity is not None and new_sensitivity is not None) else 0
        score = 0.6 * fp_reduction + 0.3 * anomaly_reduction - 2.0 * sensitivity_penalty
        
        return {
            'metrics': {
                'total_anomalies': len(representatives),
                'true_positives': true_positives,
                'false_positives': false_positives,
                'total_seizures': total_seizures,
                'detected_seizures': len(detected_seizures),
                'sensitivity': new_sensitivity,
                'false_alarms_per_hour': new_false_alarms_per_hour
            },
            'improvements': {
                'anomaly_reduction': anomaly_reduction,
                'fp_reduction': fp_reduction,
                'sensitivity_change': sensitivity_change,
                'score': score
            },
            'representatives': representatives
        }
    
    def process_single_file(self, filepath: Path) -> Dict[str, Any]:
        """Process a single file through clustering analysis."""
        print(f"Processing {filepath.name}...")
        
        # Load result data
        result_data = self.load_result_file(filepath)
        if result_data is None:
            return None
        
        # Calculate base metrics
        base_metrics = self.calculate_base_metrics(result_data)
        
        if base_metrics['metrics']['total_anomalies'] == 0:
            print(f"  No anomalies found in {filepath.name}, skipping clustering")
            return {
                'file_info': base_metrics['file_info'],
                'base_metrics': base_metrics['metrics'],
                'best_strategy': None,
                'strategy_results': {}
            }
        
        # Test all clustering strategies
        strategy_results = {}
        anomalies = base_metrics['anomalies']
        
        for threshold in self.time_thresholds:
            # Perform clustering
            clusters = self.time_based_clustering(anomalies, threshold)
            
            # Select representatives
            representatives = [self.select_cluster_representative(cluster) for cluster in clusters]
            
            # Evaluate strategy
            evaluation = self.evaluate_clustering_strategy(base_metrics, representatives)
            
            strategy_results[f"time_{threshold}s"] = {
                'time_threshold_seconds': threshold,
                'num_clusters': len(clusters),
                'cluster_sizes': [len(cluster) for cluster in clusters],
                **evaluation
            }
        
        # Find best strategy based on score
        best_strategy_name = None
        best_score = float('-inf')
        
        for strategy_name, results in strategy_results.items():
            if results['improvements']['score'] > best_score:
                best_score = results['improvements']['score']
                best_strategy_name = strategy_name
        
        return {
            'file_info': base_metrics['file_info'],
            'base_metrics': base_metrics['metrics'],
            'best_strategy': {
                'name': best_strategy_name,
                'score': best_score,
                **strategy_results[best_strategy_name]
            } if best_strategy_name else None,
            'strategy_results': strategy_results
        }
    
    def process_all_files(self) -> Dict[str, Any]:
        """Process all Madrid result files."""
        json_files = list(self.results_dir.glob("madrid_windowed_results_*.json"))
        
        if not json_files:
            print(f"No Madrid result files found in {self.results_dir}")
            return None
        
        print(f"Found {len(json_files)} Madrid result files")
        
        all_file_results = {}
        overall_base_stats = {
            'total_files': 0,
            'total_anomalies': 0,
            'total_true_positives': 0,
            'total_false_positives': 0,
            'total_seizures': 0,
            'total_detected_seizures': 0,
            'total_duration_hours': 0.0
        }
        
        overall_best_stats = {
            'total_anomalies': 0,
            'total_true_positives': 0,
            'total_false_positives': 0,
            'total_detected_seizures': 0
        }
        
        # Process each file
        for json_file in sorted(json_files):
            file_result = self.process_single_file(json_file)
            if file_result is None:
                continue
            
            all_file_results[json_file.name] = file_result
            
            # Update overall base stats
            base_metrics = file_result['base_metrics']
            overall_base_stats['total_files'] += 1
            overall_base_stats['total_anomalies'] += base_metrics['total_anomalies']
            overall_base_stats['total_true_positives'] += base_metrics['true_positives']
            overall_base_stats['total_false_positives'] += base_metrics['false_positives']
            overall_base_stats['total_seizures'] += base_metrics['total_seizures']
            overall_base_stats['total_detected_seizures'] += base_metrics['detected_seizures']
            overall_base_stats['total_duration_hours'] += file_result['file_info']['total_duration_hours']
            
            # Update overall best stats
            if file_result['best_strategy']:
                best_metrics = file_result['best_strategy']['metrics']
                overall_best_stats['total_anomalies'] += best_metrics['total_anomalies']
                overall_best_stats['total_true_positives'] += best_metrics['true_positives']
                overall_best_stats['total_false_positives'] += best_metrics['false_positives']
                overall_best_stats['total_detected_seizures'] += best_metrics['detected_seizures']
            else:
                # No clustering applied
                overall_best_stats['total_anomalies'] += base_metrics['total_anomalies']
                overall_best_stats['total_true_positives'] += base_metrics['true_positives']
                overall_best_stats['total_false_positives'] += base_metrics['false_positives']
                overall_best_stats['total_detected_seizures'] += base_metrics['detected_seizures']
        
        # Calculate overall metrics
        overall_base_sensitivity = (overall_base_stats['total_detected_seizures'] / 
                                  overall_base_stats['total_seizures'] 
                                  if overall_base_stats['total_seizures'] > 0 else None)
        
        overall_base_false_alarms_per_hour = (overall_base_stats['total_false_positives'] / 
                                            overall_base_stats['total_duration_hours'] 
                                            if overall_base_stats['total_duration_hours'] > 0 else 0)
        
        overall_best_sensitivity = (overall_best_stats['total_detected_seizures'] / 
                                  overall_base_stats['total_seizures'] 
                                  if overall_base_stats['total_seizures'] > 0 else None)
        
        overall_best_false_alarms_per_hour = (overall_best_stats['total_false_positives'] / 
                                            overall_base_stats['total_duration_hours'] 
                                            if overall_base_stats['total_duration_hours'] > 0 else 0)
        
        # Calculate overall improvements
        overall_anomaly_reduction = ((overall_base_stats['total_anomalies'] - overall_best_stats['total_anomalies']) / 
                                   overall_base_stats['total_anomalies'] 
                                   if overall_base_stats['total_anomalies'] > 0 else 0)
        
        overall_fp_reduction = ((overall_base_stats['total_false_positives'] - overall_best_stats['total_false_positives']) / 
                              overall_base_stats['total_false_positives'] 
                              if overall_base_stats['total_false_positives'] > 0 else 0)
        
        overall_sensitivity_change = ((overall_best_sensitivity - overall_base_sensitivity) 
                                    if (overall_best_sensitivity is not None and overall_base_sensitivity is not None) else 0)
        
        return {
            'analysis_metadata': {
                'calculation_timestamp': datetime.now().isoformat(),
                'results_directory': str(self.results_dir),
                'output_directory': str(self.output_dir),
                'threshold_used': self.threshold,
                'detection_strategy': 'threshold' if self.threshold is not None else 'top_ranked',
                'time_thresholds_tested': self.time_thresholds,
                'pre_seizure_minutes': self.pre_seizure_seconds / 60.0,
                'post_seizure_minutes': self.post_seizure_seconds / 60.0
            },
            'overall_results': {
                'base_metrics': {
                    **overall_base_stats,
                    'sensitivity': overall_base_sensitivity,
                    'false_alarms_per_hour': overall_base_false_alarms_per_hour
                },
                'best_metrics': {
                    **overall_best_stats,
                    'sensitivity': overall_best_sensitivity,
                    'false_alarms_per_hour': overall_best_false_alarms_per_hour
                },
                'improvements': {
                    'anomaly_reduction': overall_anomaly_reduction,
                    'fp_reduction': overall_fp_reduction,
                    'sensitivity_change': overall_sensitivity_change
                }
            },
            'individual_results': all_file_results
        }
    
    def save_results(self, results: Dict[str, Any]):
        """Save all clustering results in organized directory structure."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save base metrics (before clustering)
        base_metrics_summary = {
            'calculation_timestamp': results['analysis_metadata']['calculation_timestamp'],
            'total_files': results['overall_results']['base_metrics']['total_files'],
            'base_metrics': results['overall_results']['base_metrics']
        }
        
        base_metrics_path = self.output_dir / "metrics_before" / f"metrics_summary_{timestamp}.json"
        with open(base_metrics_path, 'w') as f:
            json.dump(base_metrics_summary, f, indent=2)
        print(f"Base metrics saved to: {base_metrics_path}")
        
        # 2. Save strategy comparison
        strategy_comparison_path = self.output_dir / "strategy_comparison" / f"comparison_{timestamp}.json"
        with open(strategy_comparison_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Strategy comparison saved to: {strategy_comparison_path}")
        
        # 3. Save best representatives
        all_representatives = []
        for file_name, file_result in results['individual_results'].items():
            if file_result['best_strategy'] and 'representatives' in file_result['best_strategy']:
                for rep in file_result['best_strategy']['representatives']:
                    rep_with_file = {
                        'file': file_name,
                        'subject_id': file_result['file_info']['subject_id'],
                        'run_id': file_result['file_info']['run_id'],
                        **rep
                    }
                    all_representatives.append(rep_with_file)
        
        best_representatives = {
            'calculation_timestamp': results['analysis_metadata']['calculation_timestamp'],
            'total_representatives': len(all_representatives),
            'representatives': all_representatives
        }
        
        representatives_path = self.output_dir / "clusters" / f"best_representatives_{timestamp}.json"
        with open(representatives_path, 'w') as f:
            json.dump(best_representatives, f, indent=2)
        print(f"Best representatives saved to: {representatives_path}")
        
        # 4. Save final metrics (after clustering)
        final_metrics = {
            'calculation_timestamp': results['analysis_metadata']['calculation_timestamp'],
            'before_clustering': results['overall_results']['base_metrics'],
            'after_clustering': results['overall_results']['best_metrics'],
            'improvements': results['overall_results']['improvements']
        }
        
        final_metrics_path = self.output_dir / "metrics_after" / f"final_metrics_{timestamp}.json"
        with open(final_metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        print(f"Final metrics saved to: {final_metrics_path}")
        
        # 5. Save console summary
        self.print_console_summary(results)
    
    def print_console_summary(self, results: Dict[str, Any]):
        """Print summary to console."""
        base = results['overall_results']['base_metrics']
        best = results['overall_results']['best_metrics']
        improvements = results['overall_results']['improvements']
        
        print(f"\n{'='*60}")
        print("CLUSTERING-BASED FALSE ALARM REDUCTION SUMMARY")
        print(f"{'='*60}")
        
        print(f"Files processed: {base['total_files']}")
        print(f"Total duration: {base['total_duration_hours']:.2f} hours")
        print(f"Detection strategy: {results['analysis_metadata']['detection_strategy']}")
        if results['analysis_metadata']['threshold_used'] is not None:
            print(f"Threshold used: {results['analysis_metadata']['threshold_used']}")
        
        print(f"\nBEFORE CLUSTERING:")
        print(f"  Total anomalies: {base['total_anomalies']}")
        print(f"  True positives: {base['total_true_positives']}")
        print(f"  False positives: {base['total_false_positives']}")
        if base['sensitivity'] is not None:
            print(f"  Sensitivity: {base['sensitivity']:.4f} ({base['sensitivity']*100:.2f}%)")
        print(f"  False alarms/hour: {base['false_alarms_per_hour']:.4f}")
        
        print(f"\nAFTER CLUSTERING (BEST STRATEGIES):")
        print(f"  Total anomalies: {best['total_anomalies']}")
        print(f"  True positives: {best['total_true_positives']}")
        print(f"  False positives: {best['total_false_positives']}")
        if best['sensitivity'] is not None:
            print(f"  Sensitivity: {best['sensitivity']:.4f} ({best['sensitivity']*100:.2f}%)")
        print(f"  False alarms/hour: {best['false_alarms_per_hour']:.4f}")
        
        print(f"\nIMPROVEMENTS:")
        print(f"  Anomaly reduction: {improvements['anomaly_reduction']:.4f} ({improvements['anomaly_reduction']*100:.2f}%)")
        print(f"  FP reduction: {improvements['fp_reduction']:.4f} ({improvements['fp_reduction']*100:.2f}%)")
        print(f"  Sensitivity change: {improvements['sensitivity_change']:+.4f}")
        
        print(f"\nOUTPUT DIRECTORY: {self.output_dir}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Reduce false alarms using temporal clustering while maintaining sensitivity"
    )
    parser.add_argument(
        "results_dir", 
        help="Directory containing Madrid windowed results JSON files"
    )
    parser.add_argument(
        "-o", "--output-dir", 
        help="Output directory for clustering results (default: results_dir/clustered_results)"
    )
    parser.add_argument(
        "-t", "--threshold", 
        type=float,
        help="Anomaly score threshold for detection (default: use top-ranked anomaly per window)"
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
    
    # Initialize reducer
    reducer = MadridClusteringFalseAlarmReducer(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
        pre_seizure_minutes=args.pre_seizure_minutes,
        post_seizure_minutes=args.post_seizure_minutes
    )
    
    # Process all files
    results = reducer.process_all_files()
    
    if results is None:
        print("No results to process.")
        return
    
    # Save results
    reducer.save_results(results)


if __name__ == "__main__":
    main()