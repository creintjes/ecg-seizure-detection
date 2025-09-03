#!/usr/bin/env python3
"""
Madrid Clustering-based False Alarm Reducer with Train/Test Split
Uses sub001-sub096 for strategy selection (training)
Uses sub097-sub125 for final evaluation (testing)
"""

import json
import os
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import argparse
from collections import defaultdict


class MadridClusteringFalseAlarmReducerTrainTest:
    def __init__(self, results_dir: str, output_dir: str = None, threshold: float = None,
                 pre_seizure_minutes: float = 5.0, post_seizure_minutes: float = 3.0):
        """
        Initialize the clustering-based false alarm reducer with train/test split.
        
        Args:
            results_dir: Directory containing Madrid windowed results JSON files
            output_dir: Directory to save clustering results (default: same as results_dir)
            threshold: Anomaly score threshold for detection (if None, uses top-ranked anomaly per window)
            pre_seizure_minutes: Minutes before seizure start to consider as detection window
            post_seizure_minutes: Minutes after seizure end to consider as detection window
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir) if output_dir else self.results_dir / "train_test_clustered_results"
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
    
    def calculate_responder_metrics(self, patient_metrics: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Calculate responder metrics for the test set.
        A responder is a patient where at least 2/3 of seizures are detected.
        
        Args:
            patient_metrics: Dictionary with patient-specific seizure detection metrics
        
        Returns:
            Dictionary with responder analysis results
        """
        responders = []
        non_responders = []
        patients_with_seizures = []
        
        for patient_id, metrics in patient_metrics.items():
            if metrics['total_seizures'] > 0:
                patients_with_seizures.append(patient_id)
                detection_rate = metrics['detected_seizures'] / metrics['total_seizures']
                
                patient_info = {
                    'patient_id': patient_id,
                    'total_seizures': metrics['total_seizures'],
                    'detected_seizures': metrics['detected_seizures'],
                    'detection_rate': detection_rate,
                    'files': metrics['files']
                }
                
                # Check if patient is a responder (≥2/3 seizures detected)
                if detection_rate >= 2/3:
                    responders.append(patient_info)
                else:
                    non_responders.append(patient_info)
        
        # Calculate responder-specific sensitivity
        total_seizures_responders = sum(p['total_seizures'] for p in responders)
        detected_seizures_responders = sum(p['detected_seizures'] for p in responders)
        
        responder_sensitivity = (detected_seizures_responders / total_seizures_responders 
                                if total_seizures_responders > 0 else None)
        
        # Calculate non-responder sensitivity for comparison
        total_seizures_non_responders = sum(p['total_seizures'] for p in non_responders)
        detected_seizures_non_responders = sum(p['detected_seizures'] for p in non_responders)
        
        non_responder_sensitivity = (detected_seizures_non_responders / total_seizures_non_responders 
                                    if total_seizures_non_responders > 0 else None)
        
        return {
            'total_patients_with_seizures': len(patients_with_seizures),
            'num_responders': len(responders),
            'num_non_responders': len(non_responders),
            'responder_rate': len(responders) / len(patients_with_seizures) if patients_with_seizures else 0,
            'responder_sensitivity': responder_sensitivity,
            'non_responder_sensitivity': non_responder_sensitivity,
            'responders': responders,
            'non_responders': non_responders,
            'total_seizures_responders': total_seizures_responders,
            'detected_seizures_responders': detected_seizures_responders,
            'total_seizures_non_responders': total_seizures_non_responders,
            'detected_seizures_non_responders': detected_seizures_non_responders
        }
    
    def is_training_file(self, filename: str) -> bool:
        """
        Determine if a file belongs to the training set (sub001-sub096).
        
        Args:
            filename: Name of the file (e.g., "madrid_windowed_results_sub-077_run-04_20250730_040717.json")
        
        Returns:
            True if file is in training set, False otherwise
        """
        # Extract subject ID from filename
        import re
        match = re.search(r'sub-(\d{3})', filename)
        if match:
            subject_num = int(match.group(1))
            return 1 <= subject_num <= 96
        return False
    
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
    
    def process_single_file_for_strategy_evaluation(self, filepath: Path) -> Dict[str, Any]:
        """Process a single file and evaluate all clustering strategies (Phase 1)."""
        print(f"Evaluating strategies for {filepath.name}...")
        
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
        
        return {
            'file_info': base_metrics['file_info'],
            'base_metrics': base_metrics['metrics'],
            'strategy_results': strategy_results
        }
    
    def process_single_file_with_global_strategy(self, filepath: Path, global_strategy_name: str) -> Dict[str, Any]:
        """Process a single file with the globally selected best strategy (Phase 2)."""
        print(f"Applying global strategy {global_strategy_name} to {filepath.name}...")
        
        # Load result data
        result_data = self.load_result_file(filepath)
        if result_data is None:
            return None
        
        # Calculate base metrics
        base_metrics = self.calculate_base_metrics(result_data)
        
        if base_metrics['metrics']['total_anomalies'] == 0:
            return {
                'file_info': base_metrics['file_info'],
                'base_metrics': base_metrics['metrics'],
                'applied_strategy': None
            }
        
        # Extract time threshold from strategy name (e.g., "time_30s" -> 30)
        time_threshold = int(global_strategy_name.replace('time_', '').replace('s', ''))
        anomalies = base_metrics['anomalies']
        
        # Apply the global strategy
        clusters = self.time_based_clustering(anomalies, time_threshold)
        representatives = [self.select_cluster_representative(cluster) for cluster in clusters]
        evaluation = self.evaluate_clustering_strategy(base_metrics, representatives)
        
        applied_strategy = {
            'name': global_strategy_name,
            'time_threshold_seconds': time_threshold,
            'num_clusters': len(clusters),
            'cluster_sizes': [len(cluster) for cluster in clusters],
            **evaluation
        }
        
        return {
            'file_info': base_metrics['file_info'],
            'base_metrics': base_metrics['metrics'],
            'applied_strategy': applied_strategy
        }
    
    def process_all_files(self) -> Dict[str, Any]:
        """Process all Madrid result files with train/test split for global strategy selection."""
        json_files = list(self.results_dir.glob("madrid_windowed_results_*.json"))
        
        if not json_files:
            print(f"No Madrid result files found in {self.results_dir}")
            return None
        
        # Split files into training and test sets
        training_files = []
        test_files = []
        
        for json_file in json_files:
            if self.is_training_file(json_file.name):
                training_files.append(json_file)
            else:
                test_files.append(json_file)
        
        print(f"Found {len(json_files)} Madrid result files")
        print(f"Training set (sub001-sub096): {len(training_files)} files")
        print(f"Test set (sub097-sub125): {len(test_files)} files")
        
        print(f"\nPHASE 1: Evaluating all strategies on TRAINING SET (sub001-sub096)...")
        
        # Phase 1: Evaluate all strategies on TRAINING SET ONLY
        training_file_evaluations = {}
        training_base_stats = {
            'total_files': 0,
            'total_anomalies': 0,
            'total_true_positives': 0,
            'total_false_positives': 0,
            'total_seizures': 0,
            'total_detected_seizures': 0,
            'total_duration_hours': 0.0
        }
        
        # Collect strategy performance on TRAINING SET
        training_strategy_performance = defaultdict(lambda: {
            'total_score': 0.0,
            'total_anomaly_reduction': 0.0,
            'total_fp_reduction': 0.0,
            'total_sensitivity_change': 0.0,
            'total_anomalies_before': 0,
            'total_anomalies_after': 0,
            'total_fp_before': 0,
            'total_fp_after': 0,
            'total_tp_after': 0,
            'total_detected_seizures': 0,
            'file_count': 0
        })
        
        for json_file in sorted(training_files):
            file_result = self.process_single_file_for_strategy_evaluation(json_file)
            if file_result is None:
                continue
            
            training_file_evaluations[json_file.name] = file_result
            
            # Update training base stats
            base_metrics = file_result['base_metrics']
            training_base_stats['total_files'] += 1
            training_base_stats['total_anomalies'] += base_metrics['total_anomalies']
            training_base_stats['total_true_positives'] += base_metrics['true_positives']
            training_base_stats['total_false_positives'] += base_metrics['false_positives']
            training_base_stats['total_seizures'] += base_metrics['total_seizures']
            training_base_stats['total_detected_seizures'] += base_metrics['detected_seizures']
            training_base_stats['total_duration_hours'] += file_result['file_info']['total_duration_hours']
            
            # Aggregate strategy performance
            strategy_results = file_result['strategy_results']
            for strategy_name, strategy_data in strategy_results.items():
                perf = training_strategy_performance[strategy_name]
                improvements = strategy_data['improvements']
                metrics = strategy_data['metrics']
                
                perf['total_score'] += improvements['score']
                perf['total_anomaly_reduction'] += improvements['anomaly_reduction']
                perf['total_fp_reduction'] += improvements['fp_reduction']
                perf['total_sensitivity_change'] += improvements['sensitivity_change']
                perf['total_anomalies_before'] += base_metrics['total_anomalies']
                perf['total_anomalies_after'] += metrics['total_anomalies']
                perf['total_fp_before'] += base_metrics['false_positives']
                perf['total_fp_after'] += metrics['false_positives']
                perf['total_tp_after'] += metrics['true_positives']
                perf['total_detected_seizures'] += metrics['detected_seizures']
                perf['file_count'] += 1
        
        # Phase 2: Find best strategy based on TRAINING SET ONLY
        print(f"\nPHASE 2: Selecting best strategy based on TRAINING SET...")
        
        best_global_strategy = None
        best_global_score = float('-inf')
        
        training_strategy_summary = {}
        for strategy_name, perf in training_strategy_performance.items():
            if perf['file_count'] > 0:
                # Calculate global metrics for this strategy on training set
                avg_score = perf['total_score'] / perf['file_count']
                global_anomaly_reduction = ((perf['total_anomalies_before'] - perf['total_anomalies_after']) / 
                                           perf['total_anomalies_before'] if perf['total_anomalies_before'] > 0 else 0)
                global_fp_reduction = ((perf['total_fp_before'] - perf['total_fp_after']) / 
                                     perf['total_fp_before'] if perf['total_fp_before'] > 0 else 0)
                
                # Use average score as global score
                global_score = avg_score
                
                training_strategy_summary[strategy_name] = {
                    'avg_score': avg_score,
                    'global_score': global_score,
                    'file_count': perf['file_count'],
                    'global_anomaly_reduction': global_anomaly_reduction,
                    'global_fp_reduction': global_fp_reduction,
                    'total_anomalies_before': perf['total_anomalies_before'],
                    'total_anomalies_after': perf['total_anomalies_after'],
                    'total_fp_before': perf['total_fp_before'],
                    'total_fp_after': perf['total_fp_after'],
                    'total_tp_after': perf['total_tp_after'],
                    'total_detected_seizures': perf['total_detected_seizures']
                }
                
                if global_score > best_global_score:
                    best_global_score = global_score
                    best_global_strategy = strategy_name
        
        print(f"Selected best strategy from training set: {best_global_strategy} (score: {best_global_score:.4f})")
        
        # Phase 3: Apply best strategy to TEST SET
        print(f"\nPHASE 3: Applying best strategy to TEST SET (sub097-sub125)...")
        
        test_file_results = {}
        test_base_stats = {
            'total_files': 0,
            'total_anomalies': 0,
            'total_true_positives': 0,
            'total_false_positives': 0,
            'total_seizures': 0,
            'total_detected_seizures': 0,
            'total_duration_hours': 0.0
        }
        test_final_stats = {
            'total_anomalies': 0,
            'total_true_positives': 0,
            'total_false_positives': 0,
            'total_detected_seizures': 0
        }
        
        # Track per-patient metrics for responder analysis
        patient_metrics = {}
        
        for json_file in sorted(test_files):
            file_result = self.process_single_file_with_global_strategy(json_file, best_global_strategy)
            if file_result is None:
                continue
            
            test_file_results[json_file.name] = file_result
            
            # Update test base stats (before clustering)
            base_metrics = file_result['base_metrics']
            test_base_stats['total_files'] += 1
            test_base_stats['total_anomalies'] += base_metrics['total_anomalies']
            test_base_stats['total_true_positives'] += base_metrics['true_positives']
            test_base_stats['total_false_positives'] += base_metrics['false_positives']
            test_base_stats['total_seizures'] += base_metrics['total_seizures']
            test_base_stats['total_detected_seizures'] += base_metrics['detected_seizures']
            test_base_stats['total_duration_hours'] += file_result['file_info']['total_duration_hours']
            
            # Track per-patient seizure detection for responder analysis
            subject_id = file_result['file_info']['subject_id']
            if subject_id not in patient_metrics:
                patient_metrics[subject_id] = {
                    'total_seizures': 0,
                    'detected_seizures': 0,
                    'files': []
                }
            
            patient_metrics[subject_id]['files'].append(json_file.name)
            patient_metrics[subject_id]['total_seizures'] += base_metrics['total_seizures']
            
            # Update test final stats (after clustering)
            if file_result['applied_strategy']:
                final_metrics = file_result['applied_strategy']['metrics']
                test_final_stats['total_anomalies'] += final_metrics['total_anomalies']
                test_final_stats['total_true_positives'] += final_metrics['true_positives']
                test_final_stats['total_false_positives'] += final_metrics['false_positives']
                test_final_stats['total_detected_seizures'] += final_metrics['detected_seizures']
                
                # Update patient-specific detected seizures
                patient_metrics[subject_id]['detected_seizures'] += final_metrics['detected_seizures']
            else:
                # No clustering applied
                test_final_stats['total_anomalies'] += base_metrics['total_anomalies']
                test_final_stats['total_true_positives'] += base_metrics['true_positives']
                test_final_stats['total_false_positives'] += base_metrics['false_positives']
                test_final_stats['total_detected_seizures'] += base_metrics['detected_seizures']
                
                # Update patient-specific detected seizures
                patient_metrics[subject_id]['detected_seizures'] += base_metrics['detected_seizures']
        
        # Calculate responder metrics
        responder_analysis = self.calculate_responder_metrics(patient_metrics)
        
        # Calculate test set metrics
        test_base_sensitivity = (test_base_stats['total_detected_seizures'] / 
                                test_base_stats['total_seizures'] 
                                if test_base_stats['total_seizures'] > 0 else None)
        
        test_base_false_alarms_per_hour = (test_base_stats['total_false_positives'] / 
                                          test_base_stats['total_duration_hours'] 
                                          if test_base_stats['total_duration_hours'] > 0 else 0)
        
        test_final_sensitivity = (test_final_stats['total_detected_seizures'] / 
                                 test_base_stats['total_seizures'] 
                                 if test_base_stats['total_seizures'] > 0 else None)
        
        test_final_false_alarms_per_hour = (test_final_stats['total_false_positives'] / 
                                           test_base_stats['total_duration_hours'] 
                                           if test_base_stats['total_duration_hours'] > 0 else 0)
        
        # Calculate test set improvements
        test_anomaly_reduction = ((test_base_stats['total_anomalies'] - test_final_stats['total_anomalies']) / 
                                 test_base_stats['total_anomalies'] 
                                 if test_base_stats['total_anomalies'] > 0 else 0)
        
        test_fp_reduction = ((test_base_stats['total_false_positives'] - test_final_stats['total_false_positives']) / 
                            test_base_stats['total_false_positives'] 
                            if test_base_stats['total_false_positives'] > 0 else 0)
        
        test_sensitivity_change = ((test_final_sensitivity - test_base_sensitivity) 
                                  if (test_final_sensitivity is not None and test_base_sensitivity is not None) else 0)
        
        return {
            'analysis_metadata': {
                'calculation_timestamp': datetime.now().isoformat(),
                'results_directory': str(self.results_dir),
                'output_directory': str(self.output_dir),
                'threshold_used': self.threshold,
                'detection_strategy': 'threshold' if self.threshold is not None else 'top_ranked',
                'time_thresholds_tested': self.time_thresholds,
                'pre_seizure_minutes': self.pre_seizure_seconds / 60.0,
                'post_seizure_minutes': self.post_seizure_seconds / 60.0,
                'best_strategy_from_training': best_global_strategy,
                'best_strategy_training_score': best_global_score,
                'training_files_count': len(training_files),
                'test_files_count': len(test_files)
            },
            'training_results': {
                'file_evaluations': training_file_evaluations,
                'strategy_summary': training_strategy_summary,
                'base_stats': training_base_stats
            },
            'test_results': {
                'base_metrics': {
                    **test_base_stats,
                    'sensitivity': test_base_sensitivity,
                    'false_alarms_per_hour': test_base_false_alarms_per_hour
                },
                'final_metrics': {
                    **test_final_stats,
                    'sensitivity': test_final_sensitivity,
                    'false_alarms_per_hour': test_final_false_alarms_per_hour
                },
                'improvements': {
                    'anomaly_reduction': test_anomaly_reduction,
                    'fp_reduction': test_fp_reduction,
                    'sensitivity_change': test_sensitivity_change
                },
                'responder_analysis': responder_analysis,
                'individual_results': test_file_results
            }
        }
    
    def save_results(self, results: Dict[str, Any]):
        """Save all clustering results in organized directory structure."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save test set base metrics (before clustering)
        test_base_metrics_summary = {
            'calculation_timestamp': results['analysis_metadata']['calculation_timestamp'],
            'test_files_count': results['analysis_metadata']['test_files_count'],
            'test_base_metrics': results['test_results']['base_metrics']
        }
        
        base_metrics_path = self.output_dir / "metrics_before" / f"test_metrics_summary_{timestamp}.json"
        with open(base_metrics_path, 'w') as f:
            json.dump(test_base_metrics_summary, f, indent=2)
        print(f"Test set base metrics saved to: {base_metrics_path}")
        
        # 2. Save training strategy comparison
        strategy_comparison = {
            'metadata': results['analysis_metadata'],
            'training_strategy_summary': results['training_results']['strategy_summary'],
            'best_strategy': results['analysis_metadata']['best_strategy_from_training']
        }
        
        strategy_comparison_path = self.output_dir / "strategy_comparison" / f"training_strategy_selection_{timestamp}.json"
        with open(strategy_comparison_path, 'w') as f:
            json.dump(strategy_comparison, f, indent=2)
        print(f"Training strategy selection saved to: {strategy_comparison_path}")
        
        # 3. Save test set representatives (from best strategy)
        test_representatives = []
        best_strategy = results['analysis_metadata']['best_strategy_from_training']
        
        for file_name, file_result in results['test_results']['individual_results'].items():
            if file_result['applied_strategy'] and 'representatives' in file_result['applied_strategy']:
                for rep in file_result['applied_strategy']['representatives']:
                    rep_with_file = {
                        'file': file_name,
                        'subject_id': file_result['file_info']['subject_id'],
                        'run_id': file_result['file_info']['run_id'],
                        'strategy_applied': best_strategy,
                        **rep
                    }
                    test_representatives.append(rep_with_file)
        
        test_representatives_data = {
            'calculation_timestamp': results['analysis_metadata']['calculation_timestamp'],
            'strategy_applied': best_strategy,
            'dataset': 'test_set (sub097-sub125)',
            'total_representatives': len(test_representatives),
            'representatives': test_representatives
        }
        
        representatives_path = self.output_dir / "clusters" / f"test_representatives_{timestamp}.json"
        with open(representatives_path, 'w') as f:
            json.dump(test_representatives_data, f, indent=2)
        print(f"Test set representatives saved to: {representatives_path}")
        
        # 4. Save test set final metrics (after clustering)
        test_final_metrics = {
            'calculation_timestamp': results['analysis_metadata']['calculation_timestamp'],
            'strategy_applied': best_strategy,
            'strategy_selected_from': 'training_set (sub001-sub096)',
            'evaluated_on': 'test_set (sub097-sub125)',
            'before_clustering': results['test_results']['base_metrics'],
            'after_clustering': results['test_results']['final_metrics'],
            'improvements': results['test_results']['improvements'],
            'responder_analysis': results['test_results']['responder_analysis']
        }
        
        final_metrics_path = self.output_dir / "metrics_after" / f"test_final_metrics_{timestamp}.json"
        with open(final_metrics_path, 'w') as f:
            json.dump(test_final_metrics, f, indent=2)
        print(f"Test set final metrics saved to: {final_metrics_path}")
        
        # 5. Save complete results
        complete_results_path = self.output_dir / f"complete_train_test_results_{timestamp}.json"
        with open(complete_results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Complete results saved to: {complete_results_path}")
        
        # 6. Print console summary
        self.print_console_summary(results)
    
    def print_console_summary(self, results: Dict[str, Any]):
        """Print summary to console."""
        test_base = results['test_results']['base_metrics']
        test_final = results['test_results']['final_metrics']
        test_improvements = results['test_results']['improvements']
        best_strategy = results['analysis_metadata']['best_strategy_from_training']
        best_score = results['analysis_metadata']['best_strategy_training_score']
        
        print(f"\n{'='*70}")
        print("TRAIN/TEST SPLIT CLUSTERING-BASED FALSE ALARM REDUCTION SUMMARY")
        print(f"{'='*70}")
        
        print(f"\nDATA SPLIT:")
        print(f"Training set (sub001-sub096): {results['analysis_metadata']['training_files_count']} files")
        print(f"Test set (sub097-sub125): {results['analysis_metadata']['test_files_count']} files")
        
        print(f"\nDETECTION STRATEGY: {results['analysis_metadata']['detection_strategy']}")
        if results['analysis_metadata']['threshold_used'] is not None:
            print(f"Threshold used: {results['analysis_metadata']['threshold_used']}")
        
        print(f"\nBEST STRATEGY SELECTED FROM TRAINING SET: {best_strategy}")
        print(f"Training set score: {best_score:.4f}")
        
        # Show top 3 strategies from training
        training_summary = results['training_results']['strategy_summary']
        sorted_strategies = sorted(training_summary.items(), key=lambda x: x[1]['global_score'], reverse=True)
        print(f"\nTOP 3 STRATEGIES FROM TRAINING SET:")
        for i, (strategy_name, strategy_data) in enumerate(sorted_strategies[:3], 1):
            print(f"  {i}. {strategy_name}: score {strategy_data['global_score']:.4f}, "
                  f"anomaly reduction {strategy_data['global_anomaly_reduction']:.4f}, "
                  f"FP reduction {strategy_data['global_fp_reduction']:.4f}")
        
        print(f"\nTEST SET RESULTS:")
        print(f"Total files: {test_base['total_files']}")
        print(f"Total duration: {test_base['total_duration_hours']:.2f} hours")
        
        print(f"\nBEFORE CLUSTERING (Test Set):")
        print(f"  Total anomalies: {test_base['total_anomalies']}")
        print(f"  True positives: {test_base['total_true_positives']}")
        print(f"  False positives: {test_base['total_false_positives']}")
        if test_base['sensitivity'] is not None:
            print(f"  Sensitivity: {test_base['sensitivity']:.4f} ({test_base['sensitivity']*100:.2f}%)")
        print(f"  False alarms/hour: {test_base['false_alarms_per_hour']:.4f}")
        
        print(f"\nAFTER CLUSTERING WITH {best_strategy} (Test Set):")
        print(f"  Total anomalies: {test_final['total_anomalies']}")
        print(f"  True positives: {test_final['total_true_positives']}")
        print(f"  False positives: {test_final['total_false_positives']}")
        if test_final['sensitivity'] is not None:
            print(f"  Sensitivity: {test_final['sensitivity']:.4f} ({test_final['sensitivity']*100:.2f}%)")
        print(f"  False alarms/hour: {test_final['false_alarms_per_hour']:.4f}")
        
        print(f"\nTEST SET IMPROVEMENTS:")
        print(f"  Anomaly reduction: {test_improvements['anomaly_reduction']:.4f} ({test_improvements['anomaly_reduction']*100:.2f}%)")
        print(f"  FP reduction: {test_improvements['fp_reduction']:.4f} ({test_improvements['fp_reduction']*100:.2f}%)")
        print(f"  Sensitivity change: {test_improvements['sensitivity_change']:+.4f}")
        
        # Print responder analysis
        responder_analysis = results['test_results'].get('responder_analysis', {})
        if responder_analysis:
            print(f"\nRESPONDER ANALYSIS (Test Set):")
            print(f"  Total patients with seizures: {responder_analysis['total_patients_with_seizures']}")
            print(f"  Responders (≥2/3 seizures detected): {responder_analysis['num_responders']}")
            print(f"  Non-responders: {responder_analysis['num_non_responders']}")
            print(f"  Responder rate: {responder_analysis['responder_rate']:.4f} ({responder_analysis['responder_rate']*100:.2f}%)")
            
            if responder_analysis['responder_sensitivity'] is not None:
                print(f"\n  Responder Sensitivity: {responder_analysis['responder_sensitivity']:.4f} ({responder_analysis['responder_sensitivity']*100:.2f}%)")
                print(f"    (Seizures in responders: {responder_analysis['detected_seizures_responders']}/{responder_analysis['total_seizures_responders']})")
            
            if responder_analysis['non_responder_sensitivity'] is not None:
                print(f"  Non-responder Sensitivity: {responder_analysis['non_responder_sensitivity']:.4f} ({responder_analysis['non_responder_sensitivity']*100:.2f}%)")
                print(f"    (Seizures in non-responders: {responder_analysis['detected_seizures_non_responders']}/{responder_analysis['total_seizures_non_responders']})")
        
        print(f"\nOUTPUT DIRECTORY: {self.output_dir}")
        print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Reduce false alarms using temporal clustering with train/test split"
    )
    parser.add_argument(
        "results_dir", 
        help="Directory containing Madrid windowed results JSON files"
    )
    parser.add_argument(
        "-o", "--output-dir", 
        help="Output directory for clustering results (default: results_dir/train_test_clustered_results)"
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
    reducer = MadridClusteringFalseAlarmReducerTrainTest(
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