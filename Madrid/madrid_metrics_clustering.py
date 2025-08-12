#!/usr/bin/env python3
"""
Madrid Metrics-Based Smart Clustering Analysis

This script implements smart clustering on the output of madrid_extended_time_metrics.py
and madrid_file_level_metrics.py. It clusters the overlapping_detections and non_seizure_detections
to reduce the total number of anomalies while maintaining detection sensitivity.
"""

import json
import os
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict


class MadridMetricsClusteringAnalyzer:
    """Smart clustering analyzer for Madrid metrics calculator output."""
    
    def __init__(self, metrics_file: str, output_folder: str = None):
        self.metrics_file = Path(metrics_file)
        self.output_folder = Path(output_folder or f"{self.metrics_file.stem}_clustered")
        self.output_folder.mkdir(exist_ok=True)
        
        # Create subfolders
        (self.output_folder / "metrics_before").mkdir(exist_ok=True)
        (self.output_folder / "clusters").mkdir(exist_ok=True)
        (self.output_folder / "metrics_after").mkdir(exist_ok=True)
        (self.output_folder / "strategy_comparison").mkdir(exist_ok=True)
        
        self.metrics_data = None
        self.all_anomalies = []
        
    def load_metrics_results(self) -> None:
        """Load Madrid metrics calculator results."""
        print(f"Loading Madrid metrics results from: {self.metrics_file}")
        
        try:
            with open(self.metrics_file, 'r') as f:
                self.metrics_data = json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading metrics file {self.metrics_file}: {e}")
        
        # Extract all anomalies from individual results
        individual_results = self.metrics_data.get('individual_results', {})
        
        for filename, file_result in individual_results.items():
            file_info = file_result.get('file_info', {})
            file_summary = file_result.get('file_summary', {})
            detection_details = file_result.get('detection_details', {})
            
            subject_id = file_info.get('subject_id', 'unknown')
            run_id = file_info.get('run_id', 'unknown')
            file_id = f"{subject_id}_{run_id}"
            seizure_present = file_summary.get('seizure_present', False)
            
            # Extract overlapping detections (True Positives from extended time window)
            detected_seizures = detection_details.get('detected_seizures', [])
            for seizure in detected_seizures:
                overlapping_detections = seizure.get('overlapping_detections', [])
                
                for detection in overlapping_detections:
                    anomaly_with_metadata = {
                        **detection,
                        'file_id': file_id,
                        'filename': filename,
                        'subject_id': subject_id,
                        'run_id': run_id,
                        'seizure_present': seizure_present,
                        'seizure_hit': True,  # These are true positives
                        'seizure_id': seizure.get('seizure_id'),
                        'seizure_detection_type': detection.get('detection_type', 'unknown'),
                        'location_time_seconds': detection.get('anomaly_absolute_time', 0)
                    }
                    self.all_anomalies.append(anomaly_with_metadata)
            
            # Extract non-seizure detections (False Positives) 
            non_seizure_detections = detection_details.get('non_seizure_detections', [])
            for detection in non_seizure_detections:
                # Calculate absolute time for non-seizure detections
                window_start_time = detection.get('window_start_time', 0)
                location_time_in_window = detection.get('location_time_in_window', 0)
                absolute_time = window_start_time + location_time_in_window
                
                anomaly_with_metadata = {
                    **detection,
                    'file_id': file_id,
                    'filename': filename,
                    'subject_id': subject_id,
                    'run_id': run_id,
                    'seizure_present': seizure_present,
                    'seizure_hit': False,  # These are false positives
                    'seizure_id': None,
                    'seizure_detection_type': 'false_positive',
                    'location_time_seconds': absolute_time
                }
                self.all_anomalies.append(anomaly_with_metadata)
        
        print(f"Loaded {len(individual_results)} files with {len(self.all_anomalies)} total anomalies")
        
    def calculate_base_metrics(self) -> Dict[str, Any]:
        """Calculate base metrics from the loaded anomalies."""
        true_positives = sum(1 for a in self.all_anomalies if a.get('seizure_hit', False))
        false_positives = sum(1 for a in self.all_anomalies if not a.get('seizure_hit', False))
        total_anomalies = len(self.all_anomalies)
        
        # Count files and seizures
        files_with_seizures = len(set(a['file_id'] for a in self.all_anomalies if a.get('seizure_present', False)))
        files_with_detected_seizures = len(set(a['file_id'] for a in self.all_anomalies if a.get('seizure_hit', False)))
        
        # Count total seizures and detected seizures
        seizure_ids = set()
        detected_seizure_ids = set()
        for a in self.all_anomalies:
            if a.get('seizure_present', False) and a.get('seizure_id'):
                seizure_ids.add(f"{a['file_id']}_{a['seizure_id']}")
                if a.get('seizure_hit', False):
                    detected_seizure_ids.add(f"{a['file_id']}_{a['seizure_id']}")
        
        total_seizures = len(seizure_ids)
        detected_seizures = len(detected_seizure_ids)
        
        # Calculate metrics
        file_level_sensitivity = files_with_detected_seizures / files_with_seizures if files_with_seizures > 0 else 0
        seizure_level_sensitivity = detected_seizures / total_seizures if total_seizures > 0 else 0
        precision = true_positives / total_anomalies if total_anomalies > 0 else 0
        false_alarm_rate = false_positives / total_anomalies if total_anomalies > 0 else 0
        
        return {
            'total_files': len(set(a['file_id'] for a in self.all_anomalies)),
            'files_with_seizures': files_with_seizures,
            'files_with_detected_seizures': files_with_detected_seizures,
            'total_seizures': total_seizures,
            'detected_seizures': detected_seizures,
            'total_anomalies': total_anomalies,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'file_level_sensitivity': file_level_sensitivity,
            'seizure_level_sensitivity': seizure_level_sensitivity,
            'precision': precision,
            'false_alarm_rate': false_alarm_rate
        }
        
    def time_based_clustering(self, time_threshold: float) -> List[List[Dict]]:
        """Time-based clustering strategy - performed separately per file."""
        all_clusters = []
        
        # Group anomalies by file
        file_groups = defaultdict(list)
        for anomaly in self.all_anomalies:
            file_groups[anomaly['file_id']].append(anomaly)
        
        # Perform clustering within each file separately
        for file_id, file_anomalies in file_groups.items():
            # Sort anomalies within this file by absolute time
            sorted_anomalies = sorted(file_anomalies, key=lambda x: x['location_time_seconds'])
            
            current_cluster = []
            for anomaly in sorted_anomalies:
                if not current_cluster:
                    current_cluster = [anomaly]
                else:
                    time_diff = anomaly['location_time_seconds'] - current_cluster[-1]['location_time_seconds']
                    if time_diff <= time_threshold:
                        current_cluster.append(anomaly)
                    else:
                        if current_cluster:
                            all_clusters.append(current_cluster)
                        current_cluster = [anomaly]
                        
            # Add the last cluster from this file
            if current_cluster:
                all_clusters.append(current_cluster)
        
        return all_clusters
        
    def score_aware_clustering(self, time_threshold: float = 60.0) -> List[List[Dict]]:
        """Score-aware clustering that prioritizes high-score anomalies."""
        all_clusters = []
        
        # Group anomalies by file
        file_groups = defaultdict(list)
        for anomaly in self.all_anomalies:
            file_groups[anomaly['file_id']].append(anomaly)
        
        # Perform clustering within each file separately
        for file_id, file_anomalies in file_groups.items():
            # Sort anomalies by score (descending) then by time
            sorted_anomalies = sorted(file_anomalies, 
                                    key=lambda x: (-x['anomaly_score'], x['location_time_seconds']))
            
            current_cluster = []
            for anomaly in sorted_anomalies:
                if not current_cluster:
                    current_cluster = [anomaly]
                else:
                    # Check time distance to any anomaly in current cluster
                    min_time_diff = min(abs(anomaly['location_time_seconds'] - 
                                          cluster_anomaly['location_time_seconds']) 
                                       for cluster_anomaly in current_cluster)
                    
                    if min_time_diff <= time_threshold:
                        current_cluster.append(anomaly)
                    else:
                        if current_cluster:
                            all_clusters.append(current_cluster)
                        current_cluster = [anomaly]
                        
            # Add the last cluster from this file
            if current_cluster:
                all_clusters.append(current_cluster)
        
        return all_clusters
        
    def detection_type_aware_clustering(self, time_threshold: float = 120.0) -> List[List[Dict]]:
        """Clustering that separates different detection types (pre/during/post seizure)."""
        all_clusters = []
        
        # Group anomalies by file and detection type
        file_type_groups = defaultdict(lambda: defaultdict(list))
        for anomaly in self.all_anomalies:
            file_id = anomaly['file_id']
            detection_type = anomaly['seizure_detection_type']
            file_type_groups[file_id][detection_type].append(anomaly)
        
        # Perform clustering within each file and detection type separately
        for file_id, type_groups in file_type_groups.items():
            for detection_type, type_anomalies in type_groups.items():
                # Sort anomalies within this type by absolute time
                sorted_anomalies = sorted(type_anomalies, key=lambda x: x['location_time_seconds'])
                
                current_cluster = []
                for anomaly in sorted_anomalies:
                    if not current_cluster:
                        current_cluster = [anomaly]
                    else:
                        time_diff = anomaly['location_time_seconds'] - current_cluster[-1]['location_time_seconds']
                        if time_diff <= time_threshold:
                            current_cluster.append(anomaly)
                        else:
                            if current_cluster:
                                all_clusters.append(current_cluster)
                            current_cluster = [anomaly]
                            
                # Add the last cluster from this type
                if current_cluster:
                    all_clusters.append(current_cluster)
        
        return all_clusters
    
    def subject_aware_clustering(self, base_threshold: float = 180.0) -> List[List[Dict]]:
        """Subject-aware clustering with adaptive thresholds per subject."""
        all_clusters = []
        
        # Group anomalies by subject and file
        subject_groups = defaultdict(lambda: defaultdict(list))
        for anomaly in self.all_anomalies:
            subject_id = anomaly['subject_id']
            file_id = anomaly['file_id']
            subject_groups[subject_id][file_id].append(anomaly)
            
        # Calculate subject-specific thresholds
        subject_thresholds = {}
        for subject_id, files in subject_groups.items():
            subject_time_gaps = []
            
            for file_id, file_anomalies in files.items():
                sorted_anomalies = sorted(file_anomalies, key=lambda x: x['location_time_seconds'])
                for i in range(1, len(sorted_anomalies)):
                    gap = sorted_anomalies[i]['location_time_seconds'] - sorted_anomalies[i-1]['location_time_seconds']
                    subject_time_gaps.append(gap)
                    
            if subject_time_gaps:
                # Use median gap time as adaptive threshold
                subject_time_gaps.sort()
                median_gap = subject_time_gaps[len(subject_time_gaps) // 2]
                # Adaptive threshold: base_threshold adjusted by subject pattern
                adaptive_factor = min(2.0, max(0.5, median_gap / base_threshold))
                subject_thresholds[subject_id] = base_threshold * adaptive_factor
            else:
                subject_thresholds[subject_id] = base_threshold
        
        # Perform clustering with subject-specific thresholds
        for subject_id, files in subject_groups.items():
            threshold = subject_thresholds[subject_id]
            
            for file_id, file_anomalies in files.items():
                sorted_anomalies = sorted(file_anomalies, key=lambda x: x['location_time_seconds'])
                
                current_cluster = []
                for anomaly in sorted_anomalies:
                    if not current_cluster:
                        current_cluster = [anomaly]
                    else:
                        time_diff = anomaly['location_time_seconds'] - current_cluster[-1]['location_time_seconds']
                        if time_diff <= threshold:
                            current_cluster.append(anomaly)
                        else:
                            if current_cluster:
                                all_clusters.append(current_cluster)
                            current_cluster = [anomaly]
                            
                # Add the last cluster from this file
                if current_cluster:
                    all_clusters.append(current_cluster)
        
        return all_clusters
        
    def select_smart_representatives(self, clusters: List[List[Dict]]) -> List[Dict]:
        """Select representative based on minimal time distance to all cluster members."""
        representatives = []
        
        for i, cluster in enumerate(clusters):
            if not cluster:
                continue
                
            if len(cluster) == 1:
                # Single anomaly cluster
                representative = cluster[0]
                min_avg_distance = 0.0
            else:
                # Find anomaly with minimal average time distance to all others
                min_avg_distance = float('inf')
                best_representative = None
                
                for candidate in cluster:
                    candidate_time = candidate['location_time_seconds']
                    
                    # Calculate average time distance to all other cluster members
                    total_distance = 0
                    for other in cluster:
                        if other != candidate:
                            total_distance += abs(candidate_time - other['location_time_seconds'])
                    
                    avg_distance = total_distance / (len(cluster) - 1)
                    
                    if avg_distance < min_avg_distance:
                        min_avg_distance = avg_distance
                        best_representative = candidate
                
                representative = best_representative
                
            # Add cluster metadata
            true_positives = [a for a in cluster if a.get('seizure_hit', False)]
            representative = representative.copy()  # Don't modify original
            representative['cluster_id'] = i
            representative['cluster_size'] = len(cluster)
            representative['cluster_tp_count'] = len(true_positives)
            representative['avg_time_distance'] = min_avg_distance
            
            # Add detection type distribution
            detection_types = {}
            for a in cluster:
                det_type = a['seizure_detection_type']
                detection_types[det_type] = detection_types.get(det_type, 0) + 1
            representative['cluster_detection_types'] = detection_types
            
            representatives.append(representative)
            
        return representatives
        
    def select_seizure_preserving_representatives(self, clusters: List[List[Dict]]) -> List[Dict]:
        """Select representatives that preserve seizure detection capability."""
        representatives = []
        
        for i, cluster in enumerate(clusters):
            if not cluster:
                continue
                
            # Prioritize true positive detections
            true_positives = [a for a in cluster if a.get('seizure_hit', False)]
            
            if true_positives:
                # If cluster has TPs, select the best TP (highest score)
                representative = max(true_positives, key=lambda x: x['anomaly_score'])
            else:
                # If no TPs, select highest scoring anomaly
                representative = max(cluster, key=lambda x: x['anomaly_score'])
                
            # Add cluster metadata
            representative = representative.copy()  # Don't modify original
            representative['cluster_id'] = i
            representative['cluster_size'] = len(cluster)
            representative['cluster_tp_count'] = len(true_positives)
            representative['cluster_max_score'] = max(a['anomaly_score'] for a in cluster)
            representative['cluster_min_score'] = min(a['anomaly_score'] for a in cluster)
            
            # Add detection type distribution
            detection_types = {}
            for a in cluster:
                det_type = a['seizure_detection_type']
                detection_types[det_type] = detection_types.get(det_type, 0) + 1
            representative['cluster_detection_types'] = detection_types
            
            representatives.append(representative)
            
        return representatives
        
    def evaluate_clustering_strategy(self, representatives: List[Dict], base_metrics: Dict) -> Dict:
        """Evaluate a clustering strategy."""
        true_positives = sum(1 for r in representatives if r.get('seizure_hit', False))
        false_positives = sum(1 for r in representatives if not r.get('seizure_hit', False))
        total_anomalies = len(representatives)
        
        # Count files with detected seizures
        files_with_representatives = set()
        files_with_tp_representatives = set()
        
        # Count seizures with detected representatives
        seizure_ids_with_representatives = set()
        
        for rep in representatives:
            files_with_representatives.add(rep['file_id'])
            if rep.get('seizure_hit', False):
                files_with_tp_representatives.add(rep['file_id'])
                if rep.get('seizure_id'):
                    seizure_ids_with_representatives.add(f"{rep['file_id']}_{rep['seizure_id']}")
        
        file_level_sensitivity = len(files_with_tp_representatives) / base_metrics['files_with_seizures'] if base_metrics['files_with_seizures'] > 0 else 0
        seizure_level_sensitivity = len(seizure_ids_with_representatives) / base_metrics['total_seizures'] if base_metrics['total_seizures'] > 0 else 0
        
        precision = true_positives / total_anomalies if total_anomalies > 0 else 0
        false_alarm_rate = false_positives / total_anomalies if total_anomalies > 0 else 0
        
        # Calculate reduction metrics
        anomaly_reduction = (base_metrics['total_anomalies'] - total_anomalies) / base_metrics['total_anomalies'] if base_metrics['total_anomalies'] > 0 else 0
        fp_reduction = (base_metrics['false_positives'] - false_positives) / base_metrics['false_positives'] if base_metrics['false_positives'] > 0 else 0
        
        # Calculate score: prioritize maintaining sensitivity while reducing FPs
        file_sensitivity_penalty = max(0, base_metrics['file_level_sensitivity'] - file_level_sensitivity)
        seizure_sensitivity_penalty = max(0, base_metrics['seizure_level_sensitivity'] - seizure_level_sensitivity)
        
        score = (fp_reduction * 0.5) + (anomaly_reduction * 0.2) - (file_sensitivity_penalty * 1.5) - (seizure_sensitivity_penalty * 1.5)
        
        return {
            'total_anomalies': total_anomalies,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'file_level_sensitivity': file_level_sensitivity,
            'seizure_level_sensitivity': seizure_level_sensitivity,
            'precision': precision,
            'false_alarm_rate': false_alarm_rate,
            'anomaly_reduction': anomaly_reduction,
            'fp_reduction': fp_reduction,
            'file_sensitivity_change': file_level_sensitivity - base_metrics['file_level_sensitivity'],
            'seizure_sensitivity_change': seizure_level_sensitivity - base_metrics['seizure_level_sensitivity'],
            'score': score
        }
        
    def run_metrics_clustering_analysis(self) -> Dict[str, Any]:
        """Run smart clustering analysis on Madrid metrics results."""
        print("Starting Madrid metrics clustering analysis...")
        
        self.load_metrics_results()
        if not self.all_anomalies:
            raise ValueError("No anomalies found in metrics data")
            
        base_metrics = self.calculate_base_metrics()
        
        print(f"\nBase metrics from extended time window:")
        print(f"Total anomalies: {base_metrics['total_anomalies']}")
        print(f"True positives: {base_metrics['true_positives']}")
        print(f"False positives: {base_metrics['false_positives']}")
        print(f"File-level sensitivity: {base_metrics['file_level_sensitivity']:.3f}")
        print(f"Seizure-level sensitivity: {base_metrics['seizure_level_sensitivity']:.3f}")
        print(f"Precision: {base_metrics['precision']:.3f}")
        
        # Save base metrics
        with open(self.output_folder / "metrics_before" / "extended_metrics_summary.json", 'w') as f:
            json.dump(base_metrics, f, indent=2)
            
        # Test different strategies adapted for extended time metrics
        strategies = {}
        
        print("\nTesting clustering strategies for extended time metrics...")
        
        # 1. Time-based clustering with various thresholds
        time_thresholds = [30, 60, 120, 180, 300, 450, 600, 900, 1200, 1800, 3600]
        print(f"Testing {len(time_thresholds)} time-based configurations...")
        for threshold in time_thresholds:
            clusters = self.time_based_clustering(threshold)
            representatives = self.select_seizure_preserving_representatives(clusters)
            metrics = self.evaluate_clustering_strategy(representatives, base_metrics)
            strategies[f'time_{threshold}s'] = {
                'method': 'time_based',
                'parameters': {'threshold': threshold},
                'clusters': len(clusters),
                'representatives': representatives,
                'metrics': metrics
            }
            print(f"Time-based ({threshold}s): {len(clusters)} clusters, score: {metrics['score']:.3f}")
                
        # 2. Score-aware clustering
        print("Testing score-aware clustering strategies...")
        score_thresholds = [60, 120, 300, 600, 900]
        for time_threshold in score_thresholds:
            clusters = self.score_aware_clustering(time_threshold)
            representatives = self.select_seizure_preserving_representatives(clusters)
            metrics = self.evaluate_clustering_strategy(representatives, base_metrics)
            strategies[f'score_aware_{time_threshold}s'] = {
                'method': 'score_aware',
                'parameters': {'threshold': time_threshold},
                'clusters': len(clusters),
                'representatives': representatives,
                'metrics': metrics
            }
            print(f"Score-aware ({time_threshold}s): {len(clusters)} clusters, score: {metrics['score']:.3f}")
            
        # 3. Detection type aware clustering
        print("Testing detection type aware clustering...")
        type_thresholds = [60, 120, 180, 300]
        for threshold in type_thresholds:
            clusters = self.detection_type_aware_clustering(threshold)
            representatives = self.select_seizure_preserving_representatives(clusters)
            metrics = self.evaluate_clustering_strategy(representatives, base_metrics)
            strategies[f'type_aware_{threshold}s'] = {
                'method': 'detection_type_aware',
                'parameters': {'threshold': threshold},
                'clusters': len(clusters),
                'representatives': representatives,
                'metrics': metrics
            }
            print(f"Type-aware ({threshold}s): {len(clusters)} clusters, score: {metrics['score']:.3f}")
            
        # 4. Subject-aware clustering
        print("Testing subject-aware clustering...")
        subject_thresholds = [90, 180, 300, 450]
        for threshold in subject_thresholds:
            clusters = self.subject_aware_clustering(threshold)
            representatives = self.select_seizure_preserving_representatives(clusters)
            metrics = self.evaluate_clustering_strategy(representatives, base_metrics)
            strategies[f'subject_aware_{threshold}s'] = {
                'method': 'subject_aware',
                'parameters': {'threshold': threshold},
                'clusters': len(clusters),
                'representatives': representatives,
                'metrics': metrics
            }
            print(f"Subject-aware ({threshold}s): {len(clusters)} clusters, score: {metrics['score']:.3f}")
        
        # Select best strategy
        if strategies:
            best_strategy_name = max(strategies.keys(), key=lambda k: strategies[k]['metrics']['score'])
            best_strategy = strategies[best_strategy_name]
            
            print(f"\nBest strategy: {best_strategy_name}")
            print(f"Score: {best_strategy['metrics']['score']:.3f}")
            print(f"Clusters: {best_strategy['clusters']}")
            print(f"Representatives: {len(best_strategy['representatives'])}")
            
            # Save results
            self.save_metrics_clustering_results(base_metrics, strategies, best_strategy_name)
            
            return {
                'base_metrics': base_metrics,
                'strategies': strategies,
                'best_strategy': best_strategy_name,
                'best_results': best_strategy
            }
        else:
            print("No strategies could be evaluated")
            return {'base_metrics': base_metrics, 'strategies': {}}
        
    def save_metrics_clustering_results(self, base_metrics: Dict, strategies: Dict, best_strategy_name: str):
        """Save all analysis results."""
        
        # Save strategy comparison
        comparison_data = {
            'base_metrics': base_metrics,
            'strategies_tested': len(strategies),
            'best_strategy': best_strategy_name,
            'data_format': 'extended_time_metrics',
            'source_file': str(self.metrics_file),
            'timestamp': datetime.now().isoformat(),
            'strategy_results': {
                name: {
                    'method': data['method'],
                    'parameters': data['parameters'],
                    'clusters': data['clusters'],
                    'metrics': data['metrics']
                } for name, data in strategies.items()
            }
        }
        
        with open(self.output_folder / "strategy_comparison" / "metrics_clustering_comparison.json", 'w') as f:
            json.dump(comparison_data, f, indent=2)
            
        # Save best strategy results
        best_strategy = strategies[best_strategy_name]
        
        # Save representatives
        representatives_output = {
            'clustering_method': best_strategy['method'],
            'parameters': best_strategy['parameters'],
            'timestamp': datetime.now().isoformat(),
            'data_format': 'extended_time_metrics',
            'source_file': str(self.metrics_file),
            'total_representatives': len(best_strategy['representatives']),
            'original_anomalies': len(self.all_anomalies),
            'representatives': best_strategy['representatives']
        }
        
        with open(self.output_folder / "clusters" / "best_metrics_representatives.json", 'w') as f:
            json.dump(representatives_output, f, indent=2)
            
        # Save final metrics
        final_metrics = {
            'strategy': best_strategy_name,
            'data_format': 'extended_time_metrics',
            'source_file': str(self.metrics_file),
            'before': base_metrics,
            'after': best_strategy['metrics'],
            'improvements': {
                'anomaly_reduction_pct': best_strategy['metrics']['anomaly_reduction'] * 100,
                'fp_reduction_pct': best_strategy['metrics']['fp_reduction'] * 100,
                'file_sensitivity_change': best_strategy['metrics']['file_sensitivity_change'],
                'seizure_sensitivity_change': best_strategy['metrics']['seizure_sensitivity_change'],
                'precision_improvement': best_strategy['metrics']['precision'] - base_metrics['precision'],
                'far_reduction': base_metrics['false_alarm_rate'] - best_strategy['metrics']['false_alarm_rate']
            }
        }
        
        with open(self.output_folder / "metrics_after" / "final_metrics_clustering.json", 'w') as f:
            json.dump(final_metrics, f, indent=2)
            
        # Print final summary
        print("\n" + "="*70)
        print("MADRID EXTENDED METRICS CLUSTERING ANALYSIS SUMMARY")
        print("="*70)
        print(f"Source: {self.metrics_file.name}")
        print(f"Best strategy: {best_strategy_name}")
        print(f"Anomalies reduced: {len(self.all_anomalies)} → {best_strategy['metrics']['total_anomalies']} "
              f"({best_strategy['metrics']['anomaly_reduction']*100:.1f}% reduction)")
        print(f"False positives: {base_metrics['false_positives']} → {best_strategy['metrics']['false_positives']} "
              f"({best_strategy['metrics']['fp_reduction']*100:.1f}% reduction)")
        print(f"File-level sensitivity: {base_metrics['file_level_sensitivity']:.3f} → {best_strategy['metrics']['file_level_sensitivity']:.3f} "
              f"({best_strategy['metrics']['file_sensitivity_change']:+.3f})")
        print(f"Seizure-level sensitivity: {base_metrics['seizure_level_sensitivity']:.3f} → {best_strategy['metrics']['seizure_level_sensitivity']:.3f} "
              f"({best_strategy['metrics']['seizure_sensitivity_change']:+.3f})")
        print(f"Precision: {base_metrics['precision']:.3f} → {best_strategy['metrics']['precision']:.3f} "
              f"({best_strategy['metrics']['precision'] - base_metrics['precision']:+.3f})")
        print("="*70)
        print(f"Results saved to: {self.output_folder}")


def main():
    """Main function."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 madrid_metrics_clustering.py <metrics_file.json>")
        print("Example: python3 madrid_metrics_clustering.py Madrid/example_results/madrid_extended_for_clustering.json")
        sys.exit(1)
        
    metrics_file = sys.argv[1]
    
    analyzer = MadridMetricsClusteringAnalyzer(metrics_file)
    
    try:
        results = analyzer.run_metrics_clustering_analysis()
        print("\nMetrics clustering analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()