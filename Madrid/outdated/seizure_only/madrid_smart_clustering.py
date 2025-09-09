#!/usr/bin/env python3
"""
Improved Madrid Clustering Analysis with Smart Clustering Strategy

This version implements multiple clustering strategies and selects the best one
based on maintaining sensitivity while maximizing false positive reduction.
"""

import json
import os
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict


class SmartMadridClusteringAnalyzer:
    """Smart analyzer with multiple clustering strategies."""
    
    def __init__(self, results_folder: str, output_folder: str = None):
        self.results_folder = Path(results_folder)
        self.output_folder = Path(output_folder or f"{results_folder}_smart_clustered")
        self.output_folder.mkdir(exist_ok=True)
        
        # Create subfolders
        (self.output_folder / "metrics_before").mkdir(exist_ok=True)
        (self.output_folder / "clusters").mkdir(exist_ok=True)
        (self.output_folder / "metrics_after").mkdir(exist_ok=True)
        (self.output_folder / "strategy_comparison").mkdir(exist_ok=True)
        
        self.results_data = []
        self.all_anomalies = []
        
    def load_madrid_results(self) -> None:
        """Load all Madrid result JSON files."""
        print(f"Loading Madrid results from: {self.results_folder}")
        
        json_files = list(self.results_folder.glob("madrid_results_*.json"))
        print(f"Found {len(json_files)} result files")
        
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    self.results_data.append(data)
                    
                    # Extract anomalies with metadata
                    for anomaly in data.get('analysis_results', {}).get('anomalies', []):
                        anomaly_with_metadata = anomaly.copy()
                        anomaly_with_metadata['file_id'] = (
                            data['input_data']['subject_id'] + '_' + 
                            data['input_data']['run_id'] + '_' + 
                            data['input_data']['seizure_id']
                        )
                        anomaly_with_metadata['subject_id'] = data['input_data']['subject_id']
                        anomaly_with_metadata['seizure_present'] = data['validation_data']['ground_truth']['seizure_present']
                        self.all_anomalies.append(anomaly_with_metadata)
                        
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        print(f"Loaded {len(self.results_data)} files with {len(self.all_anomalies)} total anomalies")
        
    def calculate_base_metrics(self) -> Dict[str, Any]:
        """Calculate base metrics before any clustering."""
        true_positives = sum(1 for a in self.all_anomalies if a.get('seizure_hit', False))
        false_positives = sum(1 for a in self.all_anomalies if not a.get('seizure_hit', False))
        total_anomalies = len(self.all_anomalies)
        
        files_with_seizures = sum(1 for data in self.results_data 
                                if data['validation_data']['ground_truth']['seizure_present'])
        files_with_detected_seizures = sum(1 for data in self.results_data 
                                         if data['analysis_results']['performance_metrics']['true_positives'] > 0)
        
        sensitivity = files_with_detected_seizures / files_with_seizures if files_with_seizures > 0 else 0
        precision = true_positives / total_anomalies if total_anomalies > 0 else 0
        false_alarm_rate = false_positives / total_anomalies if total_anomalies > 0 else 0
        
        return {
            'total_files': len(self.results_data),
            'files_with_seizures': files_with_seizures,
            'files_with_detected_seizures': files_with_detected_seizures,
            'total_anomalies': total_anomalies,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'sensitivity': sensitivity,
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
            # Sort anomalies within this file by time
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
        
    def score_based_clustering(self, time_threshold: float = 30.0) -> List[List[Dict]]:
        """Time-based clustering with different threshold (renamed for compatibility)."""
        return self.time_based_clustering(time_threshold)
        
    def file_based_clustering(self, time_threshold: float = 30.0) -> List[List[Dict]]:
        """Time-based clustering with file separation (uses only location_time_seconds)."""
        return self.time_based_clustering(time_threshold)
        
    def hybrid_clustering(self, time_threshold: float = 45.0) -> List[List[Dict]]:
        """Time-based clustering with different threshold (uses only location_time_seconds)."""
        return self.time_based_clustering(time_threshold)
        
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
        
    def select_score_aware_representatives(self, clusters: List[List[Dict]]) -> List[Dict]:
        """Select representatives prioritizing highest scores."""
        representatives = []
        
        for i, cluster in enumerate(clusters):
            if not cluster:
                continue
                
            if len(cluster) == 1:
                representative = cluster[0]
            else:
                # Select the anomaly with highest score
                representative = max(cluster, key=lambda x: x['anomaly_score'])
                
            # Add cluster metadata
            true_positives = [a for a in cluster if a.get('seizure_hit', False)]
            representative['cluster_id'] = i
            representative['cluster_size'] = len(cluster)
            representative['cluster_tp_count'] = len(true_positives)
            representative['cluster_max_score'] = max(a['anomaly_score'] for a in cluster)
            representative['cluster_min_score'] = min(a['anomaly_score'] for a in cluster)
            representatives.append(representative)
            
        return representatives
        
    def multi_threshold_clustering(self, primary_threshold: float, secondary_threshold: float) -> List[List[Dict]]:
        """Multi-threshold clustering using two different time windows."""
        all_clusters = []
        
        # Group anomalies by file
        file_groups = defaultdict(list)
        for anomaly in self.all_anomalies:
            file_groups[anomaly['file_id']].append(anomaly)
        
        # Perform clustering within each file separately
        for file_id, file_anomalies in file_groups.items():
            # Sort anomalies within this file by time
            sorted_anomalies = sorted(file_anomalies, key=lambda x: x['location_time_seconds'])
            
            # First pass: tight clustering with primary threshold
            primary_clusters = []
            current_cluster = []
            
            for anomaly in sorted_anomalies:
                if not current_cluster:
                    current_cluster = [anomaly]
                else:
                    time_diff = anomaly['location_time_seconds'] - current_cluster[-1]['location_time_seconds']
                    if time_diff <= primary_threshold:
                        current_cluster.append(anomaly)
                    else:
                        if current_cluster:
                            primary_clusters.append(current_cluster)
                        current_cluster = [anomaly]
                        
            if current_cluster:
                primary_clusters.append(current_cluster)
            
            # Second pass: merge clusters if they're within secondary threshold
            final_clusters = []
            i = 0
            while i < len(primary_clusters):
                merged_cluster = primary_clusters[i][:]
                j = i + 1
                
                while j < len(primary_clusters):
                    # Check if clusters can be merged
                    last_time_in_merged = max(a['location_time_seconds'] for a in merged_cluster)
                    first_time_in_next = min(a['location_time_seconds'] for a in primary_clusters[j])
                    
                    if first_time_in_next - last_time_in_merged <= secondary_threshold:
                        merged_cluster.extend(primary_clusters[j])
                        primary_clusters.pop(j)
                    else:
                        j += 1
                        
                final_clusters.append(merged_cluster)
                i += 1
                
            all_clusters.extend(final_clusters)
        
        return all_clusters
        
    def subject_aware_clustering(self, base_threshold: float = 90.0) -> List[List[Dict]]:
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
            representative['cluster_id'] = i
            representative['cluster_size'] = len(cluster)
            representative['cluster_tp_count'] = len(true_positives)
            representative['avg_time_distance'] = min_avg_distance if len(cluster) > 1 else 0.0
            representatives.append(representative)
            
        return representatives
        
    def evaluate_clustering_strategy(self, representatives: List[Dict], base_metrics: Dict) -> Dict:
        """Evaluate a clustering strategy."""
        true_positives = sum(1 for r in representatives if r.get('seizure_hit', False))
        false_positives = sum(1 for r in representatives if not r.get('seizure_hit', False))
        total_anomalies = len(representatives)
        
        # Count files with detected seizures
        files_with_representatives = {}
        for rep in representatives:
            file_id = rep['file_id']
            if file_id not in files_with_representatives:
                files_with_representatives[file_id] = {'has_tp': False}
            if rep.get('seizure_hit', False):
                files_with_representatives[file_id]['has_tp'] = True
                
        files_with_detected_seizures = sum(1 for file_data in files_with_representatives.values() 
                                         if file_data['has_tp'])
        
        sensitivity = files_with_detected_seizures / base_metrics['files_with_seizures'] if base_metrics['files_with_seizures'] > 0 else 0
        precision = true_positives / total_anomalies if total_anomalies > 0 else 0
        false_alarm_rate = false_positives / total_anomalies if total_anomalies > 0 else 0
        
        # Calculate reduction metrics
        anomaly_reduction = (base_metrics['total_anomalies'] - total_anomalies) / base_metrics['total_anomalies'] if base_metrics['total_anomalies'] > 0 else 0
        fp_reduction = (base_metrics['false_positives'] - false_positives) / base_metrics['false_positives'] if base_metrics['false_positives'] > 0 else 0
        
        # Calculate score: prioritize maintaining sensitivity while reducing FPs
        sensitivity_penalty = max(0, base_metrics['sensitivity'] - sensitivity)
        score = (fp_reduction * 0.6) + (anomaly_reduction * 0.3) - (sensitivity_penalty * 2.0)
        
        return {
            'total_anomalies': total_anomalies,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'sensitivity': sensitivity,
            'precision': precision,
            'false_alarm_rate': false_alarm_rate,
            'anomaly_reduction': anomaly_reduction,
            'fp_reduction': fp_reduction,
            'sensitivity_change': sensitivity - base_metrics['sensitivity'],
            'score': score
        }
        
    def run_smart_analysis(self) -> Dict[str, Any]:
        """Run smart analysis comparing multiple strategies."""
        print("Starting smart Madrid clustering analysis...")
        
        self.load_madrid_results()
        if not self.all_anomalies:
            raise ValueError("No anomalies found")
            
        base_metrics = self.calculate_base_metrics()
        
        print(f"\nBase metrics:")
        print(f"Total anomalies: {base_metrics['total_anomalies']}")
        print(f"True positives: {base_metrics['true_positives']}")
        print(f"False positives: {base_metrics['false_positives']}")
        print(f"Sensitivity: {base_metrics['sensitivity']:.3f}")
        print(f"Precision: {base_metrics['precision']:.3f}")
        print(f"False alarm rate: {base_metrics['false_alarm_rate']:.3f}")
        
        # Save base metrics
        with open(self.output_folder / "metrics_before" / "metrics_summary.json", 'w') as f:
            json.dump(base_metrics, f, indent=2)
            
        # Test different strategies
        strategies = {}
        
        print("\nTesting clustering strategies...")
        
        # 1. Comprehensive time-based clustering with extended range
        time_thresholds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 60, 75, 90, 120, 150, 180, 240, 300]
        print(f"Testing {len(time_thresholds)} time-based configurations...")
        for threshold in time_thresholds:
            clusters = self.time_based_clustering(threshold)
            representatives = self.select_smart_representatives(clusters)
            metrics = self.evaluate_clustering_strategy(representatives, base_metrics)
            strategies[f'time_{threshold}s'] = {
                'method': 'time_based',
                'parameters': {'threshold': threshold},
                'clusters': len(clusters),
                'representatives': representatives,
                'metrics': metrics
            }
            if threshold in [5, 10, 15, 30, 60, 120, 180, 300] or threshold % 50 == 0:
                print(f"Time-based ({threshold}s): {len(clusters)} clusters, score: {metrics['score']:.3f}")
            
        # 2. Fine-grained time clustering (high precision range)
        fine_thresholds = [2, 3, 6, 8, 12, 18, 22, 28, 32, 38, 42, 48, 55, 65, 85, 105]
        print(f"Testing {len(fine_thresholds)} fine-grained time configurations...")
        for threshold in fine_thresholds:
            clusters = self.time_based_clustering(threshold)
            representatives = self.select_smart_representatives(clusters)
            metrics = self.evaluate_clustering_strategy(representatives, base_metrics)
            strategies[f'fine_time_{threshold}s'] = {
                'method': 'fine_time_based',
                'parameters': {'threshold': threshold},
                'clusters': len(clusters),
                'representatives': representatives,
                'metrics': metrics
            }
            if threshold in [2, 8, 18, 32, 55, 85]:
                print(f"Fine time ({threshold}s): {len(clusters)} clusters, score: {metrics['score']:.3f}")
                
        # 3. Extended range time clustering (very large windows)
        extended_thresholds = [360, 450, 600, 900, 1200, 1800]
        print(f"Testing {len(extended_thresholds)} extended range configurations...")
        for threshold in extended_thresholds:
            clusters = self.time_based_clustering(threshold)
            representatives = self.select_smart_representatives(clusters)
            metrics = self.evaluate_clustering_strategy(representatives, base_metrics)
            strategies[f'extended_time_{threshold}s'] = {
                'method': 'extended_time_based',
                'parameters': {'threshold': threshold},
                'clusters': len(clusters),
                'representatives': representatives,
                'metrics': metrics
            }
            print(f"Extended time ({threshold}s): {len(clusters)} clusters, score: {metrics['score']:.3f}")
            
        # 4. Adaptive time clustering (based on percentiles of time gaps)
        print("Testing adaptive time clustering...")
        all_time_gaps = []
        file_groups = defaultdict(list)
        for anomaly in self.all_anomalies:
            file_groups[anomaly['file_id']].append(anomaly)
            
        for file_id, file_anomalies in file_groups.items():
            sorted_anomalies = sorted(file_anomalies, key=lambda x: x['location_time_seconds'])
            for i in range(1, len(sorted_anomalies)):
                gap = sorted_anomalies[i]['location_time_seconds'] - sorted_anomalies[i-1]['location_time_seconds']
                all_time_gaps.append(gap)
                
        if all_time_gaps:
            adaptive_thresholds = []
            all_time_gaps.sort()
            n_gaps = len(all_time_gaps)
            for percentile in [10, 25, 50, 75, 90, 95, 99]:
                idx = int(percentile / 100 * (n_gaps - 1))
                adaptive_thresholds.append(int(all_time_gaps[idx]))
                
            for i, threshold in enumerate(adaptive_thresholds):
                percentile = [10, 25, 50, 75, 90, 95, 99][i] if i < 7 else 99
                if threshold > 0:
                    clusters = self.time_based_clustering(threshold)
                    representatives = self.select_smart_representatives(clusters)
                    metrics = self.evaluate_clustering_strategy(representatives, base_metrics)
                    strategies[f'adaptive_p{percentile}_{threshold}s'] = {
                        'method': 'adaptive_time_based',
                        'parameters': {'threshold': threshold, 'percentile_basis': percentile},
                        'clusters': len(clusters),
                        'representatives': representatives,
                        'metrics': metrics
                    }
                    print(f"Adaptive P{percentile} ({threshold}s): {len(clusters)} clusters, score: {metrics['score']:.3f}")
                    
        # 5. Score-aware clustering (prioritize high-score anomalies in clustering)
        print("Testing score-aware clustering strategies...")
        score_thresholds = [30, 60, 120, 180]
        for time_threshold in score_thresholds:
            clusters = self.score_aware_clustering(time_threshold)
            representatives = self.select_score_aware_representatives(clusters)
            metrics = self.evaluate_clustering_strategy(representatives, base_metrics)
            strategies[f'score_aware_{time_threshold}s'] = {
                'method': 'score_aware',
                'parameters': {'threshold': time_threshold},
                'clusters': len(clusters),
                'representatives': representatives,
                'metrics': metrics
            }
            print(f"Score-aware ({time_threshold}s): {len(clusters)} clusters, score: {metrics['score']:.3f}")
            
        # 6. Hybrid multi-threshold clustering
        print("Testing hybrid multi-threshold clustering...")
        hybrid_configs = [
            {'primary': 60, 'secondary': 120},
            {'primary': 30, 'secondary': 90},
            {'primary': 45, 'secondary': 180},
            {'primary': 90, 'secondary': 240}
        ]
        for config in hybrid_configs:
            clusters = self.multi_threshold_clustering(config['primary'], config['secondary'])
            representatives = self.select_smart_representatives(clusters)
            metrics = self.evaluate_clustering_strategy(representatives, base_metrics)
            strategies[f'hybrid_{config["primary"]}_{config["secondary"]}s'] = {
                'method': 'multi_threshold',
                'parameters': config,
                'clusters': len(clusters),
                'representatives': representatives,
                'metrics': metrics
            }
            print(f"Hybrid ({config['primary']}/{config['secondary']}s): {len(clusters)} clusters, score: {metrics['score']:.3f}")
            
        # 7. Subject-aware clustering (different thresholds per subject pattern)
        print("Testing subject-aware clustering...")
        subject_thresholds = [45, 90, 135]
        for threshold in subject_thresholds:
            clusters = self.subject_aware_clustering(threshold)
            representatives = self.select_smart_representatives(clusters)
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
        best_strategy_name = max(strategies.keys(), key=lambda k: strategies[k]['metrics']['score'])
        best_strategy = strategies[best_strategy_name]
        
        print(f"\nBest strategy: {best_strategy_name}")
        print(f"Score: {best_strategy['metrics']['score']:.3f}")
        print(f"Clusters: {best_strategy['clusters']}")
        print(f"Representatives: {len(best_strategy['representatives'])}")
        
        # Save results
        self.save_smart_results(base_metrics, strategies, best_strategy_name)
        
        return {
            'base_metrics': base_metrics,
            'strategies': strategies,
            'best_strategy': best_strategy_name,
            'best_results': best_strategy
        }
        
    def save_smart_results(self, base_metrics: Dict, strategies: Dict, best_strategy_name: str):
        """Save all analysis results."""
        
        # Save strategy comparison
        comparison_data = {
            'base_metrics': base_metrics,
            'strategies_tested': len(strategies),
            'best_strategy': best_strategy_name,
            'strategy_results': {
                name: {
                    'method': data['method'],
                    'parameters': data['parameters'],
                    'clusters': data['clusters'],
                    'metrics': data['metrics']
                } for name, data in strategies.items()
            }
        }
        
        with open(self.output_folder / "strategy_comparison" / "comparison.json", 'w') as f:
            json.dump(comparison_data, f, indent=2)
            
        # Save best strategy results
        best_strategy = strategies[best_strategy_name]
        
        # Save representatives
        representatives_output = {
            'clustering_method': best_strategy['method'],
            'parameters': best_strategy['parameters'],
            'timestamp': datetime.now().isoformat(),
            'total_representatives': len(best_strategy['representatives']),
            'original_anomalies': len(self.all_anomalies),
            'representatives': best_strategy['representatives']
        }
        
        with open(self.output_folder / "clusters" / "best_representatives.json", 'w') as f:
            json.dump(representatives_output, f, indent=2)
            
        # Save final metrics
        final_metrics = {
            'strategy': best_strategy_name,
            'before': base_metrics,
            'after': best_strategy['metrics'],
            'improvements': {
                'anomaly_reduction_pct': best_strategy['metrics']['anomaly_reduction'] * 100,
                'fp_reduction_pct': best_strategy['metrics']['fp_reduction'] * 100,
                'sensitivity_change': best_strategy['metrics']['sensitivity_change'],
                'precision_improvement': best_strategy['metrics']['precision'] - base_metrics['precision'],
                'far_reduction': base_metrics['false_alarm_rate'] - best_strategy['metrics']['false_alarm_rate']
            }
        }
        
        with open(self.output_folder / "metrics_after" / "final_metrics.json", 'w') as f:
            json.dump(final_metrics, f, indent=2)
            
        # Print final summary
        print("\n" + "="*60)
        print("SMART CLUSTERING ANALYSIS SUMMARY")
        print("="*60)
        print(f"Best strategy: {best_strategy_name}")
        print(f"Anomalies reduced: {len(self.all_anomalies)} → {best_strategy['metrics']['total_anomalies']} "
              f"({best_strategy['metrics']['anomaly_reduction']*100:.1f}% reduction)")
        print(f"False positives: {base_metrics['false_positives']} → {best_strategy['metrics']['false_positives']} "
              f"({best_strategy['metrics']['fp_reduction']*100:.1f}% reduction)")
        print(f"Sensitivity: {base_metrics['sensitivity']:.3f} → {best_strategy['metrics']['sensitivity']:.3f} "
              f"({best_strategy['metrics']['sensitivity_change']:+.3f})")
        print(f"Precision: {base_metrics['precision']:.3f} → {best_strategy['metrics']['precision']:.3f} "
              f"({best_strategy['metrics']['precision'] - base_metrics['precision']:+.3f})")
        print(f"False alarm rate: {base_metrics['false_alarm_rate']:.3f} → {best_strategy['metrics']['false_alarm_rate']:.3f} "
              f"({base_metrics['false_alarm_rate'] - best_strategy['metrics']['false_alarm_rate']:+.3f})")
        print("="*60)
        print(f"Results saved to: {self.output_folder}")


def main():
    """Main function."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 madrid_smart_clustering.py <input_folder>")
        print("Example: python3 madrid_smart_clustering.py madrid_seizure_results_parallel_example")
        sys.exit(1)
        
    input_folder = sys.argv[1]
    
    analyzer = SmartMadridClusteringAnalyzer(input_folder)
    
    try:
        results = analyzer.run_smart_analysis()
        print("\nSmart analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()