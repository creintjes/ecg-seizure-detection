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
        
        # 1. Time-based with different thresholds
        for threshold in [15, 30, 60, 120]:
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
            print(f"Time-based ({threshold}s): {len(clusters)} clusters, score: {metrics['score']:.3f}")
            
        # 2. Time-based with different thresholds (renamed from score-based)
        for threshold in [20, 25, 35, 45]:
            clusters = self.score_based_clustering(threshold)
            representatives = self.select_smart_representatives(clusters)
            metrics = self.evaluate_clustering_strategy(representatives, base_metrics)
            strategies[f'time_alt_{threshold}s'] = {
                'method': 'time_based_alt',
                'parameters': {'threshold': threshold},
                'clusters': len(clusters),
                'representatives': representatives,
                'metrics': metrics
            }
            print(f"Time-based alt ({threshold}s): {len(clusters)} clusters, score: {metrics['score']:.3f}")
            
        # 3. Time-based with medium threshold
        clusters = self.file_based_clustering(30)
        representatives = self.select_smart_representatives(clusters)
        metrics = self.evaluate_clustering_strategy(representatives, base_metrics)
        strategies['time_30s'] = {
            'method': 'time_based',
            'parameters': {'threshold': 30},
            'clusters': len(clusters),
            'representatives': representatives,
            'metrics': metrics
        }
        print(f"Time-based (30s): {len(clusters)} clusters, score: {metrics['score']:.3f}")
        
        # 4. Time-based with larger threshold
        clusters = self.hybrid_clustering(45)
        representatives = self.select_smart_representatives(clusters)
        metrics = self.evaluate_clustering_strategy(representatives, base_metrics)
        strategies['time_45s'] = {
            'method': 'time_based',
            'parameters': {'threshold': 45},
            'clusters': len(clusters),
            'representatives': representatives,
            'metrics': metrics
        }
        print(f"Time-based (45s): {len(clusters)} clusters, score: {metrics['score']:.3f}")
        
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