#!/usr/bin/env python3
"""
Simple script to run Madrid clustering analysis without external dependencies.
This version uses only built-in Python libraries for basic clustering analysis.
"""

import json
import os
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict


class SimpleMadridClusteringAnalyzer:
    """Simplified analyzer for clustering Madrid seizure detection results."""
    
    def __init__(self, results_folder: str, output_folder: str = None):
        self.results_folder = Path(results_folder)
        self.output_folder = Path(output_folder or f"{results_folder}_clustered")
        self.output_folder.mkdir(exist_ok=True)
        
        # Create subfolders
        (self.output_folder / "metrics_before").mkdir(exist_ok=True)
        (self.output_folder / "clusters").mkdir(exist_ok=True)
        (self.output_folder / "metrics_after").mkdir(exist_ok=True)
        
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
        
    def calculate_metrics_before_clustering(self) -> Dict[str, Any]:
        """Calculate detection metrics before clustering."""
        print("Calculating metrics before clustering...")
        
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
        
        metrics_before = {
            'total_files': len(self.results_data),
            'files_with_seizures': files_with_seizures,
            'files_with_detected_seizures': files_with_detected_seizures,
            'total_anomalies': total_anomalies,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'sensitivity': sensitivity,
            'precision': precision,
            'false_alarm_rate': false_alarm_rate,
            'anomalies_per_file': total_anomalies / len(self.results_data) if len(self.results_data) > 0 else 0
        }
        
        with open(self.output_folder / "metrics_before" / "metrics_summary.json", 'w') as f:
            json.dump(metrics_before, f, indent=2)
            
        print(f"Before clustering - Total anomalies: {total_anomalies}, TP: {true_positives}, FP: {false_positives}")
        print(f"Sensitivity: {sensitivity:.3f}, Precision: {precision:.3f}, FAR: {false_alarm_rate:.3f}")
        
        return metrics_before
        
    def simple_time_based_clustering(self, time_threshold: float = 300.0) -> List[List[Dict]]:
        """Simple time-based clustering: group anomalies within time_threshold seconds per file."""
        print(f"Performing simple time-based clustering per file (threshold: {time_threshold}s)")
        
        all_clusters = []
        
        # Group anomalies by file
        file_groups = defaultdict(list)
        for anomaly in self.all_anomalies:
            file_groups[anomaly['file_id']].append(anomaly)
        
        print(f"Processing {len(file_groups)} files separately...")
        
        # Perform clustering within each file separately
        for file_id, file_anomalies in file_groups.items():
            print(f"  File {file_id}: {len(file_anomalies)} anomalies")
            
            # Sort anomalies within this file by time
            sorted_anomalies = sorted(file_anomalies, key=lambda x: x['location_time_seconds'])
            
            current_cluster = []
            file_clusters = 0
            
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
                            file_clusters += 1
                        current_cluster = [anomaly]
                        
            # Add the last cluster from this file
            if current_cluster:
                all_clusters.append(current_cluster)
                file_clusters += 1
                
            print(f"    → {file_clusters} clusters")
        
        print(f"Found {len(all_clusters)} total clusters from {len(self.all_anomalies)} anomalies across {len(file_groups)} files")
        
        return all_clusters
        
    def select_cluster_representatives(self, clusters: List[List[Dict]]) -> List[Dict]:
        """Select representative based on minimal time distance to all cluster members."""
        print("Selecting cluster representatives based on time centrality...")
        
        representatives = []
        for i, cluster in enumerate(clusters):
            if not cluster:
                continue
                
            if len(cluster) == 1:
                # Single anomaly cluster
                representative = cluster[0]
                avg_distance = 0.0
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
                avg_distance = min_avg_distance
                
            # Add cluster metadata
            true_positives = [a for a in cluster if a.get('seizure_hit', False)]
            representative['cluster_id'] = i
            representative['cluster_size'] = len(cluster)
            representative['cluster_tp_count'] = len(true_positives)
            representative['avg_time_distance'] = avg_distance
            representatives.append(representative)
            
            print(f"  Cluster {i}: Size {len(cluster)}, Representative at {representative['location_time_seconds']:.1f}s (avg dist: {avg_distance:.1f}s)")
            
        print(f"Selected {len(representatives)} representatives from {len(clusters)} clusters")
        return representatives
        
    def calculate_metrics_after_clustering(self, representatives: List[Dict]) -> Dict[str, Any]:
        """Calculate metrics after clustering."""
        print("Calculating metrics after clustering...")
        
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
        
        files_with_seizures = sum(1 for data in self.results_data 
                                if data['validation_data']['ground_truth']['seizure_present'])
        
        sensitivity = files_with_detected_seizures / files_with_seizures if files_with_seizures > 0 else 0
        precision = true_positives / total_anomalies if total_anomalies > 0 else 0
        false_alarm_rate = false_positives / total_anomalies if total_anomalies > 0 else 0
        
        metrics_after = {
            'total_files': len(self.results_data),
            'files_with_seizures': files_with_seizures,
            'files_with_detected_seizures': files_with_detected_seizures,
            'total_anomalies': total_anomalies,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'sensitivity': sensitivity,
            'precision': precision,
            'false_alarm_rate': false_alarm_rate,
            'anomalies_per_file': total_anomalies / len(self.results_data) if len(self.results_data) > 0 else 0,
            'cluster_representatives': len(representatives)
        }
        
        with open(self.output_folder / "metrics_after" / "metrics_summary.json", 'w') as f:
            json.dump(metrics_after, f, indent=2)
            
        print(f"After clustering - Total anomalies: {total_anomalies}, TP: {true_positives}, FP: {false_positives}")
        print(f"Sensitivity: {sensitivity:.3f}, Precision: {precision:.3f}, FAR: {false_alarm_rate:.3f}")
        
        return metrics_after
        
    def compare_and_save_results(self, metrics_before: Dict, metrics_after: Dict, 
                                representatives: List[Dict], clusters: List[List[Dict]]) -> Dict[str, Any]:
        """Compare metrics and save all results."""
        print("Comparing metrics and saving results...")
        
        comparison = {
            'reduction_in_anomalies': {
                'absolute': metrics_before['total_anomalies'] - metrics_after['total_anomalies'],
                'percentage': ((metrics_before['total_anomalies'] - metrics_after['total_anomalies']) / 
                             metrics_before['total_anomalies'] * 100) if metrics_before['total_anomalies'] > 0 else 0
            },
            'reduction_in_false_positives': {
                'absolute': metrics_before['false_positives'] - metrics_after['false_positives'],
                'percentage': ((metrics_before['false_positives'] - metrics_after['false_positives']) / 
                             metrics_before['false_positives'] * 100) if metrics_before['false_positives'] > 0 else 0
            },
            'sensitivity_change': metrics_after['sensitivity'] - metrics_before['sensitivity'],
            'precision_improvement': metrics_after['precision'] - metrics_before['precision'],
            'false_alarm_rate_reduction': metrics_before['false_alarm_rate'] - metrics_after['false_alarm_rate'],
            'metrics_before': metrics_before,
            'metrics_after': metrics_after
        }
        
        # Save comparison
        with open(self.output_folder / "metrics_comparison.json", 'w') as f:
            json.dump(comparison, f, indent=2)
            
        # Save cluster representatives
        representatives_output = {
            'clustering_method': 'time_based',
            'timestamp': datetime.now().isoformat(),
            'total_representatives': len(representatives),
            'original_anomalies': len(self.all_anomalies),
            'reduction_ratio': len(representatives) / len(self.all_anomalies) if len(self.all_anomalies) > 0 else 0,
            'representatives': representatives
        }
        
        with open(self.output_folder / "clusters" / "cluster_representatives.json", 'w') as f:
            json.dump(representatives_output, f, indent=2)
            
        # Save cluster details
        cluster_details = {
            'clustering_method': 'time_based',
            'total_clusters': len(clusters),
            'cluster_sizes': [len(cluster) for cluster in clusters],
            'clusters': [
                {
                    'cluster_id': i,
                    'size': len(cluster),
                    'anomalies': cluster
                } for i, cluster in enumerate(clusters)
            ]
        }
        
        with open(self.output_folder / "clusters" / "clustering_details.json", 'w') as f:
            json.dump(cluster_details, f, indent=2)
            
        return comparison
        
    def run_analysis(self, time_threshold: float = 300.0) -> Dict[str, Any]:
        """Run the complete analysis."""
        print("Starting Madrid clustering analysis...")
        print(f"Input folder: {self.results_folder}")
        print(f"Output folder: {self.output_folder}")
        
        # Load data
        self.load_madrid_results()
        
        if not self.all_anomalies:
            raise ValueError("No anomalies found in the results folder")
            
        # Calculate metrics before
        metrics_before = self.calculate_metrics_before_clustering()
        
        # Perform clustering
        clusters = self.simple_time_based_clustering(time_threshold)
        
        # Select representatives
        representatives = self.select_cluster_representatives(clusters)
        
        # Calculate metrics after
        metrics_after = self.calculate_metrics_after_clustering(representatives)
        
        # Compare and save results
        comparison = self.compare_and_save_results(metrics_before, metrics_after, representatives, clusters)
        
        # Print summary
        print("\n" + "="*60)
        print("CLUSTERING ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total anomalies reduced: {comparison['reduction_in_anomalies']['absolute']} "
              f"({comparison['reduction_in_anomalies']['percentage']:.1f}%)")
        print(f"False positives reduced: {comparison['reduction_in_false_positives']['absolute']} "
              f"({comparison['reduction_in_false_positives']['percentage']:.1f}%)")
        print(f"Precision improved by: {comparison['precision_improvement']:.3f}")
        print(f"False alarm rate reduced by: {comparison['false_alarm_rate_reduction']:.3f}")
        print(f"Sensitivity change: {comparison['sensitivity_change']:.3f}")
        print("="*60)
        print(f"\nResults saved to: {self.output_folder}")
        
        return comparison


def main():
    """Main function."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 run_clustering_analysis.py <input_folder> [time_threshold]")
        print("Example: python3 run_clustering_analysis.py madrid_seizure_results_parallel_example 300")
        sys.exit(1)
        
    input_folder = sys.argv[1]
    time_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
    
    analyzer = SimpleMadridClusteringAnalyzer(input_folder)
    
    try:
        comparison = analyzer.run_analysis(time_threshold)
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()