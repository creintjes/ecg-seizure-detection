#!/usr/bin/env python3
"""
Madrid Clustering Analysis Script

This script analyzes Madrid seizure detection results, performs clustering on detected anomalies 
to reduce false positives, and compares metrics before and after clustering.


"""

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


class MadridClusteringAnalyzer:
    """Analyzer for clustering Madrid seizure detection results to reduce false positives."""
    
    def __init__(self, results_folder: str, output_folder: str = None):
        """
        Initialize the analyzer.
        
        Args:
            results_folder: Path to folder containing Madrid results JSON files
            output_folder: Path to save clustering results (default: results_folder + '_clustered')
        """
        self.results_folder = Path(results_folder)
        self.output_folder = Path(output_folder or f"{results_folder}_clustered")
        self.output_folder.mkdir(exist_ok=True)
        
        # Create subfolders for organized output
        (self.output_folder / "metrics_before").mkdir(exist_ok=True)
        (self.output_folder / "clusters").mkdir(exist_ok=True)
        (self.output_folder / "metrics_after").mkdir(exist_ok=True)
        (self.output_folder / "visualizations").mkdir(exist_ok=True)
        
        self.results_data = []
        self.all_anomalies = []
        self.clustering_results = {}
        
    def load_madrid_results(self) -> None:
        """Load all Madrid result JSON files from the results folder."""
        print(f"Loading Madrid results from: {self.results_folder}")
        
        json_files = list(self.results_folder.glob("madrid_results_*.json"))
        print(f"Found {len(json_files)} result files")
        
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    self.results_data.append(data)
                    
                    # Extract anomalies with file metadata
                    for anomaly in data.get('analysis_results', {}).get('anomalies', []):
                        anomaly_with_metadata = anomaly.copy()
                        anomaly_with_metadata['file_id'] = data['input_data']['subject_id'] + '_' + \
                                                          data['input_data']['run_id'] + '_' + \
                                                          data['input_data']['seizure_id']
                        anomaly_with_metadata['subject_id'] = data['input_data']['subject_id']
                        anomaly_with_metadata['seizure_present'] = data['validation_data']['ground_truth']['seizure_present']
                        self.all_anomalies.append(anomaly_with_metadata)
                        
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        print(f"Loaded {len(self.results_data)} files with {len(self.all_anomalies)} total anomalies")
        
    def calculate_metrics_before_clustering(self) -> Dict[str, Any]:
        """Calculate detection metrics before clustering."""
        print("Calculating metrics before clustering...")
        
        # Count true positives and false positives
        true_positives = sum(1 for a in self.all_anomalies if a.get('seizure_hit', False))
        false_positives = sum(1 for a in self.all_anomalies if not a.get('seizure_hit', False))
        total_anomalies = len(self.all_anomalies)
        
        # Count seizures detected vs total seizures present
        files_with_seizures = sum(1 for data in self.results_data 
                                if data['validation_data']['ground_truth']['seizure_present'])
        files_with_detected_seizures = sum(1 for data in self.results_data 
                                         if data['analysis_results']['performance_metrics']['true_positives'] > 0)
        
        # Calculate metrics
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
        
        # Save metrics before clustering
        with open(self.output_folder / "metrics_before" / "metrics_summary.json", 'w') as f:
            json.dump(metrics_before, f, indent=2)
            
        print(f"Before clustering - Total anomalies: {total_anomalies}, "
              f"True positives: {true_positives}, False positives: {false_positives}")
        print(f"Sensitivity: {sensitivity:.3f}, Precision: {precision:.3f}, FAR: {false_alarm_rate:.3f}")
        
        return metrics_before
        
    def prepare_clustering_features(self) -> np.ndarray:
        """Prepare features for clustering anomalies."""
        print("Preparing features for clustering...")
        
        features = []
        for anomaly in self.all_anomalies:
            # Use time location, anomaly score, m_value, and rank as features
            feature_vector = [
                anomaly['location_time_seconds'],
                anomaly['anomaly_score'],
                anomaly['m_value'],
                anomaly['rank']
            ]
            features.append(feature_vector)
            
        features_array = np.array(features)
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_array)
        
        return features_scaled, scaler
        
    def perform_clustering(self, features: np.ndarray) -> Dict[str, Any]:
        """Perform clustering on anomalies using multiple algorithms."""
        print("Performing clustering analysis...")
        
        clustering_results = {}
        
        # Try different clustering algorithms
        
        # 1. DBSCAN - good for finding clusters of varying shapes and sizes
        print("Running DBSCAN clustering...")
        dbscan = DBSCAN(eps=0.5, min_samples=3)
        dbscan_labels = dbscan.fit_predict(features)
        n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_noise_dbscan = list(dbscan_labels).count(-1)
        
        clustering_results['dbscan'] = {
            'labels': dbscan_labels.tolist(),
            'n_clusters': n_clusters_dbscan,
            'n_noise': n_noise_dbscan,
            'silhouette_score': silhouette_score(features, dbscan_labels) if n_clusters_dbscan > 1 else -1
        }
        
        # 2. K-Means with different k values
        print("Running K-Means clustering...")
        best_kmeans = None
        best_k = 0
        best_silhouette = -1
        
        for k in range(2, min(20, len(self.all_anomalies) // 3)):
            if k >= len(self.all_anomalies):
                break
                
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(features)
            silhouette = silhouette_score(features, kmeans_labels)
            
            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_kmeans = kmeans
                best_k = k
                
        if best_kmeans is not None:
            clustering_results['kmeans'] = {
                'labels': best_kmeans.labels_.tolist(),
                'n_clusters': best_k,
                'silhouette_score': best_silhouette,
                'cluster_centers': best_kmeans.cluster_centers_.tolist()
            }
            
        print(f"DBSCAN found {n_clusters_dbscan} clusters ({n_noise_dbscan} noise points)")
        print(f"Best K-Means: k={best_k}, silhouette={best_silhouette:.3f}")
        
        return clustering_results
        
    def select_cluster_representatives(self, clustering_method: str = 'dbscan') -> List[Dict]:
        """Select representative anomalies from each cluster."""
        print(f"Selecting cluster representatives using {clustering_method}...")
        
        if clustering_method not in self.clustering_results:
            raise ValueError(f"Clustering method {clustering_method} not found in results")
            
        labels = self.clustering_results[clustering_method]['labels']
        unique_labels = set(labels)
        
        # Remove noise label (-1) if present
        if -1 in unique_labels:
            unique_labels.remove(-1)
            
        cluster_representatives = []
        
        for cluster_label in unique_labels:
            # Get all anomalies in this cluster
            cluster_anomalies = [self.all_anomalies[i] for i, label in enumerate(labels) 
                               if label == cluster_label]
            
            if not cluster_anomalies:
                continue
                
            # Select representative based on highest anomaly score
            representative = max(cluster_anomalies, key=lambda x: x['anomaly_score'])
            representative['cluster_id'] = cluster_label
            representative['cluster_size'] = len(cluster_anomalies)
            cluster_representatives.append(representative)
            
        # Add noise points as individual anomalies if using DBSCAN
        if clustering_method == 'dbscan':
            noise_anomalies = [self.all_anomalies[i] for i, label in enumerate(labels) 
                             if label == -1]
            for noise_anomaly in noise_anomalies:
                noise_anomaly['cluster_id'] = -1
                noise_anomaly['cluster_size'] = 1
                cluster_representatives.append(noise_anomaly)
                
        print(f"Selected {len(cluster_representatives)} cluster representatives from "
              f"{len(self.all_anomalies)} original anomalies")
        
        return cluster_representatives
        
    def calculate_metrics_after_clustering(self, representatives: List[Dict]) -> Dict[str, Any]:
        """Calculate detection metrics after clustering."""
        print("Calculating metrics after clustering...")
        
        # Count true positives and false positives in representatives
        true_positives = sum(1 for r in representatives if r.get('seizure_hit', False))
        false_positives = sum(1 for r in representatives if not r.get('seizure_hit', False))
        total_anomalies = len(representatives)
        
        # Count files with detected seizures (at least one representative is TP)
        files_with_representatives = {}
        for rep in representatives:
            file_id = rep['file_id']
            if file_id not in files_with_representatives:
                files_with_representatives[file_id] = {'has_tp': False}
            if rep.get('seizure_hit', False):
                files_with_representatives[file_id]['has_tp'] = True
                
        files_with_detected_seizures = sum(1 for file_data in files_with_representatives.values() 
                                         if file_data['has_tp'])
        
        # Get total files with seizures (unchanged)
        files_with_seizures = sum(1 for data in self.results_data 
                                if data['validation_data']['ground_truth']['seizure_present'])
        
        # Calculate metrics
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
        
        # Save metrics after clustering
        with open(self.output_folder / "metrics_after" / "metrics_summary.json", 'w') as f:
            json.dump(metrics_after, f, indent=2)
            
        print(f"After clustering - Total anomalies: {total_anomalies}, "
              f"True positives: {true_positives}, False positives: {false_positives}")
        print(f"Sensitivity: {sensitivity:.3f}, Precision: {precision:.3f}, FAR: {false_alarm_rate:.3f}")
        
        return metrics_after
        
    def create_visualizations(self, features: np.ndarray, clustering_method: str = 'dbscan') -> None:
        """Create visualizations of clustering results."""
        print("Creating visualizations...")
        
        labels = self.clustering_results[clustering_method]['labels']
        
        # 1. Cluster visualization (2D projection using first two features)
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        scatter = plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='tab10', alpha=0.7)
        plt.xlabel('Feature 1 (Time Location - Scaled)')
        plt.ylabel('Feature 2 (Anomaly Score - Scaled)')
        plt.title(f'Clustering Results ({clustering_method.upper()})')
        plt.colorbar(scatter)
        
        # 2. Anomaly scores distribution
        plt.subplot(2, 2, 2)
        scores = [a['anomaly_score'] for a in self.all_anomalies]
        tp_scores = [a['anomaly_score'] for a in self.all_anomalies if a.get('seizure_hit', False)]
        fp_scores = [a['anomaly_score'] for a in self.all_anomalies if not a.get('seizure_hit', False)]
        
        plt.hist(fp_scores, alpha=0.7, label='False Positives', bins=20, color='red')
        plt.hist(tp_scores, alpha=0.7, label='True Positives', bins=20, color='green')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.title('Anomaly Score Distribution')
        plt.legend()
        
        # 3. Cluster sizes
        plt.subplot(2, 2, 3)
        unique_labels, counts = np.unique(labels, return_counts=True)
        plt.bar(range(len(unique_labels)), counts)
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Anomalies')
        plt.title('Cluster Sizes')
        plt.xticks(range(len(unique_labels)), unique_labels)
        
        # 4. Time location of anomalies
        plt.subplot(2, 2, 4)
        times = [a['location_time_seconds'] for a in self.all_anomalies]
        tp_times = [a['location_time_seconds'] for a in self.all_anomalies if a.get('seizure_hit', False)]
        fp_times = [a['location_time_seconds'] for a in self.all_anomalies if not a.get('seizure_hit', False)]
        
        plt.scatter(fp_times, [0]*len(fp_times), alpha=0.7, color='red', label='False Positives')
        plt.scatter(tp_times, [0]*len(tp_times), alpha=0.7, color='green', label='True Positives', s=100)
        plt.xlabel('Time (seconds)')
        plt.ylabel('')
        plt.title('Anomaly Locations in Time')
        plt.legend()
        plt.yticks([])
        
        plt.tight_layout()
        plt.savefig(self.output_folder / "visualizations" / f"clustering_analysis_{clustering_method}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def compare_metrics(self, metrics_before: Dict, metrics_after: Dict) -> Dict[str, Any]:
        """Compare metrics before and after clustering."""
        print("Comparing metrics before and after clustering...")
        
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
        
        return comparison
        
    def save_clustering_results(self, representatives: List[Dict], clustering_method: str) -> None:
        """Save detailed clustering results."""
        print("Saving clustering results...")
        
        # Save cluster representatives
        representatives_output = {
            'clustering_method': clustering_method,
            'timestamp': datetime.now().isoformat(),
            'total_representatives': len(representatives),
            'original_anomalies': len(self.all_anomalies),
            'reduction_ratio': len(representatives) / len(self.all_anomalies) if len(self.all_anomalies) > 0 else 0,
            'representatives': representatives
        }
        
        with open(self.output_folder / "clusters" / f"cluster_representatives_{clustering_method}.json", 'w') as f:
            json.dump(representatives_output, f, indent=2)
            
        # Save full clustering results
        with open(self.output_folder / "clusters" / f"clustering_details_{clustering_method}.json", 'w') as f:
            json.dump(self.clustering_results, f, indent=2)
            
        # Create CSV for easy analysis
        df_representatives = pd.DataFrame(representatives)
        df_representatives.to_csv(self.output_folder / "clusters" / f"representatives_{clustering_method}.csv", 
                                index=False)
        
    def run_full_analysis(self, clustering_method: str = 'dbscan') -> Dict[str, Any]:
        """Run the complete clustering analysis pipeline."""
        print("Starting Madrid clustering analysis...")
        print(f"Input folder: {self.results_folder}")
        print(f"Output folder: {self.output_folder}")
        
        # Load data
        self.load_madrid_results()
        
        if not self.all_anomalies:
            raise ValueError("No anomalies found in the results folder")
            
        # Calculate metrics before clustering
        metrics_before = self.calculate_metrics_before_clustering()
        
        # Prepare features and perform clustering
        features, scaler = self.prepare_clustering_features()
        self.clustering_results = self.perform_clustering(features)
        
        # Select cluster representatives
        representatives = self.select_cluster_representatives(clustering_method)
        
        # Calculate metrics after clustering
        metrics_after = self.calculate_metrics_after_clustering(representatives)
        
        # Create visualizations
        self.create_visualizations(features, clustering_method)
        
        # Compare metrics
        comparison = self.compare_metrics(metrics_before, metrics_after)
        
        # Save results
        self.save_clustering_results(representatives, clustering_method)
        
        print(f"\nAnalysis complete! Results saved to: {self.output_folder}")
        
        return comparison


def main():
    """Main function to run the clustering analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Madrid seizure detection results with clustering')
    parser.add_argument('--input_folder', type=str, required=True,
                       help='Path to folder containing Madrid result JSON files')
    parser.add_argument('--output_folder', type=str, default=None,
                       help='Path to save clustering results (default: input_folder + "_clustered")')
    parser.add_argument('--clustering_method', type=str, default='dbscan',
                       choices=['dbscan', 'kmeans'],
                       help='Clustering method to use (default: dbscan)')
    
    args = parser.parse_args()
    
    # Create analyzer and run analysis
    analyzer = MadridClusteringAnalyzer(args.input_folder, args.output_folder)
    
    try:
        comparison = analyzer.run_full_analysis(args.clustering_method)
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()