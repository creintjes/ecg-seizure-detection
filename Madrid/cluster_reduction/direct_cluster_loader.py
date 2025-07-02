#!/usr/bin/env python3
"""
Direct Cluster Loader for existing clustered anomaly data.

This module handles cluster data that already contains cluster assignments
and anomaly information in a structured format.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

# Import from original module
from cluster_analyzer import Anomaly, Cluster

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DirectClusterLoader:
    """Load cluster data that already contains cluster assignments."""
    
    def __init__(self, clustered_data_path: str):
        """
        Initialize with path to clustered data.
        
        Args:
            clustered_data_path: Path to file containing clustered anomaly data
        """
        self.clustered_data_path = Path(clustered_data_path)
        self.clusters = {}
        self.anomalies = {}
        
    def load_clustered_data(self) -> Dict[str, Any]:
        """Load clustered anomaly data from file."""
        try:
            if self.clustered_data_path.suffix == '.json':
                with open(self.clustered_data_path, 'r') as f:
                    data = json.load(f)
            elif self.clustered_data_path.suffix == '.pkl':
                with open(self.clustered_data_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                raise ValueError(f"Unsupported file format: {self.clustered_data_path.suffix}")
            
            logger.info(f"Loaded clustered data from {self.clustered_data_path}")
            
            # Handle different data structures
            if isinstance(data, list):
                # List of anomaly dictionaries
                return self.process_anomaly_list(data)
            elif isinstance(data, dict):
                # Dictionary with clusters
                return self.process_cluster_dict(data)
            else:
                raise ValueError(f"Unexpected data structure: {type(data)}")
                
        except Exception as e:
            logger.error(f"Error loading clustered data: {e}")
            raise
    
    def process_anomaly_list(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process list of anomaly dictionaries with cluster assignments."""
        
        # Group anomalies by cluster_id
        clusters_data = defaultdict(list)
        
        for anomaly_data in data:
            # Create Anomaly object
            anomaly = self.create_anomaly_from_dict(anomaly_data)
            if anomaly:
                self.anomalies[anomaly.anomaly_id] = anomaly
                
                # Group by cluster_id
                cluster_id = str(anomaly_data.get('cluster_id', 'unknown'))
                clusters_data[cluster_id].append(anomaly)
        
        # Create Cluster objects
        for cluster_id, anomaly_list in clusters_data.items():
            if cluster_id != 'unknown' and anomaly_list:
                cluster = self.create_cluster_from_anomalies(cluster_id, anomaly_list)
                if cluster:
                    self.clusters[cluster_id] = cluster
        
        logger.info(f"Processed {len(data)} anomalies into {len(self.clusters)} clusters")
        
        return {
            'clusters': self.clusters,
            'anomalies': self.anomalies,
            'raw_data': data
        }
    
    def process_cluster_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process dictionary structure with cluster information."""
        # This would handle dictionary-based cluster data
        # Implementation depends on your specific data structure
        
        logger.warning("Dictionary-based cluster data not yet implemented")
        return {'clusters': {}, 'anomalies': {}, 'raw_data': data}
    
    def create_anomaly_from_dict(self, anomaly_data: Dict[str, Any]) -> Optional[Anomaly]:
        """Create Anomaly object from dictionary data."""
        try:
            # Extract required fields with fallbacks
            anomaly_id = anomaly_data.get('anomaly_id', f"anomaly_{anomaly_data.get('rank', 0)}")
            
            # Parse subject info
            file_id = anomaly_data.get('file_id', '')
            if '_' in file_id:
                parts = file_id.split('_')
                subject_id = parts[0] if len(parts) > 0 else 'unknown'
                run_id = parts[1] if len(parts) > 1 else 'unknown' 
                seizure_id = '_'.join(parts[2:]) if len(parts) > 2 else 'unknown'
            else:
                subject_id = anomaly_data.get('subject_id', 'unknown')
                run_id = 'unknown'
                seizure_id = 'unknown'
            
            anomaly = Anomaly(
                anomaly_id=anomaly_id,
                rank=anomaly_data.get('rank', 0),
                m_value=anomaly_data.get('m_value', 0),
                anomaly_score=anomaly_data.get('anomaly_score', 0.0),
                location_sample=anomaly_data.get('location_sample', 0),
                location_time_seconds=anomaly_data.get('location_time_seconds', 0.0),
                seizure_hit=anomaly_data.get('seizure_hit', False),
                normalized_score=anomaly_data.get('normalized_score', anomaly_data.get('anomaly_score', 0.0)),
                confidence=anomaly_data.get('confidence', anomaly_data.get('anomaly_score', 0.0)),
                subject_id=subject_id,
                run_id=run_id,
                seizure_id=seizure_id
            )
            
            return anomaly
            
        except Exception as e:
            logger.warning(f"Error creating anomaly from data: {e}")
            return None
    
    def create_cluster_from_anomalies(self, cluster_id: str, anomalies: List[Anomaly]) -> Optional[Cluster]:
        """Create Cluster object from list of anomalies."""
        try:
            if not anomalies:
                return None
            
            # Calculate cluster statistics
            has_seizure_hit = any(a.seizure_hit for a in anomalies)
            cluster_size = len(anomalies)
            scores = [a.anomaly_score for a in anomalies]
            times = [a.location_time_seconds for a in anomalies]
            locations = [a.location_sample for a in anomalies]
            
            # Temporal and spatial statistics
            temporal_span = max(times) - min(times) if len(times) > 1 else 0.0
            spatial_span = max(locations) - min(locations) if len(locations) > 1 else 0.0
            
            # Score statistics
            avg_score = np.mean(scores)
            max_score = np.max(scores)
            score_variance = np.var(scores) if len(scores) > 1 else 0.0
            
            # Spatial consistency (inverse of coefficient of variation for locations)
            if len(locations) > 1 and np.mean(locations) > 0:
                spatial_cv = np.std(locations) / np.mean(locations)
                spatial_consistency = 1.0 / (1.0 + spatial_cv)
            else:
                spatial_consistency = 1.0
            
            # Representative anomaly (highest score)
            representative_anomaly = max(anomalies, key=lambda a: a.anomaly_score)
            
            cluster = Cluster(
                cluster_id=cluster_id,
                representative_anomaly_id=representative_anomaly.anomaly_id,
                anomalies=anomalies,
                has_seizure_hit=has_seizure_hit,
                cluster_size=cluster_size,
                temporal_span=temporal_span,
                spatial_consistency=spatial_consistency,
                avg_anomaly_score=avg_score,
                max_anomaly_score=max_score,
                score_variance=score_variance
            )
            
            return cluster
            
        except Exception as e:
            logger.warning(f"Error creating cluster {cluster_id}: {e}")
            return None
    
    def analyze_cluster_distribution(self) -> Dict[str, Any]:
        """Analyze the distribution of clusters."""
        if not self.clusters:
            return {}
        
        total_clusters = len(self.clusters)
        seizure_clusters = sum(1 for c in self.clusters.values() if c.has_seizure_hit)
        fp_clusters = total_clusters - seizure_clusters
        
        # Subject-level analysis
        subject_stats = defaultdict(lambda: {'total': 0, 'seizure_hit': 0, 'fp': 0})
        for cluster in self.clusters.values():
            if cluster.anomalies:
                subject_id = cluster.anomalies[0].subject_id
                subject_stats[subject_id]['total'] += 1
                if cluster.has_seizure_hit:
                    subject_stats[subject_id]['seizure_hit'] += 1
                else:
                    subject_stats[subject_id]['fp'] += 1
        
        # Calculate statistics
        cluster_sizes = [c.cluster_size for c in self.clusters.values()]
        seizure_scores = [c.avg_anomaly_score for c in self.clusters.values() if c.has_seizure_hit]
        fp_scores = [c.avg_anomaly_score for c in self.clusters.values() if not c.has_seizure_hit]
        
        analysis = {
            'total_clusters': total_clusters,
            'seizure_hit_clusters': seizure_clusters,
            'false_positive_clusters': fp_clusters,
            'seizure_hit_ratio': seizure_clusters / total_clusters if total_clusters > 0 else 0,
            'avg_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
            'avg_seizure_score': np.mean(seizure_scores) if seizure_scores else 0,
            'avg_fp_score': np.mean(fp_scores) if fp_scores else 0,
            'subjects_analyzed': len(subject_stats),
            'subject_statistics': dict(subject_stats)
        }
        
        return analysis
    
    def get_balanced_dataset(self, balance_strategy: str = 'undersample') -> Tuple[List[Cluster], List[int]]:
        """Create a balanced dataset for training."""
        if not self.clusters:
            raise ValueError("No clusters loaded")
        
        # Separate positive and negative samples
        positive_clusters = [c for c in self.clusters.values() if c.has_seizure_hit]
        negative_clusters = [c for c in self.clusters.values() if not c.has_seizure_hit]
        
        logger.info(f"Dataset composition: {len(positive_clusters)} positive, {len(negative_clusters)} negative")
        
        if balance_strategy == 'undersample':
            # Undersample majority class
            min_size = min(len(positive_clusters), len(negative_clusters))
            if len(negative_clusters) > min_size:
                negative_clusters = np.random.choice(negative_clusters, min_size, replace=False).tolist()
            if len(positive_clusters) > min_size:
                positive_clusters = np.random.choice(positive_clusters, min_size, replace=False).tolist()
        
        # Combine and create labels
        all_clusters = positive_clusters + negative_clusters
        labels = [1] * len(positive_clusters) + [0] * len(negative_clusters)
        
        logger.info(f"Balanced dataset: {len(all_clusters)} clusters, {sum(labels)} positive")
        return all_clusters, labels
    
    def save_analysis_report(self, output_path: str):
        """Save analysis report to file."""
        analysis = self.analyze_cluster_distribution()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write("Direct Cluster Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Clusters: {analysis.get('total_clusters', 0)}\n")
            f.write(f"Seizure Hit Clusters: {analysis.get('seizure_hit_clusters', 0)}\n")
            f.write(f"False Positive Clusters: {analysis.get('false_positive_clusters', 0)}\n")
            f.write(f"Seizure Hit Ratio: {analysis.get('seizure_hit_ratio', 0):.3f}\n")
            f.write(f"Average Cluster Size: {analysis.get('avg_cluster_size', 0):.2f}\n")
            f.write(f"Subjects Analyzed: {analysis.get('subjects_analyzed', 0)}\n\n")
            
            f.write("Subject Statistics:\n")
            f.write("-" * 30 + "\n")
            for subject_id, stats in analysis.get('subject_statistics', {}).items():
                f.write(f"{subject_id}: {stats['seizure_hit']}/{stats['total']} seizure hits "
                       f"({stats['seizure_hit']/stats['total']*100:.1f}%)\n")
        
        logger.info(f"Analysis report saved to {output_file}")

def main():
    """Example usage."""
    # Example path - adjust to your clustered data file
    clustered_data_path = "path/to/your/clustered_data.json"
    
    try:
        loader = DirectClusterLoader(clustered_data_path)
        results = loader.load_clustered_data()
        
        print(f"Loaded {len(results['clusters'])} clusters")
        print(f"Loaded {len(results['anomalies'])} anomalies")
        
        # Analyze
        analysis = loader.analyze_cluster_distribution()
        print("Analysis:", analysis)
        
        return loader
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return None

if __name__ == "__main__":
    loader = main()