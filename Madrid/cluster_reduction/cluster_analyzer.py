#!/usr/bin/env python3
"""
Cluster Analysis Module for Madrid False Positive Reduction

This module loads and analyzes cluster results from tolerance_adjusted_smart_clustered
and maps them to original Madrid results for feature extraction.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Anomaly:
    """Data class for individual anomaly information."""
    anomaly_id: str
    rank: int
    m_value: int
    anomaly_score: float
    location_sample: int
    location_time_seconds: float
    seizure_hit: bool
    normalized_score: float
    confidence: float
    subject_id: str
    run_id: str
    seizure_id: str

@dataclass
class Cluster:
    """Data class for cluster information."""
    cluster_id: str
    representative_anomaly_id: str
    anomalies: List[Anomaly]
    has_seizure_hit: bool
    cluster_size: int
    temporal_span: float
    spatial_consistency: float
    avg_anomaly_score: float
    max_anomaly_score: float
    score_variance: float

class ClusterAnalyzer:
    """Main class for analyzing Madrid cluster results."""
    
    def __init__(self, 
                 cluster_results_dir: str,
                 madrid_results_dir: str,
                 raw_data_dir: Optional[str] = None):
        """
        Initialize the cluster analyzer.
        
        Args:
            cluster_results_dir: Path to tolerance_adjusted_smart_clustered directory
            madrid_results_dir: Path to original Madrid results directory
            raw_data_dir: Path to raw ECG data (optional)
        """
        self.cluster_results_dir = Path(cluster_results_dir)
        self.madrid_results_dir = Path(madrid_results_dir)
        self.raw_data_dir = Path(raw_data_dir) if raw_data_dir else None
        
        # Data storage
        self.clusters = {}
        self.anomalies = {}
        self.madrid_results = {}
        self.cluster_metrics = {}
        
    def load_cluster_results(self) -> Dict[str, Any]:
        """Load cluster results from best_representatives.json."""
        try:
            cluster_file = self.cluster_results_dir / "clusters" / "best_representatives.json"
            if not cluster_file.exists():
                raise FileNotFoundError(f"Cluster file not found: {cluster_file}")
            
            with open(cluster_file, 'r') as f:
                cluster_data = json.load(f)
            
            logger.info(f"Loaded cluster data with {len(cluster_data)} entries")
            return cluster_data
            
        except Exception as e:
            logger.error(f"Error loading cluster results: {e}")
            raise
    
    def load_madrid_results(self) -> Dict[str, Any]:
        """Load original Madrid results from JSON files."""
        try:
            madrid_files = list(self.madrid_results_dir.glob("madrid_results_*.json"))
            if not madrid_files:
                raise FileNotFoundError(f"No Madrid result files found in {self.madrid_results_dir}")
            
            madrid_data = {}
            for file_path in madrid_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Extract identifiers
                    subject_id = data['input_data']['subject_id']
                    run_id = data['input_data']['run_id']
                    seizure_id = data['input_data']['seizure_id']
                    key = f"{subject_id}_{run_id}_{seizure_id}"
                    
                    madrid_data[key] = data
                    
                except Exception as e:
                    logger.warning(f"Error loading {file_path}: {e}")
                    continue
            
            logger.info(f"Loaded {len(madrid_data)} Madrid result files")
            self.madrid_results = madrid_data
            return madrid_data
            
        except Exception as e:
            logger.error(f"Error loading Madrid results: {e}")
            raise
    
    def create_anomaly_objects(self) -> Dict[str, Anomaly]:
        """Create Anomaly objects from Madrid results."""
        anomalies = {}
        
        for session_key, madrid_data in self.madrid_results.items():
            subject_id = madrid_data['input_data']['subject_id']
            run_id = madrid_data['input_data']['run_id']
            seizure_id = madrid_data['input_data']['seizure_id']
            
            # Extract anomalies from analysis results
            if 'analysis_results' in madrid_data and 'anomalies' in madrid_data['analysis_results']:
                for anomaly_data in madrid_data['analysis_results']['anomalies']:
                    anomaly = Anomaly(
                        anomaly_id=anomaly_data['anomaly_id'],
                        rank=anomaly_data['rank'],
                        m_value=anomaly_data['m_value'],
                        anomaly_score=anomaly_data['anomaly_score'],
                        location_sample=anomaly_data['location_sample'],
                        location_time_seconds=anomaly_data['location_time_seconds'],
                        seizure_hit=anomaly_data['seizure_hit'],
                        normalized_score=anomaly_data['normalized_score'],
                        confidence=anomaly_data['confidence'],
                        subject_id=subject_id,
                        run_id=run_id,
                        seizure_id=seizure_id
                    )
                    anomalies[anomaly.anomaly_id] = anomaly
        
        logger.info(f"Created {len(anomalies)} anomaly objects")
        self.anomalies = anomalies
        return anomalies
    
    def map_clusters_to_anomalies(self, cluster_data: Dict[str, Any]) -> Dict[str, Cluster]:
        """Map cluster representatives to full anomaly information."""
        clusters = {}
        
        for cluster_id, cluster_info in cluster_data.items():
            try:
                # Get representative anomaly ID
                if isinstance(cluster_info, dict) and 'representative' in cluster_info:
                    rep_anomaly_id = cluster_info['representative']
                elif isinstance(cluster_info, str):
                    rep_anomaly_id = cluster_info
                else:
                    logger.warning(f"Unexpected cluster info format for {cluster_id}: {cluster_info}")
                    continue
                
                # Find all anomalies belonging to this cluster
                # For now, we'll use the representative as a single-member cluster
                # This can be extended when cluster membership data is available
                cluster_anomalies = []
                if rep_anomaly_id in self.anomalies:
                    cluster_anomalies = [self.anomalies[rep_anomaly_id]]
                
                if not cluster_anomalies:
                    logger.warning(f"No anomalies found for cluster {cluster_id}")
                    continue
                
                # Calculate cluster statistics
                has_seizure_hit = any(a.seizure_hit for a in cluster_anomalies)
                cluster_size = len(cluster_anomalies)
                scores = [a.anomaly_score for a in cluster_anomalies]
                times = [a.location_time_seconds for a in cluster_anomalies]
                
                temporal_span = max(times) - min(times) if len(times) > 1 else 0.0
                avg_score = np.mean(scores)
                max_score = np.max(scores)
                score_variance = np.var(scores) if len(scores) > 1 else 0.0
                
                # Calculate spatial consistency (placeholder)
                spatial_consistency = 1.0  # Will be improved with more cluster data
                
                # Create cluster object
                cluster = Cluster(
                    cluster_id=cluster_id,
                    representative_anomaly_id=rep_anomaly_id,
                    anomalies=cluster_anomalies,
                    has_seizure_hit=has_seizure_hit,
                    cluster_size=cluster_size,
                    temporal_span=temporal_span,
                    spatial_consistency=spatial_consistency,
                    avg_anomaly_score=avg_score,
                    max_anomaly_score=max_score,
                    score_variance=score_variance
                )
                
                clusters[cluster_id] = cluster
                
            except Exception as e:
                logger.warning(f"Error processing cluster {cluster_id}: {e}")
                continue
        
        logger.info(f"Created {len(clusters)} cluster objects")
        self.clusters = clusters
        return clusters
    
    def analyze_cluster_distribution(self) -> Dict[str, Any]:
        """Analyze the distribution of seizure hits vs false positives."""
        if not self.clusters:
            raise ValueError("No clusters loaded. Run load_and_process_data() first.")
        
        total_clusters = len(self.clusters)
        seizure_clusters = sum(1 for c in self.clusters.values() if c.has_seizure_hit)
        fp_clusters = total_clusters - seizure_clusters
        
        # Analyze by subject
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
        
        self.cluster_metrics = analysis
        return analysis
    
    def get_balanced_dataset(self, balance_strategy: str = 'undersample') -> Tuple[List[Cluster], List[int]]:
        """
        Create a balanced dataset for training.
        
        Args:
            balance_strategy: 'undersample', 'oversample', or 'none'
            
        Returns:
            Tuple of (cluster_list, labels)
        """
        if not self.clusters:
            raise ValueError("No clusters loaded. Run load_and_process_data() first.")
        
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
                
        elif balance_strategy == 'oversample':
            # This will be handled by SMOTE in the ML pipeline
            pass
        
        # Combine and create labels
        all_clusters = positive_clusters + negative_clusters
        labels = [1] * len(positive_clusters) + [0] * len(negative_clusters)
        
        logger.info(f"Balanced dataset: {len(all_clusters)} clusters, {sum(labels)} positive")
        return all_clusters, labels
    
    def load_and_process_data(self) -> Dict[str, Any]:
        """Complete data loading and processing pipeline."""
        logger.info("Starting cluster analysis pipeline...")
        
        # Load cluster results
        cluster_data = self.load_cluster_results()
        
        # Load Madrid results
        madrid_data = self.load_madrid_results()
        
        # Create anomaly objects
        anomalies = self.create_anomaly_objects()
        
        # Map clusters to anomalies
        clusters = self.map_clusters_to_anomalies(cluster_data)
        
        # Analyze distribution
        analysis = self.analyze_cluster_distribution()
        
        logger.info("Cluster analysis pipeline completed successfully")
        return {
            'clusters': clusters,
            'anomalies': anomalies,
            'madrid_results': madrid_data,
            'analysis': analysis
        }
    
    def save_analysis_report(self, output_path: str):
        """Save analysis report to file."""
        if not self.cluster_metrics:
            raise ValueError("No analysis performed. Run analyze_cluster_distribution() first.")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write("Cluster Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            metrics = self.cluster_metrics
            f.write(f"Total Clusters: {metrics['total_clusters']}\n")
            f.write(f"Seizure Hit Clusters: {metrics['seizure_hit_clusters']}\n")
            f.write(f"False Positive Clusters: {metrics['false_positive_clusters']}\n")
            f.write(f"Seizure Hit Ratio: {metrics['seizure_hit_ratio']:.3f}\n")
            f.write(f"Average Cluster Size: {metrics['avg_cluster_size']:.2f}\n")
            f.write(f"Average Seizure Score: {metrics['avg_seizure_score']:.3f}\n")
            f.write(f"Average FP Score: {metrics['avg_fp_score']:.3f}\n")
            f.write(f"Subjects Analyzed: {metrics['subjects_analyzed']}\n\n")
            
            f.write("Subject Statistics:\n")
            f.write("-" * 30 + "\n")
            for subject_id, stats in metrics['subject_statistics'].items():
                f.write(f"{subject_id}: {stats['seizure_hit']}/{stats['total']} seizure hits "
                       f"({stats['seizure_hit']/stats['total']*100:.1f}%)\n")
        
        logger.info(f"Analysis report saved to {output_file}")

def main():
    """Example usage of ClusterAnalyzer."""
    # Configure paths
    cluster_results_dir = "madrid_results copy/tolerance_adjusted_smart_clustered"
    madrid_results_dir = "madrid_results copy/madrid_dir_400_examples_tolerance"
    
    # Initialize analyzer
    analyzer = ClusterAnalyzer(
        cluster_results_dir=cluster_results_dir,
        madrid_results_dir=madrid_results_dir
    )
    
    try:
        # Run analysis
        results = analyzer.load_and_process_data()
        
        # Save report
        analyzer.save_analysis_report("cluster_reduction/analysis_report.txt")
        
        # Print summary
        print("Cluster Analysis Summary:")
        print(f"Total clusters: {len(results['clusters'])}")
        print(f"Total anomalies: {len(results['anomalies'])}")
        print(f"Analysis metrics: {results['analysis']}")
        
        return analyzer
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    analyzer = main()