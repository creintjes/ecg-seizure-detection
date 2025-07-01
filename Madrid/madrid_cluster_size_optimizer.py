#!/usr/bin/env python3
"""
Madrid Cluster Size Optimizer

This script analyzes the clustering results and tests different minimum cluster size thresholds
to find the optimal balance between FP reduction and sensitivity preservation.
Similar methodology to madrid_smart_clustering.py but focused on cluster size filtering.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import statistics


class MadridClusterSizeOptimizer:
    """Optimizer for finding optimal minimum cluster size threshold."""
    
    def __init__(self, clustering_results_folder: str, output_folder: str = None):
        self.clustering_results_folder = Path(clustering_results_folder)
        self.output_folder = Path(output_folder or f"{clustering_results_folder}_size_optimized")
        self.output_folder.mkdir(exist_ok=True)
        
        # Create subfolders
        (self.output_folder / "size_analysis").mkdir(exist_ok=True)
        (self.output_folder / "filtered_results").mkdir(exist_ok=True)
        (self.output_folder / "optimization_summary").mkdir(exist_ok=True)
        
        self.representatives = []
        self.base_metrics = {}
        
    def load_clustering_results(self) -> None:
        """Load clustering results from best_representatives.json."""
        print(f"Loading clustering results from: {self.clustering_results_folder}")
        
        # Load representatives
        representatives_file = self.clustering_results_folder / "clusters" / "best_representatives.json"
        if not representatives_file.exists():
            raise FileNotFoundError(f"Representatives file not found: {representatives_file}")
            
        with open(representatives_file, 'r') as f:
            data = json.load(f)
            self.representatives = data['representatives']
            
        # Load base metrics (before clustering)
        metrics_file = self.clustering_results_folder / "metrics_before" / "metrics_summary.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                self.base_metrics = json.load(f)
        else:
            # Calculate base metrics from representatives
            self.base_metrics = self.calculate_base_metrics_from_representatives()
            
        print(f"Loaded {len(self.representatives)} representatives")
        
    def calculate_base_metrics_from_representatives(self) -> Dict[str, Any]:
        """Calculate base metrics from representatives (clustering results)."""
        true_positives = sum(1 for r in self.representatives if r.get('seizure_hit', False))
        false_positives = sum(1 for r in self.representatives if not r.get('seizure_hit', False))
        total_anomalies = len(self.representatives)
        
        # Count unique files
        unique_files = set(r['file_id'] for r in self.representatives)
        files_with_seizures = len(set(r['file_id'] for r in self.representatives if r.get('seizure_present', False)))
        files_with_detected_seizures = len(set(r['file_id'] for r in self.representatives 
                                            if r.get('seizure_hit', False)))
        
        sensitivity = files_with_detected_seizures / files_with_seizures if files_with_seizures > 0 else 0
        precision = true_positives / total_anomalies if total_anomalies > 0 else 0
        false_alarm_rate = false_positives / total_anomalies if total_anomalies > 0 else 0
        
        return {
            'total_files': len(unique_files),
            'files_with_seizures': files_with_seizures,
            'files_with_detected_seizures': files_with_detected_seizures,
            'total_anomalies': total_anomalies,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'sensitivity': sensitivity,
            'precision': precision,
            'false_alarm_rate': false_alarm_rate
        }
        
    def analyze_cluster_size_distribution(self) -> Dict[str, Any]:
        """Analyze the distribution of cluster sizes."""
        cluster_sizes = [r['cluster_size'] for r in self.representatives]
        tp_by_size = defaultdict(int)
        fp_by_size = defaultdict(int)
        count_by_size = defaultdict(int)
        
        for rep in self.representatives:
            size = rep['cluster_size']
            count_by_size[size] += 1
            if rep.get('seizure_hit', False):
                tp_by_size[size] += 1
            else:
                fp_by_size[size] += 1
                
        # Calculate statistics
        size_stats = {
            'min_size': min(cluster_sizes),
            'max_size': max(cluster_sizes),
            'mean_size': statistics.mean(cluster_sizes),
            'median_size': statistics.median(cluster_sizes),
            'std_size': statistics.stdev(cluster_sizes) if len(cluster_sizes) > 1 else 0,
            'unique_sizes': sorted(set(cluster_sizes))
        }
        
        # Create size distribution analysis
        size_analysis = {}
        for size in size_stats['unique_sizes']:
            size_analysis[size] = {
                'count': count_by_size[size],
                'tp_count': tp_by_size[size],
                'fp_count': fp_by_size[size],
                'tp_rate': tp_by_size[size] / count_by_size[size] if count_by_size[size] > 0 else 0,
                'fp_rate': fp_by_size[size] / count_by_size[size] if count_by_size[size] > 0 else 0
            }
            
        return {
            'statistics': size_stats,
            'size_analysis': size_analysis,
            'distribution': {
                'sizes': cluster_sizes,
                'tp_by_size': dict(tp_by_size),
                'fp_by_size': dict(fp_by_size),
                'count_by_size': dict(count_by_size)
            }
        }
        
    def filter_by_minimum_size(self, min_size: int) -> List[Dict]:
        """Filter representatives by minimum cluster size."""
        return [r for r in self.representatives if r['cluster_size'] >= min_size]
        
    def evaluate_size_threshold(self, filtered_representatives: List[Dict], min_size: int) -> Dict:
        """Evaluate metrics for a given minimum size threshold."""
        if not filtered_representatives:
            return {
                'min_size': min_size,
                'total_anomalies': 0,
                'true_positives': 0,
                'false_positives': 0,
                'sensitivity': 0.0,
                'precision': 0.0,
                'false_alarm_rate': 0.0,
                'anomaly_reduction': 1.0,
                'fp_reduction': 1.0,
                'sensitivity_change': -self.base_metrics['sensitivity'],
                'score': -10.0  # Very bad score for empty results
            }
            
        true_positives = sum(1 for r in filtered_representatives if r.get('seizure_hit', False))
        false_positives = sum(1 for r in filtered_representatives if not r.get('seizure_hit', False))
        total_anomalies = len(filtered_representatives)
        
        # Count files with detected seizures after filtering
        files_with_representatives = {}
        for rep in filtered_representatives:
            file_id = rep['file_id']
            if file_id not in files_with_representatives:
                files_with_representatives[file_id] = {'has_tp': False}
            if rep.get('seizure_hit', False):
                files_with_representatives[file_id]['has_tp'] = True
                
        files_with_detected_seizures = sum(1 for file_data in files_with_representatives.values() 
                                         if file_data['has_tp'])
        
        sensitivity = files_with_detected_seizures / self.base_metrics['files_with_seizures'] if self.base_metrics['files_with_seizures'] > 0 else 0
        precision = true_positives / total_anomalies if total_anomalies > 0 else 0
        false_alarm_rate = false_positives / total_anomalies if total_anomalies > 0 else 0
        
        # Calculate reduction metrics
        anomaly_reduction = (self.base_metrics['total_anomalies'] - total_anomalies) / self.base_metrics['total_anomalies'] if self.base_metrics['total_anomalies'] > 0 else 0
        fp_reduction = (self.base_metrics['false_positives'] - false_positives) / self.base_metrics['false_positives'] if self.base_metrics['false_positives'] > 0 else 0
        
        # Calculate score: prioritize maintaining sensitivity while reducing FPs
        sensitivity_penalty = max(0, self.base_metrics['sensitivity'] - sensitivity)
        score = (fp_reduction * 0.6) + (anomaly_reduction * 0.3) - (sensitivity_penalty * 2.0)
        
        return {
            'min_size': min_size,
            'total_anomalies': total_anomalies,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'sensitivity': sensitivity,
            'precision': precision,
            'false_alarm_rate': false_alarm_rate,
            'anomaly_reduction': anomaly_reduction,
            'fp_reduction': fp_reduction,
            'sensitivity_change': sensitivity - self.base_metrics['sensitivity'],
            'score': score,
            'files_with_detected_seizures': files_with_detected_seizures
        }
        
    def run_size_optimization(self) -> Dict[str, Any]:
        """Run size optimization analysis."""
        print("Starting cluster size optimization analysis...")
        
        self.load_clustering_results()
        if not self.representatives:
            raise ValueError("No representatives found")
            
        print(f"\nBase metrics (after clustering):")
        print(f"Total anomalies: {self.base_metrics['total_anomalies']}")
        print(f"True positives: {self.base_metrics['true_positives']}")
        print(f"False positives: {self.base_metrics['false_positives']}")
        print(f"Sensitivity: {self.base_metrics['sensitivity']:.3f}")
        print(f"Precision: {self.base_metrics['precision']:.3f}")
        print(f"False alarm rate: {self.base_metrics['false_alarm_rate']:.3f}")
        
        # Analyze cluster size distribution
        size_analysis = self.analyze_cluster_size_distribution()
        
        print(f"\nCluster size distribution:")
        print(f"Min size: {size_analysis['statistics']['min_size']}")
        print(f"Max size: {size_analysis['statistics']['max_size']}")
        print(f"Mean size: {size_analysis['statistics']['mean_size']:.1f}")
        print(f"Median size: {size_analysis['statistics']['median_size']:.1f}")
        print(f"Unique sizes: {len(size_analysis['statistics']['unique_sizes'])}")
        
        # Save size analysis
        with open(self.output_folder / "size_analysis" / "size_distribution.json", 'w') as f:
            json.dump(size_analysis, f, indent=2)
            
        # Test different minimum size thresholds
        size_thresholds = []
        unique_sizes = size_analysis['statistics']['unique_sizes']
        
        # Test sizes from 1 to reasonable upper bound
        max_test_size = min(50, max(unique_sizes))  # Don't go too high
        
        # Create a range of test sizes
        test_sizes = list(range(1, max_test_size + 1))
        
        # Add some specific percentile-based sizes
        cluster_sizes_list = [r['cluster_size'] for r in self.representatives]
        cluster_sizes_list.sort()
        n = len(cluster_sizes_list)
        
        percentiles = [25, 50, 75, 90, 95]
        for p in percentiles:
            index = int((p / 100) * (n - 1))
            size_p = cluster_sizes_list[index]
            if size_p not in test_sizes and size_p <= max_test_size:
                test_sizes.append(size_p)
                
        test_sizes = sorted(set(test_sizes))
        
        print(f"\nTesting {len(test_sizes)} different minimum size thresholds...")
        
        results = {}
        for min_size in test_sizes:
            filtered_reps = self.filter_by_minimum_size(min_size)
            metrics = self.evaluate_size_threshold(filtered_reps, min_size)
            results[min_size] = {
                'metrics': metrics,
                'representatives': filtered_reps
            }
            
            if min_size % 5 == 0 or min_size <= 10:  # Print progress for every 5th size or small sizes
                print(f"Min size {min_size:2d}: {metrics['total_anomalies']:4d} anomalies, "
                      f"sensitivity: {metrics['sensitivity']:.3f}, score: {metrics['score']:.3f}")
                      
        # Find best threshold
        best_min_size = max(results.keys(), key=lambda k: results[k]['metrics']['score'])
        best_result = results[best_min_size]
        
        print(f"\nBest minimum cluster size: {best_min_size}")
        print(f"Score: {best_result['metrics']['score']:.3f}")
        print(f"Anomalies: {best_result['metrics']['total_anomalies']}")
        print(f"Sensitivity: {best_result['metrics']['sensitivity']:.3f}")
        print(f"Precision: {best_result['metrics']['precision']:.3f}")
        
        # Save results
        self.save_optimization_results(size_analysis, results, best_min_size)
        
        return {
            'size_analysis': size_analysis,
            'threshold_results': results,
            'best_min_size': best_min_size,
            'best_metrics': best_result['metrics']
        }
        
    def save_optimization_results(self, size_analysis: Dict, results: Dict, best_min_size: int):
        """Save all optimization results."""
        
        # Save detailed results for all thresholds
        threshold_results = {
            'base_metrics': self.base_metrics,
            'size_analysis': size_analysis,
            'tested_thresholds': len(results),
            'best_min_size': best_min_size,
            'threshold_results': {
                str(min_size): data['metrics'] for min_size, data in results.items()
            }
        }
        
        with open(self.output_folder / "optimization_summary" / "threshold_comparison.json", 'w') as f:
            json.dump(threshold_results, f, indent=2)
            
        # Save best filtered representatives
        best_representatives = results[best_min_size]['representatives']
        best_output = {
            'optimization_method': 'minimum_cluster_size',
            'best_min_size': best_min_size,
            'timestamp': datetime.now().isoformat(),
            'total_representatives': len(best_representatives),
            'original_representatives': len(self.representatives),
            'representatives': best_representatives
        }
        
        with open(self.output_folder / "filtered_results" / "best_size_filtered.json", 'w') as f:
            json.dump(best_output, f, indent=2)
            
        # Save optimization summary
        best_metrics = results[best_min_size]['metrics']
        summary = {
            'optimization_method': 'minimum_cluster_size',
            'best_min_size': best_min_size,
            'before_filtering': self.base_metrics,
            'after_filtering': best_metrics,
            'improvements': {
                'anomaly_reduction_pct': best_metrics['anomaly_reduction'] * 100,
                'fp_reduction_pct': best_metrics['fp_reduction'] * 100,
                'sensitivity_change': best_metrics['sensitivity_change'],
                'precision_improvement': best_metrics['precision'] - self.base_metrics['precision'],
                'far_reduction': self.base_metrics['false_alarm_rate'] - best_metrics['false_alarm_rate']
            },
            'size_distribution': size_analysis['statistics']
        }
        
        with open(self.output_folder / "optimization_summary" / "final_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Print final summary
        print("\n" + "="*70)
        print("CLUSTER SIZE OPTIMIZATION SUMMARY")
        print("="*70)
        print(f"Best minimum cluster size: {best_min_size}")
        print(f"Representatives: {len(self.representatives)} → {best_metrics['total_anomalies']} "
              f"({best_metrics['anomaly_reduction']*100:.1f}% reduction)")
        print(f"False positives: {self.base_metrics['false_positives']} → {best_metrics['false_positives']} "
              f"({best_metrics['fp_reduction']*100:.1f}% reduction)")
        print(f"Sensitivity: {self.base_metrics['sensitivity']:.3f} → {best_metrics['sensitivity']:.3f} "
              f"({best_metrics['sensitivity_change']:+.3f})")
        print(f"Precision: {self.base_metrics['precision']:.3f} → {best_metrics['precision']:.3f} "
              f"({best_metrics['precision'] - self.base_metrics['precision']:+.3f})")
        print(f"False alarm rate: {self.base_metrics['false_alarm_rate']:.3f} → {best_metrics['false_alarm_rate']:.3f} "
              f"({self.base_metrics['false_alarm_rate'] - best_metrics['false_alarm_rate']:+.3f})")
        
        # Print size distribution insight
        print(f"\nCluster size insights:")
        print(f"Mean cluster size: {size_analysis['statistics']['mean_size']:.1f}")
        print(f"Median cluster size: {size_analysis['statistics']['median_size']:.1f}")
        print(f"Clusters >= {best_min_size}: {len(best_representatives)} / {len(self.representatives)}")
        
        print("="*70)
        print(f"Results saved to: {self.output_folder}")


def main():
    """Main function."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 madrid_cluster_size_optimizer.py <clustering_results_folder>")
        print("Example: python3 madrid_cluster_size_optimizer.py madrid_results_smart_clustered")
        sys.exit(1)
        
    input_folder = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else None
    
    optimizer = MadridClusterSizeOptimizer(input_folder, output_folder)
    
    try:
        results = optimizer.run_size_optimization()
        print("\nCluster size optimization completed successfully!")
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        raise


if __name__ == "__main__":
    main()