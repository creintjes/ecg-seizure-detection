#!/usr/bin/env python3
"""
Smart clustering for AD time series (TimeVQVAE-AD friendly)

This version accepts your model outputs directly:
- boolean detection mask (True = anomaly)
- samplerate (Hz)
- optional per-sample anomaly scores
- optional ground-truth seizure intervals for metrics

It then builds anomaly events, runs the same clustering strategies as the
original Madrid script, chooses the best one, and returns/saves:
- metrics_before/metrics_after
- strategy_comparison
- clusters/best_representatives.json
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict


def _contiguous_true_segments(mask: List[bool]) -> List[Tuple[int, int]]:
    """Return [(start_idx, end_idx_inclusive), ...] for contiguous True runs."""
    segs = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            start = i
            i += 1
            while i < n and mask[i]:
                i += 1
            segs.append((start, i - 1))
        else:
            i += 1
    return segs


def _overlaps(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    """Whether intervals [a0, a1] and [b0, b1] overlap (>0)."""
    return max(a[0], b[0]) < min(a[1], b[1])


class SmartTimeSeriesClusteringAnalyzer:
    """Smart analyzer operating on anomalies derived from boolean masks & fs."""

    def __init__(self, anomalies: List[Dict[str, Any]], output_folder: Optional[str] = None,
                 gt_intervals: Optional[List[Tuple[float, float]]] = None,
                 sampling_rate: float = 8.0, total_samples: int = 0, 
                 true_anomaly_samples: Optional[int] = None,
                 true_anomaly_events: Optional[int] = None,
                 fixed_strategy: Optional[str] = None):
        """
        anomalies: flat list of dicts with at least:
          - file_id: str
          - subject_id: str
          - location_time_seconds: float (representative time of anomaly)
          - anomaly_score: float
          - seizure_hit: bool (optional, False if unknown)
          - seizure_present: bool (optional, False if unknown)
        gt_intervals: List of seizure ground truth intervals (start_sec, end_sec)
        sampling_rate: Sampling rate in Hz for event-based metrics calculation
        total_samples: Total number of samples in the recording for FAR calculation
        true_anomaly_samples: Number of true anomaly samples in ground truth (e.g., 2568)
        true_anomaly_events: Number of true anomaly events in ground truth (e.g., 4)
        fixed_strategy: If provided, only run this specific clustering strategy (for fixed-strategy evaluation)
        """
        self.all_anomalies = anomalies[:]  # already per-file
        self.output_folder = Path(output_folder or "smart_clustered_results")
        self.gt_intervals = gt_intervals or []
        self.sampling_rate = sampling_rate
        self.total_samples = total_samples
        self.true_anomaly_samples = true_anomaly_samples  # NEW: GT true anomaly sample count
        self.true_anomaly_events = true_anomaly_events    # NEW: GT true event count
        self.fixed_strategy = fixed_strategy
        
        # Calculate ground truth percentage if provided
        self.true_anomaly_percentage = (true_anomaly_samples / total_samples 
                                       if true_anomaly_samples and total_samples > 0 else None)
        
        if output_folder is not None:  # Only create folders if explicitly requested
            self.output_folder.mkdir(exist_ok=True)
            (self.output_folder / "metrics_before").mkdir(exist_ok=True)
            (self.output_folder / "clusters").mkdir(exist_ok=True)
            (self.output_folder / "metrics_after").mkdir(exist_ok=True)
            (self.output_folder / "strategy_comparison").mkdir(exist_ok=True)

    # ---- metrics & clustering helpers ----
    def calculate_base_metrics(self) -> Dict[str, Any]:
        """Calculate base metrics including event-based metrics for all anomalies."""
        from metrics import event_metrics_for_clusters  # Use separate metrics module
        
        true_positives = sum(1 for a in self.all_anomalies if a.get('seizure_hit', False))
        false_positives = sum(1 for a in self.all_anomalies if not a.get('seizure_hit', False))
        total_anomalies = len(self.all_anomalies)
        
        # Calculate event-based metrics for all anomalies
        event_metrics = event_metrics_for_clusters(
            self.all_anomalies, self.gt_intervals, self.total_samples, self.sampling_rate
        )

        return {
            'total_anomalies': total_anomalies,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'event_recall': event_metrics.get('event_recall', 0.0),
            'event_iou': event_metrics.get('event_iou', 0.0),
            'event_far_per_hour': event_metrics.get('event_far_per_hour', 0.0)
        }

    def time_based_clustering(self, time_threshold: float) -> List[List[Dict]]:
        """
        Group anomalies that occur within time_threshold seconds of each other.
        This is the core clustering method for seizure detection.
        """
        all_clusters = []
        file_groups = defaultdict(list)
        for anomaly in self.all_anomalies:
            file_groups[anomaly['file_id']].append(anomaly)
        for file_id, file_anomalies in file_groups.items():
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
            if current_cluster:
                all_clusters.append(current_cluster)
        return all_clusters

    def select_smart_representatives(self, clusters: List[List[Dict]]) -> List[Dict]:
        representatives = []
        for i, cluster in enumerate(clusters):
            if not cluster:
                continue
                
            # Calculate cluster time span
            all_start_times = [a.get('segment_start_seconds', a['location_time_seconds']) for a in cluster]
            all_end_times = [a.get('segment_end_seconds', a['location_time_seconds']) for a in cluster]
            cluster_start = min(all_start_times)
            cluster_end = max(all_end_times)
            
            if len(cluster) == 1:
                representative = cluster[0].copy()
                avgd = 0.0
            else:
                min_avg_distance = float('inf')
                best = None
                for candidate in cluster:
                    ct = candidate['location_time_seconds']
                    total = sum(abs(ct - other['location_time_seconds']) for other in cluster if other is not candidate)
                    avg = total / (len(cluster) - 1)
                    if avg < min_avg_distance:
                        min_avg_distance = avg
                        best = candidate
                representative = best.copy()
                avgd = min_avg_distance
                
            true_positives = [a for a in cluster if a.get('seizure_hit', False)]
            
            # Add cluster span information (critical fix)
            representative['cluster_start_seconds'] = cluster_start
            representative['cluster_end_seconds'] = cluster_end
            representative['cluster_duration_seconds'] = cluster_end - cluster_start
            representative['cluster_id'] = i
            representative['cluster_size'] = len(cluster)
            representative['cluster_tp_count'] = len(true_positives)
            representative['avg_time_distance'] = avgd
            representatives.append(representative)
        return representatives

    def evaluate_clustering_strategy(self, representatives: List[Dict], base_metrics: Dict) -> Dict:
        """Evaluate clustering strategy with ground-truth informed sanity checks."""
        from metrics import event_metrics_for_clusters  # Use separate metrics module
        
        # Calculate event-based metrics for the representatives
        event_metrics = event_metrics_for_clusters(
            representatives, self.gt_intervals, self.total_samples, self.sampling_rate
        )
        
        # Basic counting metrics
        total_anomalies = len(representatives)
        true_positives = sum(1 for r in representatives if r.get('seizure_hit', False))
        false_positives = total_anomalies - true_positives
        
        # Calculate reductions compared to base metrics
        anomaly_reduction = ((base_metrics['total_anomalies'] - total_anomalies) / base_metrics['total_anomalies']
                             if base_metrics['total_anomalies'] > 0 else 0.0)
        fp_reduction = ((base_metrics['false_positives'] - false_positives) / base_metrics['false_positives']
                        if base_metrics['false_positives'] > 0 else 0.0)
        
        # Use Madrid's scoring function: prioritize FP reduction while maintaining sensitivity
        base_sensitivity = base_metrics.get('event_recall', 0.0)
        new_sensitivity = event_metrics.get('event_recall', 0.0)
        sensitivity_penalty = max(0.0, base_sensitivity - new_sensitivity)
        
        # ENHANCED SANITY CHECKS using ground truth statistics
        over_clustering_penalty = 0.0
        
        # 1. Ground truth event-based penalty
        if self.true_anomaly_events:
            # We should have at least as many clusters as true events
            min_reasonable_clusters = self.true_anomaly_events
            max_reasonable_clusters = self.true_anomaly_events * 5  # Allow up to 5x true events
            
            if total_anomalies < min_reasonable_clusters:
                # Heavy penalty for having fewer clusters than true events
                over_clustering_penalty += 4.0
                # print(f"DEBUG: Heavy penalty - {total_anomalies} clusters < {min_reasonable_clusters} true events")
            elif total_anomalies > max_reasonable_clusters:
                # Light penalty for too many clusters (under-clustering)
                over_clustering_penalty += 0.5
        
        # 2. Ground truth sample-based penalty (adaptive to actual GT percentage)
        if self.true_anomaly_samples and self.true_anomaly_percentage:
            # Check if we're creating unreasonably large clusters compared to true anomaly density
            
            # Calculate average cluster span in samples
            if representatives:
                total_cluster_span_seconds = 0
                for rep in representatives:
                    cluster_start = rep.get('cluster_start_seconds', rep['location_time_seconds'])
                    cluster_end = rep.get('cluster_end_seconds', rep['location_time_seconds'])
                    total_cluster_span_seconds += (cluster_end - cluster_start)
                
                avg_cluster_span_samples = (total_cluster_span_seconds / len(representatives)) * self.sampling_rate
                
                # Adaptive penalty based on actual GT anomaly percentage
                # For very sparse anomalies (< 1%), be strict about cluster size
                # For denser anomalies (> 5%), be more lenient
                if self.true_anomaly_percentage < 0.01:  # Less than 1% true anomalies
                    max_reasonable_cluster_samples = self.true_anomaly_samples * 10  # Allow 10x inflation
                    penalty_strength = 2.0
                elif self.true_anomaly_percentage < 0.05:  # 1-5% true anomalies
                    max_reasonable_cluster_samples = self.true_anomaly_samples * 5   # Allow 5x inflation
                    penalty_strength = 1.5
                else:  # > 5% true anomalies
                    max_reasonable_cluster_samples = self.true_anomaly_samples * 3   # Allow 3x inflation
                    penalty_strength = 1.0
                
                if avg_cluster_span_samples > max_reasonable_cluster_samples:
                    penalty_factor = min(penalty_strength, avg_cluster_span_samples / max_reasonable_cluster_samples)
                    over_clustering_penalty += penalty_factor
                    # print(f"DEBUG: Cluster span penalty - avg span {avg_cluster_span_samples:.0f} samples > max {max_reasonable_cluster_samples:.0f} "
                    #       f"(GT: {self.true_anomaly_percentage*100:.3f}%, penalty: {penalty_factor:.2f})")
        
        # 3. "Giant cluster" detection
        if total_anomalies == 1 and base_metrics['total_anomalies'] > 100:
            # Single cluster from many anomalies is almost always wrong
            over_clustering_penalty += 5.0
            #print(f"DEBUG: Giant cluster penalty - 1 cluster from {base_metrics['total_anomalies']} anomalies")
        
        # 4. Coverage expansion penalty (CRITICAL: prevent clustering from expanding coverage)
        if representatives:
            # Calculate total clustered coverage in samples
            total_clustered_duration = 0.0
            for rep in representatives:
                cluster_start = rep.get('cluster_start_seconds', rep['location_time_seconds'])
                cluster_end = rep.get('cluster_end_seconds', rep['location_time_seconds'])
                total_clustered_duration += (cluster_end - cluster_start)
            
            clustered_samples = total_clustered_duration * self.sampling_rate
            original_predicted_samples = base_metrics['total_anomalies']  # This approximates original coverage
            
            # Strong penalty if clustering expands coverage beyond original predictions
            if clustered_samples > original_predicted_samples * 2:  # Allow 2x expansion max
                coverage_expansion_ratio = clustered_samples / original_predicted_samples
                # Exponential penalty for coverage expansion
                expansion_penalty = max(20.0, (coverage_expansion_ratio - 2.0) * 0.1)
                over_clustering_penalty += expansion_penalty
                # print(f"DEBUG: Coverage expansion penalty - clustered {clustered_samples:.0f} vs predicted {original_predicted_samples} "
                #       f"(ratio: {coverage_expansion_ratio:.1f}x, penalty: {expansion_penalty:.1f})")
        
        # 5. Extreme reduction penalty (more aggressive)
        if anomaly_reduction > 0.99:  # >99% reduction
            over_clustering_penalty += 3.0
            # print(f"DEBUG: Extreme reduction penalty - {anomaly_reduction*100:.1f}% reduction")
        elif anomaly_reduction > 0.95:  # >95% reduction
            over_clustering_penalty += 1.5


        
        # Madrid's proven scoring function with enhanced over-clustering penalty
        score = 0.6 * fp_reduction + 0.3 * anomaly_reduction - 2.0 * sensitivity_penalty - over_clustering_penalty
        
        return {
            'total_anomalies': total_anomalies,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'anomaly_reduction': anomaly_reduction,
            'fp_reduction': fp_reduction,
            'event_recall': event_metrics.get('event_recall', 0.0),
            'event_iou': event_metrics.get('event_iou', 0.0),
            'event_far_per_hour': event_metrics.get('event_far_per_hour', 0.0),
            'event_recall_change': new_sensitivity - base_sensitivity,
            'sensitivity_penalty': sensitivity_penalty,
            'over_clustering_penalty': over_clustering_penalty,
            'score': score
        }

    def run_fixed_strategy_analysis(self, strategy_name: str, strategy_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run clustering analysis with a single fixed strategy.
        Used for the second evaluation run with predetermined optimal strategies.
        
        Args:
            strategy_name: Name of the clustering strategy to use
            strategy_params: Parameters for the clustering strategy
        
        Returns:
            Dictionary with base metrics, strategy results, and final metrics
        """
        if not self.all_anomalies:
            raise ValueError("No anomalies found")

        base_metrics = self.calculate_base_metrics()
        
        # Apply the specified clustering strategy
        clusters, representatives, strategy_metrics = self._apply_strategy(strategy_name, strategy_params, base_metrics)
        
        # Save base metrics only if output folder was explicitly provided
        if self.output_folder and self.output_folder.name != "smart_clustered_results":
            with open(self.output_folder / "metrics_before" / "metrics_summary.json", 'w') as f:
                json.dump(base_metrics, f, indent=2)
            
            # Save fixed strategy results
            self._save_fixed_strategy_results(base_metrics, strategy_name, strategy_params, 
                                            clusters, representatives, strategy_metrics)
        
        return {
            'base_metrics': base_metrics,
            'strategy_used': strategy_name,
            'strategy_parameters': strategy_params,
            'clusters': clusters,
            'representatives': representatives,
            'final_metrics': strategy_metrics,
            'n_clusters': len(clusters),
            'n_representatives': len(representatives)
        }
    
    def _apply_strategy(self, strategy_name: str, strategy_params: Dict[str, Any], base_metrics: Dict) -> Tuple[List, List, Dict]:
        """
        Apply a specific clustering strategy based on the strategy name and parameters.
        Only supports time-based strategies for seizure detection.
        
        Returns:
            Tuple of (clusters, representatives, metrics)
        """
        # Parse strategy name and apply appropriate method
        if strategy_name.startswith('time_') and strategy_name.endswith('s'):
            # Extract threshold from strategy name like 'time_60s'
            threshold = strategy_params.get('threshold', int(strategy_name.split('_')[1][:-1]))
            clusters = self.time_based_clustering(threshold)
            representatives = self.select_smart_representatives(clusters)
            
        elif strategy_name.startswith('fine_time_') and strategy_name.endswith('s'):
            # Extract threshold from strategy name like 'fine_time_30s'
            threshold = strategy_params.get('threshold', int(strategy_name.split('_')[2][:-1]))
            clusters = self.time_based_clustering(threshold)
            representatives = self.select_smart_representatives(clusters)
            
        elif strategy_name.startswith('extended_time_') and strategy_name.endswith('s'):
            # Extract threshold from strategy name like 'extended_time_300s'
            threshold = strategy_params.get('threshold', int(strategy_name.split('_')[2][:-1]))
            clusters = self.time_based_clustering(threshold)
            representatives = self.select_smart_representatives(clusters)
            
        else:
            raise ValueError(f"Unsupported clustering strategy for seizure detection: {strategy_name}. "
                           f"Only time-based strategies are supported: 'time_*', 'fine_time_*', 'extended_time_*'")
        
        # Evaluate the strategy
        metrics = self.evaluate_clustering_strategy(representatives, base_metrics)
        
        return clusters, representatives, metrics
    
    def _save_fixed_strategy_results(self, base_metrics: Dict, strategy_name: str, strategy_params: Dict,
                                   clusters: List, representatives: List, final_metrics: Dict):
        """Save results from fixed strategy analysis."""
        
        # Save final metrics
        with open(self.output_folder / "metrics_after" / "metrics_summary.json", 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        # Save cluster information
        cluster_info = {
            'strategy_used': strategy_name,
            'strategy_parameters': strategy_params,
            'total_clusters': len(clusters),
            'total_representatives': len(representatives),
            'clusters': []
        }
        
        for i, cluster in enumerate(clusters):
            cluster_info['clusters'].append({
                'cluster_id': i,
                'size': len(cluster),
                'anomaly_locations': [a['location_time_seconds'] for a in cluster],
                'representative_time': representatives[i]['location_time_seconds'] if i < len(representatives) else None
            })
        
        with open(self.output_folder / "clusters" / "cluster_info.json", 'w') as f:
            json.dump(cluster_info, f, indent=2)
        
        # Save representatives
        with open(self.output_folder / "clusters" / "best_representatives.json", 'w') as f:
            json.dump(representatives, f, indent=2, default=str)
        
        # Save strategy comparison (single strategy in this case)
        comparison_data = {
            'strategy_name': strategy_name,
            'base_metrics': base_metrics,
            'final_metrics': final_metrics,
            'improvement': {
                'anomaly_reduction': final_metrics.get('anomaly_reduction', 0.0),
                'fp_reduction': final_metrics.get('fp_reduction', 0.0),
                'event_recall_change': final_metrics.get('event_recall_change', 0.0)
            }
        }
        
        with open(self.output_folder / "strategy_comparison" / "fixed_strategy_results.json", 'w') as f:
            json.dump(comparison_data, f, indent=2)

    def run_smart_analysis(self) -> Dict[str, Any]:
        if not self.all_anomalies:
            raise ValueError("No anomalies found")

        base_metrics = self.calculate_base_metrics()

        # Save base metrics only if output folder was explicitly provided
        if self.output_folder and self.output_folder.name != "smart_clustered_results":
            with open(self.output_folder / "metrics_before" / "metrics_summary.json", 'w') as f:
                json.dump(base_metrics, f, indent=2)

        strategies = {}

        # Use Madrid's proven time thresholds (more conservative)
        time_thresholds = [2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 300, 420, 600, 900]
        
        for th in time_thresholds:
            clusters = self.time_based_clustering(th)
            reps = self.select_smart_representatives(clusters)
            m = self.evaluate_clustering_strategy(reps, base_metrics)
            strategies[f'time_{th}s'] = {
                'method': 'time_based',
                'parameters': {'threshold': th},
                'clusters': len(clusters),
                'representatives': reps,
                'metrics': m
            }

        best_strategy_name = max(strategies.keys(), key=lambda k: strategies[k]['metrics']['score'])
        best_strategy = strategies[best_strategy_name]

        # Only save results if output folder was explicitly requested
        if self.output_folder and self.output_folder.name != "smart_clustered_results":  # Only save if custom folder was provided
            self.save_smart_results(self.calculate_base_metrics(), strategies, best_strategy_name)

        return {
            'base_metrics': self.calculate_base_metrics(),
            'strategies': strategies,
            'best_strategy': best_strategy_name,
            'best_results': best_strategy
        }

    def save_smart_results(self, base_metrics: Dict, strategies: Dict, best_strategy_name: str):
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

        best_strategy = strategies[best_strategy_name]
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

        # Use the correct metric keys that actually exist
        final_metrics = {
            'strategy': best_strategy_name,
            'before': base_metrics,
            'after': best_strategy['metrics'],
            'improvements': {
                'anomaly_reduction_pct': best_strategy['metrics']['anomaly_reduction'] * 100,
                'fp_reduction_pct': best_strategy['metrics']['fp_reduction'] * 100,
                'event_recall_change': best_strategy['metrics']['event_recall_change'],
                'event_iou_improvement': best_strategy['metrics']['event_iou'] - base_metrics['event_iou'],
                'event_far_change': best_strategy['metrics']['event_far_per_hour'] - base_metrics['event_far_per_hour']
            }
        }
        with open(self.output_folder / "metrics_after" / "final_metrics.json", 'w') as f:
            json.dump(final_metrics, f, indent=2)

        print("\n" + "="*60)
        print("SMART CLUSTERING ANALYSIS SUMMARY")
        print("="*60)
        print(f"Best strategy: {best_strategy_name}")
        print(f"Anomalies reduced: {len(self.all_anomalies)} → {best_strategy['metrics']['total_anomalies']} "
              f"({best_strategy['metrics']['anomaly_reduction']*100:.1f}% reduction)")
        print(f"False positives: {base_metrics['false_positives']} → {best_strategy['metrics']['false_positives']} "
              f"({best_strategy['metrics']['fp_reduction']*100:.1f}% reduction)")
        print(f"Event Recall: {base_metrics['event_recall']:.3f} → {best_strategy['metrics']['event_recall']:.3f} "
              f"({best_strategy['metrics']['event_recall_change']:+.3f})")
        print(f"Event IoU: {base_metrics['event_iou']:.3f} → {best_strategy['metrics']['event_iou']:.3f} "
              f"({best_strategy['metrics']['event_iou'] - base_metrics['event_iou']:+.3f})")
        print(f"Event FAR/hr: {base_metrics['event_far_per_hour']:.2f} → {best_strategy['metrics']['event_far_per_hour']:.2f} "
              f"({best_strategy['metrics']['event_far_per_hour'] - base_metrics['event_far_per_hour']:+.2f})")
        print("="*60)
        print(f"Results saved to: {self.output_folder}")


def build_anomalies_from_mask(
    mask: List[bool],
    fs: float,
    file_id: str = "sample_0",
    subject_id: str = "subject_0",
    scores: Optional[List[float]] = None,
    gt_intervals: Optional[List[Tuple[float, float]]] = None,
) -> List[Dict[str, Any]]:
    """
    Convert boolean mask (+ optional scores) into anomaly points compatible with the analyzer.
    Each contiguous True run becomes one anomaly at its onset time.
    """
    segs = _contiguous_true_segments(list(map(bool, mask)))
    anomalies = []
    seizure_present = bool(gt_intervals)  # file-level flag if GT provided

    for (s, e) in segs:
        start_sec = s / fs
        end_sec = (e + 1) / fs
        duration_sec = end_sec - start_sec

        score = 1.0
        if scores is not None and len(scores) == len(mask):
            # take max score within the segment if available
            score = float(max(scores[s:e+1])) if e >= s else float(scores[s])

        hit = False
        if gt_intervals:
            for gt in gt_intervals:
                if _overlaps((start_sec, end_sec), gt):
                    hit = True
                    break

        anomalies.append({
            'file_id': file_id,
            'subject_id': subject_id,
            'location_time_seconds': float(start_sec),  # representative time = onset
            'segment_start_seconds': float(start_sec),
            'segment_end_seconds': float(end_sec),
            'segment_duration_seconds': float(duration_sec),
            'anomaly_score': score,
            'seizure_hit': bool(hit),
            'seizure_present': seizure_present
        })

    return anomalies


def analyze_detection_mask(
    mask,
    fs: float,
    file_id: str = "sample_0",
    scores: Optional[List[float]] = None,
    subject_id: str = "subject_0",
    gt_intervals: Optional[List[Tuple[float, float]]] = None,
    output_folder: Optional[str] = None,
    fixed_strategy: Optional[str] = None,
    true_anomaly_samples: Optional[int] = None,  # NEW
    true_anomaly_events: Optional[int] = None    # NEW
) -> Dict[str, Any]:
    """
    Convenience wrapper for a single recording.
    Returns the same structure as the original smart analysis (dict with base_metrics, strategies, best_*).
    
    Args:
        mask: Boolean detection mask
        fs: Sampling rate in Hz
        file_id: Identifier for the file
        scores: Optional anomaly scores corresponding to mask
        subject_id: Identifier for the subject
        gt_intervals: Ground truth seizure intervals
        output_folder: Output folder for results
        fixed_strategy: If provided, use this specific strategy instead of testing all strategies
        true_anomaly_samples: Number of true anomaly samples in ground truth (e.g., 2568)
        true_anomaly_events: Number of true anomaly events in ground truth (e.g., 4)
    """
    anomalies = build_anomalies_from_mask(
        mask=list(mask), fs=fs, file_id=file_id, subject_id=subject_id,
        scores=list(scores) if scores is not None else None,
        gt_intervals=gt_intervals
    )
    analyzer = SmartTimeSeriesClusteringAnalyzer(
        anomalies, 
        output_folder=output_folder,
        gt_intervals=gt_intervals,
        sampling_rate=fs,
        total_samples=len(mask),
        true_anomaly_samples=true_anomaly_samples,  # NEW
        true_anomaly_events=true_anomaly_events,    # NEW
        fixed_strategy=fixed_strategy
    )
    
    if fixed_strategy:
        # Parse strategy name and parameters
        strategy_params = {}
        
        # Extract parameters from strategy name with error handling
        try:
            if fixed_strategy.startswith('time_') and fixed_strategy.endswith('s'):
                # Extract threshold from strategy name like 'time_60s'
                threshold = int(fixed_strategy.split('_')[1][:-1])
                strategy_params['threshold'] = threshold
            elif fixed_strategy.startswith('fine_time_') and fixed_strategy.endswith('s'):
                # Extract threshold from strategy name like 'fine_time_30s'
                threshold = int(fixed_strategy.split('_')[2][:-1])
                strategy_params['threshold'] = threshold
            elif fixed_strategy.startswith('extended_time_') and fixed_strategy.endswith('s'):
                # Extract threshold from strategy name like 'extended_time_300s'
                threshold = int(fixed_strategy.split('_')[2][:-1])
                strategy_params['threshold'] = threshold
            else:
                raise ValueError(f"Unsupported fixed strategy format: {fixed_strategy}. "
                               f"Only time-based strategies are supported: 'time_*s', 'fine_time_*s', 'extended_time_*s'")
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid fixed strategy name '{fixed_strategy}': {str(e)}. "
                           f"Expected format like 'time_60s', 'fine_time_30s', 'extended_time_300s'.")
        
        # Run fixed strategy analysis
        result = analyzer.run_fixed_strategy_analysis(fixed_strategy, strategy_params)
        
        # Convert to expected format for backward compatibility
        strategy_method = 'time_based'
        if fixed_strategy.startswith('fine_time_'):
            strategy_method = 'fine_time_based'
        elif fixed_strategy.startswith('extended_time_'):
            strategy_method = 'extended_time_based'
            
        return {
            'base_metrics': result['base_metrics'],
            'strategies': {fixed_strategy: {
                'method': strategy_method,
                'parameters': result['strategy_parameters'],
                'clusters': result['n_clusters'],
                'representatives': result['representatives'],
                'metrics': result['final_metrics']
            }},
            'best_strategy': fixed_strategy,
            'best_results': {
                'clusters': result['n_clusters'],
                'representatives': result['representatives'],
                'metrics': result['final_metrics']
            }
        }
    else:
        # Run normal adaptive analysis
        return analyzer.run_smart_analysis()


def analyze_multiple_records(
    records: List[Dict[str, Any]],
    output_folder: Optional[str] = None,
    true_anomaly_samples: Optional[int] = None,  # NEW
    true_anomaly_events: Optional[int] = None    # NEW
) -> Dict[str, Any]:
    """
    Analyze multiple recordings at once.

    records: list of dicts, each with:
      - mask: List[bool]
      - fs: float
      - file_id: str
      - subject_id: str (optional)
      - scores: List[float] (optional)
      - gt_intervals: List[(start_sec, end_sec)] (optional)
    true_anomaly_samples: Total number of true anomaly samples across all records
    true_anomaly_events: Total number of true anomaly events across all records
    """
    all_anomalies = []
    for r in records:
        anomalies = build_anomalies_from_mask(
            mask=list(r['mask']),
            fs=float(r['fs']),
            file_id=str(r.get('file_id', 'sample_0')),
            subject_id=str(r.get('subject_id', 'subject_0')),
            scores=list(r['scores']) if r.get('scores') is not None else None,
            gt_intervals=r.get('gt_intervals')
        )
        all_anomalies.extend(anomalies)

    # For multiple records, collect all GT intervals and calculate total samples
    all_gt_intervals = []
    total_samples = 0
    sampling_rate = records[0]['fs'] if records else 8.0  # Use first record's fs
    
    for r in records:
        total_samples += len(r['mask'])
        if r.get('gt_intervals'):
            all_gt_intervals.extend(r['gt_intervals'])

    analyzer = SmartTimeSeriesClusteringAnalyzer(
        all_anomalies, 
        output_folder=output_folder,
        gt_intervals=all_gt_intervals,
        sampling_rate=sampling_rate,
        total_samples=total_samples,
        true_anomaly_samples=true_anomaly_samples,  # NEW
        true_anomaly_events=true_anomaly_events     # NEW
    )
    return analyzer.run_smart_analysis()


def main():
    """
    Example CLI:
    python3 smart_timeseries_clustering.py

    (This demo builds a fake mask; for real use, import analyze_detection_mask/analyze_multiple_records.)
    """
    # Minimal demo (no GT -> metrics will reflect that)
    fs = 256.0
    n = 10_000
    mask = [False]*n
    # fake anomalies
    for i in range(2000, 2100): mask[i] = True
    for i in range(5000, 5040): mask[i] = True
    # optional scores (same length)
    scores = [0.0]*n
    for i in range(2000, 2100): scores[i] = 0.7
    for i in range(5000, 5040): scores[i] = 0.9

    results = analyze_detection_mask(
        mask=mask,
        fs=fs,
        file_id="demo_rec",
        subject_id="demo_subj",
        scores=scores,
        gt_intervals=[(7.8, 8.7)],  # optional
        output_folder="demo_rec_smart_clustered"
    )
    print("\nSmart analysis completed.\nBest strategy:", results["best_strategy"])


if __name__ == "__main__":
    main()