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
                 sampling_rate: float = 8.0, total_samples: int = 0, fixed_strategy: Optional[str] = None):
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
        fixed_strategy: If provided, only run this specific clustering strategy (for fixed-strategy evaluation)
        """
        self.all_anomalies = anomalies[:]  # already per-file
        self.output_folder = Path(output_folder or "smart_clustered_results")
        self.gt_intervals = gt_intervals or []
        self.sampling_rate = sampling_rate
        self.total_samples = total_samples
        self.fixed_strategy = fixed_strategy  # New parameter for fixed strategy mode
        
        if output_folder is not None:  # Only create folders if explicitly requested
            self.output_folder.mkdir(exist_ok=True)
            (self.output_folder / "metrics_before").mkdir(exist_ok=True)
            (self.output_folder / "clusters").mkdir(exist_ok=True)
            (self.output_folder / "metrics_after").mkdir(exist_ok=True)
            (self.output_folder / "strategy_comparison").mkdir(exist_ok=True)

    # ---- metrics & clustering helpers ----
    def calculate_base_metrics(self) -> Dict[str, Any]:
        """Calculate base metrics including event-based metrics for all anomalies."""
        from eval import event_metrics_for_clusters  # Avoid circular import
        
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

    def score_based_clustering(self, time_threshold: float = 30.0) -> List[List[Dict]]:
        return self.time_based_clustering(time_threshold)

    def file_based_clustering(self, time_threshold: float = 30.0) -> List[List[Dict]]:
        return self.time_based_clustering(time_threshold)

    def hybrid_clustering(self, time_threshold: float = 45.0) -> List[List[Dict]]:
        return self.time_based_clustering(time_threshold)

    def score_aware_clustering(self, time_threshold: float = 60.0) -> List[List[Dict]]:
        all_clusters = []
        file_groups = defaultdict(list)
        for anomaly in self.all_anomalies:
            file_groups[anomaly['file_id']].append(anomaly)
        for file_id, file_anomalies in file_groups.items():
            sorted_anomalies = sorted(file_anomalies,
                                      key=lambda x: (-x.get('anomaly_score', 0.0),
                                                     x['location_time_seconds']))
            current_cluster = []
            for anomaly in sorted_anomalies:
                if not current_cluster:
                    current_cluster = [anomaly]
                else:
                    min_time_diff = min(abs(anomaly['location_time_seconds'] -
                                            c['location_time_seconds']) for c in current_cluster)
                    if min_time_diff <= time_threshold:
                        current_cluster.append(anomaly)
                    else:
                        if current_cluster:
                            all_clusters.append(current_cluster)
                        current_cluster = [anomaly]
            if current_cluster:
                all_clusters.append(current_cluster)
        return all_clusters

    def select_score_aware_representatives(self, clusters: List[List[Dict]]) -> List[Dict]:
        representatives = []
        for i, cluster in enumerate(clusters):
            if not cluster:
                continue
            representative = max(cluster, key=lambda x: x.get('anomaly_score', 0.0))
            true_positives = [a for a in cluster if a.get('seizure_hit', False)]
            representative = representative.copy()
            representative['cluster_id'] = i
            representative['cluster_size'] = len(cluster)
            representative['cluster_tp_count'] = len(true_positives)
            representative['cluster_max_score'] = max(a.get('anomaly_score', 0.0) for a in cluster)
            representative['cluster_min_score'] = min(a.get('anomaly_score', 0.0) for a in cluster)
            representatives.append(representative)
        return representatives

    def multi_threshold_clustering(self, primary_threshold: float, secondary_threshold: float) -> List[List[Dict]]:
        all_clusters = []
        file_groups = defaultdict(list)
        for anomaly in self.all_anomalies:
            file_groups[anomaly['file_id']].append(anomaly)
        for file_id, file_anomalies in file_groups.items():
            sorted_anomalies = sorted(file_anomalies, key=lambda x: x['location_time_seconds'])
            primary_clusters, current_cluster = [], []
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

            final_clusters = []
            i = 0
            while i < len(primary_clusters):
                merged = primary_clusters[i][:]
                j = i + 1
                while j < len(primary_clusters):
                    last_time_in_merged = max(a['location_time_seconds'] for a in merged)
                    first_time_in_next = min(a['location_time_seconds'] for a in primary_clusters[j])
                    if first_time_in_next - last_time_in_merged <= secondary_threshold:
                        merged.extend(primary_clusters[j])
                        primary_clusters.pop(j)
                    else:
                        j += 1
                final_clusters.append(merged)
                i += 1
            all_clusters.extend(final_clusters)
        return all_clusters

    def subject_aware_clustering(self, base_threshold: float = 90.0) -> List[List[Dict]]:
        all_clusters = []
        subject_groups = defaultdict(lambda: defaultdict(list))
        for anomaly in self.all_anomalies:
            subject_groups[anomaly.get('subject_id', 'unknown')][anomaly['file_id']].append(anomaly)

        subject_thresholds = {}
        for subject_id, files in subject_groups.items():
            gaps = []
            for file_id, file_anomalies in files.items():
                s = sorted(file_anomalies, key=lambda x: x['location_time_seconds'])
                for i in range(1, len(s)):
                    gaps.append(s[i]['location_time_seconds'] - s[i-1]['location_time_seconds'])
            if gaps:
                gaps.sort()
                median_gap = gaps[len(gaps)//2]
                adaptive_factor = min(2.0, max(0.5, median_gap / base_threshold))
                subject_thresholds[subject_id] = base_threshold * adaptive_factor
            else:
                subject_thresholds[subject_id] = base_threshold

        for subject_id, files in subject_groups.items():
            threshold = subject_thresholds[subject_id]
            for file_id, file_anomalies in files.items():
                s = sorted(file_anomalies, key=lambda x: x['location_time_seconds'])
                current_cluster = []
                for anomaly in s:
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
                if current_cluster:
                    all_clusters.append(current_cluster)
        return all_clusters

    def select_smart_representatives(self, clusters: List[List[Dict]]) -> List[Dict]:
        representatives = []
        for i, cluster in enumerate(clusters):
            if not cluster:
                continue
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
            representative['cluster_id'] = i
            representative['cluster_size'] = len(cluster)
            representative['cluster_tp_count'] = len(true_positives)
            representative['avg_time_distance'] = avgd
            representatives.append(representative)
        return representatives

    def evaluate_clustering_strategy(self, representatives: List[Dict], base_metrics: Dict) -> Dict:
        """Evaluate clustering strategy using event-based metrics only."""
        from eval import event_metrics_for_clusters  # Avoid circular import
        
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
        
        # Event-based performance penalty - penalize if event recall drops significantly
        event_recall_penalty = max(0.0, base_metrics.get('event_recall', 0.0) - event_metrics.get('event_recall', 0.0))
        
        # Score based on event-based metrics: prioritize maintaining event recall while reducing false alarms
        score = (fp_reduction * 0.5) + (anomaly_reduction * 0.3) + (event_metrics.get('event_recall', 0.0) * 0.4) - (event_recall_penalty * 3.0)
        
        return {
            'total_anomalies': total_anomalies,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'anomaly_reduction': anomaly_reduction,
            'fp_reduction': fp_reduction,
            'event_recall': event_metrics.get('event_recall', 0.0),
            'event_iou': event_metrics.get('event_iou', 0.0),
            'event_far_per_hour': event_metrics.get('event_far_per_hour', 0.0),
            'event_recall_change': event_metrics.get('event_recall', 0.0) - base_metrics.get('event_recall', 0.0),
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
        if self.output_folder.name != "smart_clustered_results":
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
            
        elif strategy_name.startswith('adaptive_p') and strategy_name.endswith('s'):
            # Extract threshold from strategy name like 'adaptive_p75_120s'
            parts = strategy_name.split('_')
            threshold = strategy_params.get('threshold', int(parts[2][:-1]))
            clusters = self.time_based_clustering(threshold)
            representatives = self.select_smart_representatives(clusters)
            
        elif strategy_name.startswith('score_aware_') and strategy_name.endswith('s'):
            # Extract threshold from strategy name like 'score_aware_60s'
            threshold = strategy_params.get('threshold', int(strategy_name.split('_')[2][:-1]))
            clusters = self.score_aware_clustering(threshold)
            representatives = self.select_score_aware_representatives(clusters)
            
        elif strategy_name.startswith('hybrid_') and strategy_name.endswith('s'):
            # Extract thresholds from strategy name like 'hybrid_60_120s'
            parts = strategy_name.split('_')
            primary = strategy_params.get('primary', int(parts[1]))
            secondary = strategy_params.get('secondary', int(parts[2][:-1]))
            clusters = self.multi_threshold_clustering(primary, secondary)
            representatives = self.select_smart_representatives(clusters)
            
        elif strategy_name.startswith('subject_aware_') and strategy_name.endswith('s'):
            # Extract threshold from strategy name like 'subject_aware_90s'
            threshold = strategy_params.get('threshold', int(strategy_name.split('_')[2][:-1]))
            clusters = self.subject_aware_clustering(threshold)
            representatives = self.select_smart_representatives(clusters)
            
        else:
            raise ValueError(f"Unknown clustering strategy: {strategy_name}")
        
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
        if self.output_folder.name != "smart_clustered_results":
            with open(self.output_folder / "metrics_before" / "metrics_summary.json", 'w') as f:
                json.dump(base_metrics, f, indent=2)

        strategies = {}

        # 1) time thresholds (broad)
        time_thresholds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 60, 75, 90, 120, 150, 180, 240, 300]
        for th in time_thresholds:
            clusters = self.time_based_clustering(th)
            reps = self.select_smart_representatives(clusters)
            m = self.evaluate_clustering_strategy(reps, base_metrics)
            strategies[f'time_{th}s'] = {'method': 'time_based',
                                         'parameters': {'threshold': th},
                                         'clusters': len(clusters),
                                         'representatives': reps,
                                         'metrics': m}

        # 2) fine thresholds
        fine_thresholds = [2, 3, 6, 8, 12, 18, 22, 28, 32, 38, 42, 48, 55, 65, 85, 105]
        for th in fine_thresholds:
            clusters = self.time_based_clustering(th)
            reps = self.select_smart_representatives(clusters)
            m = self.evaluate_clustering_strategy(reps, base_metrics)
            strategies[f'fine_time_{th}s'] = {'method': 'fine_time_based',
                                              'parameters': {'threshold': th},
                                              'clusters': len(clusters),
                                              'representatives': reps,
                                              'metrics': m}

        # 3) extended thresholds
        extended_thresholds = [360, 450, 600, 900, 1200, 1800]
        for th in extended_thresholds:
            clusters = self.time_based_clustering(th)
            reps = self.select_smart_representatives(clusters)
            m = self.evaluate_clustering_strategy(reps, base_metrics)
            strategies[f'extended_time_{th}s'] = {'method': 'extended_time_based',
                                                  'parameters': {'threshold': th},
                                                  'clusters': len(clusters),
                                                  'representatives': reps,
                                                  'metrics': m}

        # 4) adaptive thresholds from empirical gaps
        all_time_gaps = []
        file_groups = defaultdict(list)
        for a in self.all_anomalies:
            file_groups[a['file_id']].append(a)
        for fid, fas in file_groups.items():
            s = sorted(fas, key=lambda x: x['location_time_seconds'])
            for i in range(1, len(s)):
                all_time_gaps.append(s[i]['location_time_seconds'] - s[i-1]['location_time_seconds'])
        if all_time_gaps:
            all_time_gaps.sort()
            n = len(all_time_gaps)
            for p in [10, 25, 50, 75, 90, 95, 99]:
                idx = int(p / 100 * (n - 1))
                th = int(all_time_gaps[idx])
                if th > 0:
                    clusters = self.time_based_clustering(th)
                    reps = self.select_smart_representatives(clusters)
                    m = self.evaluate_clustering_strategy(reps, base_metrics)
                    strategies[f'adaptive_p{p}_{th}s'] = {'method': 'adaptive_time_based',
                                                          'parameters': {'threshold': th, 'percentile_basis': p},
                                                          'clusters': len(clusters),
                                                          'representatives': reps,
                                                          'metrics': m}

        # 5) score-aware
        for th in [30, 60, 120, 180]:
            clusters = self.score_aware_clustering(th)
            reps = self.select_score_aware_representatives(clusters)
            m = self.evaluate_clustering_strategy(reps, base_metrics)
            strategies[f'score_aware_{th}s'] = {'method': 'score_aware',
                                                'parameters': {'threshold': th},
                                                'clusters': len(clusters),
                                                'representatives': reps,
                                                'metrics': m}

        # 6) hybrid multi-threshold
        for cfg in [{'primary': 60, 'secondary': 120},
                    {'primary': 30, 'secondary': 90},
                    {'primary': 45, 'secondary': 180},
                    {'primary': 90, 'secondary': 240}]:
            clusters = self.multi_threshold_clustering(cfg['primary'], cfg['secondary'])
            reps = self.select_smart_representatives(clusters)
            m = self.evaluate_clustering_strategy(reps, base_metrics)
            strategies[f'hybrid_{cfg["primary"]}_{cfg["secondary"]}s'] = {'method': 'multi_threshold',
                                                                          'parameters': cfg,
                                                                          'clusters': len(clusters),
                                                                          'representatives': reps,
                                                                          'metrics': m}

        # 7) subject-aware
        for th in [45, 90, 135]:
            clusters = self.subject_aware_clustering(th)
            reps = self.select_smart_representatives(clusters)
            m = self.evaluate_clustering_strategy(reps, base_metrics)
            strategies[f'subject_aware_{th}s'] = {'method': 'subject_aware',
                                                  'parameters': {'threshold': th},
                                                  'clusters': len(clusters),
                                                  'representatives': reps,
                                                  'metrics': m}

        best_strategy_name = max(strategies.keys(), key=lambda k: strategies[k]['metrics']['score'])
        best_strategy = strategies[best_strategy_name]

        # Only save results if output folder was explicitly requested
        if self.output_folder.name != "smart_clustered_results":  # Only save if custom folder was provided
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
    fixed_strategy: Optional[str] = None
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
        fixed_strategy=fixed_strategy
    )
    
    if fixed_strategy:
        # Parse strategy name and parameters
        strategy_params = {}
        
        # Extract parameters from strategy name
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
        elif fixed_strategy.startswith('adaptive_p') and fixed_strategy.endswith('s'):
            # Extract threshold from strategy name like 'adaptive_p75_120s'
            parts = fixed_strategy.split('_')
            threshold = int(parts[2][:-1])
            percentile = int(parts[1][1:])  # Remove 'p' prefix
            strategy_params = {'threshold': threshold, 'percentile_basis': percentile}
        elif fixed_strategy.startswith('score_aware_') and fixed_strategy.endswith('s'):
            # Extract threshold from strategy name like 'score_aware_60s'
            threshold = int(fixed_strategy.split('_')[2][:-1])
            strategy_params['threshold'] = threshold
        elif fixed_strategy.startswith('hybrid_') and fixed_strategy.endswith('s'):
            # Extract thresholds from strategy name like 'hybrid_60_120s'
            parts = fixed_strategy.split('_')
            primary = int(parts[1])
            secondary = int(parts[2][:-1])
            strategy_params = {'primary': primary, 'secondary': secondary}
        elif fixed_strategy.startswith('subject_aware_') and fixed_strategy.endswith('s'):
            # Extract threshold from strategy name like 'subject_aware_90s'
            threshold = int(fixed_strategy.split('_')[2][:-1])
            strategy_params['threshold'] = threshold
        
        # Run fixed strategy analysis
        result = analyzer.run_fixed_strategy_analysis(fixed_strategy, strategy_params)
        
        # Convert to expected format for backward compatibility
        return {
            'base_metrics': result['base_metrics'],
            'strategies': {fixed_strategy: {
                'method': fixed_strategy.split('_')[0] + '_based' if not fixed_strategy.startswith('hybrid') and not fixed_strategy.startswith('subject') else fixed_strategy.split('_')[0],
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
    output_folder: Optional[str] = None
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
        total_samples=total_samples
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