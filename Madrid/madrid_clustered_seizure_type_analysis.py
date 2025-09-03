#!/usr/bin/env python3
"""
Madrid Clustered Seizure Type Analysis
Analyzes which seizure types are best detected using time_180s clustering strategy.
Uses the same clustering approach as madrid_clustering_false_alarm_reducer_train_test.py
"""

import json
import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import argparse

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import the SeizeIT2 annotation class if available
try:
    from Information.Data.seizeit2_main.classes.annotation import Annotation
    SEIZEIT_AVAILABLE = True
except ImportError:
    print("Warning: SeizeIT2 annotation class not available. Using fallback method.")
    SEIZEIT_AVAILABLE = False


class MadridClusteredSeizureTypeAnalyzer:
    def __init__(self, results_dir: str, seizeit2_data_path: str = None, output_dir: str = None,
                 pre_seizure_minutes: float = 5.0, post_seizure_minutes: float = 3.0):
        """
        Initialize the clustered seizure type analyzer.
        
        Args:
            results_dir: Directory containing Madrid windowed results JSON files
            seizeit2_data_path: Path to SeizeIT2 dataset for annotations (optional)
            output_dir: Directory to save analysis results
            pre_seizure_minutes: Minutes before seizure start to consider as detection window
            post_seizure_minutes: Minutes after seizure end to consider as detection window
        """
        self.results_dir = Path(results_dir)
        self.seizeit2_data_path = Path(seizeit2_data_path) if seizeit2_data_path else None
        self.output_dir = Path(output_dir) if output_dir else self.results_dir / "seizure_type_analysis"
        self.pre_seizure_seconds = pre_seizure_minutes * 60.0
        self.post_seizure_seconds = post_seizure_minutes * 60.0
        self.clustering_time_threshold = 180  # Fixed time_180s strategy
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize storage for results
        self.seizure_type_data = defaultdict(lambda: {
            'total_seizures': 0,
            'detected_seizures': 0,
            'total_anomalies_before': 0,
            'total_anomalies_after': 0,
            'true_positives': 0,
            'false_positives': 0,
            'total_duration_hours': 0.0,
            'patients': set(),
            'seizure_details': []
        })
        
        self.seizure_categories = {
            'focal_aware': ['sz_foc_a_m', 'sz_foc_a_nm', 'sz_foc_a_um'],
            'focal_impaired': ['sz_foc_ia_m', 'sz_foc_ia_nm', 'sz_foc_ia_um'],
            'focal_unknown': ['sz_foc_ua_m', 'sz_foc_ua_nm', 'sz_foc_ua_um'],
            'focal_bilateral': ['sz_foc_f2b'],
            'generalized_motor': ['sz_gen_m'],
            'generalized_nonmotor': ['sz_gen_nm'],
            'unknown_motor': ['sz_uo_m'],
            'unknown_nonmotor': ['sz_uo_nm']
        }
    
    def get_seizure_category(self, event_type: str) -> str:
        """
        Categorize seizure types into broader categories.
        
        Args:
            event_type: Original event type string
            
        Returns:
            Category string
        """
        if not event_type or event_type == 'unknown':
            return 'unknown'
        
        # Check each category
        for category, prefixes in self.seizure_categories.items():
            for prefix in prefixes:
                if event_type.startswith(prefix):
                    return category
        
        # Fallback categories
        if event_type.startswith('sz_foc'):
            return 'focal_other'
        elif event_type.startswith('sz_gen'):
            return 'generalized_other'
        elif event_type.startswith('sz_uo'):
            return 'unknown_onset'
        elif event_type.startswith('sz'):
            return 'seizure_unclassified'
        else:
            return 'non_seizure'
    
    def get_motor_classification(self, event_type: str) -> str:
        """
        Classify seizures as motor, non-motor, or mixed.
        
        Args:
            event_type: Original event type string
            
        Returns:
            'motor', 'non-motor', or 'unknown'
        """
        if not event_type or event_type == 'unknown':
            return 'unknown'
        
        # Motor patterns
        motor_indicators = ['_m_', '_m', '_f2b', 'tonic', 'clonic', 'myoclonic', 'atonic', 
                          'automatisms', 'hyperkinetic', 'spasms']
        
        # Non-motor patterns
        nonmotor_indicators = ['_nm_', '_nm', 'sensory', 'cognitive', 'emotional', 
                              'autonomic', 'behavior', 'typical', 'atypical']
        
        event_lower = event_type.lower()
        
        # Check for motor indicators
        has_motor = any(indicator in event_lower for indicator in motor_indicators)
        has_nonmotor = any(indicator in event_lower for indicator in nonmotor_indicators)
        
        if has_motor and not has_nonmotor:
            return 'motor'
        elif has_nonmotor and not has_motor:
            return 'non-motor'
        elif has_motor and has_nonmotor:
            return 'mixed'
        else:
            return 'unknown'
    
    def load_result_file(self, filepath: Path) -> Dict[str, Any]:
        """Load and parse a Madrid results JSON file."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def get_seizure_eventtype_from_data(self, result_data: Dict[str, Any]) -> str:
        """
        Extract seizure event type from the result data.
        First tries to get from validation_data, then falls back to SeizeIT2 if available.
        
        Args:
            result_data: Madrid result data dictionary
            
        Returns:
            Event type string or 'unknown'
        """
        # Try to get from validation data first
        validation_data = result_data.get('validation_data', {})
        ground_truth = validation_data.get('ground_truth', {})
        seizure_windows = ground_truth.get('seizure_windows', [])
        
        if seizure_windows:
            # Get the most common event type from seizure windows
            event_types = []
            for window in seizure_windows:
                segments = window.get('seizure_segments', [])
                for segment in segments:
                    event_type = segment.get('event_type', segment.get('eventType', ''))
                    if event_type:
                        event_types.append(event_type)
            
            if event_types:
                # Return most common event type
                from collections import Counter
                most_common = Counter(event_types).most_common(1)
                return most_common[0][0] if most_common else 'unknown'
        
        # Fallback: try to get from SeizeIT2 annotations if available
        if SEIZEIT_AVAILABLE and self.seizeit2_data_path:
            try:
                input_data = result_data.get('input_data', {})
                subject_id = input_data.get('subject_id', '')
                run_id = input_data.get('run_id', '')
                
                if subject_id and run_id:
                    annotation = Annotation.loadAnnotation(
                        str(self.seizeit2_data_path),
                        [subject_id, run_id]
                    )
                    # Assuming first seizure if not specified
                    if annotation.types and len(annotation.types) > 0:
                        return annotation.types[0]
            except:
                pass
        
        return 'unknown'
    
    def extract_file_level_anomalies(self, result_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract all anomalies from a file using top-ranked strategy.
        """
        analysis_results = result_data.get('analysis_results', {})
        window_results = analysis_results.get('window_results', [])
        
        all_anomalies = []
        
        for window in window_results:
            window_index = window.get('window_index')
            window_start_time = window.get('window_start_time', 0)
            anomalies = window.get('anomalies', [])
            
            if not anomalies:
                continue
            
            # Use top-ranked anomaly only (matching train_test script)
            selected_anomalies = [anomalies[0]]
            
            # Convert to file-level representation
            for anomaly in selected_anomalies:
                location_time_in_window = anomaly.get('location_time_in_window', 0)
                absolute_time = window_start_time + location_time_in_window
                
                file_level_anomaly = {
                    'absolute_time': absolute_time,
                    'anomaly_score': anomaly.get('anomaly_score', 0),
                    'original_window_index': window_index,
                    'window_start_time': window_start_time
                }
                
                all_anomalies.append(file_level_anomaly)
        
        # Sort by absolute time
        all_anomalies.sort(key=lambda x: x['absolute_time'])
        
        return all_anomalies
    
    def time_based_clustering(self, anomalies: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Perform time-based clustering with time_180s strategy.
        """
        if not anomalies:
            return []
        
        clusters = []
        current_cluster = [anomalies[0]]
        
        for i in range(1, len(anomalies)):
            time_diff = anomalies[i]['absolute_time'] - current_cluster[-1]['absolute_time']
            
            if time_diff <= self.clustering_time_threshold:
                current_cluster.append(anomalies[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [anomalies[i]]
        
        if current_cluster:
            clusters.append(current_cluster)
        
        return clusters
    
    def select_cluster_representative(self, cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select representative from cluster based on minimal mean time distance.
        """
        if len(cluster) == 1:
            return cluster[0].copy()
        
        # Calculate mean time distance for each anomaly
        times = [a['absolute_time'] for a in cluster]
        best_idx = 0
        min_mean_distance = float('inf')
        
        for i, anomaly in enumerate(cluster):
            mean_distance = np.mean([abs(anomaly['absolute_time'] - other_time) 
                                   for other_time in times if other_time != anomaly['absolute_time']])
            
            if mean_distance < min_mean_distance:
                min_mean_distance = mean_distance
                best_idx = i
        
        return cluster[best_idx].copy()
    
    def group_seizures_by_time(self, seizure_windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group seizure windows into individual seizures with extended time windows."""
        if not seizure_windows:
            return []
        
        # Collect all seizure segments
        all_segments = []
        for window in seizure_windows:
            seizure_segments = window.get('seizure_segments', [])
            for segment in seizure_segments:
                # Also capture event type from segment
                segment_with_type = segment.copy()
                segment_with_type['event_type'] = segment.get('event_type', segment.get('eventType', 'unknown'))
                all_segments.append(segment_with_type)
        
        # Group segments by their absolute time intervals
        seizure_groups = {}
        for segment in all_segments:
            start_time = round(segment.get('start_time_absolute', 0), 1)
            end_time = round(segment.get('end_time_absolute', 0), 1)
            time_key = (start_time, end_time)
            
            if time_key not in seizure_groups:
                seizure_groups[time_key] = {
                    'start_time_absolute': start_time,
                    'end_time_absolute': end_time,
                    'duration_seconds': end_time - start_time,
                    'extended_start_time': start_time - self.pre_seizure_seconds,
                    'extended_end_time': end_time + self.post_seizure_seconds,
                    'event_type': segment['event_type']
                }
        
        # Convert to list and sort by start time
        seizures = list(seizure_groups.values())
        seizures.sort(key=lambda x: x['start_time_absolute'])
        
        return seizures
    
    def process_single_file(self, filepath: Path) -> Dict[str, Any]:
        """Process a single file and analyze seizure type detection with clustering."""
        
        result_data = self.load_result_file(filepath)
        if result_data is None:
            return None
        
        # Get basic info
        input_data = result_data.get('input_data', {})
        validation_data = result_data.get('validation_data', {})
        
        subject_id = input_data.get('subject_id', 'unknown')
        run_id = input_data.get('run_id', 'unknown')
        
        # Get signal duration
        signal_metadata = input_data.get('signal_metadata', {})
        file_duration_hours = signal_metadata.get('total_duration_seconds', 0) / 3600.0
        
        # Get seizure event type
        event_type = self.get_seizure_eventtype_from_data(result_data)
        category = self.get_seizure_category(event_type)
        motor_class = self.get_motor_classification(event_type)
        
        # Get anomalies at file level
        all_anomalies = self.extract_file_level_anomalies(result_data)
        
        # Apply time_180s clustering
        clusters = self.time_based_clustering(all_anomalies)
        representatives = [self.select_cluster_representative(cluster) for cluster in clusters]
        
        # Get seizures with extended windows
        ground_truth = validation_data.get('ground_truth', {})
        seizure_windows = ground_truth.get('seizure_windows', [])
        individual_seizures = self.group_seizures_by_time(seizure_windows)
        
        # Calculate detection metrics
        detected_seizures = set()
        true_positives = 0
        false_positives = 0
        
        for rep in representatives:
            rep_time = rep['absolute_time']
            is_true_positive = False
            
            # Check if representative overlaps with any extended seizure window
            for i, seizure in enumerate(individual_seizures):
                if seizure['extended_start_time'] <= rep_time <= seizure['extended_end_time']:
                    is_true_positive = True
                    detected_seizures.add(i)
                    break
            
            if is_true_positive:
                true_positives += 1
            else:
                false_positives += 1
        
        # Store results
        file_result = {
            'subject_id': subject_id,
            'run_id': run_id,
            'event_type': event_type,
            'category': category,
            'motor_class': motor_class,
            'duration_hours': file_duration_hours,
            'total_seizures': len(individual_seizures),
            'detected_seizures': len(detected_seizures),
            'anomalies_before_clustering': len(all_anomalies),
            'anomalies_after_clustering': len(representatives),
            'true_positives': true_positives,
            'false_positives': false_positives,
            'seizure_detected': len(detected_seizures) > 0
        }
        
        # Update aggregated data for event type
        self.seizure_type_data[event_type]['total_seizures'] += len(individual_seizures)
        self.seizure_type_data[event_type]['detected_seizures'] += len(detected_seizures)
        self.seizure_type_data[event_type]['total_anomalies_before'] += len(all_anomalies)
        self.seizure_type_data[event_type]['total_anomalies_after'] += len(representatives)
        self.seizure_type_data[event_type]['true_positives'] += true_positives
        self.seizure_type_data[event_type]['false_positives'] += false_positives
        self.seizure_type_data[event_type]['total_duration_hours'] += file_duration_hours
        self.seizure_type_data[event_type]['patients'].add(subject_id)
        self.seizure_type_data[event_type]['seizure_details'].append(file_result)
        
        # Also update category data
        self.seizure_type_data[f'CATEGORY_{category}']['total_seizures'] += len(individual_seizures)
        self.seizure_type_data[f'CATEGORY_{category}']['detected_seizures'] += len(detected_seizures)
        self.seizure_type_data[f'CATEGORY_{category}']['total_anomalies_before'] += len(all_anomalies)
        self.seizure_type_data[f'CATEGORY_{category}']['total_anomalies_after'] += len(representatives)
        self.seizure_type_data[f'CATEGORY_{category}']['true_positives'] += true_positives
        self.seizure_type_data[f'CATEGORY_{category}']['false_positives'] += false_positives
        self.seizure_type_data[f'CATEGORY_{category}']['total_duration_hours'] += file_duration_hours
        self.seizure_type_data[f'CATEGORY_{category}']['patients'].add(subject_id)
        
        # Update motor classification data
        self.seizure_type_data[f'MOTOR_{motor_class}']['total_seizures'] += len(individual_seizures)
        self.seizure_type_data[f'MOTOR_{motor_class}']['detected_seizures'] += len(detected_seizures)
        self.seizure_type_data[f'MOTOR_{motor_class}']['total_anomalies_before'] += len(all_anomalies)
        self.seizure_type_data[f'MOTOR_{motor_class}']['total_anomalies_after'] += len(representatives)
        self.seizure_type_data[f'MOTOR_{motor_class}']['true_positives'] += true_positives
        self.seizure_type_data[f'MOTOR_{motor_class}']['false_positives'] += false_positives
        self.seizure_type_data[f'MOTOR_{motor_class}']['total_duration_hours'] += file_duration_hours
        self.seizure_type_data[f'MOTOR_{motor_class}']['patients'].add(subject_id)
        
        return file_result
    
    def calculate_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate final metrics for each seizure type."""
        metrics = {}
        
        for seizure_type, data in self.seizure_type_data.items():
            if data['total_seizures'] == 0:
                continue
            
            sensitivity = data['detected_seizures'] / data['total_seizures'] if data['total_seizures'] > 0 else 0
            
            precision = (data['true_positives'] / (data['true_positives'] + data['false_positives']) 
                        if (data['true_positives'] + data['false_positives']) > 0 else 0)
            
            false_alarms_per_hour = (data['false_positives'] / data['total_duration_hours'] 
                                    if data['total_duration_hours'] > 0 else 0)
            
            anomaly_reduction = ((data['total_anomalies_before'] - data['total_anomalies_after']) / 
                               data['total_anomalies_before'] 
                               if data['total_anomalies_before'] > 0 else 0)
            
            metrics[seizure_type] = {
                'sensitivity': sensitivity,
                'precision': precision,
                'false_alarms_per_hour': false_alarms_per_hour,
                'anomaly_reduction': anomaly_reduction,
                'total_seizures': data['total_seizures'],
                'detected_seizures': data['detected_seizures'],
                'num_patients': len(data['patients']),
                'total_duration_hours': data['total_duration_hours']
            }
        
        return metrics
    
    def create_visualizations(self, metrics: Dict[str, Dict[str, float]]):
        """Create visualizations for seizure type analysis."""
        
        # Separate different metric types
        event_types = {}
        categories = {}
        motor_classes = {}
        
        for key, value in metrics.items():
            if key.startswith('CATEGORY_'):
                categories[key.replace('CATEGORY_', '')] = value
            elif key.startswith('MOTOR_'):
                motor_classes[key.replace('MOTOR_', '')] = value
            else:
                event_types[key] = value
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Sensitivity by seizure type
        ax1 = plt.subplot(3, 3, 1)
        if event_types:
            types_sorted = sorted(event_types.items(), key=lambda x: x[1]['sensitivity'], reverse=True)[:15]
            types = [t[0] for t in types_sorted]
            sensitivities = [t[1]['sensitivity'] for t in types_sorted]
            ax1.barh(types, sensitivities, color='skyblue')
            ax1.set_xlabel('Sensitivity')
            ax1.set_title('Top 15 Seizure Types by Sensitivity')
            ax1.set_xlim(0, 1)
            for i, v in enumerate(sensitivities):
                ax1.text(v + 0.01, i, f'{v:.2%}', va='center')
        
        # 2. Sensitivity by category
        ax2 = plt.subplot(3, 3, 2)
        if categories:
            cat_sorted = sorted(categories.items(), key=lambda x: x[1]['sensitivity'], reverse=True)
            cat_names = [c[0] for c in cat_sorted]
            cat_sens = [c[1]['sensitivity'] for c in cat_sorted]
            cat_counts = [c[1]['total_seizures'] for c in cat_sorted]
            
            bars = ax2.bar(range(len(cat_names)), cat_sens, color='lightgreen')
            ax2.set_xlabel('Category')
            ax2.set_ylabel('Sensitivity')
            ax2.set_title('Sensitivity by Seizure Category')
            ax2.set_xticks(range(len(cat_names)))
            ax2.set_xticklabels(cat_names, rotation=45, ha='right')
            ax2.set_ylim(0, 1)
            
            # Add sample size on top of bars
            for i, (bar, count) in enumerate(zip(bars, cat_counts)):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'n={count}', ha='center', va='bottom', fontsize=8)
        
        # 3. Motor vs Non-Motor
        ax3 = plt.subplot(3, 3, 3)
        if motor_classes:
            motor_sorted = sorted(motor_classes.items(), key=lambda x: x[1]['sensitivity'], reverse=True)
            motor_names = [m[0] for m in motor_sorted]
            motor_sens = [m[1]['sensitivity'] for m in motor_sorted]
            motor_counts = [m[1]['total_seizures'] for m in motor_sorted]
            
            bars = ax3.bar(range(len(motor_names)), motor_sens, color='salmon')
            ax3.set_xlabel('Motor Classification')
            ax3.set_ylabel('Sensitivity')
            ax3.set_title('Sensitivity by Motor Classification')
            ax3.set_xticks(range(len(motor_names)))
            ax3.set_xticklabels(motor_names, rotation=45, ha='right')
            ax3.set_ylim(0, 1)
            
            for i, (bar, count) in enumerate(zip(bars, motor_counts)):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'n={count}', ha='center', va='bottom', fontsize=8)
        
        # 4. False Alarms per Hour by category
        ax4 = plt.subplot(3, 3, 4)
        if categories:
            cat_fa = [(c, v['false_alarms_per_hour']) for c, v in categories.items()]
            cat_fa_sorted = sorted(cat_fa, key=lambda x: x[1])
            cat_names = [c[0] for c in cat_fa_sorted]
            fa_rates = [c[1] for c in cat_fa_sorted]
            
            ax4.bar(range(len(cat_names)), fa_rates, color='orange')
            ax4.set_xlabel('Category')
            ax4.set_ylabel('False Alarms per Hour')
            ax4.set_title('False Alarm Rate by Category')
            ax4.set_xticks(range(len(cat_names)))
            ax4.set_xticklabels(cat_names, rotation=45, ha='right')
        
        # 5. Scatter plot: Sensitivity vs False Alarms
        ax5 = plt.subplot(3, 3, 5)
        if event_types:
            sens_values = [v['sensitivity'] for v in event_types.values()]
            fa_values = [v['false_alarms_per_hour'] for v in event_types.values()]
            sizes = [v['total_seizures'] * 10 for v in event_types.values()]
            
            scatter = ax5.scatter(fa_values, sens_values, s=sizes, alpha=0.6, c=range(len(sens_values)), cmap='viridis')
            ax5.set_xlabel('False Alarms per Hour')
            ax5.set_ylabel('Sensitivity')
            ax5.set_title('Sensitivity vs False Alarms Trade-off')
            ax5.grid(True, alpha=0.3)
            
            # Add ideal point
            ax5.scatter([0], [1], s=100, marker='*', color='red', label='Ideal', zorder=5)
            ax5.legend()
        
        # 6. Anomaly Reduction by Category
        ax6 = plt.subplot(3, 3, 6)
        if categories:
            cat_reduction = [(c, v['anomaly_reduction']) for c, v in categories.items()]
            cat_reduction_sorted = sorted(cat_reduction, key=lambda x: x[1], reverse=True)
            cat_names = [c[0] for c in cat_reduction_sorted]
            reductions = [c[1] * 100 for c in cat_reduction_sorted]
            
            ax6.bar(range(len(cat_names)), reductions, color='purple', alpha=0.7)
            ax6.set_xlabel('Category')
            ax6.set_ylabel('Anomaly Reduction (%)')
            ax6.set_title('Clustering Effectiveness by Category')
            ax6.set_xticks(range(len(cat_names)))
            ax6.set_xticklabels(cat_names, rotation=45, ha='right')
            ax6.set_ylim(0, 100)
        
        # 7. Sample Size Distribution
        ax7 = plt.subplot(3, 3, 7)
        if categories:
            cat_samples = [(c, v['total_seizures']) for c, v in categories.items()]
            cat_samples_sorted = sorted(cat_samples, key=lambda x: x[1], reverse=True)
            cat_names = [c[0] for c in cat_samples_sorted]
            samples = [c[1] for c in cat_samples_sorted]
            
            ax7.bar(range(len(cat_names)), samples, color='teal')
            ax7.set_xlabel('Category')
            ax7.set_ylabel('Number of Seizures')
            ax7.set_title('Sample Size by Category')
            ax7.set_xticks(range(len(cat_names)))
            ax7.set_xticklabels(cat_names, rotation=45, ha='right')
        
        # 8. Precision by Category
        ax8 = plt.subplot(3, 3, 8)
        if categories:
            cat_prec = [(c, v['precision']) for c, v in categories.items()]
            cat_prec_sorted = sorted(cat_prec, key=lambda x: x[1], reverse=True)
            cat_names = [c[0] for c in cat_prec_sorted]
            precisions = [c[1] for c in cat_prec_sorted]
            
            ax8.bar(range(len(cat_names)), precisions, color='gold')
            ax8.set_xlabel('Category')
            ax8.set_ylabel('Precision')
            ax8.set_title('Precision by Category')
            ax8.set_xticks(range(len(cat_names)))
            ax8.set_xticklabels(cat_names, rotation=45, ha='right')
            ax8.set_ylim(0, 1)
        
        # 9. Heatmap of metrics by category
        ax9 = plt.subplot(3, 3, 9)
        if categories:
            cat_names = list(categories.keys())
            metric_names = ['Sensitivity', 'Precision', 'FA/h', 'Reduction']
            
            heatmap_data = []
            for cat in cat_names:
                row = [
                    categories[cat]['sensitivity'],
                    categories[cat]['precision'],
                    min(categories[cat]['false_alarms_per_hour'] / 10, 1),  # Normalize FA/h
                    categories[cat]['anomaly_reduction']
                ]
                heatmap_data.append(row)
            
            im = ax9.imshow(np.array(heatmap_data).T, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
            ax9.set_xticks(range(len(cat_names)))
            ax9.set_yticks(range(len(metric_names)))
            ax9.set_xticklabels(cat_names, rotation=45, ha='right')
            ax9.set_yticklabels(metric_names)
            ax9.set_title('Metrics Heatmap by Category')
            
            # Add colorbar
            plt.colorbar(im, ax=ax9, fraction=0.046, pad=0.04)
        
        plt.suptitle(f'Seizure Type Detection Analysis with time_{self.clustering_time_threshold}s Clustering', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / f'seizure_type_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualizations saved to: {output_path}")
        plt.close()
    
    def save_results(self, metrics: Dict[str, Dict[str, float]]):
        """Save analysis results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed metrics as JSON
        json_path = self.output_dir / f"seizure_type_metrics_{timestamp}.json"
        with open(json_path, 'w') as f:
            # Convert sets to lists for JSON serialization
            serializable_data = {}
            for key, value in self.seizure_type_data.items():
                serializable_data[key] = {
                    k: list(v) if isinstance(v, set) else v
                    for k, v in value.items()
                    if k != 'seizure_details'  # Exclude detailed list for summary
                }
            
            json.dump({
                'analysis_metadata': {
                    'timestamp': timestamp,
                    'clustering_strategy': f'time_{self.clustering_time_threshold}s',
                    'pre_seizure_minutes': self.pre_seizure_seconds / 60,
                    'post_seizure_minutes': self.post_seizure_seconds / 60
                },
                'metrics': metrics,
                'detailed_data': serializable_data
            }, f, indent=2)
        print(f"Metrics saved to: {json_path}")
        
        # Save human-readable report
        report_path = self.output_dir / f"seizure_type_report_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SEIZURE TYPE DETECTION ANALYSIS REPORT\n")
            f.write(f"Clustering Strategy: time_{self.clustering_time_threshold}s\n")
            f.write(f"Extended Window: -{self.pre_seizure_seconds/60:.1f} min to +{self.post_seizure_seconds/60:.1f} min\n")
            f.write("="*80 + "\n\n")
            
            # Separate metrics by type
            event_types = {k: v for k, v in metrics.items() if not (k.startswith('CATEGORY_') or k.startswith('MOTOR_'))}
            categories = {k.replace('CATEGORY_', ''): v for k, v in metrics.items() if k.startswith('CATEGORY_')}
            motor_classes = {k.replace('MOTOR_', ''): v for k, v in metrics.items() if k.startswith('MOTOR_')}
            
            # Sort by sensitivity
            if event_types:
                f.write("TOP PERFORMING SEIZURE TYPES (by Sensitivity):\n")
                f.write("-"*40 + "\n")
                sorted_types = sorted(event_types.items(), key=lambda x: x[1]['sensitivity'], reverse=True)[:10]
                for event_type, m in sorted_types:
                    f.write(f"\n{event_type}:\n")
                    f.write(f"  Sensitivity: {m['sensitivity']:.4f} ({m['sensitivity']*100:.2f}%)\n")
                    f.write(f"  Detected: {m['detected_seizures']}/{m['total_seizures']} seizures\n")
                    f.write(f"  Precision: {m['precision']:.4f}\n")
                    f.write(f"  False Alarms/Hour: {m['false_alarms_per_hour']:.4f}\n")
                    f.write(f"  Anomaly Reduction: {m['anomaly_reduction']*100:.1f}%\n")
                    f.write(f"  Patients: {m['num_patients']}\n")
            
            if categories:
                f.write("\n" + "="*40 + "\n")
                f.write("PERFORMANCE BY CATEGORY:\n")
                f.write("-"*40 + "\n")
                sorted_cats = sorted(categories.items(), key=lambda x: x[1]['sensitivity'], reverse=True)
                for category, m in sorted_cats:
                    f.write(f"\n{category}:\n")
                    f.write(f"  Sensitivity: {m['sensitivity']:.4f} ({m['sensitivity']*100:.2f}%)\n")
                    f.write(f"  Detected: {m['detected_seizures']}/{m['total_seizures']} seizures\n")
                    f.write(f"  Precision: {m['precision']:.4f}\n")
                    f.write(f"  False Alarms/Hour: {m['false_alarms_per_hour']:.4f}\n")
                    f.write(f"  Anomaly Reduction: {m['anomaly_reduction']*100:.1f}%\n")
                    f.write(f"  Patients: {m['num_patients']}\n")
                    f.write(f"  Total Duration: {m['total_duration_hours']:.1f} hours\n")
            
            if motor_classes:
                f.write("\n" + "="*40 + "\n")
                f.write("PERFORMANCE BY MOTOR CLASSIFICATION:\n")
                f.write("-"*40 + "\n")
                sorted_motor = sorted(motor_classes.items(), key=lambda x: x[1]['sensitivity'], reverse=True)
                for motor_class, m in sorted_motor:
                    f.write(f"\n{motor_class}:\n")
                    f.write(f"  Sensitivity: {m['sensitivity']:.4f} ({m['sensitivity']*100:.2f}%)\n")
                    f.write(f"  Detected: {m['detected_seizures']}/{m['total_seizures']} seizures\n")
                    f.write(f"  Precision: {m['precision']:.4f}\n")
                    f.write(f"  False Alarms/Hour: {m['false_alarms_per_hour']:.4f}\n")
                    f.write(f"  Anomaly Reduction: {m['anomaly_reduction']*100:.1f}%\n")
                    f.write(f"  Patients: {m['num_patients']}\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"Report saved to: {report_path}")
    
    def run_analysis(self):
        """Run complete seizure type analysis."""
        json_files = list(self.results_dir.glob("madrid_windowed_results_*.json"))
        
        if not json_files:
            print(f"No Madrid result files found in {self.results_dir}")
            return None
        
        print(f"Found {len(json_files)} Madrid result files")
        print(f"Using clustering strategy: time_{self.clustering_time_threshold}s")
        print(f"Extended window: -{self.pre_seizure_seconds/60:.1f} min to +{self.post_seizure_seconds/60:.1f} min")
        
        # Process each file
        for i, json_file in enumerate(sorted(json_files), 1):
            print(f"Processing file {i}/{len(json_files)}: {json_file.name}")
            self.process_single_file(json_file)
        
        # Calculate final metrics
        metrics = self.calculate_metrics()
        
        # Create visualizations
        self.create_visualizations(metrics)
        
        # Save results
        self.save_results(metrics)
        
        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        
        # Find best performing types
        event_types = {k: v for k, v in metrics.items() if not (k.startswith('CATEGORY_') or k.startswith('MOTOR_'))}
        if event_types:
            best_types = sorted(event_types.items(), key=lambda x: x[1]['sensitivity'], reverse=True)[:5]
            print("\nTop 5 Best Detected Seizure Types:")
            for event_type, m in best_types:
                print(f"  {event_type}: {m['sensitivity']*100:.1f}% sensitivity ({m['detected_seizures']}/{m['total_seizures']} detected)")
        
        categories = {k.replace('CATEGORY_', ''): v for k, v in metrics.items() if k.startswith('CATEGORY_')}
        if categories:
            print("\nCategory Performance:")
            sorted_cats = sorted(categories.items(), key=lambda x: x[1]['sensitivity'], reverse=True)
            for category, m in sorted_cats:
                print(f"  {category}: {m['sensitivity']*100:.1f}% sensitivity, {m['false_alarms_per_hour']:.2f} FA/h")
        
        motor_classes = {k.replace('MOTOR_', ''): v for k, v in metrics.items() if k.startswith('MOTOR_')}
        if motor_classes:
            print("\nMotor Classification Performance:")
            sorted_motor = sorted(motor_classes.items(), key=lambda x: x[1]['sensitivity'], reverse=True)
            for motor_class, m in sorted_motor:
                print(f"  {motor_class}: {m['sensitivity']*100:.1f}% sensitivity ({m['total_seizures']} seizures)")
        
        return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Analyze which seizure types are best detected using time_180s clustering"
    )
    parser.add_argument(
        "results_dir",
        help="Directory containing Madrid windowed results JSON files"
    )
    parser.add_argument(
        "-s", "--seizeit2-path",
        help="Path to SeizeIT2 dataset for annotations (optional)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        help="Output directory for analysis results"
    )
    parser.add_argument(
        "--pre-seizure-minutes",
        type=float,
        default=5.0,
        help="Minutes before seizure start to consider as detection window (default: 5.0)"
    )
    parser.add_argument(
        "--post-seizure-minutes",
        type=float,
        default=3.0,
        help="Minutes after seizure end to consider as detection window (default: 3.0)"
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = MadridClusteredSeizureTypeAnalyzer(
        results_dir=args.results_dir,
        seizeit2_data_path=args.seizeit2_path,
        output_dir=args.output_dir,
        pre_seizure_minutes=args.pre_seizure_minutes,
        post_seizure_minutes=args.post_seizure_minutes
    )
    
    # Run analysis
    results = analyzer.run_analysis()
    
    if results is None:
        print("Analysis failed.")
        return 1
    
    print("\nAnalysis completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())