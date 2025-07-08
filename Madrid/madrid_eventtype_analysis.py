#!/usr/bin/env python3
"""
Analysis script to evaluate Madrid metrics by eventType.

This script:
1. Loads Madrid clustering results from tolerance_adjusted_smart_clustered
2. Cross-references with original SeizeIT2 annotations to get eventType information
3. Calculates sensitivity, precision, and FAR metrics per eventType
4. Generates comparative visualizations
"""

import json
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from Information.Data.seizeit2_main.classes.annotation import Annotation

class MadridEventTypeAnalyzer:
    """Analyzer for Madrid results by seizure eventType."""
    
    def __init__(self, madrid_results_dir: str, seizeit2_data_path: str):
        """
        Initialize the analyzer.
        
        Args:
            madrid_results_dir: Path to Madrid results directory
            seizeit2_data_path: Path to SeizeIT2 dataset for annotations
        """
        self.madrid_results_dir = Path(madrid_results_dir)
        self.seizeit2_data_path = Path(seizeit2_data_path)
        self.results_by_eventtype = defaultdict(list)
        self.eventtype_metrics = {}
        
    def assign_seizure_category(self, event_type: str) -> str:
        """
        Assign seizure category based on event type.
        
        Args:
            event_type: Original event type string
            
        Returns:
            Category string
        """
        focal_mit_moption = [
            "sz_foc_a_m", "sz_foc_a_m_automatisms", "sz_foc_a_m_atonic", "sz_foc_a_m_clonic", 
            "sz_foc_a_m_spasms", "sz_foc_a_m_hyperkinetic", "sz_foc_a_m_myoclonic", "sz_foc_a_m_tonic", 
            "sz_foc_ia_m", "sz_foc_ia_m_automatisms", "sz_foc_ia_m_atonic", "sz_foc_ia_m_clonic", 
            "sz_foc_ia_m_spasms", "sz_foc_ia_m_hyperkinetic", "sz_foc_ia_m_myoclonic", "sz_foc_ia_m_tonic",
            "sz_foc_f2b", "sz_foc_a_um", "sz_foc_ia", "sz_foc_ua_m_hyperkinetic", "sz_foc_ua_m", "sz_foc_ua_m_tonic"
        ]
        focal_ohne_moption = [
            "sz_foc_a_nm", "sz_foc_ia_nm", "sz_foc_ia_nm_autonomic", "sz_foc_ia_nm_behavior", 
            "sz_foc_ia_nm_cognitive", "sz_foc_ia_nm_emotional", "sz_foc_ia_nm_sensory", 
            "sz_foc_ua_nm", "sz_foc_ua_nm_autonomic", "sz_foc_ua_nm_behavior", "sz_foc_ua_nm_cognitive", 
            "sz_foc_ua_nm_emotional", "sz_foc_ua_nm_sensory", "sz_foc_a_nm_behavior", "sz_foc_ia_um", "sz_foc_ia_um"
        ]
        gen_mit_moption = [
            "sz_gen_m", "sz_gen_m_tonicClonic", "sz_gen_m_clonic", "sz_gen_m_tonic", 
            "sz_gen_m_myoTC", "sz_gen_m_myoAtonic", "sz_gen_m_atonic", "sz_gen_m_spasms"
        ]
        gen_ohne_moption = [
            "sz_gen_nm", "sz_gen_nm_typical", "sz_gen_nm_atypical", "sz_gen_nm_myoclonic", 
            "sz_gen_nm_eyelidMyio"
        ]
        unknown_mit_moption = [
            "sz_uo_m", "sz_uo_m_myoclonic", "sz_uo_m_tonicClonic", "sz_uo_m_tonicMyio", 
            "sz_uo_m_hyperkinetic", "sz_uo_m_spasms"
        ]
        unknown_ohne_moption = [
            "sz_uo_nm", "sz_uo_nm_behavior"
        ]

        # Zuordnung zu den Hauptkategorien
        if event_type in focal_mit_moption:
            return 'focal_mit_moption'
        elif event_type in focal_ohne_moption:
            return 'focal_ohne_moption'
        elif event_type in gen_mit_moption:
            return 'gen_mit_moption'
        elif event_type in gen_ohne_moption:
            return 'gen_ohne_moption'
        elif event_type in unknown_mit_moption:
            return 'unknown_mit_moption'
        elif event_type in unknown_ohne_moption:
            return 'unknown_ohne_moption'
        else:
            return 'unclassified'
    
    def assign_seizure_category_v2(self, event_type: str) -> str:
        """
        Assign seizure category based on event type using v2 classification.
        
        Args:
            event_type: Original event type string
            
        Returns:
            Category string based on v2 classification
        """
        # Non-Seizure
        if event_type.startswith('bckg') or event_type.startswith('impd'):
            return 'Non-Seizure'
        
        # Seizure Unknown Classification (sz but not more specific)
        if event_type == 'sz':
            return 'Seizure Unknown Classification'
        
        # Focal Seizures - more granular subcategories
        if event_type.startswith('sz_foc_'):
            # Focal Aware
            if event_type.startswith(('sz_foc_a_m', 'sz_foc_a_nm', 'sz_foc_a_um')):
                return 'Focal Aware'
            # Focal Impaired-Awareness
            elif event_type.startswith(('sz_foc_ia_m', 'sz_foc_ia_nm', 'sz_foc_ia_um')):
                return 'Focal Impaired-Awareness'
            # Focal Unknown-Awareness
            elif event_type.startswith(('sz_foc_ua_m', 'sz_foc_ua_nm', 'sz_foc_ua_um')):
                return 'Focal Unknown-Awareness'
            # Focal to Bilateral TC
            elif event_type.startswith('sz_foc_f2b'):
                return 'Focal to Bilateral TC'
            # General Focal if no specific match
            else:
                return 'Focal Seizures'
        
        # Generalized Seizures
        if event_type.startswith(('sz_gen_m', 'sz_gen_nm')):
            return 'Generalized Seizures'
        
        # Seizures Unknown Onset
        if event_type.startswith(('sz_uo_m', 'sz_uo_nm')):
            return 'Seizures Unknown Onset'
        
        # Any other seizure type
        if event_type.startswith('sz'):
            return 'Other Seizures'
        
        return 'Unclassified'
        
    def load_madrid_results(self) -> Dict[str, Any]:
        """Load Madrid results - preferring clustered results if available."""
        results = {}
        
        # First, try to load clustered results (best representatives)
        clustered_results_file = self.madrid_results_dir / "clusters" / "best_representatives.json"
        if clustered_results_file.exists():
            print(f"Loading clustered results from: {clustered_results_file}")
            try:
                with open(clustered_results_file, 'r') as f:
                    clustered_data = json.load(f)
                    
                # Process clustered representatives
                if 'representatives' in clustered_data:
                    representatives = clustered_data['representatives']
                    
                    # Group representatives by file_id to reconstruct per-file results
                    file_groups = {}
                    for rep in representatives:
                        file_id = rep['file_id']
                        if file_id not in file_groups:
                            file_groups[file_id] = {
                                'anomalies': [],
                                'file_metadata': rep  # Use first rep for metadata
                            }
                        file_groups[file_id]['anomalies'].append(rep)
                    
                    # Convert back to expected format
                    for file_id, group_data in file_groups.items():
                        # Reconstruct the analysis_results structure
                        analysis_results = {
                            'ranked_anomalies': group_data['anomalies'],
                            'anomalies': group_data['anomalies']  # Backup field name
                        }
                        
                        # Create minimal result structure
                        result_data = {
                            'input_data': {
                                'subject_id': file_id.split('_')[0],
                                'run_id': file_id.split('_')[1], 
                                'seizure_id': '_'.join(file_id.split('_')[2:]),
                                'signal_metadata': {
                                    'total_duration_seconds': 3600  # Default 1 hour - adjust if available
                                }
                            },
                            'analysis_results': analysis_results
                        }
                        
                        results[file_id] = result_data
                        
                    print(f"Loaded {len(results)} file results from clustered data")
                    return results
                    
            except Exception as e:
                print(f"Error loading clustered results: {e}")
                print("Falling back to individual results...")
        
        # Fallback: Load individual seizure results
        individual_results_dir = self.madrid_results_dir 
        if individual_results_dir.exists():
            print(f"Loading individual results from: {individual_results_dir}")
            for json_file in individual_results_dir.glob("madrid_results_*.json"):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        # Extract subject, run, seizure info from filename or data
                        subject_id = data['input_data']['subject_id']
                        run_id = data['input_data']['run_id']
                        seizure_id = data['input_data']['seizure_id']
                        
                        key = f"{subject_id}_{run_id}_{seizure_id}"
                        results[key] = data
                        
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
                    
        return results
    
    def get_seizure_eventtype(self, subject_id: str, run_id: str, seizure_id: str) -> str:
        """
        Get eventType for a specific seizure from SeizeIT2 annotations.
        
        Args:
            subject_id: Subject identifier (e.g., 'sub-001')
            run_id: Run identifier (e.g., 'run-03') 
            seizure_id: Seizure identifier (e.g., 'seizure_00')
            
        Returns:
            eventType string or 'unknown' if not found
        """
        try:
            # Load annotation using the Annotation class
            annotation = Annotation.loadAnnotation(
                str(self.seizeit2_data_path),
                [subject_id, run_id]
            )
            
            # Extract seizure index from seizure_id
            seizure_idx = int(seizure_id.split('_')[-1])
            
            if seizure_idx < len(annotation.types):
                return annotation.types[seizure_idx]
            else:
                return 'unknown'
                
        except Exception as e:
            print(f"Error getting eventType for {subject_id}_{run_id}_{seizure_id}: {e}")
            return 'unknown'
    
    def calculate_metrics_by_eventtype(self, madrid_results: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
        """
        Calculate metrics (sensitivity, precision, FAR) by eventType, category, and category_v2.
        
        Args:
            madrid_results: Dictionary of Madrid results
            
        Returns:
            Tuple of (metrics_by_eventtype, metrics_by_category, metrics_by_category_v2)
        """
        eventtype_data = defaultdict(lambda: {
            'total_seizures': 0,
            'detected_seizures': 0,
            'true_positives': 0,
            'false_positives': 0,
            'total_anomalies': 0,
            'total_recording_hours': 0.0
        })
        
        category_data = defaultdict(lambda: {
            'total_seizures': 0,
            'detected_seizures': 0,
            'true_positives': 0,
            'false_positives': 0,
            'total_anomalies': 0,
            'total_recording_hours': 0.0
        })
        
        category_v2_data = defaultdict(lambda: {
            'total_seizures': 0,
            'detected_seizures': 0,
            'true_positives': 0,
            'false_positives': 0,
            'total_anomalies': 0,
            'total_recording_hours': 0.0
        })
        
        for key, result in madrid_results.items():
            # Parse subject, run, seizure from key
            parts = key.split('_')
            subject_id = parts[0]
            run_id = parts[1]
            seizure_id = '_'.join(parts[2:])
            
            # Get eventType and categories
            eventtype = self.get_seizure_eventtype(subject_id, run_id, seizure_id)
            category = self.assign_seizure_category(eventtype)
            category_v2 = self.assign_seizure_category_v2(eventtype)
            
            # Extract metrics from Madrid results
            if 'analysis_results' in result:
                analysis = result['analysis_results']
                
                # Get recording duration in hours
                recording_duration_hours = 0.0
                if 'input_data' in result and 'signal_metadata' in result['input_data']:
                    # Try to get duration from signal metadata
                    signal_meta = result['input_data']['signal_metadata']
                    if 'total_duration_seconds' in signal_meta:
                        recording_duration_hours = signal_meta['total_duration_seconds'] / 3600.0
                    elif 'recording_duration' in signal_meta:
                        recording_duration_hours = signal_meta['recording_duration'] / 3600.0
                
                # If no duration found, try to estimate from Madrid metadata
                if recording_duration_hours == 0.0 and 'madrid_parameters' in result:
                    # This is a fallback - you might need to adjust based on your data structure
                    print(f"Warning: No recording duration found for {key}, using default estimate")
                    recording_duration_hours = 1.0  # Default 1 hour if unknown
                
                eventtype_data[eventtype]['total_seizures'] += 1
                category_data[category]['total_seizures'] += 1
                category_v2_data[category_v2]['total_seizures'] += 1
                
                eventtype_data[eventtype]['total_recording_hours'] += recording_duration_hours
                category_data[category]['total_recording_hours'] += recording_duration_hours
                category_v2_data[category_v2]['total_recording_hours'] += recording_duration_hours
                
                # Check if seizure was detected (has any true positives)
                has_detection = False
                tp_count = 0
                fp_count = 0
                # Check both possible field names for anomalies
                anomalies_list = analysis.get('ranked_anomalies', []) or analysis.get('anomalies', [])
                total_anomalies = len(anomalies_list)
                
                for anomaly in anomalies_list:
                    if anomaly.get('seizure_hit', False):
                        tp_count += 1
                        has_detection = True
                    else:
                        fp_count += 1
                
                if has_detection:
                    eventtype_data[eventtype]['detected_seizures'] += 1
                    category_data[category]['detected_seizures'] += 1
                    category_v2_data[category_v2]['detected_seizures'] += 1
                
                eventtype_data[eventtype]['true_positives'] += tp_count
                eventtype_data[eventtype]['false_positives'] += fp_count
                eventtype_data[eventtype]['total_anomalies'] += total_anomalies
                
                category_data[category]['true_positives'] += tp_count
                category_data[category]['false_positives'] += fp_count
                category_data[category]['total_anomalies'] += total_anomalies
                
                category_v2_data[category_v2]['true_positives'] += tp_count
                category_v2_data[category_v2]['false_positives'] += fp_count
                category_v2_data[category_v2]['total_anomalies'] += total_anomalies
        
        # Calculate final metrics for event types
        metrics_by_eventtype = {}
        for eventtype, data in eventtype_data.items():
            metrics = self._calculate_metrics_from_data(data)
            metrics_by_eventtype[eventtype] = metrics
        
        # Calculate final metrics for categories
        metrics_by_category = {}
        for category, data in category_data.items():
            metrics = self._calculate_metrics_from_data(data)
            metrics_by_category[category] = metrics
        
        # Calculate final metrics for categories v2
        metrics_by_category_v2 = {}
        for category_v2, data in category_v2_data.items():
            metrics = self._calculate_metrics_from_data(data)
            metrics_by_category_v2[category_v2] = metrics
            
        return metrics_by_eventtype, metrics_by_category, metrics_by_category_v2
    
    def _calculate_metrics_from_data(self, data: Dict) -> Dict[str, float]:
        """Helper method to calculate metrics from data dictionary."""
        metrics = {}
        
        # Sensitivity (recall) = TP / (TP + FN) = detected_seizures / total_seizures
        metrics['sensitivity'] = data['detected_seizures'] / data['total_seizures'] if data['total_seizures'] > 0 else 0
        
        # Precision = TP / (TP + FP)
        total_predictions = data['true_positives'] + data['false_positives']
        metrics['precision'] = data['true_positives'] / total_predictions if total_predictions > 0 else 0
        
        # False Alarm Rate = FP per hour (FA/h)
        if data['total_recording_hours'] > 0:
            metrics['false_alarm_rate'] = data['false_positives'] / data['total_recording_hours']
        else:
            metrics['false_alarm_rate'] = 0.0
        
        # Additional metrics
        metrics['total_seizures'] = data['total_seizures']
        metrics['detected_seizures'] = data['detected_seizures']
        metrics['true_positives'] = data['true_positives']
        metrics['false_positives'] = data['false_positives']
        metrics['total_anomalies'] = data['total_anomalies']
        metrics['total_recording_hours'] = data['total_recording_hours']
        
        return metrics
    
    def create_visualizations(self, metrics_by_eventtype: Dict[str, Dict[str, float]], 
                            metrics_by_category: Dict[str, Dict[str, float]], 
                            metrics_by_category_v2: Dict[str, Dict[str, float]], output_dir: str):
        """Create visualizations comparing metrics across eventTypes and categories."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Prepare data for plotting - EventTypes
        eventtypes = list(metrics_by_eventtype.keys())
        sensitivities = [metrics_by_eventtype[et]['sensitivity'] for et in eventtypes]
        precisions = [metrics_by_eventtype[et]['precision'] for et in eventtypes]
        false_alarm_rates = [metrics_by_eventtype[et]['false_alarm_rate'] for et in eventtypes]
        total_seizures = [metrics_by_eventtype[et]['total_seizures'] for et in eventtypes]
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Bar plot comparing all metrics by EventType
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Madrid Metrics by Seizure EventType', fontsize=16, fontweight='bold')
        
        # Sensitivity
        axes[0, 0].bar(eventtypes, sensitivities, alpha=0.7)
        axes[0, 0].set_title('Sensitivity by EventType')
        axes[0, 0].set_ylabel('Sensitivity')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Precision
        axes[0, 1].bar(eventtypes, precisions, alpha=0.7, color='orange')
        axes[0, 1].set_title('Precision by EventType')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # False Alarm Rate (FA/h)
        axes[1, 0].bar(eventtypes, false_alarm_rates, alpha=0.7, color='red')
        axes[1, 0].set_title('False Alarm Rate (FA/h) by EventType')
        axes[1, 0].set_ylabel('False Alarms per Hour')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Total seizures (sample size)
        axes[1, 1].bar(eventtypes, total_seizures, alpha=0.7, color='green')
        axes[1, 1].set_title('Sample Size by EventType')
        axes[1, 1].set_ylabel('Number of Seizures')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'madrid_metrics_by_eventtype.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. NEW: Bar plot comparing all metrics by Category
        categories = list(metrics_by_category.keys())
        cat_sensitivities = [metrics_by_category[cat]['sensitivity'] for cat in categories]
        cat_precisions = [metrics_by_category[cat]['precision'] for cat in categories]
        cat_false_alarm_rates = [metrics_by_category[cat]['false_alarm_rate'] for cat in categories]
        cat_total_seizures = [metrics_by_category[cat]['total_seizures'] for cat in categories]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Madrid Metrics by Seizure Category', fontsize=16, fontweight='bold')
        
        # Sensitivity
        axes[0, 0].bar(categories, cat_sensitivities, alpha=0.7)
        axes[0, 0].set_title('Sensitivity by Category')
        axes[0, 0].set_ylabel('Sensitivity')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Precision
        axes[0, 1].bar(categories, cat_precisions, alpha=0.7, color='orange')
        axes[0, 1].set_title('Precision by Category')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # False Alarm Rate (FA/h)
        axes[1, 0].bar(categories, cat_false_alarm_rates, alpha=0.7, color='red')
        axes[1, 0].set_title('False Alarm Rate (FA/h) by Category')
        axes[1, 0].set_ylabel('False Alarms per Hour')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Total seizures (sample size)
        axes[1, 1].bar(categories, cat_total_seizures, alpha=0.7, color='green')
        axes[1, 1].set_title('Sample Size by Category')
        axes[1, 1].set_ylabel('Number of Seizures')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'madrid_metrics_by_category.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Sensitivity vs False Alarm Rate scatter plot - EventType
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot with point sizes based on sample size
        sizes = [max(50, n * 5) for n in total_seizures]  # Scale point sizes
        scatter = plt.scatter(false_alarm_rates, sensitivities, s=sizes, alpha=0.7, c=range(len(eventtypes)), cmap='tab10')
        
        # Add labels for each point
        for i, eventtype in enumerate(eventtypes):
            plt.annotate(f'{eventtype}\n(n={total_seizures[i]})', 
                        (false_alarm_rates[i], sensitivities[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, ha='left')
        
        plt.xlabel('False Alarm Rate (FA/h)')
        plt.ylabel('Sensitivity')
        plt.title('Madrid Performance: Sensitivity vs False Alarm Rate (FA/h) by EventType')
        plt.grid(True, alpha=0.3)
        
        # Add ideal performance reference lines
        plt.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Target Sensitivity (80%)')
        plt.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5, label='Target FAR (0.2/h)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'madrid_sensitivity_vs_far_by_eventtype.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. NEW: Sensitivity vs False Alarm Rate scatter plot - Category
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot with point sizes based on sample size
        cat_sizes = [max(50, n * 5) for n in cat_total_seizures]  # Scale point sizes
        scatter = plt.scatter(cat_false_alarm_rates, cat_sensitivities, s=cat_sizes, alpha=0.7, c=range(len(categories)), cmap='viridis')
        
        # Add labels for each point
        for i, category in enumerate(categories):
            plt.annotate(f'{category}\n(n={cat_total_seizures[i]})', 
                        (cat_false_alarm_rates[i], cat_sensitivities[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, ha='left')
        
        plt.xlabel('False Alarm Rate (FA/h)')
        plt.ylabel('Sensitivity')
        plt.title('Madrid Performance: Sensitivity vs False Alarm Rate (FA/h) by Category')
        plt.grid(True, alpha=0.3)
        
        # Add ideal performance reference lines
        plt.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Target Sensitivity (80%)')
        plt.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5, label='Target FAR (0.2/h)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'madrid_sensitivity_vs_far_by_category.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. NEW: Bar plot comparing all metrics by Category V2
        categories_v2 = list(metrics_by_category_v2.keys())
        cat_v2_sensitivities = [metrics_by_category_v2[cat]['sensitivity'] for cat in categories_v2]
        cat_v2_precisions = [metrics_by_category_v2[cat]['precision'] for cat in categories_v2]
        cat_v2_false_alarm_rates = [metrics_by_category_v2[cat]['false_alarm_rate'] for cat in categories_v2]
        cat_v2_total_seizures = [metrics_by_category_v2[cat]['total_seizures'] for cat in categories_v2]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Madrid Metrics by Seizure Category V2', fontsize=16, fontweight='bold')
        
        # Sensitivity
        axes[0, 0].bar(categories_v2, cat_v2_sensitivities, alpha=0.7)
        axes[0, 0].set_title('Sensitivity by Category V2')
        axes[0, 0].set_ylabel('Sensitivity')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Precision
        axes[0, 1].bar(categories_v2, cat_v2_precisions, alpha=0.7, color='orange')
        axes[0, 1].set_title('Precision by Category V2')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # False Alarm Rate (FA/h)
        axes[1, 0].bar(categories_v2, cat_v2_false_alarm_rates, alpha=0.7, color='red')
        axes[1, 0].set_title('False Alarm Rate (FA/h) by Category V2')
        axes[1, 0].set_ylabel('False Alarms per Hour')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Total seizures (sample size)
        axes[1, 1].bar(categories_v2, cat_v2_total_seizures, alpha=0.7, color='green')
        axes[1, 1].set_title('Sample Size by Category V2')
        axes[1, 1].set_ylabel('Number of Seizures')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'madrid_metrics_by_category_v2.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. NEW: Sensitivity vs False Alarm Rate scatter plot - Category V2
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot with point sizes based on sample size
        cat_v2_sizes = [max(50, n * 5) for n in cat_v2_total_seizures]  # Scale point sizes
        scatter = plt.scatter(cat_v2_false_alarm_rates, cat_v2_sensitivities, s=cat_v2_sizes, 
                             alpha=0.7, c=range(len(categories_v2)), cmap='plasma')
        
        # Add labels for each point
        for i, category_v2 in enumerate(categories_v2):
            plt.annotate(f'{category_v2}\n(n={cat_v2_total_seizures[i]})', 
                        (cat_v2_false_alarm_rates[i], cat_v2_sensitivities[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, ha='left')
        
        plt.xlabel('False Alarm Rate (FA/h)')
        plt.ylabel('Sensitivity')
        plt.title('Madrid Performance: Sensitivity vs False Alarm Rate (FA/h) by Category V2')
        plt.grid(True, alpha=0.3)
        
        # Add ideal performance reference lines
        plt.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Target Sensitivity (80%)')
        plt.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5, label='Target FAR (0.2/h)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'madrid_sensitivity_vs_far_by_category_v2.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_path}")
    
    def generate_report(self, metrics_by_eventtype: Dict[str, Dict[str, float]], 
                       metrics_by_category: Dict[str, Dict[str, float]], 
                       metrics_by_category_v2: Dict[str, Dict[str, float]], output_dir: str):
        """Generate detailed text report of metrics by eventType and category."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        report_file = output_path / 'madrid_eventtype_analysis_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("Madrid Seizure Detection: Analysis by EventType and Categories\n")
            f.write("=" * 65 + "\n\n")
            
            # Sort by total seizures for better readability
            sorted_eventtypes = sorted(metrics_by_eventtype.items(), 
                                     key=lambda x: x[1]['total_seizures'], reverse=True)
            
            f.write("SUMMARY BY EVENTTYPE:\n")
            f.write("-" * 30 + "\n")
            
            for eventtype, metrics in sorted_eventtypes:
                f.write(f"\nEventType: {eventtype}\n")
                f.write(f"  Total Seizures: {metrics['total_seizures']}\n")
                f.write(f"  Detected Seizures: {metrics['detected_seizures']}\n")
                f.write(f"  Sensitivity: {metrics['sensitivity']:.3f} ({metrics['sensitivity']*100:.1f}%)\n")
                f.write(f"  Precision: {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)\n")
                f.write(f"  False Alarm Rate (FA/h): {metrics['false_alarm_rate']:.3f}\n")
                f.write(f"  True Positives: {metrics['true_positives']}\n")
                f.write(f"  False Positives: {metrics['false_positives']}\n")
                f.write(f"  Total Anomalies: {metrics['total_anomalies']}\n")
                f.write(f"  Total Recording Hours: {metrics['total_recording_hours']:.1f}\n")
            
            # Summary by Category
            f.write(f"\n\nSUMMARY BY CATEGORY (V1):\n")
            f.write("-" * 30 + "\n")
            
            sorted_categories = sorted(metrics_by_category.items(), 
                                     key=lambda x: x[1]['total_seizures'], reverse=True)
            
            for category, metrics in sorted_categories:
                f.write(f"\nCategory: {category}\n")
                f.write(f"  Total Seizures: {metrics['total_seizures']}\n")
                f.write(f"  Detected Seizures: {metrics['detected_seizures']}\n")
                f.write(f"  Sensitivity: {metrics['sensitivity']:.3f} ({metrics['sensitivity']*100:.1f}%)\n")
                f.write(f"  Precision: {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)\n")
                f.write(f"  False Alarm Rate (FA/h): {metrics['false_alarm_rate']:.3f}\n")
                f.write(f"  True Positives: {metrics['true_positives']}\n")
                f.write(f"  False Positives: {metrics['false_positives']}\n")
                f.write(f"  Total Anomalies: {metrics['total_anomalies']}\n")
                f.write(f"  Total Recording Hours: {metrics['total_recording_hours']:.1f}\n")
            
            # NEW: Summary by Category V2
            f.write(f"\n\nSUMMARY BY CATEGORY (V2):\n")
            f.write("-" * 30 + "\n")
            
            sorted_categories_v2 = sorted(metrics_by_category_v2.items(), 
                                        key=lambda x: x[1]['total_seizures'], reverse=True)
            
            for category_v2, metrics in sorted_categories_v2:
                f.write(f"\nCategory V2: {category_v2}\n")
                f.write(f"  Total Seizures: {metrics['total_seizures']}\n")
                f.write(f"  Detected Seizures: {metrics['detected_seizures']}\n")
                f.write(f"  Sensitivity: {metrics['sensitivity']:.3f} ({metrics['sensitivity']*100:.1f}%)\n")
                f.write(f"  Precision: {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)\n")
                f.write(f"  False Alarm Rate (FA/h): {metrics['false_alarm_rate']:.3f}\n")
                f.write(f"  True Positives: {metrics['true_positives']}\n")
                f.write(f"  False Positives: {metrics['false_positives']}\n")
                f.write(f"  Total Anomalies: {metrics['total_anomalies']}\n")
                f.write(f"  Total Recording Hours: {metrics['total_recording_hours']:.1f}\n")
            
            # Overall statistics
            f.write(f"\n\nOVERALL STATISTICS:\n")
            f.write("-" * 20 + "\n")
            
            total_seizures = sum(m['total_seizures'] for m in metrics_by_eventtype.values())
            total_detected = sum(m['detected_seizures'] for m in metrics_by_eventtype.values())
            total_tp = sum(m['true_positives'] for m in metrics_by_eventtype.values())
            total_fp = sum(m['false_positives'] for m in metrics_by_eventtype.values())
            
            overall_sensitivity = total_detected / total_seizures if total_seizures > 0 else 0
            overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            # Calculate overall FA/h
            total_recording_hours = sum(m['total_recording_hours'] for m in metrics_by_eventtype.values())
            overall_far_per_hour = total_fp / total_recording_hours if total_recording_hours > 0 else 0
            
            f.write(f"Total Seizures Analyzed: {total_seizures}\n")
            f.write(f"Total EventTypes Found: {len(metrics_by_eventtype)}\n")
            f.write(f"Total Categories V1 Found: {len(metrics_by_category)}\n")
            f.write(f"Total Categories V2 Found: {len(metrics_by_category_v2)}\n")
            f.write(f"Total Recording Hours: {total_recording_hours:.1f}\n")
            f.write(f"Overall Sensitivity: {overall_sensitivity:.3f} ({overall_sensitivity*100:.1f}%)\n")
            f.write(f"Overall Precision: {overall_precision:.3f} ({overall_precision*100:.1f}%)\n")
            f.write(f"Overall False Alarm Rate (FA/h): {overall_far_per_hour:.3f}\n")
            
        print(f"Report saved to {report_file}")
    
    def run_analysis(self, output_dir: str = "madrid_eventtype_analysis"):
        """Run complete analysis pipeline."""
        print("Loading Madrid results...")
        madrid_results = self.load_madrid_results()
        print(f"Loaded {len(madrid_results)} Madrid result files")
        
        print("Calculating metrics by eventType, category, and category v2...")
        metrics_by_eventtype, metrics_by_category, metrics_by_category_v2 = self.calculate_metrics_by_eventtype(madrid_results)
        print(f"Found {len(metrics_by_eventtype)} different eventTypes")
        print(f"Found {len(metrics_by_category)} different categories (v1)")
        print(f"Found {len(metrics_by_category_v2)} different categories (v2)")
        
        print("Generating visualizations...")
        self.create_visualizations(metrics_by_eventtype, metrics_by_category, metrics_by_category_v2, output_dir)
        
        print("Generating report...")
        self.generate_report(metrics_by_eventtype, metrics_by_category, metrics_by_category_v2, output_dir)
        
        return metrics_by_eventtype, metrics_by_category, metrics_by_category_v2

def main():
    """Main execution function."""
    # Configuration
    madrid_results_dir = "Madrid/madrid_results/madrid_seizure_results_parallel_400/tolerance_adjusted"
    seizeit2_data_path = "/home/swolf/asim_shared/raw_data/ds005873-1.1.0"  
    output_dir = "Madrid/madrid_results/madrid_seizure_results_parallel_400/madrid_eventtype_analysis"
    
    # Initialize analyzer
    analyzer = MadridEventTypeAnalyzer(madrid_results_dir, seizeit2_data_path)
    
    # Run analysis
    try:
        results = analyzer.run_analysis(output_dir)
        print("\nAnalysis completed successfully!")
        print(f"Results saved to: {output_dir}/")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()