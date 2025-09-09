#!/usr/bin/env python3
"""
Madrid Responder Analysis

This script analyzes the differences in metrics between Responders and Non-Responders
based on Madrid clustering results.

Responders: Patients where >= 2/3 (66.7%) of seizures are detected
Non-Responders: Patients where < 2/3 (66.7%) of seizures are detected
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter
import statistics

# Try to import plotting libraries with fallback
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib not available. Plotting will be disabled.")
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    print("Warning: seaborn not available. Using default styling.")
    SEABORN_AVAILABLE = False


class MadridResponderAnalyzer:
    """Analyzer for Responder vs Non-Responder comparison."""
    
    def __init__(self, clustering_results_folder: str, output_folder: str = None, responder_threshold: float = 2/3):
        self.clustering_results_folder = Path(clustering_results_folder)
        if output_folder:
            self.output_folder = Path(output_folder)
        else:
            input_path = Path(clustering_results_folder)
            self.output_folder = input_path.parent / f"{input_path.name}_responder_analysis"
        self.output_folder.mkdir(exist_ok=True, parents=True)
        
        # Create subfolders
        (self.output_folder / "patient_analysis").mkdir(exist_ok=True)
        (self.output_folder / "group_comparison").mkdir(exist_ok=True)
        (self.output_folder / "visualizations").mkdir(exist_ok=True)
        
        self.responder_threshold = responder_threshold
        self.representatives = []
        self.original_data = []  # Store original data files for patient analysis
        
        self.responders = {}
        self.non_responders = {}
        
        # Set up plotting style if available
        if MATPLOTLIB_AVAILABLE:
            plt.style.use('default')
        if SEABORN_AVAILABLE:
            sns.set_palette("husl")
            
    def load_clustering_results(self) -> None:
        """Load clustering results and original data."""
        print(f"Loading clustering results from: {self.clustering_results_folder}")
        
        # Load representatives
        representatives_file = self.clustering_results_folder / "clusters" / "best_representatives.json"
        if not representatives_file.exists():
            raise FileNotFoundError(f"Representatives file not found: {representatives_file}")
            
        with open(representatives_file, 'r') as f:
            data = json.load(f)
            self.representatives = data['representatives']
            
        print(f"Loaded {len(self.representatives)} representatives")
        
        # Try to load original Madrid results for comprehensive patient analysis
        self._load_original_madrid_data()
        
    def _load_original_madrid_data(self) -> None:
        """Load original Madrid result files to get complete patient data."""
        # Look for original Madrid results in parent directories
        possible_paths = [
            self.clustering_results_folder.parent / "madrid_dir_400_examples_tolerance",
            self.clustering_results_folder.parent / "madrid_dir_400_examples",
            self.clustering_results_folder.parent.parent / "madrid_dir_400_examples_tolerance",
            self.clustering_results_folder.parent.parent / "madrid_dir_400_examples"
        ]
        
        original_data_path = None
        for path in possible_paths:
            if path.exists() and list(path.glob("madrid_results_*.json")):
                original_data_path = path
                break
                
        if original_data_path:
            print(f"Loading original Madrid data from: {original_data_path}")
            json_files = list(original_data_path.glob("madrid_results_*.json"))
            
            for file_path in json_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        self.original_data.append(data)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    
            print(f"Loaded {len(self.original_data)} original Madrid result files")
        else:
            print("Warning: Could not find original Madrid data. Analysis will be based on representatives only.")
            
    def calculate_patient_seizure_detection_rates(self) -> Dict[str, Dict]:
        """Calculate seizure detection rates per patient."""
        patient_stats = defaultdict(lambda: {
            'total_seizure_files': 0,
            'detected_seizure_files': 0,
            'total_anomalies': 0,
            'true_positives': 0,
            'false_positives': 0,
            'seizure_files': [],
            'detection_rate': 0.0
        })
        
        # Use original data if available, otherwise use representatives
        data_source = self.original_data if self.original_data else []
        
        if data_source:
            print("Calculating patient detection rates from original data...")
            for data in data_source:
                subject_id = data['input_data']['subject_id']
                seizure_present = data['validation_data']['ground_truth']['seizure_present']
                has_tp = data['analysis_results']['performance_metrics']['true_positives'] > 0
                
                if seizure_present:
                    patient_stats[subject_id]['total_seizure_files'] += 1
                    patient_stats[subject_id]['seizure_files'].append({
                        'file_id': f"{subject_id}_{data['input_data']['run_id']}_{data['input_data']['seizure_id']}",
                        'detected': has_tp,
                        'tp_count': data['analysis_results']['performance_metrics']['true_positives'],
                        'fp_count': data['analysis_results']['performance_metrics']['false_positives']
                    })
                    
                    if has_tp:
                        patient_stats[subject_id]['detected_seizure_files'] += 1
                        
                patient_stats[subject_id]['total_anomalies'] += data['analysis_results']['performance_metrics'].get('total_anomalies_detected', 0)
                patient_stats[subject_id]['true_positives'] += data['analysis_results']['performance_metrics']['true_positives']
                patient_stats[subject_id]['false_positives'] += data['analysis_results']['performance_metrics']['false_positives']
        else:
            print("Calculating patient detection rates from clustering representatives...")
            # Fallback: analyze from representatives
            file_stats = defaultdict(lambda: {'has_seizure': False, 'has_tp': False})
            
            for rep in self.representatives:
                subject_id = rep['subject_id']
                file_id = rep['file_id']
                seizure_present = rep.get('seizure_present', False)
                seizure_hit = rep.get('seizure_hit', False)
                
                file_stats[file_id]['has_seizure'] = seizure_present
                if seizure_hit:
                    file_stats[file_id]['has_tp'] = True
                    
                patient_stats[subject_id]['total_anomalies'] += 1
                if seizure_hit:
                    patient_stats[subject_id]['true_positives'] += 1
                else:
                    patient_stats[subject_id]['false_positives'] += 1
                    
            # Calculate file-level detection rates
            subject_files = defaultdict(list)
            for file_id, stats in file_stats.items():
                subject_id = file_id.split('_')[0]
                subject_files[subject_id].append(stats)
                
            for subject_id, files in subject_files.items():
                seizure_files = [f for f in files if f['has_seizure']]
                detected_files = [f for f in seizure_files if f['has_tp']]
                
                patient_stats[subject_id]['total_seizure_files'] = len(seizure_files)
                patient_stats[subject_id]['detected_seizure_files'] = len(detected_files)
                
        # Calculate detection rates
        for subject_id in patient_stats:
            if patient_stats[subject_id]['total_seizure_files'] > 0:
                patient_stats[subject_id]['detection_rate'] = (
                    patient_stats[subject_id]['detected_seizure_files'] / 
                    patient_stats[subject_id]['total_seizure_files']
                )
                
        return dict(patient_stats)
        
    def classify_responders(self, patient_stats: Dict[str, Dict]) -> Tuple[Dict, Dict]:
        """Classify patients as Responders or Non-Responders."""
        responders = {}
        non_responders = {}
        
        for subject_id, stats in patient_stats.items():
            if stats['detection_rate'] >= self.responder_threshold:
                responders[subject_id] = stats
            else:
                non_responders[subject_id] = stats
                
        print(f"Classification (threshold: {self.responder_threshold:.1%}):")
        print(f"  Responders: {len(responders)} patients")
        print(f"  Non-Responders: {len(non_responders)} patients")
        
        return responders, non_responders
        
    def analyze_group_metrics(self, responders: Dict, non_responders: Dict) -> Dict[str, Any]:
        """Analyze and compare metrics between groups."""
        
        def calculate_group_stats(group: Dict, group_name: str) -> Dict:
            if not group:
                return {
                    'name': group_name,
                    'count': 0,
                    'detection_rates': [],
                    'mean_detection_rate': 0,
                    'total_anomalies': 0,
                    'total_tp': 0,
                    'total_fp': 0,
                    'precision': 0,
                    'seizure_files_detected': 0,
                    'total_seizure_files': 0,
                    'overall_sensitivity': 0
                }
                
            detection_rates = [stats['detection_rate'] for stats in group.values()]
            total_anomalies = sum(stats['total_anomalies'] for stats in group.values())
            total_tp = sum(stats['true_positives'] for stats in group.values())
            total_fp = sum(stats['false_positives'] for stats in group.values())
            seizure_files_detected = sum(stats['detected_seizure_files'] for stats in group.values())
            total_seizure_files = sum(stats['total_seizure_files'] for stats in group.values())
            
            return {
                'name': group_name,
                'count': len(group),
                'detection_rates': detection_rates,
                'mean_detection_rate': statistics.mean(detection_rates) if detection_rates else 0,
                'median_detection_rate': statistics.median(detection_rates) if detection_rates else 0,
                'std_detection_rate': statistics.stdev(detection_rates) if len(detection_rates) > 1 else 0,
                'min_detection_rate': min(detection_rates) if detection_rates else 0,
                'max_detection_rate': max(detection_rates) if detection_rates else 0,
                'total_anomalies': total_anomalies,
                'total_tp': total_tp,
                'total_fp': total_fp,
                'precision': total_tp / total_anomalies if total_anomalies > 0 else 0,
                'seizure_files_detected': seizure_files_detected,
                'total_seizure_files': total_seizure_files,
                'overall_sensitivity': seizure_files_detected / total_seizure_files if total_seizure_files > 0 else 0
            }
            
        responder_stats = calculate_group_stats(responders, 'Responders')
        non_responder_stats = calculate_group_stats(non_responders, 'Non-Responders')
        
        # Calculate differences and comparative metrics
        comparison = {
            'responders': responder_stats,
            'non_responders': non_responder_stats,
            'differences': {
                'mean_detection_rate_diff': responder_stats['mean_detection_rate'] - non_responder_stats['mean_detection_rate'],
                'precision_diff': responder_stats['precision'] - non_responder_stats['precision'],
                'sensitivity_diff': responder_stats['overall_sensitivity'] - non_responder_stats['overall_sensitivity'],
                'total_patients': len(responders) + len(non_responders),
                'responder_percentage': len(responders) / (len(responders) + len(non_responders)) * 100 if (len(responders) + len(non_responders)) > 0 else 0
            }
        }
        
        return comparison
        
    def analyze_cluster_characteristics(self, responders: Dict, non_responders: Dict) -> Dict[str, Any]:
        """Analyze cluster characteristics for each group."""
        responder_reps = [rep for rep in self.representatives if rep['subject_id'] in responders]
        non_responder_reps = [rep for rep in self.representatives if rep['subject_id'] in non_responders]
        
        def analyze_rep_group(reps: List[Dict], group_name: str) -> Dict:
            if not reps:
                return {'name': group_name, 'count': 0}
                
            cluster_sizes = [rep['cluster_size'] for rep in reps]
            anomaly_scores = [rep['anomaly_score'] for rep in reps]
            tp_counts = [rep.get('cluster_tp_count', 0) for rep in reps]
            seizure_hits = [rep for rep in reps if rep.get('seizure_hit', False)]
            
            return {
                'name': group_name,
                'count': len(reps),
                'cluster_sizes': {
                    'mean': statistics.mean(cluster_sizes),
                    'median': statistics.median(cluster_sizes),
                    'std': statistics.stdev(cluster_sizes) if len(cluster_sizes) > 1 else 0,
                    'min': min(cluster_sizes),
                    'max': max(cluster_sizes)
                },
                'anomaly_scores': {
                    'mean': statistics.mean(anomaly_scores),
                    'median': statistics.median(anomaly_scores),
                    'std': statistics.stdev(anomaly_scores) if len(anomaly_scores) > 1 else 0,
                    'min': min(anomaly_scores),
                    'max': max(anomaly_scores)
                },
                'cluster_tp_counts': {
                    'mean': statistics.mean(tp_counts) if tp_counts else 0,
                    'median': statistics.median(tp_counts) if tp_counts else 0,
                    'total': sum(tp_counts)
                },
                'seizure_hit_rate': len(seizure_hits) / len(reps) if reps else 0,
                'representatives_per_patient': len(reps) / len(set(rep['subject_id'] for rep in reps)) if reps else 0
            }
            
        responder_cluster_analysis = analyze_rep_group(responder_reps, 'Responders')
        non_responder_cluster_analysis = analyze_rep_group(non_responder_reps, 'Non-Responders')
        
        return {
            'responders': responder_cluster_analysis,
            'non_responders': non_responder_cluster_analysis,
            'differences': {
                'mean_cluster_size_diff': (responder_cluster_analysis['cluster_sizes']['mean'] - 
                                         non_responder_cluster_analysis['cluster_sizes']['mean']) if responder_cluster_analysis['count'] > 0 and non_responder_cluster_analysis['count'] > 0 else 0,
                'mean_score_diff': (responder_cluster_analysis['anomaly_scores']['mean'] - 
                                  non_responder_cluster_analysis['anomaly_scores']['mean']) if responder_cluster_analysis['count'] > 0 and non_responder_cluster_analysis['count'] > 0 else 0,
                'seizure_hit_rate_diff': (responder_cluster_analysis['seizure_hit_rate'] - 
                                        non_responder_cluster_analysis['seizure_hit_rate']) if responder_cluster_analysis['count'] > 0 and non_responder_cluster_analysis['count'] > 0 else 0
            }
        }
        
    def create_visualizations(self, group_comparison: Dict, cluster_analysis: Dict) -> None:
        """Create comprehensive visualizations comparing groups."""
        
        if not MATPLOTLIB_AVAILABLE:
            print("Skipping visualizations - matplotlib not available")
            return
            
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Patient Detection Rates Distribution
        ax1 = plt.subplot(3, 3, 1)
        responder_rates = group_comparison['responders']['detection_rates']
        non_responder_rates = group_comparison['non_responders']['detection_rates']
        
        if responder_rates and non_responder_rates:
            ax1.hist([responder_rates, non_responder_rates], bins=20, alpha=0.7, 
                    label=['Responders', 'Non-Responders'], color=['green', 'red'])
        elif responder_rates:
            ax1.hist(responder_rates, bins=20, alpha=0.7, label='Responders', color='green')
        elif non_responder_rates:
            ax1.hist(non_responder_rates, bins=20, alpha=0.7, label='Non-Responders', color='red')
            
        ax1.axvline(x=self.responder_threshold, color='black', linestyle='--', 
                   label=f'Threshold ({self.responder_threshold:.1%})')
        ax1.set_xlabel('Detection Rate')
        ax1.set_ylabel('Number of Patients')
        ax1.set_title('Patient Detection Rate Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Group Comparison Bar Chart
        ax2 = plt.subplot(3, 3, 2)
        metrics = ['Mean Detection Rate', 'Precision', 'Overall Sensitivity']
        responder_values = [
            group_comparison['responders']['mean_detection_rate'],
            group_comparison['responders']['precision'],
            group_comparison['responders']['overall_sensitivity']
        ]
        non_responder_values = [
            group_comparison['non_responders']['mean_detection_rate'],
            group_comparison['non_responders']['precision'],
            group_comparison['non_responders']['overall_sensitivity']
        ]
        
        x = range(len(metrics))
        width = 0.35
        ax2.bar([i - width/2 for i in x], responder_values, width, 
               label='Responders', color='green', alpha=0.7)
        ax2.bar([i + width/2 for i in x], non_responder_values, width,
               label='Non-Responders', color='red', alpha=0.7)
        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Value')
        ax2.set_title('Group Performance Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Cluster Size Comparison
        ax3 = plt.subplot(3, 3, 3)
        responder_reps = [rep for rep in self.representatives if rep['subject_id'] in self.responders]
        non_responder_reps = [rep for rep in self.representatives if rep['subject_id'] in self.non_responders]
        
        if responder_reps and non_responder_reps:
            responder_sizes = [rep['cluster_size'] for rep in responder_reps]
            non_responder_sizes = [rep['cluster_size'] for rep in non_responder_reps]
            
            box_data = [responder_sizes, non_responder_sizes]
            box_plot = ax3.boxplot(box_data, labels=['Responders', 'Non-Responders'], patch_artist=True)
            box_plot['boxes'][0].set_facecolor('green')
            box_plot['boxes'][1].set_facecolor('red')
            
        ax3.set_ylabel('Cluster Size')
        ax3.set_title('Cluster Size Distribution by Group')
        ax3.grid(True, alpha=0.3)
        
        # 4. Anomaly Score Comparison
        ax4 = plt.subplot(3, 3, 4)
        if responder_reps and non_responder_reps:
            responder_scores = [rep['anomaly_score'] for rep in responder_reps]
            non_responder_scores = [rep['anomaly_score'] for rep in non_responder_reps]
            
            ax4.hist([responder_scores, non_responder_scores], bins=30, alpha=0.7,
                    label=['Responders', 'Non-Responders'], color=['green', 'red'])
                    
        ax4.set_xlabel('Anomaly Score')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Anomaly Score Distribution by Group')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Patient Count Pie Chart
        ax5 = plt.subplot(3, 3, 5)
        sizes = [len(self.responders), len(self.non_responders)]
        labels = ['Responders', 'Non-Responders']
        colors = ['green', 'red']
        ax5.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax5.set_title('Patient Distribution')
        
        # 6. Seizure Hit Rate by Group
        ax6 = plt.subplot(3, 3, 6)
        if cluster_analysis['responders']['count'] > 0 and cluster_analysis['non_responders']['count'] > 0:
            hit_rates = [
                cluster_analysis['responders']['seizure_hit_rate'],
                cluster_analysis['non_responders']['seizure_hit_rate']
            ]
            ax6.bar(['Responders', 'Non-Responders'], hit_rates, 
                   color=['green', 'red'], alpha=0.7)
                   
        ax6.set_ylabel('Seizure Hit Rate')
        ax6.set_title('Seizure Hit Rate by Group')
        ax6.grid(True, alpha=0.3)
        
        # 7. Detection Rate vs Cluster Size Scatter
        ax7 = plt.subplot(3, 3, 7)
        patient_detection_rates = []
        patient_mean_cluster_sizes = []
        patient_colors = []
        
        for subject_id in set(rep['subject_id'] for rep in self.representatives):
            patient_reps = [rep for rep in self.representatives if rep['subject_id'] == subject_id]
            if subject_id in self.responders:
                detection_rate = self.responders[subject_id]['detection_rate']
                color = 'green'
            elif subject_id in self.non_responders:
                detection_rate = self.non_responders[subject_id]['detection_rate']
                color = 'red'
            else:
                continue
                
            mean_cluster_size = statistics.mean([rep['cluster_size'] for rep in patient_reps])
            patient_detection_rates.append(detection_rate)
            patient_mean_cluster_sizes.append(mean_cluster_size)
            patient_colors.append(color)
            
        ax7.scatter(patient_mean_cluster_sizes, patient_detection_rates, c=patient_colors, alpha=0.6)
        ax7.axhline(y=self.responder_threshold, color='black', linestyle='--', alpha=0.7)
        ax7.set_xlabel('Mean Cluster Size')
        ax7.set_ylabel('Detection Rate')
        ax7.set_title('Detection Rate vs Mean Cluster Size')
        ax7.grid(True, alpha=0.3)
        
        # 8. Summary Statistics Text
        ax8 = plt.subplot(3, 3, 8)
        ax8.axis('off')
        
        summary_text = f"""
RESPONDER ANALYSIS SUMMARY

Threshold: {self.responder_threshold:.1%}

RESPONDERS ({len(self.responders)} patients):
• Mean Detection Rate: {group_comparison['responders']['mean_detection_rate']:.1%}
• Mean Precision: {group_comparison['responders']['precision']:.1%}
• Overall Sensitivity: {group_comparison['responders']['overall_sensitivity']:.1%}

NON-RESPONDERS ({len(self.non_responders)} patients):
• Mean Detection Rate: {group_comparison['non_responders']['mean_detection_rate']:.1%}
• Mean Precision: {group_comparison['non_responders']['precision']:.1%}
• Overall Sensitivity: {group_comparison['non_responders']['overall_sensitivity']:.1%}

DIFFERENCES:
• Detection Rate: {group_comparison['differences']['mean_detection_rate_diff']:+.1%}
• Precision: {group_comparison['differences']['precision_diff']:+.1%}
• Sensitivity: {group_comparison['differences']['sensitivity_diff']:+.1%}
        """
        
        ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # 9. Additional cluster metrics
        ax9 = plt.subplot(3, 3, 9)
        if cluster_analysis['responders']['count'] > 0 and cluster_analysis['non_responders']['count'] > 0:
            cluster_metrics = ['Mean Cluster Size', 'Mean Anomaly Score', 'Reps per Patient']
            responder_cluster_values = [
                cluster_analysis['responders']['cluster_sizes']['mean'],
                cluster_analysis['responders']['anomaly_scores']['mean'],
                cluster_analysis['responders']['representatives_per_patient']
            ]
            non_responder_cluster_values = [
                cluster_analysis['non_responders']['cluster_sizes']['mean'],
                cluster_analysis['non_responders']['anomaly_scores']['mean'],
                cluster_analysis['non_responders']['representatives_per_patient']
            ]
            
            x = range(len(cluster_metrics))
            width = 0.35
            ax9.bar([i - width/2 for i in x], responder_cluster_values, width, 
                   label='Responders', color='green', alpha=0.7)
            ax9.bar([i + width/2 for i in x], non_responder_cluster_values, width,
                   label='Non-Responders', color='red', alpha=0.7)
            ax9.set_xlabel('Cluster Metrics')
            ax9.set_ylabel('Value')
            ax9.set_title('Cluster Characteristics by Group')
            ax9.set_xticks(x)
            ax9.set_xticklabels(cluster_metrics, rotation=45, ha='right')
            ax9.legend()
            ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_folder / "visualizations" / "responder_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_folder / "visualizations" / "responder_analysis.pdf", 
                   bbox_inches='tight')
        print(f"Visualizations saved to: {self.output_folder / 'visualizations'}")
        
    def save_detailed_results(self, patient_stats: Dict, group_comparison: Dict, cluster_analysis: Dict) -> None:
        """Save detailed analysis results."""
        
        # Save patient-level statistics
        patient_output = {
            'analysis_type': 'responder_analysis',
            'responder_threshold': self.responder_threshold,
            'timestamp': datetime.now().isoformat(),
            'patient_statistics': patient_stats,
            'responder_classification': {
                'responders': list(self.responders.keys()),
                'non_responders': list(self.non_responders.keys())
            }
        }
        
        with open(self.output_folder / "patient_analysis" / "patient_statistics.json", 'w') as f:
            json.dump(patient_output, f, indent=2)
            
        # Save group comparison
        with open(self.output_folder / "group_comparison" / "group_metrics.json", 'w') as f:
            json.dump(group_comparison, f, indent=2)
            
        # Save cluster analysis
        with open(self.output_folder / "group_comparison" / "cluster_analysis.json", 'w') as f:
            json.dump(cluster_analysis, f, indent=2)
            
        # Save comprehensive report
        self._create_text_report(patient_stats, group_comparison, cluster_analysis)
        
    def _create_text_report(self, patient_stats: Dict, group_comparison: Dict, cluster_analysis: Dict) -> None:
        """Create a comprehensive text report."""
        
        report = f"""
MADRID RESPONDER ANALYSIS REPORT
Generated: {datetime.now().isoformat()}
Source: {self.clustering_results_folder}
Responder Threshold: {self.responder_threshold:.1%}

=== OVERVIEW ===
Total Patients: {len(patient_stats)}
Responders: {len(self.responders)} ({len(self.responders)/len(patient_stats)*100:.1f}%)
Non-Responders: {len(self.non_responders)} ({len(self.non_responders)/len(patient_stats)*100:.1f}%)

=== RESPONDER GROUP ===
Patient Count: {group_comparison['responders']['count']}
Detection Rate: {group_comparison['responders']['mean_detection_rate']:.1%} ± {group_comparison['responders']['std_detection_rate']:.1%}
Range: {group_comparison['responders']['min_detection_rate']:.1%} - {group_comparison['responders']['max_detection_rate']:.1%}
Precision: {group_comparison['responders']['precision']:.1%}
Overall Sensitivity: {group_comparison['responders']['overall_sensitivity']:.1%}
Total Anomalies: {group_comparison['responders']['total_anomalies']}
True Positives: {group_comparison['responders']['total_tp']}
False Positives: {group_comparison['responders']['total_fp']}

=== NON-RESPONDER GROUP ===
Patient Count: {group_comparison['non_responders']['count']}
Detection Rate: {group_comparison['non_responders']['mean_detection_rate']:.1%} ± {group_comparison['non_responders']['std_detection_rate']:.1%}
Range: {group_comparison['non_responders']['min_detection_rate']:.1%} - {group_comparison['non_responders']['max_detection_rate']:.1%}
Precision: {group_comparison['non_responders']['precision']:.1%}
Overall Sensitivity: {group_comparison['non_responders']['overall_sensitivity']:.1%}
Total Anomalies: {group_comparison['non_responders']['total_anomalies']}
True Positives: {group_comparison['non_responders']['total_tp']}
False Positives: {group_comparison['non_responders']['total_fp']}

=== GROUP DIFFERENCES ===
Detection Rate Difference: {group_comparison['differences']['mean_detection_rate_diff']:+.1%}
Precision Difference: {group_comparison['differences']['precision_diff']:+.1%}
Sensitivity Difference: {group_comparison['differences']['sensitivity_diff']:+.1%}

=== CLUSTER CHARACTERISTICS ===

RESPONDERS:
• Mean Cluster Size: {cluster_analysis['responders']['cluster_sizes']['mean']:.1f}
• Mean Anomaly Score: {cluster_analysis['responders']['anomaly_scores']['mean']:.3f}
• Seizure Hit Rate: {cluster_analysis['responders']['seizure_hit_rate']:.1%}
• Representatives per Patient: {cluster_analysis['responders']['representatives_per_patient']:.1f}

NON-RESPONDERS:
• Mean Cluster Size: {cluster_analysis['non_responders']['cluster_sizes']['mean']:.1f}
• Mean Anomaly Score: {cluster_analysis['non_responders']['anomaly_scores']['mean']:.3f}
• Seizure Hit Rate: {cluster_analysis['non_responders']['seizure_hit_rate']:.1%}
• Representatives per Patient: {cluster_analysis['non_responders']['representatives_per_patient']:.1f}

CLUSTER DIFFERENCES:
• Cluster Size Difference: {cluster_analysis['differences']['mean_cluster_size_diff']:+.1f}
• Anomaly Score Difference: {cluster_analysis['differences']['mean_score_diff']:+.3f}
• Seizure Hit Rate Difference: {cluster_analysis['differences']['seizure_hit_rate_diff']:+.1%}

=== PATIENT DETAILS ===

RESPONDER PATIENTS:
"""
        
        for subject_id in sorted(self.responders.keys()):
            stats = self.responders[subject_id]
            report += f"• {subject_id}: {stats['detection_rate']:.1%} ({stats['detected_seizure_files']}/{stats['total_seizure_files']})\n"
            
        report += "\nNON-RESPONDER PATIENTS:\n"
        for subject_id in sorted(self.non_responders.keys()):
            stats = self.non_responders[subject_id]
            report += f"• {subject_id}: {stats['detection_rate']:.1%} ({stats['detected_seizure_files']}/{stats['total_seizure_files']})\n"
            
        with open(self.output_folder / "responder_analysis_report.txt", 'w') as f:
            f.write(report)
            
    def run_responder_analysis(self) -> Dict[str, Any]:
        """Run complete responder analysis."""
        print("Starting Madrid Responder Analysis...")
        
        self.load_clustering_results()
        if not self.representatives:
            raise ValueError("No representatives found")
            
        # Calculate patient detection rates
        print("Calculating patient seizure detection rates...")
        patient_stats = self.calculate_patient_seizure_detection_rates()
        
        # Classify responders
        print("Classifying patients as Responders/Non-Responders...")
        self.responders, self.non_responders = self.classify_responders(patient_stats)
        
        # Analyze group metrics
        print("Analyzing group metrics...")
        group_comparison = self.analyze_group_metrics(self.responders, self.non_responders)
        
        # Analyze cluster characteristics
        print("Analyzing cluster characteristics...")
        cluster_analysis = self.analyze_cluster_characteristics(self.responders, self.non_responders)
        
        # Create visualizations
        if MATPLOTLIB_AVAILABLE:
            print("Creating visualizations...")
            self.create_visualizations(group_comparison, cluster_analysis)
        else:
            print("Skipping visualizations - matplotlib not available")
        
        # Save results
        print("Saving detailed results...")
        self.save_detailed_results(patient_stats, group_comparison, cluster_analysis)
        
        # Print summary
        print(f"\n=== RESPONDER ANALYSIS SUMMARY ===")
        print(f"Threshold: {self.responder_threshold:.1%}")
        print(f"Responders: {len(self.responders)} patients (mean detection: {group_comparison['responders']['mean_detection_rate']:.1%})")
        print(f"Non-Responders: {len(self.non_responders)} patients (mean detection: {group_comparison['non_responders']['mean_detection_rate']:.1%})")
        print(f"Results saved to: {self.output_folder}")
        
        return {
            'patient_stats': patient_stats,
            'group_comparison': group_comparison,
            'cluster_analysis': cluster_analysis,
            'responders': self.responders,
            'non_responders': self.non_responders
        }


def main():
    """Main function."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 madrid_responder_analysis.py <clustering_results_folder> [output_folder] [threshold]")
        print("Example: python3 madrid_responder_analysis.py madrid_results_smart_clustered")
        print("Threshold: Responder threshold (default: 0.667 = 2/3)")
        sys.exit(1)
        
    input_folder = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else None
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 2/3
    
    analyzer = MadridResponderAnalyzer(input_folder, output_folder, threshold)
    
    try:
        results = analyzer.run_responder_analysis()
        print("\nResponder analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()