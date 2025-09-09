#!/usr/bin/env python3
"""
Madrid Cluster Size Distribution Visualizer

This script creates visual analysis of cluster size distributions, comparing
seizure-hit clusters vs non-seizure-hit clusters from Madrid clustering results.
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


class MadridClusterSizeVisualizer:
    """Visualizer for cluster size distribution analysis."""
    
    def __init__(self, clustering_results_folder: str, output_folder: str = None):
        self.clustering_results_folder = Path(clustering_results_folder)
        if output_folder:
            self.output_folder = Path(output_folder)
        else:
            input_path = Path(clustering_results_folder)
            self.output_folder = input_path.parent / f"{input_path.name}_visualizations"
        self.output_folder.mkdir(exist_ok=True, parents=True)
        
        self.representatives = []
        self.seizure_hit_clusters = []
        self.non_seizure_hit_clusters = []
        
        # Set up plotting style if available
        if MATPLOTLIB_AVAILABLE:
            plt.style.use('default')
        if SEABORN_AVAILABLE:
            sns.set_palette("husl")
        
    def load_clustering_results(self) -> None:
        """Load clustering results from best_representatives.json."""
        print(f"Loading clustering results from: {self.clustering_results_folder}")
        
        representatives_file = self.clustering_results_folder / "clusters" / "best_representatives.json"
        if not representatives_file.exists():
            raise FileNotFoundError(f"Representatives file not found: {representatives_file}")
            
        with open(representatives_file, 'r') as f:
            data = json.load(f)
            self.representatives = data['representatives']
            
        # Separate by seizure hit status
        for rep in self.representatives:
            if rep.get('seizure_hit', False):
                self.seizure_hit_clusters.append(rep)
            else:
                self.non_seizure_hit_clusters.append(rep)
                
        print(f"Loaded {len(self.representatives)} representatives:")
        print(f"  - Seizure hit clusters: {len(self.seizure_hit_clusters)}")
        print(f"  - Non-seizure hit clusters: {len(self.non_seizure_hit_clusters)}")
        
    def calculate_distribution_stats(self) -> Dict[str, Any]:
        """Calculate detailed statistics for both distributions."""
        
        # Get cluster sizes
        all_sizes = [r['cluster_size'] for r in self.representatives]
        seizure_sizes = [r['cluster_size'] for r in self.seizure_hit_clusters]
        non_seizure_sizes = [r['cluster_size'] for r in self.non_seizure_hit_clusters]
        
        def calc_stats(sizes: List[int], name: str) -> Dict:
            if not sizes:
                return {
                    'name': name,
                    'count': 0,
                    'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'std': 0,
                    'q25': 0, 'q75': 0, 'mode': 0
                }
                
            sizes_sorted = sorted(sizes)
            n = len(sizes_sorted)
            
            return {
                'name': name,
                'count': len(sizes),
                'min': min(sizes),
                'max': max(sizes),
                'mean': statistics.mean(sizes),
                'median': statistics.median(sizes),
                'std': statistics.stdev(sizes) if len(sizes) > 1 else 0,
                'q25': sizes_sorted[int(0.25 * (n-1))],
                'q75': sizes_sorted[int(0.75 * (n-1))],
                'mode': Counter(sizes).most_common(1)[0][0] if sizes else 0
            }
        
        stats = {
            'all': calc_stats(all_sizes, 'All Clusters'),
            'seizure_hit': calc_stats(seizure_sizes, 'Seizure Hit Clusters'),
            'non_seizure_hit': calc_stats(non_seizure_sizes, 'Non-Seizure Hit Clusters')
        }
        
        # Additional comparative statistics
        if seizure_sizes and non_seizure_sizes:
            stats['comparison'] = {
                'mean_difference': stats['seizure_hit']['mean'] - stats['non_seizure_hit']['mean'],
                'median_difference': stats['seizure_hit']['median'] - stats['non_seizure_hit']['median'],
                'seizure_hit_larger_pct': sum(1 for s in seizure_sizes if s > stats['non_seizure_hit']['mean']) / len(seizure_sizes) * 100,
                'non_seizure_hit_larger_pct': sum(1 for s in non_seizure_sizes if s > stats['seizure_hit']['mean']) / len(non_seizure_sizes) * 100
            }
        
        return stats
        
    def create_distribution_plots(self, stats: Dict[str, Any]) -> None:
        """Create comprehensive distribution visualization plots."""
        
        if not MATPLOTLIB_AVAILABLE:
            print("Skipping plot creation - matplotlib not available")
            return
        
        # Get data
        all_sizes = [r['cluster_size'] for r in self.representatives]
        seizure_sizes = [r['cluster_size'] for r in self.seizure_hit_clusters]
        non_seizure_sizes = [r['cluster_size'] for r in self.non_seizure_hit_clusters]
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Side-by-side histograms
        ax1 = plt.subplot(3, 3, 1)
        bins = range(1, max(all_sizes) + 2)
        ax1.hist([seizure_sizes, non_seizure_sizes], bins=bins, alpha=0.7, 
                label=['Seizure Hit', 'Non-Seizure Hit'], color=['red', 'blue'])
        ax1.set_xlabel('Cluster Size')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Cluster Size Distribution - Overlaid Histograms')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Box plots comparison
        ax2 = plt.subplot(3, 3, 2)
        data_for_box = [seizure_sizes, non_seizure_sizes]
        labels_for_box = ['Seizure Hit', 'Non-Seizure Hit']
        box_plot = ax2.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
        box_plot['boxes'][0].set_facecolor('red')
        box_plot['boxes'][1].set_facecolor('blue')
        ax2.set_ylabel('Cluster Size')
        ax2.set_title('Cluster Size Distribution - Box Plot Comparison')
        ax2.grid(True, alpha=0.3)
        
        # 3. Violin plots
        ax3 = plt.subplot(3, 3, 3)
        violin_data = [seizure_sizes, non_seizure_sizes]
        parts = ax3.violinplot(violin_data, positions=[1, 2], showmeans=True, showmedians=True)
        ax3.set_xticks([1, 2])
        ax3.set_xticklabels(['Seizure Hit', 'Non-Seizure Hit'])
        ax3.set_ylabel('Cluster Size')
        ax3.set_title('Cluster Size Distribution - Violin Plot')
        ax3.grid(True, alpha=0.3)
        
        # 4. Cumulative distribution
        ax4 = plt.subplot(3, 3, 4)
        seizure_sorted = sorted(seizure_sizes)
        non_seizure_sorted = sorted(non_seizure_sizes)
        
        seizure_cumulative = [i/len(seizure_sorted) for i in range(1, len(seizure_sorted)+1)]
        non_seizure_cumulative = [i/len(non_seizure_sorted) for i in range(1, len(non_seizure_sorted)+1)]
        
        ax4.plot(seizure_sorted, seizure_cumulative, 'r-', label='Seizure Hit', linewidth=2)
        ax4.plot(non_seizure_sorted, non_seizure_cumulative, 'b-', label='Non-Seizure Hit', linewidth=2)
        ax4.set_xlabel('Cluster Size')
        ax4.set_ylabel('Cumulative Probability')
        ax4.set_title('Cumulative Distribution Function')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Separate histograms
        ax5 = plt.subplot(3, 3, 5)
        bins = range(1, max(all_sizes) + 2)
        ax5.hist(seizure_sizes, bins=bins, alpha=0.7, color='red', label=f'Seizure Hit (n={len(seizure_sizes)})')
        ax5.set_xlabel('Cluster Size')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Seizure Hit Clusters Only')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        ax6 = plt.subplot(3, 3, 6)
        ax6.hist(non_seizure_sizes, bins=bins, alpha=0.7, color='blue', label=f'Non-Seizure Hit (n={len(non_seizure_sizes)})')
        ax6.set_xlabel('Cluster Size')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Non-Seizure Hit Clusters Only')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 6. Proportional stacked bar chart
        ax7 = plt.subplot(3, 3, 7)
        max_size = max(all_sizes)
        size_ranges = [(1, 5), (6, 10), (11, 20), (21, 30), (31, max_size)]
        
        seizure_counts = []
        non_seizure_counts = []
        range_labels = []
        
        for start, end in size_ranges:
            seizure_count = sum(1 for s in seizure_sizes if start <= s <= end)
            non_seizure_count = sum(1 for s in non_seizure_sizes if start <= s <= end)
            seizure_counts.append(seizure_count)
            non_seizure_counts.append(non_seizure_count)
            range_labels.append(f'{start}-{end}')
            
        x_pos = range(len(range_labels))
        total_counts = [s + n for s, n in zip(seizure_counts, non_seizure_counts)]
        seizure_props = [s/t*100 if t > 0 else 0 for s, t in zip(seizure_counts, total_counts)]
        non_seizure_props = [n/t*100 if t > 0 else 0 for n, t in zip(non_seizure_counts, total_counts)]
        
        ax7.bar(x_pos, seizure_props, color='red', alpha=0.7, label='Seizure Hit %')
        ax7.bar(x_pos, non_seizure_props, bottom=seizure_props, color='blue', alpha=0.7, label='Non-Seizure Hit %')
        ax7.set_xlabel('Cluster Size Range')
        ax7.set_ylabel('Percentage')
        ax7.set_title('Proportional Distribution by Size Range')
        ax7.set_xticks(x_pos)
        ax7.set_xticklabels(range_labels)
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 7. Statistical summary plot
        ax8 = plt.subplot(3, 3, 8)
        metrics = ['Mean', 'Median', 'Q75', 'Max']
        seizure_values = [stats['seizure_hit']['mean'], stats['seizure_hit']['median'], 
                         stats['seizure_hit']['q75'], stats['seizure_hit']['max']]
        non_seizure_values = [stats['non_seizure_hit']['mean'], stats['non_seizure_hit']['median'],
                             stats['non_seizure_hit']['q75'], stats['non_seizure_hit']['max']]
        
        x_metrics = range(len(metrics))
        width = 0.35
        ax8.bar([x - width/2 for x in x_metrics], seizure_values, width, 
               label='Seizure Hit', color='red', alpha=0.7)
        ax8.bar([x + width/2 for x in x_metrics], non_seizure_values, width,
               label='Non-Seizure Hit', color='blue', alpha=0.7)
        ax8.set_xlabel('Statistical Metrics')
        ax8.set_ylabel('Cluster Size')
        ax8.set_title('Statistical Comparison')
        ax8.set_xticks(x_metrics)
        ax8.set_xticklabels(metrics)
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 8. Text summary
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        summary_text = f"""
DISTRIBUTION SUMMARY

Total Clusters: {len(self.representatives)}
• Seizure Hit: {len(self.seizure_hit_clusters)} ({len(self.seizure_hit_clusters)/len(self.representatives)*100:.1f}%)
• Non-Seizure Hit: {len(self.non_seizure_hit_clusters)} ({len(self.non_seizure_hit_clusters)/len(self.representatives)*100:.1f}%)

SEIZURE HIT CLUSTERS:
• Mean Size: {stats['seizure_hit']['mean']:.1f}
• Median Size: {stats['seizure_hit']['median']:.1f}
• Range: {stats['seizure_hit']['min']}-{stats['seizure_hit']['max']}

NON-SEIZURE HIT CLUSTERS:
• Mean Size: {stats['non_seizure_hit']['mean']:.1f}
• Median Size: {stats['non_seizure_hit']['median']:.1f}
• Range: {stats['non_seizure_hit']['min']}-{stats['non_seizure_hit']['max']}

COMPARISON:
• Mean Difference: {stats['comparison']['mean_difference']:+.1f}
• Median Difference: {stats['comparison']['median_difference']:+.1f}
        """
        
        ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_folder / 'cluster_size_distribution_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_folder / 'cluster_size_distribution_analysis.pdf', 
                   bbox_inches='tight')
        print(f"Distribution plots saved to: {self.output_folder}")
        
    def create_detailed_size_analysis(self, stats: Dict[str, Any]) -> None:
        """Create detailed size-by-size analysis."""
        
        if not MATPLOTLIB_AVAILABLE:
            print("Skipping detailed plots - matplotlib not available")
            return
        
        # Count occurrences by size
        seizure_size_counts = Counter([r['cluster_size'] for r in self.seizure_hit_clusters])
        non_seizure_size_counts = Counter([r['cluster_size'] for r in self.non_seizure_hit_clusters])
        
        # Get all unique sizes
        all_unique_sizes = sorted(set(seizure_size_counts.keys()) | set(non_seizure_size_counts.keys()))
        
        # Create detailed comparison plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Count comparison
        seizure_counts = [seizure_size_counts.get(size, 0) for size in all_unique_sizes]
        non_seizure_counts = [non_seizure_size_counts.get(size, 0) for size in all_unique_sizes]
        
        width = 0.35
        x_pos = range(len(all_unique_sizes))
        
        ax1.bar([x - width/2 for x in x_pos], seizure_counts, width, 
               label='Seizure Hit', color='red', alpha=0.7)
        ax1.bar([x + width/2 for x in x_pos], non_seizure_counts, width,
               label='Non-Seizure Hit', color='blue', alpha=0.7)
        
        ax1.set_xlabel('Cluster Size')
        ax1.set_ylabel('Number of Clusters')
        ax1.set_title('Cluster Count by Size - Seizure Hit vs Non-Seizure Hit')
        ax1.set_xticks(x_pos[::2])  # Show every 2nd tick to avoid crowding
        ax1.set_xticklabels([str(size) for size in all_unique_sizes[::2]])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Proportion analysis
        total_counts = [s + n for s, n in zip(seizure_counts, non_seizure_counts)]
        seizure_props = [s/t*100 if t > 0 else 0 for s, t in zip(seizure_counts, total_counts)]
        
        ax2.plot(all_unique_sizes, seizure_props, 'ro-', linewidth=2, markersize=4, 
                label='% Seizure Hit')
        ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.7, label='50% line')
        ax2.set_xlabel('Cluster Size')
        ax2.set_ylabel('Percentage Seizure Hit')
        ax2.set_title('Seizure Hit Rate by Cluster Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(self.output_folder / 'detailed_size_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_folder / 'detailed_size_analysis.pdf', 
                   bbox_inches='tight')
        
    def save_statistics_report(self, stats: Dict[str, Any]) -> None:
        """Save detailed statistics report as JSON and text."""
        
        # Save as JSON
        with open(self.output_folder / 'cluster_size_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
            
        # Save as readable text report
        report = f"""
MADRID CLUSTER SIZE DISTRIBUTION ANALYSIS REPORT
Generated: {datetime.now().isoformat()}
Source: {self.clustering_results_folder}

=== OVERVIEW ===
Total Representatives: {len(self.representatives)}
Seizure Hit Clusters: {len(self.seizure_hit_clusters)} ({len(self.seizure_hit_clusters)/len(self.representatives)*100:.1f}%)
Non-Seizure Hit Clusters: {len(self.non_seizure_hit_clusters)} ({len(self.non_seizure_hit_clusters)/len(self.representatives)*100:.1f}%)

=== SEIZURE HIT CLUSTERS ===
Count: {stats['seizure_hit']['count']}
Size Range: {stats['seizure_hit']['min']} - {stats['seizure_hit']['max']}
Mean: {stats['seizure_hit']['mean']:.2f}
Median: {stats['seizure_hit']['median']:.2f}
Standard Deviation: {stats['seizure_hit']['std']:.2f}
25th Percentile: {stats['seizure_hit']['q25']}
75th Percentile: {stats['seizure_hit']['q75']}
Mode: {stats['seizure_hit']['mode']}

=== NON-SEIZURE HIT CLUSTERS ===
Count: {stats['non_seizure_hit']['count']}
Size Range: {stats['non_seizure_hit']['min']} - {stats['non_seizure_hit']['max']}
Mean: {stats['non_seizure_hit']['mean']:.2f}
Median: {stats['non_seizure_hit']['median']:.2f}
Standard Deviation: {stats['non_seizure_hit']['std']:.2f}
25th Percentile: {stats['non_seizure_hit']['q25']}
75th Percentile: {stats['non_seizure_hit']['q75']}
Mode: {stats['non_seizure_hit']['mode']}

=== COMPARISON ===
Mean Size Difference: {stats['comparison']['mean_difference']:+.2f} (Seizure Hit - Non-Seizure Hit)
Median Size Difference: {stats['comparison']['median_difference']:+.1f}
Seizure Hit clusters larger than Non-Seizure Hit mean: {stats['comparison']['seizure_hit_larger_pct']:.1f}%
Non-Seizure Hit clusters larger than Seizure Hit mean: {stats['comparison']['non_seizure_hit_larger_pct']:.1f}%

=== ALL CLUSTERS ===
Total Count: {stats['all']['count']}
Size Range: {stats['all']['min']} - {stats['all']['max']}
Mean: {stats['all']['mean']:.2f}
Median: {stats['all']['median']:.2f}
Standard Deviation: {stats['all']['std']:.2f}
25th Percentile: {stats['all']['q25']}
75th Percentile: {stats['all']['q75']}
Mode: {stats['all']['mode']}
        """
        
        with open(self.output_folder / 'cluster_size_analysis_report.txt', 'w') as f:
            f.write(report)
            
    def run_visualization_analysis(self) -> Dict[str, Any]:
        """Run complete visualization analysis."""
        print("Starting Madrid cluster size visualization analysis...")
        
        self.load_clustering_results()
        if not self.representatives:
            raise ValueError("No representatives found")
            
        # Calculate statistics
        print("Calculating distribution statistics...")
        stats = self.calculate_distribution_stats()
        
        # Create visualizations
        if MATPLOTLIB_AVAILABLE:
            print("Creating distribution plots...")
            self.create_distribution_plots(stats)
            
            print("Creating detailed size analysis...")
            self.create_detailed_size_analysis(stats)
        else:
            print("Skipping visualizations - matplotlib not available")
        
        # Save reports
        print("Saving statistics report...")
        self.save_statistics_report(stats)
        
        print(f"\n=== ANALYSIS SUMMARY ===")
        print(f"Seizure Hit Clusters: {len(self.seizure_hit_clusters)} (mean size: {stats['seizure_hit']['mean']:.1f})")
        print(f"Non-Seizure Hit Clusters: {len(self.non_seizure_hit_clusters)} (mean size: {stats['non_seizure_hit']['mean']:.1f})")
        print(f"Mean size difference: {stats['comparison']['mean_difference']:+.1f}")
        print(f"Results saved to: {self.output_folder}")
        
        return stats


def main():
    """Main function."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 madrid_cluster_size_visualizer.py <clustering_results_folder> [output_folder]")
        print("Example: python3 madrid_cluster_size_visualizer.py madrid_results_smart_clustered")
        sys.exit(1)
        
    input_folder = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else None
    
    visualizer = MadridClusterSizeVisualizer(input_folder, output_folder)
    
    try:
        results = visualizer.run_visualization_analysis()
        print("\nVisualization analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()