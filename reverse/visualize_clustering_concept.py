#!/usr/bin/env python3
"""
Visualize the clustering concept with a simple example.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


def create_clustering_visualization(output_dir: Path):
    """Create a visual explanation of how clustering works."""

    output_dir.mkdir(exist_ok=True, parents=True)

    # Example timeline (in seconds)
    total_duration = 600  # 10 minutes

    # Simulated anomaly detections (time in seconds)
    anomalies = [
        # Cluster 1: False alarms close together
        (50, 4.8), (55, 5.2), (62, 4.9), (68, 5.1),
        # Cluster 2: True positive (seizure) with some nearby FPs
        (200, 6.5), (205, 6.8), (210, 6.2), (215, 5.9),
        # Cluster 3: Isolated false alarms
        (350, 5.0), (358, 4.7),
        # Cluster 4: Another group of FPs
        (480, 5.3), (485, 5.5), (492, 5.1)
    ]

    # Ground truth seizure
    gt_seizure = (200, 220)  # 20 seconds long

    # Clustering with 30s threshold
    time_threshold = 30

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))

    # ========================
    # Plot 1: Before Clustering
    # ========================
    ax = axes[0]
    ax.set_xlim(0, total_duration)
    ax.set_ylim(0, 2)
    ax.set_title('BEFORE CLUSTERING: All Detected Anomalies',
                 fontsize=14, fontweight='bold')

    # Draw all anomalies
    for i, (time, score) in enumerate(anomalies):
        # Check if it overlaps with ground truth
        is_tp = gt_seizure[0] <= time <= gt_seizure[1]
        color = 'green' if is_tp else 'red'
        marker = 'o' if is_tp else 'x'
        label = 'TP' if is_tp else 'FP'

        ax.scatter(time, 1, s=200, c=color, marker=marker,
                  edgecolors='black', linewidth=2, zorder=3,
                  label=label if i == 0 or (i == 4 and is_tp) else "")
        ax.text(time, 1.3, f'{time}s\n{score:.1f}',
               ha='center', va='bottom', fontsize=8)

    # Draw ground truth
    ax.add_patch(mpatches.Rectangle(
        (gt_seizure[0], 0.3), gt_seizure[1]-gt_seizure[0], 0.4,
        facecolor='orange', alpha=0.3, edgecolor='orange', linewidth=2,
        label='Ground Truth Seizure'
    ))

    ax.set_ylabel('Events', fontsize=11)
    ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')

    # Statistics
    n_events = len(anomalies)
    n_tp = sum(1 for t, _ in anomalies if gt_seizure[0] <= t <= gt_seizure[1])
    n_fp = n_events - n_tp
    ax.text(10, 1.7, f'Events: {n_events} (TP={n_tp}, FP={n_fp})\nFAR: {n_fp/(total_duration/3600):.1f} events/hour',
           fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ========================
    # Plot 2: Clustering Process
    # ========================
    ax = axes[1]
    ax.set_xlim(0, total_duration)
    ax.set_ylim(0, 2)
    ax.set_title(f'CLUSTERING PROCESS: Grouping with {time_threshold}s Threshold',
                 fontsize=14, fontweight='bold')

    # Perform simple clustering
    sorted_anomalies = sorted(anomalies, key=lambda x: x[0])
    clusters = []
    current_cluster = [sorted_anomalies[0]]

    for i in range(1, len(sorted_anomalies)):
        prev_time = current_cluster[-1][0]
        curr_time = sorted_anomalies[i][0]

        if curr_time - prev_time <= time_threshold:
            current_cluster.append(sorted_anomalies[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [sorted_anomalies[i]]
    clusters.append(current_cluster)

    # Draw clusters with colored regions
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    for cluster_idx, cluster in enumerate(clusters):
        color = colors[cluster_idx % len(colors)]

        # Draw cluster region
        cluster_start = min(t for t, _ in cluster)
        cluster_end = max(t for t, _ in cluster)
        ax.add_patch(mpatches.Rectangle(
            (cluster_start, 0.5), cluster_end - cluster_start, 1,
            facecolor=color, alpha=0.3, edgecolor='black', linewidth=2
        ))

        # Draw anomalies in cluster
        for time, score in cluster:
            is_tp = gt_seizure[0] <= time <= gt_seizure[1]
            marker_color = 'green' if is_tp else 'red'
            marker = 'o' if is_tp else 'x'

            ax.scatter(time, 1, s=150, c=marker_color, marker=marker,
                      edgecolors='black', linewidth=1.5, zorder=3)

        # Label cluster
        cluster_center = (cluster_start + cluster_end) / 2
        ax.text(cluster_center, 1.7, f'Cluster {cluster_idx+1}\n({len(cluster)} events)',
               ha='center', va='center', fontsize=9, fontweight='bold')

    # Draw ground truth
    ax.add_patch(mpatches.Rectangle(
        (gt_seizure[0], 0.1), gt_seizure[1]-gt_seizure[0], 0.3,
        facecolor='orange', alpha=0.3, edgecolor='orange', linewidth=2
    ))

    ax.set_ylabel('Clusters', fontsize=11)
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, axis='x')

    ax.text(10, 0.2, f'Clusters formed: {len(clusters)}',
           fontsize=11, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # ========================
    # Plot 3: After Clustering (Representatives)
    # ========================
    ax = axes[2]
    ax.set_xlim(0, total_duration)
    ax.set_ylim(0, 2)
    ax.set_title('AFTER CLUSTERING: Cluster Representatives',
                 fontsize=14, fontweight='bold')

    # Select representatives (central anomaly in each cluster)
    representatives = []
    for cluster_idx, cluster in enumerate(clusters):
        # Find central anomaly (min average distance to others)
        if len(cluster) == 1:
            rep = cluster[0]
        else:
            min_avg_dist = float('inf')
            rep = None
            for candidate in cluster:
                avg_dist = sum(abs(candidate[0] - other[0]) for other in cluster if other != candidate) / (len(cluster) - 1)
                if avg_dist < min_avg_dist:
                    min_avg_dist = avg_dist
                    rep = candidate

        cluster_start = min(t for t, _ in cluster)
        cluster_end = max(t for t, _ in cluster)
        is_tp = any(gt_seizure[0] <= t <= gt_seizure[1] for t, _ in cluster)

        representatives.append((rep, cluster_start, cluster_end, is_tp))

    # Draw representatives and their cluster spans
    for i, (rep, c_start, c_end, is_tp) in enumerate(representatives):
        color = 'green' if is_tp else 'red'

        # Draw cluster span
        ax.add_patch(mpatches.Rectangle(
            (c_start, 0.8), c_end - c_start, 0.4,
            facecolor=color, alpha=0.2, edgecolor=color, linewidth=2
        ))

        # Draw representative
        ax.scatter(rep[0], 1, s=300, c=color, marker='*',
                  edgecolors='black', linewidth=2, zorder=3)
        ax.text(rep[0], 1.4, f'Rep {i+1}\n{rep[0]}s',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Draw ground truth
    ax.add_patch(mpatches.Rectangle(
        (gt_seizure[0], 0.3), gt_seizure[1]-gt_seizure[0], 0.4,
        facecolor='orange', alpha=0.3, edgecolor='orange', linewidth=2
    ))

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Representatives', fontsize=11)
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, axis='x')

    # Statistics
    n_rep = len(representatives)
    n_tp_rep = sum(1 for _, _, _, is_tp in representatives if is_tp)
    n_fp_rep = n_rep - n_tp_rep
    far_before = n_fp / (total_duration/3600)
    far_after = n_fp_rep / (total_duration/3600)
    far_reduction = ((far_before - far_after) / far_before) * 100

    ax.text(10, 1.7, f'Representatives: {n_rep} (TP={n_tp_rep}, FP={n_fp_rep})\n'
                     f'FAR: {far_after:.1f} events/hour\n'
                     f'FAR Reduction: {far_reduction:.1f}%',
           fontsize=11, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.tight_layout()
    output_path = output_dir / 'clustering_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

    # ========================
    # Create comparison plot
    # ========================
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    categories = ['Events', 'True Positives', 'False Positives', 'FAR (events/h)']
    before = [n_events, n_tp, n_fp, far_before]
    after = [n_rep, n_tp_rep, n_fp_rep, far_after]

    x = np.arange(len(categories))
    width = 0.35

    ax.bar(x - width/2, before, width, label='Before Clustering', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, after, width, label='After Clustering', color='seagreen', alpha=0.8)

    ax.set_ylabel('Count / Rate', fontsize=12)
    ax.set_title('Impact of Clustering on Detection Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (b, a) in enumerate(zip(before, after)):
        ax.text(i - width/2, b + 0.5, f'{b:.1f}', ha='center', va='bottom', fontsize=10)
        ax.text(i + width/2, a + 0.5, f'{a:.1f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    output_path = output_dir / 'clustering_metrics_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    output_dir = Path(__file__).parent / "clustering_visualization"

    print("="*80)
    print("CREATING CLUSTERING VISUALIZATION")
    print("="*80)

    create_clustering_visualization(output_dir)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nPlots saved to: {output_dir}")
    print("\nThis demonstrates:")
    print("  1. Before: All detected anomalies (high FP rate)")
    print("  2. Process: How clustering groups nearby anomalies")
    print("  3. After: Cluster representatives (reduced FP, same TP)")
    print()
