#!/usr/bin/env python3
"""
Madrid Single File Analysis Script - Top 3 Anomalies Version

Analysiert eine einzelne PKL-Datei mit Madrid für verschiedene m-Werte und findet
die TOP 3 Anomalien pro m-Wert. Erstellt einen Plot, der Location und Score aller
gefundenen Anomalien zeigt, farblich markiert ob es tatsächlich eine Seizure ist.

Usage:
    python madrid_single_file_analysis_top3.py <pkl_file_path> --m-range 250 3000 --step 250
"""

import os
import sys
import time
import pickle
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Madrid class
try:
    from models.madrid import MADRID
except ImportError:
    try:
        from madrid import MADRID
    except ImportError as e:
        raise ImportError(
            "MADRID konnte nicht importiert werden. "
            "Stelle sicher, dass madrid.py verfügbar ist."
        ) from e

warnings.filterwarnings("ignore", category=UserWarning)

class MadridTop3Analyzer:
    """Analysiert eine einzelne PKL-Datei mit Madrid und findet Top 3 Anomalien pro m-Wert."""
    
    def __init__(self, use_gpu: bool = True, train_ratio: float = 1/3, top_k: int = 3):
        self.use_gpu = use_gpu
        self.train_ratio = train_ratio
        self.top_k = top_k
        
    def load_pickle_file(self, file_path: Path) -> Tuple[np.ndarray, List[Tuple[int, int]], int]:
        """
        Lädt eine PKL-Datei und extrahiert Signal, Seizure-Regionen und Sampling-Rate.
        """
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        # Extract signal from first channel
        ch = data["channels"][0]
        signal = ch["data"].astype(np.float64)
        labels = ch["labels"]
        sampling_rate = data.get("sampling_rate", 125)
        
        # Find ictal (seizure) regions
        ictal_indices = np.where(labels == "ictal")[0]
        seizure_regions = []
        
        if len(ictal_indices) > 0:
            # Group consecutive indices into regions
            start = ictal_indices[0]
            for i in range(1, len(ictal_indices)):
                if ictal_indices[i] != ictal_indices[i-1] + 1:
                    seizure_regions.append((start, ictal_indices[i-1] + 1))
                    start = ictal_indices[i]
            seizure_regions.append((start, ictal_indices[-1] + 1))
        
        return signal, seizure_regions, sampling_rate
    
    def find_top_k_anomalies(self, discord_scores: np.ndarray, train_test_split: int, 
                           m: int, exclusion_factor: float = 0.5) -> List[Tuple[float, int]]:
        """
        Findet die Top-K Anomalien in einem Discord-Score Array mit Exclusion-Zone.
        
        Args:
            discord_scores: Array mit Discord-Scores für alle Positionen
            train_test_split: Index wo Test-Daten beginnen
            m: Subsequence-Länge (für Exclusion-Zone)
            exclusion_factor: Multiplikator für Exclusion-Zone Größe
            
        Returns:
            Liste von (score, location) Tupeln, sortiert nach Score (absteigend)
        """
        # Nur Test-Portion betrachten
        test_scores = discord_scores[train_test_split:]
        test_indices = np.arange(train_test_split, len(discord_scores))
        
        # Infinite und NaN Werte ausschließen
        valid_mask = np.isfinite(test_scores) & ~np.isinf(test_scores)
        if not np.any(valid_mask):
            return []
        
        valid_scores = test_scores[valid_mask]
        valid_indices = test_indices[valid_mask]
        
        anomalies = []
        used_indices = set()
        exclusion_radius = int(m * exclusion_factor)
        
        # Iterativ beste Anomalien finden
        for _ in range(self.top_k):
            # Verfügbare Scores (nicht in Exclusion-Zone)
            available_mask = np.ones(len(valid_scores), dtype=bool)
            for used_idx in used_indices:
                # Exclusion-Zone um bereits gefundene Anomalien
                distance = np.abs(valid_indices - used_idx)
                available_mask &= (distance > exclusion_radius)
            
            if not np.any(available_mask):
                break  # Keine weiteren Anomalien verfügbar
            
            # Beste verfügbare Anomalie finden
            available_scores = valid_scores.copy()
            available_scores[~available_mask] = -np.inf
            
            best_idx = np.argmax(available_scores)
            best_score = valid_scores[best_idx]
            best_location = valid_indices[best_idx]
            
            if best_score <= 0:  # Keine sinnvollen Scores mehr
                break
                
            anomalies.append((float(best_score), int(best_location)))
            used_indices.add(best_location)
        
        return anomalies
    
    def run_madrid_analysis(self, signal: np.ndarray, m_values: List[int]) -> Dict[int, List[Tuple[float, int]]]:
        """
        Führt Madrid-Analyse EINMAL für alle m-Werte durch und extrahiert Top-K Anomalien.
        
        Returns:
            Dict mit Listen von (score, location) Tupeln für jeden m-Wert
        """
        results = {}
        train_test_split = int(len(signal) * self.train_ratio)
        
        # Determine step size from m_values
        if len(m_values) > 1:
            step_size = m_values[1] - m_values[0]
        else:
            step_size = 1
            
        min_m = min(m_values)
        max_m = max(m_values)
        
        print(f"  Führe MADRID einmal durch für m={min_m} bis {max_m} (step={step_size})...")
        print(f"  Suche Top-{self.top_k} Anomalien pro m-Wert...")
        
        try:
            madrid = MADRID(use_gpu=self.use_gpu, enable_output=False)
            
            # Run Madrid ONCE for entire m range
            multi_length_table, bsf, bsf_loc = madrid.fit(
                T=signal,
                min_length=min_m,
                max_length=max_m,
                step_size=step_size,
                train_test_split=train_test_split,
                factor=1
            )
            
            print(f"    Debug: Multi-length table shape: {multi_length_table.shape}")
            print(f"    Debug: BSF shape: {bsf.shape}")
            
            # Extract Top-K anomalies from multi_length_table
            for i in range(multi_length_table.shape[0]):
                # Bestimme den entsprechenden m-Wert
                # Nach dem Fix sollte das korrekt funktionieren
                m_idx = i * step_size + min_m
                if m_idx <= max_m and m_idx in m_values:
                    m = m_idx
                    
                    # Extract discord scores für diesen m-Wert
                    discord_scores = multi_length_table[i, :]
                    
                    # Finde Top-K Anomalien für diesen m-Wert
                    top_anomalies = self.find_top_k_anomalies(
                        discord_scores, train_test_split, m
                    )
                    
                    results[m] = top_anomalies
                    
                    print(f"    m={m}: {len(top_anomalies)} Anomalien gefunden")
                    for j, (score, loc) in enumerate(top_anomalies):
                        print(f"      #{j+1}: score={score:.4f}, location={loc}")
                else:
                    print(f"    Row {i}: m={m_idx} nicht in gewünschten m_values")
            
            # Sicherstellen, dass alle m_values vertreten sind
            for m in m_values:
                if m not in results:
                    results[m] = []
                    
        except ValueError as e:
            if "constant" in str(e).lower():
                print(f"    Warnung: Konstante Regionen erkannt...")
                for m in m_values:
                    results[m] = []
            else:
                raise
        except Exception as e:
            print(f"    Fehler bei MADRID-Ausführung: {e}")
            for m in m_values:
                results[m] = []
        
        return results
    
    def check_seizure_overlap(self, location: int, m: int, seizure_regions: List[Tuple[int, int]], 
                            overlap_threshold: float = 0.1) -> bool:
        """
        Prüft, ob die gefundene Anomalie mit einer bekannten Seizure überlappt.
        """
        if location is None:
            return False
            
        anomaly_start = location
        anomaly_end = location + m
        
        for seizure_start, seizure_end in seizure_regions:
            # Calculate overlap
            overlap_start = max(anomaly_start, seizure_start)
            overlap_end = min(anomaly_end, seizure_end)
            overlap_length = max(0, overlap_end - overlap_start)
            
            # Check if overlap is significant
            anomaly_length = anomaly_end - anomaly_start
            seizure_length = seizure_end - seizure_start
            min_length = min(anomaly_length, seizure_length)
            
            if overlap_length / min_length >= overlap_threshold:
                return True
                
        return False
    
    def create_plot(self, m_values: List[int], results: Dict[int, List[Tuple[float, int]]], 
                   seizure_regions: List[Tuple[int, int]], sampling_rate: int, 
                   file_name: str, output_path: Path):
        """
        Erstellt Plots für alle Top-K Anomalien.
        """
        # Sammle alle Anomalien für den Plot
        all_m_vals = []
        all_locations = []
        all_scores = []
        all_is_seizure = []
        all_ranks = []
        
        for m in sorted(results.keys()):
            anomalies = results[m]
            for rank, (score, location) in enumerate(anomalies):
                is_seizure = self.check_seizure_overlap(location, m, seizure_regions)
                
                all_m_vals.append(m)
                all_locations.append(location)
                all_scores.append(score)
                all_is_seizure.append(is_seizure)
                all_ranks.append(rank + 1)  # 1-based ranking
        
        if not all_m_vals:
            print("Keine gültigen Ergebnisse zum Plotten gefunden.")
            return
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        fig.suptitle(f'Madrid Top-{self.top_k} Anomaly Detection Analysis\nFile: {file_name}', 
                    fontsize=14, fontweight='bold')
        
        # Define colors for different ranks - dynamically generate enough colors
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        
        # Generate colors for all ranks
        if self.top_k <= 10:
            # Use predefined colors for common cases
            base_colors = ['red', 'orange', 'gold', 'lime', 'cyan', 
                          'blue', 'purple', 'pink', 'brown', 'gray']
            rank_colors = base_colors[:self.top_k]
        else:
            # Use colormap for larger k values
            cmap = cm.get_cmap('tab20')  # or 'viridis', 'Set3', etc.
            rank_colors = [cmap(i / self.top_k) for i in range(self.top_k)]
        
        seizure_markers = ['o', 's']  # circle for seizure, square for non-seizure
        
        # Plot 1: Score vs m-value (colored by rank, shaped by seizure detection)
        for rank in range(1, self.top_k + 1):
            rank_mask = np.array(all_ranks) == rank
            if not np.any(rank_mask):
                continue
                
            rank_m = np.array(all_m_vals)[rank_mask]
            rank_scores = np.array(all_scores)[rank_mask]
            rank_seizure = np.array(all_is_seizure)[rank_mask]
            
            # Plot seizure detections
            seizure_mask = rank_seizure
            if np.any(seizure_mask):
                ax1.scatter(rank_m[seizure_mask], rank_scores[seizure_mask], 
                           c=rank_colors[rank-1], marker='o', s=80, alpha=0.8,
                           label=f'Rank {rank} - Seizure' if rank == 1 else '', 
                           edgecolors='darkred', linewidth=1)
            
            # Plot non-seizure detections
            non_seizure_mask = ~rank_seizure
            if np.any(non_seizure_mask):
                ax1.scatter(rank_m[non_seizure_mask], rank_scores[non_seizure_mask], 
                           c=rank_colors[rank-1], marker='s', s=60, alpha=0.6,
                           label=f'Rank {rank} - No Seizure' if rank == 1 else '', 
                           edgecolors='darkblue', linewidth=1)
        
        ax1.set_xlabel('m-Wert (Subsequence Length)')
        ax1.set_ylabel('Anomaly Score')
        ax1.set_title(f'Top-{self.top_k} Anomaly Scores by m-value')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Location vs m-value (colored by rank, shaped by seizure detection)
        for rank in range(1, self.top_k + 1):
            rank_mask = np.array(all_ranks) == rank
            if not np.any(rank_mask):
                continue
                
            rank_m = np.array(all_m_vals)[rank_mask]
            rank_locations = np.array(all_locations)[rank_mask]
            rank_seizure = np.array(all_is_seizure)[rank_mask]
            
            # Plot seizure detections
            seizure_mask = rank_seizure
            if np.any(seizure_mask):
                ax2.scatter(rank_m[seizure_mask], rank_locations[seizure_mask], 
                           c=rank_colors[rank-1], marker='o', s=80, alpha=0.8,
                           edgecolors='darkred', linewidth=1)
            
            # Plot non-seizure detections
            non_seizure_mask = ~rank_seizure
            if np.any(non_seizure_mask):
                ax2.scatter(rank_m[non_seizure_mask], rank_locations[non_seizure_mask], 
                           c=rank_colors[rank-1], marker='s', s=60, alpha=0.6,
                           edgecolors='darkblue', linewidth=1)
        
        ax2.set_xlabel('m-Wert (Subsequence Length)')
        ax2.set_ylabel('Anomaly Location (Sample Index)')
        ax2.set_title(f'Top-{self.top_k} Anomaly Locations by m-value')
        ax2.grid(True, alpha=0.3)
        
        # Mark seizure regions on location plot
        for start, end in seizure_regions:
            ax2.axhspan(start, end, alpha=0.2, color='red', 
                       label='True Seizure Region' if start == seizure_regions[0][0] else "")
        
        # Add legend for shapes
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                      markersize=10, label='Seizure detected'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
                      markersize=8, label='No seizure'),
        ]
        if seizure_regions:
            legend_elements.append(patches.Patch(color='red', alpha=0.2, label='True Seizure Region'))
        
        ax2.legend(handles=legend_elements, loc='upper right')
        
        # Add statistics text
        total_detections = len(all_m_vals)
        seizure_detections = sum(all_is_seizure)
        detection_rate = seizure_detections / total_detections if total_detections > 0 else 0
        
        # Count by rank
        rank_stats = []
        for rank in range(1, self.top_k + 1):
            rank_count = sum(1 for r in all_ranks if r == rank)
            rank_seizures = sum(1 for i, r in enumerate(all_ranks) 
                              if r == rank and all_is_seizure[i])
            rank_stats.append(f"Rank {rank}: {rank_seizures}/{rank_count}")
        
        stats_text = (
            f"Statistics:\n"
            f"Total detections: {total_detections}\n"
            f"Seizure hits: {seizure_detections}\n"
            f"Overall detection rate: {detection_rate:.2%}\n"
            f"Sampling rate: {sampling_rate} Hz\n"
            f"True seizure regions: {len(seizure_regions)}\n"
            f"Per rank: {', '.join(rank_stats)}"
        )
        
        # Add text box with statistics
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"madrid_top{self.top_k}_analysis_{Path(file_name).stem}_{timestamp}.png"
        plot_path = output_path / plot_filename
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot gespeichert: {plot_path}")
        
        # Show plot
        plt.show()
        
        return plot_path

def main():
    parser = argparse.ArgumentParser(description="Madrid Single File Analysis - Top K Anomalies")
    parser.add_argument("pkl_file", help="Pfad zur PKL-Datei")
    parser.add_argument("--m-range", nargs=2, type=int, metavar=("MIN", "MAX"), 
                       default=[250, 3000], help="m-Bereich (Default: 250-3000)")
    parser.add_argument("--step", type=int, default=250, help="Schrittweite zwischen m-Werten")
    parser.add_argument("--top-k", type=int, default=3, help="Anzahl Top-Anomalien pro m-Wert")
    parser.add_argument("--no-gpu", action="store_true", help="GPU deaktivieren")
    parser.add_argument("--overlap-threshold", type=float, default=0.1, 
                       help="Minimaler Überlappungsanteil für Seizure-Erkennung")
    
    args = parser.parse_args()
    
    # Validate input file
    pkl_file = Path(args.pkl_file)
    if not pkl_file.exists():
        print(f"Fehler: Datei {pkl_file} existiert nicht.")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path("Madrid/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate m values
    min_m, max_m = args.m_range
    m_values = list(range(min_m, max_m + 1, args.step))
    
    print(f"Analysiere Datei: {pkl_file}")
    print(f"m-Werte: {min_m} bis {max_m} (Schritt: {args.step})")
    print(f"Anzahl m-Werte: {len(m_values)}")
    print(f"Top-K Anomalien: {args.top_k}")
    print(f"GPU: {'Deaktiviert' if args.no_gpu else 'Aktiviert'}")
    
    # Initialize analyzer
    analyzer = MadridTop3Analyzer(use_gpu=not args.no_gpu, top_k=args.top_k)
    
    # Load data
    print("Lade Daten...")
    signal, seizure_regions, sampling_rate = analyzer.load_pickle_file(pkl_file)
    print(f"Signal-Länge: {len(signal)} Samples")
    print(f"Sampling-Rate: {sampling_rate} Hz")
    print(f"Seizure-Regionen: {len(seizure_regions)}")
    
    # Run analysis
    print("Führe Madrid-Analyse durch...")
    start_time = time.time()
    results = analyzer.run_madrid_analysis(signal, m_values)
    analysis_time = time.time() - start_time
    
    print(f"Analyse abgeschlossen in {analysis_time:.1f} Sekunden")
    
    # Create plot
    print("Erstelle Plot...")
    plot_path = analyzer.create_plot(
        m_values, results, seizure_regions, sampling_rate, 
        pkl_file.name, output_dir
    )
    
    # Save results to text file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"madrid_top{args.top_k}_results_{pkl_file.stem}_{timestamp}.txt"
    results_path = output_dir / results_filename
    
    with open(results_path, 'w') as f:
        f.write(f"Madrid Top-{args.top_k} Anomalies Analysis Results\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"File: {pkl_file}\n")
        f.write(f"Analysis time: {analysis_time:.1f} seconds\n")
        f.write(f"Signal length: {len(signal)} samples\n")
        f.write(f"Sampling rate: {sampling_rate} Hz\n")
        f.write(f"Seizure regions: {len(seizure_regions)}\n")
        f.write(f"m-range: {min_m} to {max_m} (step: {args.step})\n")
        f.write(f"Top-K: {args.top_k}\n")
        f.write(f"GPU used: {'No' if args.no_gpu else 'Yes'}\n\n")
        
        f.write("Results per m-value:\n")
        f.write("m\tRank\tScore\tLocation\tSeizure_Hit\n")
        
        for m in sorted(results.keys()):
            anomalies = results[m]
            if anomalies:
                for rank, (score, location) in enumerate(anomalies, 1):
                    is_hit = analyzer.check_seizure_overlap(
                        location, m, seizure_regions, args.overlap_threshold
                    )
                    f.write(f"{m}\t{rank}\t{score:.4f}\t{location}\t{is_hit}\n")
            else:
                f.write(f"{m}\t-\t0.0000\tNone\tFalse\n")
    
    print(f"Ergebnisse gespeichert: {results_path}")
    print("Analyse abgeschlossen!")

if __name__ == "__main__":
    main()