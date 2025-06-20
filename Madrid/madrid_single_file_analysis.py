#!/usr/bin/env python3
"""
Madrid Single File Analysis Script

Analysiert eine einzelne PKL-Datei mit Madrid für verschiedene m-Werte und erstellt
einen Plot, der Location und Score der gefundenen Anomalien zeigt, farblich markiert
ob es tatsächlich eine Seizure ist oder nicht.

Usage:
    python madrid_single_file_analysis.py <pkl_file_path> --m-range 250 3000 --step 250
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

class MadridSingleFileAnalyzer:
    """Analysiert eine einzelne PKL-Datei mit Madrid für verschiedene m-Werte."""
    
    def __init__(self, use_gpu: bool = True, train_ratio: float = 1/3):
        self.use_gpu = use_gpu
        self.train_ratio = train_ratio
        
    def load_pickle_file(self, file_path: Path) -> Tuple[np.ndarray, List[Tuple[int, int]], int]:
        """
        Lädt eine PKL-Datei und extrahiert Signal, Seizure-Regionen und Sampling-Rate.
        
        Returns:
            signal (np.ndarray): ECG Signal
            seizure_regions (List[Tuple[int, int]]): Liste von (start, end) Indizes für Seizures
            sampling_rate (int): Sampling-Rate des Signals
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
    
    def run_madrid_analysis(self, signal: np.ndarray, m_values: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Führt Madrid-Analyse EINMAL für alle m-Werte durch - viel effizienter!
        
        Args:
            signal: ECG Signal
            m_values: Liste der zu testenden m-Werte
            
        Returns:
            Dict mit Ergebnissen für jeden m-Wert: {m: {"score": float, "location": int}}
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
        
        try:
            madrid = MADRID(use_gpu=self.use_gpu, enable_output=False)
            
            # Run Madrid ONCE for entire m range
            multi_length_table, bsf, bsf_loc = madrid.fit(
                T=signal,
                min_length=min_m,
                max_length=max_m,
                step_size=step_size,
                train_test_split=train_test_split,
            )
            
            # Debug: Print array shapes and contents
            print(f"    Debug: BSF shape: {bsf.shape}, BSF_loc shape: {bsf_loc.shape}")
            print(f"    Debug: Multi-length table shape: {multi_length_table.shape}")
            print(f"    Debug: Expected m-values: {list(range(min_m, max_m + 1, step_size))}")
            print(f"    Debug: BSF values: {bsf}")
            print(f"    Debug: BSF_loc values: {bsf_loc}")
            
            # Alternative approach: Extract from multi_length_table directly
            m_range = list(range(min_m, max_m + 1, step_size))
            print(f"    Debug: Processing {len(m_range)} m-values from multi_length_table")
            
            for i, m in enumerate(m_range):
                if m in m_values and i < multi_length_table.shape[0]:
                    # Find best discord in this row (for this m-value)
                    row = multi_length_table[i, train_test_split:]  # Only test portion
                    if len(row) > 0 and not np.all(np.isinf(row)):
                        max_idx = np.nanargmax(row)
                        score = float(row[max_idx]) if not np.isnan(row[max_idx]) else 0.0
                        location = int(max_idx + train_test_split) if not np.isnan(row[max_idx]) else None
                    else:
                        score = 0.0
                        location = None
                    
                    results[m] = {
                        "score": score,
                        "location": location
                    }
                    print(f"    m={m}: score={score:.4f}, location={location}")
                elif m in m_values:
                    print(f"    m={m}: Index {i} außerhalb multi_length_table-Bereich")
            
            # Fill in any missing m values with zeros
            for m in m_values:
                if m not in results:
                    print(f"    m={m}: Nicht gefunden, setze auf 0")
                    results[m] = {"score": 0.0, "location": None}
                    
        except ValueError as e:
            if "constant" in str(e).lower():
                print(f"    Warnung: Konstante Regionen erkannt, setze alle Werte auf 0...")
                for m in m_values:
                    results[m] = {"score": 0.0, "location": None}
            else:
                raise
        except Exception as e:
            print(f"    Fehler bei MADRID-Ausführung: {e}")
            for m in m_values:
                results[m] = {"score": 0.0, "location": None}
        
        return results
    
    def check_seizure_overlap(self, location: int, m: int, seizure_regions: List[Tuple[int, int]], 
                            overlap_threshold: float = 0.1) -> bool:
        """
        Prüft, ob die gefundene Anomalie mit einer bekannten Seizure überlappt.
        
        Args:
            location: Start-Position der gefundenen Anomalie
            m: Länge der Anomalie
            seizure_regions: Liste der bekannten Seizure-Regionen
            overlap_threshold: Minimaler Überlappungsanteil
            
        Returns:
            True wenn Overlap >= threshold, sonst False
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
            
            # Check if overlap is significant relative to anomaly or seizure length
            anomaly_length = anomaly_end - anomaly_start
            seizure_length = seizure_end - seizure_start
            min_length = min(anomaly_length, seizure_length)
            
            if overlap_length / min_length >= overlap_threshold:
                return True
                
        return False
    
    def create_plot(self, m_values: List[int], results: Dict[int, Dict[str, Any]], 
                   seizure_regions: List[Tuple[int, int]], sampling_rate: int, 
                   file_name: str, output_path: Path):
        """
        Erstellt den gewünschten Plot mit m-Werten, Locations, Scores und Seizure-Markierung.
        """
        # Prepare data for plotting
        valid_results = [(m, res) for m, res in results.items() 
                        if res["location"] is not None and res["score"] > 0]
        
        if not valid_results:
            print("Keine gültigen Ergebnisse zum Plotten gefunden.")
            return
        
        m_vals = [m for m, _ in valid_results]
        locations = [res["location"] for _, res in valid_results]
        scores = [res["score"] for _, res in valid_results]
        
        # Check seizure overlaps
        is_seizure = [
            self.check_seizure_overlap(res["location"], m, seizure_regions)
            for m, res in valid_results
        ]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle(f'Madrid Anomaly Detection Analysis\nFile: {file_name}', fontsize=14, fontweight='bold')
        
        # Plot 1: Score vs m-value (colored by seizure detection)
        colors = ['red' if seizure else 'blue' for seizure in is_seizure]
        scatter1 = ax1.scatter(m_vals, scores, c=colors, alpha=0.7, s=60)
        ax1.set_xlabel('m-Wert (Subsequence Length)')
        ax1.set_ylabel('Anomaly Score')
        ax1.set_title('Anomaly Scores by m-value')
        ax1.grid(True, alpha=0.3)
        
        # Add legend for colors
        red_patch = patches.Patch(color='red', label='Seizure detected')
        blue_patch = patches.Patch(color='blue', label='No seizure')
        ax1.legend(handles=[red_patch, blue_patch], loc='upper right')
        
        # Plot 2: Location vs m-value (colored by seizure detection)
        scatter2 = ax2.scatter(m_vals, locations, c=colors, alpha=0.7, s=60)
        ax2.set_xlabel('m-Wert (Subsequence Length)')
        ax2.set_ylabel('Anomaly Location (Sample Index)')
        ax2.set_title('Anomaly Locations by m-value')
        ax2.grid(True, alpha=0.3)
        
        # Mark seizure regions on location plot
        y_min, y_max = ax2.get_ylim()
        for start, end in seizure_regions:
            ax2.axhspan(start, end, alpha=0.2, color='red', 
                       label='True Seizure Region' if start == seizure_regions[0][0] else "")
        
        if seizure_regions:
            ax2.legend(loc='upper right')
        
        # Add statistics text
        total_detections = len(valid_results)
        seizure_detections = sum(is_seizure)
        detection_rate = seizure_detections / total_detections if total_detections > 0 else 0
        
        stats_text = (
            f"Statistics:\n"
            f"Total detections: {total_detections}\n"
            f"Seizure hits: {seizure_detections}\n"
            f"Detection rate: {detection_rate:.2%}\n"
            f"Sampling rate: {sampling_rate} Hz\n"
            f"True seizure regions: {len(seizure_regions)}"
        )
        
        # Add text box with statistics
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"madrid_analysis_{Path(file_name).stem}_{timestamp}.png"
        plot_path = output_path / plot_filename
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot gespeichert: {plot_path}")
        
        # Show plot
        plt.show()
        
        return plot_path

def main():
    parser = argparse.ArgumentParser(description="Madrid Single File Analysis")
    parser.add_argument("pkl_file", help="Pfad zur PKL-Datei")
    parser.add_argument("--m-range", nargs=2, type=int, metavar=("MIN", "MAX"), 
                       default=[250, 3000], help="m-Bereich (Default: 250-3000)")
    parser.add_argument("--step", type=int, default=250, help="Schrittweite zwischen m-Werten")
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
    print(f"GPU: {'Deaktiviert' if args.no_gpu else 'Aktiviert'}")
    
    # Initialize analyzer
    analyzer = MadridSingleFileAnalyzer(use_gpu=not args.no_gpu)
    
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
    results_filename = f"madrid_results_{pkl_file.stem}_{timestamp}.txt"
    results_path = output_dir / results_filename
    
    with open(results_path, 'w') as f:
        f.write(f"Madrid Single File Analysis Results\n")
        f.write(f"===================================\n\n")
        f.write(f"File: {pkl_file}\n")
        f.write(f"Analysis time: {analysis_time:.1f} seconds\n")
        f.write(f"Signal length: {len(signal)} samples\n")
        f.write(f"Sampling rate: {sampling_rate} Hz\n")
        f.write(f"Seizure regions: {len(seizure_regions)}\n")
        f.write(f"m-range: {min_m} to {max_m} (step: {args.step})\n")
        f.write(f"GPU used: {'No' if args.no_gpu else 'Yes'}\n\n")
        
        f.write("Results per m-value:\n")
        f.write("m\tScore\tLocation\tSeizure_Hit\n")
        
        for m in sorted(results.keys()):
            res = results[m]
            if res["location"] is not None:
                is_hit = analyzer.check_seizure_overlap(
                    res["location"], m, seizure_regions, args.overlap_threshold
                )
                f.write(f"{m}\t{res['score']:.4f}\t{res['location']}\t{is_hit}\n")
            else:
                f.write(f"{m}\t{res['score']:.4f}\tNone\tFalse\n")
    
    print(f"Ergebnisse gespeichert: {results_path}")
    print("Analyse abgeschlossen!")

if __name__ == "__main__":
    main()