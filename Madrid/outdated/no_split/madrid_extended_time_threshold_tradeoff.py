#!/usr/bin/env python3
"""
Madrid Extended Time Window Threshold Trade-off Analysis
Tests multiple thresholds and plots Sensitivity vs False Alarms per Hour trade-off
with extended time windows (5 minutes before, 3 minutes after seizures).
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import argparse

# Import the main metrics calculator
from madrid_extended_time_metrics import MadridExtendedTimeMetrics


class MadridExtendedTimeThresholdTradeoffAnalyzer:
    def __init__(self, results_dir: str, output_dir: str = None, 
                 pre_seizure_minutes: float = 5.0, post_seizure_minutes: float = 3.0):
        """
        Initialize the extended time threshold trade-off analyzer.
        
        Args:
            results_dir: Directory containing Madrid windowed results JSON files
            output_dir: Directory to save plots and results (default: same as results_dir)
            pre_seizure_minutes: Minutes before seizure start to consider as detection window
            post_seizure_minutes: Minutes after seizure end to consider as detection window
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir) if output_dir else self.results_dir
        self.pre_seizure_minutes = pre_seizure_minutes
        self.post_seizure_minutes = post_seizure_minutes
        self.output_dir.mkdir(exist_ok=True)
        
    def extract_all_anomaly_scores(self) -> List[float]:
        """Extract all anomaly scores from all result files to determine threshold range."""
        json_files = list(self.results_dir.glob("madrid_windowed_results_*.json"))
        all_scores = []
        
        print(f"Extracting anomaly scores from {len(json_files)} files...")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                window_results = data.get('analysis_results', {}).get('window_results', [])
                for window in window_results:
                    anomalies = window.get('anomalies', [])
                    for anomaly in anomalies:
                        score = anomaly.get('anomaly_score', 0)
                        if score > 0:  # Only include positive scores
                            all_scores.append(score)
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue
        
        all_scores.sort()
        print(f"Extracted {len(all_scores)} anomaly scores")
        print(f"Score range: {min(all_scores):.6f} to {max(all_scores):.6f}")
        
        return all_scores
    
    def generate_threshold_range(self, all_scores: List[float], num_thresholds: int = 50) -> List[float]:
        """Generate a range of thresholds to test based on anomaly score distribution."""
        if not all_scores:
            return []
        
        # Use percentiles to generate thresholds
        percentiles = np.linspace(0, 99.9, num_thresholds)
        thresholds = [np.percentile(all_scores, p) for p in percentiles]
        
        # Remove duplicates and sort
        thresholds = sorted(list(set(thresholds)))
        
        print(f"Generated {len(thresholds)} unique thresholds")
        print(f"Threshold range: {min(thresholds):.6f} to {max(thresholds):.6f}")
        
        return thresholds
    
    def run_threshold_analysis(self, thresholds: List[float]) -> List[Dict[str, Any]]:
        """Run metrics calculation for each threshold."""
        results = []
        
        print(f"Running extended time analysis for {len(thresholds)} thresholds...")
        print(f"Extended window: -{self.pre_seizure_minutes:.1f} min to +{self.post_seizure_minutes:.1f} min")
        
        for i, threshold in enumerate(thresholds):
            print(f"Processing threshold {i+1}/{len(thresholds)}: {threshold:.6f}")
            
            # Initialize calculator with this threshold
            calculator = MadridExtendedTimeMetrics(
                results_dir=str(self.results_dir),
                output_dir=None,  # Don't save individual results
                threshold=threshold,
                pre_seizure_minutes=self.pre_seizure_minutes,
                post_seizure_minutes=self.post_seizure_minutes
            )
            
            # Process all files
            try:
                metrics_result = calculator.process_all_files()
                if metrics_result is not None:
                    summary = metrics_result['summary_statistics']
                    
                    results.append({
                        'threshold': threshold,
                        'sensitivity_file_level': summary.get('file_level_sensitivity', 0.0),
                        'sensitivity_seizure_level': summary.get('seizure_level_sensitivity', 0.0),
                        'false_alarms_per_hour': summary.get('overall_false_alarms_per_hour', 0.0),
                        'total_detections': summary.get('total_detections', 0),
                        'total_seizures': summary.get('total_seizures', 0),
                        'total_detected_seizures': summary.get('total_detected_seizures', 0),
                        'files_processed': summary.get('total_files_processed', 0),
                        'files_with_seizures': summary.get('files_with_seizures', 0),
                        'seizure_files_detected': summary.get('seizure_files_detected', 0),
                        'total_duration_hours': summary.get('total_duration_hours', 0.0),
                        'pre_seizure_minutes': self.pre_seizure_minutes,
                        'post_seizure_minutes': self.post_seizure_minutes
                    })
                else:
                    print(f"No results for threshold {threshold}")
            except Exception as e:
                print(f"Error processing threshold {threshold}: {e}")
                continue
        
        return results
    
    def create_tradeoff_plot(self, results: List[Dict[str, Any]], output_filename: str):
        """Create and save the sensitivity vs false alarms per hour trade-off plot."""
        
        # Extract data for plotting
        thresholds = [r['threshold'] for r in results]
        sensitivity_file = [r['sensitivity_file_level'] for r in results]
        sensitivity_seizure = [r['sensitivity_seizure_level'] for r in results]
        false_alarms_per_hour = [r['false_alarms_per_hour'] for r in results]
        total_detections = [r['total_detections'] for r in results]
        
        # Create the main trade-off plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: File-Level Sensitivity vs False Alarms per Hour
        ax1.plot(false_alarms_per_hour, sensitivity_file, 'b.-', linewidth=2, markersize=4)
        ax1.set_xlabel('False Alarms per Hour')
        ax1.set_ylabel('File-Level Sensitivity')
        ax1.set_title(f'File-Level Sensitivity vs False Alarms per Hour\n(Madrid Extended Time: -{self.pre_seizure_minutes:.1f}min to +{self.post_seizure_minutes:.1f}min)')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(left=0)
        ax1.set_ylim(0, 1)
        
        # Add some threshold annotations for key points
        if len(results) > 0:
            # Find some interesting points to annotate
            max_sens_idx = np.argmax(sensitivity_file)
            min_fa_idx = np.argmin([fa for fa in false_alarms_per_hour if fa > 0] or [0])
            
            if sensitivity_file[max_sens_idx] > 0:
                ax1.annotate(f'Max Sens: {sensitivity_file[max_sens_idx]:.3f}\nFA/h: {false_alarms_per_hour[max_sens_idx]:.3f}\nThreshold: {thresholds[max_sens_idx]:.4f}',
                           xy=(false_alarms_per_hour[max_sens_idx], sensitivity_file[max_sens_idx]),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           fontsize=8)
        
        # Plot 2: Seizure-Level Sensitivity vs False Alarms per Hour
        ax2.plot(false_alarms_per_hour, sensitivity_seizure, 'r.-', linewidth=2, markersize=4)
        ax2.set_xlabel('False Alarms per Hour')
        ax2.set_ylabel('Seizure-Level Sensitivity')
        ax2.set_title(f'Seizure-Level Sensitivity vs False Alarms per Hour\n(Madrid Extended Time: -{self.pre_seizure_minutes:.1f}min to +{self.post_seizure_minutes:.1f}min)')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(left=0)
        ax2.set_ylim(0, 1)
        
        # Plot 3: Both sensitivities on same plot
        ax3.plot(false_alarms_per_hour, sensitivity_file, 'b.-', label='File-Level Sensitivity', linewidth=2, markersize=4)
        ax3.plot(false_alarms_per_hour, sensitivity_seizure, 'r.-', label='Seizure-Level Sensitivity', linewidth=2, markersize=4)
        ax3.set_xlabel('False Alarms per Hour')
        ax3.set_ylabel('Sensitivity')
        ax3.set_title(f'Combined Sensitivity Comparison\n(Madrid Extended Time: -{self.pre_seizure_minutes:.1f}min to +{self.post_seizure_minutes:.1f}min)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(left=0)
        ax3.set_ylim(0, 1)
        
        # Plot 4: Total detections vs threshold
        ax4.semilogy(thresholds, total_detections, 'g.-', linewidth=2, markersize=4)
        ax4.set_xlabel('Threshold')
        ax4.set_ylabel('Total Detections (log scale)')
        ax4.set_title(f'Total Detections vs Threshold\n(Madrid Extended Time: -{self.pre_seizure_minutes:.1f}min to +{self.post_seizure_minutes:.1f}min)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = self.output_dir / output_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Extended time trade-off plot saved to: {plot_path}")
        
        # Also save as PDF for high quality
        pdf_path = self.output_dir / output_filename.replace('.png', '.pdf')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Extended time trade-off plot (PDF) saved to: {pdf_path}")
        
        plt.close()
    
    def save_results(self, results: List[Dict[str, Any]], output_filename: str):
        """Save detailed results to JSON and CSV files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_path = self.output_dir / f"{output_filename}_results.json"
        with open(json_path, 'w') as f:
            json.dump({
                'analysis_metadata': {
                    'analysis_type': 'threshold_tradeoff_extended_time',
                    'timestamp': timestamp,
                    'results_directory': str(self.results_dir),
                    'num_thresholds_tested': len(results),
                    'pre_seizure_minutes': self.pre_seizure_minutes,
                    'post_seizure_minutes': self.post_seizure_minutes
                },
                'results': results
            }, f, indent=2)
        print(f"Detailed results saved to: {json_path}")
        
        # Save CSV for easy analysis
        csv_path = self.output_dir / f"{output_filename}_results.csv"
        with open(csv_path, 'w') as f:
            f.write("threshold,sensitivity_file_level,sensitivity_seizure_level,false_alarms_per_hour,total_detections,total_seizures,total_detected_seizures,files_processed,files_with_seizures,seizure_files_detected,total_duration_hours,pre_seizure_minutes,post_seizure_minutes\n")
            for r in results:
                f.write(f"{r['threshold']},{r['sensitivity_file_level']},{r['sensitivity_seizure_level']},{r['false_alarms_per_hour']},{r['total_detections']},{r['total_seizures']},{r['total_detected_seizures']},{r['files_processed']},{r['files_with_seizures']},{r['seizure_files_detected']},{r['total_duration_hours']},{r['pre_seizure_minutes']},{r['post_seizure_minutes']}\n")
        print(f"CSV results saved to: {csv_path}")
    
    def run_full_analysis(self, num_thresholds: int = 50):
        """Run the complete threshold trade-off analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("Starting Madrid Extended Time Threshold Trade-off Analysis...")
        print(f"Results directory: {self.results_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Extended time window: -{self.pre_seizure_minutes:.1f} min to +{self.post_seizure_minutes:.1f} min")
        
        # Step 1: Extract all anomaly scores
        all_scores = self.extract_all_anomaly_scores()
        if not all_scores:
            print("No anomaly scores found!")
            return None
        
        # Step 2: Generate threshold range
        thresholds = self.generate_threshold_range(all_scores, num_thresholds)
        if not thresholds:
            print("Could not generate thresholds!")
            return None
        
        # Step 3: Run analysis for each threshold
        results = self.run_threshold_analysis(thresholds)
        if not results:
            print("No results generated!")
            return None
        
        # Step 4: Create plots
        plot_filename = f"madrid_extended_time_threshold_tradeoff_{timestamp}.png"
        self.create_tradeoff_plot(results, plot_filename)
        
        # Step 5: Save results
        results_filename = f"madrid_extended_time_threshold_tradeoff_{timestamp}"
        self.save_results(results, results_filename)
        
        # Step 6: Print summary
        print(f"\n{'='*60}")
        print("EXTENDED TIME THRESHOLD TRADE-OFF ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"Extended window: -{self.pre_seizure_minutes:.1f} min to +{self.post_seizure_minutes:.1f} min")
        print(f"Total thresholds tested: {len(results)}")
        if results:
            best_file_sens = max(results, key=lambda x: x['sensitivity_file_level'])
            best_seizure_sens = max(results, key=lambda x: x['sensitivity_seizure_level'])
            min_fa = min([r for r in results if r['false_alarms_per_hour'] > 0], 
                        key=lambda x: x['false_alarms_per_hour'], default=results[0])
            
            print(f"\nBest File-Level Sensitivity: {best_file_sens['sensitivity_file_level']:.4f}")
            print(f"  Threshold: {best_file_sens['threshold']:.6f}")
            print(f"  False Alarms/Hour: {best_file_sens['false_alarms_per_hour']:.4f}")
            
            print(f"\nBest Seizure-Level Sensitivity: {best_seizure_sens['sensitivity_seizure_level']:.4f}")
            print(f"  Threshold: {best_seizure_sens['threshold']:.6f}")
            print(f"  False Alarms/Hour: {best_seizure_sens['false_alarms_per_hour']:.4f}")
            
            print(f"\nMinimum False Alarms/Hour: {min_fa['false_alarms_per_hour']:.4f}")
            print(f"  Threshold: {min_fa['threshold']:.6f}")
            print(f"  File-Level Sensitivity: {min_fa['sensitivity_file_level']:.4f}")
            print(f"  Seizure-Level Sensitivity: {min_fa['sensitivity_seizure_level']:.4f}")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate threshold trade-off analysis for Madrid extended time window metrics"
    )
    parser.add_argument(
        "results_dir", 
        help="Directory containing Madrid windowed results JSON files"
    )
    parser.add_argument(
        "-o", "--output-dir", 
        help="Output directory for plots and results (default: same as results_dir)"
    )
    parser.add_argument(
        "-n", "--num-thresholds", 
        type=int,
        default=50,
        help="Number of thresholds to test (default: 50)"
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
    analyzer = MadridExtendedTimeThresholdTradeoffAnalyzer(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        pre_seizure_minutes=args.pre_seizure_minutes,
        post_seizure_minutes=args.post_seizure_minutes
    )
    
    # Run analysis
    results = analyzer.run_full_analysis(args.num_thresholds)
    
    if results is None:
        print("Analysis failed.")
        return 1
    
    print(f"\nAnalysis completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())