#!/usr/bin/env python3
"""
Seizure Duration Analysis Script

This script analyzes seizure durations from preprocessed seizure data files.
It calculates statistics about seizure lengths including mean, median, 
distribution, and provides insights for optimal MADRID parameter selection.

Key features:
- Analyze seizure durations from ictal phase labels
- Calculate comprehensive statistics (mean, median, percentiles, etc.)
- Generate duration distribution analysis
- Export results in human-readable format
- Support for different file formats and sampling rates
- Provide recommendations for MADRID m-parameter ranges

Usage:
    python seizure_duration_analysis.py --data-dir DATA_DIR
    python seizure_duration_analysis.py --data-dir DATA_DIR --output seizure_durations.txt
    python seizure_duration_analysis.py --data-dir DATA_DIR --subject sub-001 --detailed
    python seizure_duration_analysis.py --data-dir DATA_DIR --percentage 50 --plot


"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import os
import argparse
from pathlib import Path
import warnings
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
from collections import defaultdict
import random

warnings.filterwarnings('ignore')


class SeizureDurationAnalyzer:
    """
    Analyzer for seizure duration statistics from preprocessed data files.
    
    Extracts and analyzes ictal phase durations to understand seizure characteristics.
    """
    
    def __init__(self):
        """Initialize the seizure duration analyzer."""
        self.seizure_data = []
        self.duration_stats = {}
        
        print("Initialized SeizureDurationAnalyzer")
    
    def discover_seizure_files(self, data_dir: str, percentage: float = 100.0, 
                              subject_filter: str = None) -> List[Path]:
        """
        Discover seizure files in the given directory.
        
        Args:
            data_dir: Directory containing seizure files
            percentage: Percentage of files to analyze (0-100)
            subject_filter: Optional subject filter
            
        Returns:
            List of seizure file paths
        """
        data_path = Path(data_dir)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_path}")
        
        # Find all seizure files - try different patterns
        seizure_files = list(data_path.glob("*_seizure_*_preprocessed.pkl"))
        if not seizure_files:
            seizure_files = list(data_path.glob("*seizure*.pkl"))
            if not seizure_files:
                seizure_files = list(data_path.glob("*.pkl"))
        
        print(f"Found {len(seizure_files)} potential seizure files in {data_path}")
        
        # Apply subject filter
        if subject_filter:
            seizure_files = [f for f in seizure_files if subject_filter in f.name]
            print(f"After subject filter '{subject_filter}': {len(seizure_files)} files")
        
        if not seizure_files:
            raise ValueError("No seizure files found matching criteria")
        
        # Calculate number of files to process
        n_files_to_process = max(1, int(len(seizure_files) * percentage / 100))
        
        # Randomly sample files for representative results
        random.seed(42)  # For reproducible results
        selected_files = random.sample(seizure_files, n_files_to_process)
        
        print(f"Selected {len(selected_files)} files ({percentage}% of {len(seizure_files)} total)")
        
        return sorted(selected_files)
    
    def load_seizure_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load and analyze a single seizure file.
        
        Args:
            file_path: Path to seizure file
            
        Returns:
            Dictionary with seizure information or None if failed
        """
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Validate data structure
            if 'channels' not in data or not data['channels']:
                print(f"  Warning: No channels in {file_path.name}")
                return None
            
            channel = data['channels'][0]
            if 'data' not in channel or 'labels' not in channel:
                print(f"  Warning: Missing data or labels in {file_path.name}")
                return None
            
            # Extract metadata
            fs = data.get('sampling_rate', None)
            if fs is None and 'metadata' in data:
                fs = data['metadata'].get('sampling_rate', 125)
            if fs is None:
                fs = 125  # Default fallback
                print(f"  Warning: Using default sampling rate {fs} Hz for {file_path.name}")
            
            # Extract basic information
            labels = channel['labels']
            data_length = len(channel['data'])
            total_duration = data_length / fs
            
            # Find seizure durations
            seizure_regions = self._find_seizure_regions(labels, fs)
            
            # Extract file metadata
            file_info = {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'subject_id': data.get('subject_id', 'unknown'),
                'run_id': data.get('run_id', 'unknown'), 
                'seizure_index': data.get('seizure_index', 0),
                'sampling_rate': fs,
                'total_duration': total_duration,
                'data_length': data_length,
                'seizure_regions': seizure_regions,
                'n_seizures': len(seizure_regions)
            }
            
            # Calculate seizure-specific statistics
            if seizure_regions:
                seizure_durations = [region['duration'] for region in seizure_regions]
                file_info.update({
                    'seizure_durations': seizure_durations,
                    'total_seizure_duration': sum(seizure_durations),
                    'mean_seizure_duration': np.mean(seizure_durations),
                    'median_seizure_duration': np.median(seizure_durations),
                    'min_seizure_duration': min(seizure_durations),
                    'max_seizure_duration': max(seizure_durations),
                    'std_seizure_duration': np.std(seizure_durations),
                    'seizure_coverage': sum(seizure_durations) / total_duration * 100
                })
            else:
                file_info.update({
                    'seizure_durations': [],
                    'total_seizure_duration': 0.0,
                    'mean_seizure_duration': 0.0,
                    'median_seizure_duration': 0.0,
                    'min_seizure_duration': 0.0,
                    'max_seizure_duration': 0.0,
                    'std_seizure_duration': 0.0,
                    'seizure_coverage': 0.0
                })
            
            return file_info
            
        except Exception as e:
            print(f"  Error loading {file_path.name}: {e}")
            return None
    
    def _find_seizure_regions(self, labels: np.ndarray, fs: int) -> List[Dict[str, Any]]:
        """Find continuous seizure (ictal) regions in the data."""
        seizure_regions = []
        
        # Find ictal segments
        ictal_mask = (labels == 'ictal')
        if not np.any(ictal_mask):
            return seizure_regions
        
        # Find continuous ictal regions
        ictal_indices = np.where(ictal_mask)[0]
        
        if len(ictal_indices) == 0:
            return seizure_regions
        
        # Group consecutive indices
        region_start = ictal_indices[0]
        region_end = ictal_indices[0]
        
        for i in range(1, len(ictal_indices)):
            if ictal_indices[i] == region_end + 1:
                region_end = ictal_indices[i]
            else:
                # Add completed region
                seizure_regions.append({
                    'start_sample': region_start,
                    'end_sample': region_end,
                    'start_time': region_start / fs,
                    'end_time': region_end / fs,
                    'duration': (region_end - region_start + 1) / fs,
                    'n_samples': region_end - region_start + 1
                })
                region_start = ictal_indices[i]
                region_end = ictal_indices[i]
        
        # Add final region
        seizure_regions.append({
            'start_sample': region_start,
            'end_sample': region_end,
            'start_time': region_start / fs,
            'end_time': region_end / fs,
            'duration': (region_end - region_start + 1) / fs,
            'n_samples': region_end - region_start + 1
        })
        
        return seizure_regions
    
    def analyze_seizure_durations(self, file_paths: List[Path]) -> Dict[str, Any]:
        """
        Analyze seizure durations across multiple files.
        
        Args:
            file_paths: List of seizure file paths
            
        Returns:
            Comprehensive duration analysis results
        """
        print(f"\n{'='*60}")
        print(f"SEIZURE DURATION ANALYSIS")
        print(f"{'='*60}")
        print(f"Analyzing {len(file_paths)} files...")
        
        all_durations = []
        file_results = []
        subject_durations = defaultdict(list)
        sampling_rates = []
        
        successful_files = 0
        total_seizures = 0
        
        # Process each file
        for i, file_path in enumerate(file_paths):
            print(f"\nProcessing file {i+1}/{len(file_paths)}: {file_path.name}")
            
            file_info = self.load_seizure_file(file_path)
            if file_info is None:
                continue
            
            file_results.append(file_info)
            successful_files += 1
            
            # Collect data for aggregate analysis
            if file_info['seizure_durations']:
                all_durations.extend(file_info['seizure_durations'])
                subject_durations[file_info['subject_id']].extend(file_info['seizure_durations'])
                total_seizures += len(file_info['seizure_durations'])
                sampling_rates.append(file_info['sampling_rate'])
                
                print(f"  ✓ Found {file_info['n_seizures']} seizures")
                print(f"  ✓ Durations: {file_info['min_seizure_duration']:.2f}s - {file_info['max_seizure_duration']:.2f}s")
                print(f"  ✓ Mean: {file_info['mean_seizure_duration']:.2f}s")
            else:
                print(f"  ⚠️  No seizures found")
        
        print(f"\n✓ Successfully processed {successful_files}/{len(file_paths)} files")
        print(f"✓ Total seizures found: {total_seizures}")
        
        if not all_durations:
            print("❌ No seizure durations found in any files")
            return {
                'success': False,
                'message': 'No seizure durations found',
                'file_results': file_results
            }
        
        # Calculate comprehensive statistics
        all_durations = np.array(all_durations)
        
        # Basic statistics
        duration_stats = {
            'n_files': successful_files,
            'n_seizures': total_seizures,
            'n_subjects': len(subject_durations),
            'mean_sampling_rate': np.mean(sampling_rates) if sampling_rates else 125,
            
            # Duration statistics
            'mean_duration': float(np.mean(all_durations)),
            'median_duration': float(np.median(all_durations)),
            'std_duration': float(np.std(all_durations)),
            'min_duration': float(np.min(all_durations)),
            'max_duration': float(np.max(all_durations)),
            
            # Percentiles
            'percentile_5': float(np.percentile(all_durations, 5)),
            'percentile_25': float(np.percentile(all_durations, 25)),
            'percentile_75': float(np.percentile(all_durations, 75)),
            'percentile_95': float(np.percentile(all_durations, 95)),
            
            # Additional statistics
            'range': float(np.max(all_durations) - np.min(all_durations)),
            'coefficient_of_variation': float(np.std(all_durations) / np.mean(all_durations)),
        }
        
        # Distribution analysis
        duration_categories = {
            'very_short': (0.0, 2.0),      # < 2s
            'short': (2.0, 5.0),           # 2-5s
            'medium': (5.0, 15.0),         # 5-15s
            'long': (15.0, 60.0),          # 15-60s
            'very_long': (60.0, float('inf'))  # > 60s
        }
        
        category_counts = {}
        for category, (min_dur, max_dur) in duration_categories.items():
            if max_dur == float('inf'):
                count = np.sum(all_durations >= min_dur)
            else:
                count = np.sum((all_durations >= min_dur) & (all_durations < max_dur))
            category_counts[category] = {
                'count': int(count),
                'percentage': float(count / len(all_durations) * 100),
                'range': (min_dur, max_dur)
            }
        
        # Subject-wise analysis
        subject_stats = {}
        for subject_id, durations in subject_durations.items():
            if durations:
                subject_stats[subject_id] = {
                    'n_seizures': len(durations),
                    'mean_duration': float(np.mean(durations)),
                    'median_duration': float(np.median(durations)),
                    'std_duration': float(np.std(durations)),
                    'min_duration': float(np.min(durations)),
                    'max_duration': float(np.max(durations))
                }
        
        # MADRID parameter recommendations
        madrid_recommendations = self._generate_madrid_recommendations(
            all_durations, duration_stats, category_counts
        )
        
        results = {
            'success': True,
            'duration_stats': duration_stats,
            'category_distribution': category_counts,
            'subject_stats': subject_stats,
            'madrid_recommendations': madrid_recommendations,
            'all_durations': all_durations.tolist(),
            'file_results': file_results
        }
        
        self.seizure_data = file_results
        self.duration_stats = results
        
        return results
    
    def _generate_madrid_recommendations(self, durations: np.ndarray, 
                                       stats: Dict[str, float],
                                       categories: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate MADRID parameter recommendations based on seizure durations."""
        fs = stats['mean_sampling_rate']
        
        # Calculate optimal m-parameter ranges
        recommendations = {
            'sampling_rate': fs,
            
            # Conservative approach - cover most seizures
            'conservative': {
                'min_m_seconds': max(0.1, stats['percentile_5'] * 0.5),
                'max_m_seconds': min(60.0, stats['percentile_95'] * 1.5),
                'description': 'Covers 90% of seizures with safety margin'
            },
            
            # Focused approach - target typical seizures
            'focused': {
                'min_m_seconds': max(0.2, stats['percentile_25'] * 0.8),
                'max_m_seconds': min(30.0, stats['percentile_75'] * 1.2),
                'description': 'Targets central 50% of seizures'
            },
            
            # Aggressive approach - tight around mean
            'aggressive': {
                'min_m_seconds': max(0.1, stats['mean_duration'] * 0.3),
                'max_m_seconds': min(20.0, stats['mean_duration'] * 2.0),
                'description': 'Tight range around mean duration'
            }
        }
        
        # Convert to samples and add practical parameters
        for approach_name, approach in recommendations.items():
            if approach_name == 'sampling_rate':
                continue
                
            min_m = max(2, int(approach['min_m_seconds'] * fs))
            max_m = int(approach['max_m_seconds'] * fs)
            step_m = max(1, int((max_m - min_m) / 20))  # ~20 steps
            
            approach.update({
                'min_m_samples': min_m,
                'max_m_samples': max_m,
                'step_m_samples': step_m,
                'madrid_config': {
                    'min_length': min_m,
                    'max_length': max_m,
                    'step_size': step_m
                }
            })
        
        return recommendations
    
    def save_duration_analysis(self, output_file: str, include_detailed: bool = True,
                             include_recommendations: bool = True):
        """
        Save duration analysis results in human-readable format.
        
        Args:
            output_file: Output file path
            include_detailed: Whether to include detailed per-file results
            include_recommendations: Whether to include MADRID recommendations
        """
        if not self.duration_stats:
            print("No results to save")
            return
        
        results = self.duration_stats
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # Header
            f.write("="*80 + "\n")
            f.write("SEIZURE DURATION ANALYSIS RESULTS\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Files analyzed: {results['duration_stats']['n_files']}\n")
            f.write(f"Total seizures: {results['duration_stats']['n_seizures']}\n")
            f.write(f"Subjects: {results['duration_stats']['n_subjects']}\n")
            f.write(f"Mean sampling rate: {results['duration_stats']['mean_sampling_rate']:.1f} Hz\n\n")
            
            # Executive Summary
            stats = results['duration_stats']
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*20 + "\n")
            f.write(f"Average seizure duration: {stats['mean_duration']:.2f} seconds\n")
            f.write(f"Median seizure duration: {stats['median_duration']:.2f} seconds\n")
            f.write(f"Duration range: {stats['min_duration']:.2f}s - {stats['max_duration']:.2f}s\n")
            f.write(f"Standard deviation: {stats['std_duration']:.2f} seconds\n")
            f.write(f"Coefficient of variation: {stats['coefficient_of_variation']:.2f}\n\n")
            
            # Duration Statistics
            f.write("DETAILED DURATION STATISTICS\n")
            f.write("-"*35 + "\n")
            f.write(f"Mean:                {stats['mean_duration']:8.2f} seconds\n")
            f.write(f"Median:              {stats['median_duration']:8.2f} seconds\n")
            f.write(f"Standard deviation:  {stats['std_duration']:8.2f} seconds\n")
            f.write(f"Minimum:             {stats['min_duration']:8.2f} seconds\n")
            f.write(f"Maximum:             {stats['max_duration']:8.2f} seconds\n")
            f.write(f"Range:               {stats['range']:8.2f} seconds\n\n")
            
            f.write("Percentiles:\n")
            f.write(f"  5th percentile:    {stats['percentile_5']:8.2f} seconds\n")
            f.write(f"  25th percentile:   {stats['percentile_25']:8.2f} seconds\n")
            f.write(f"  75th percentile:   {stats['percentile_75']:8.2f} seconds\n")
            f.write(f"  95th percentile:   {stats['percentile_95']:8.2f} seconds\n\n")
            
            # Duration Categories
            f.write("DURATION DISTRIBUTION\n")
            f.write("-"*25 + "\n")
            categories = results['category_distribution']
            f.write(f"{'Category':<12} {'Range':<15} {'Count':<8} {'Percentage':<10}\n")
            f.write("-"*50 + "\n")
            
            for category, data in categories.items():
                range_str = f"{data['range'][0]:.1f}-{data['range'][1]:.1f}s" if data['range'][1] != float('inf') else f">{data['range'][0]:.1f}s"
                f.write(f"{category:<12} {range_str:<15} {data['count']:<8} {data['percentage']:<10.1f}%\n")
            
            f.write("\n")
            
            # Subject-wise Analysis
            if results['subject_stats']:
                f.write("SUBJECT-WISE ANALYSIS\n")
                f.write("-"*25 + "\n")
                f.write(f"{'Subject':<12} {'N':<4} {'Mean':<8} {'Median':<8} {'Std':<8} {'Min':<8} {'Max':<8}\n")
                f.write("-"*65 + "\n")
                
                subject_stats = results['subject_stats']
                for subject_id, stats in sorted(subject_stats.items()):
                    f.write(f"{subject_id:<12} {stats['n_seizures']:<4} {stats['mean_duration']:<8.2f} "
                           f"{stats['median_duration']:<8.2f} {stats['std_duration']:<8.2f} "
                           f"{stats['min_duration']:<8.2f} {stats['max_duration']:<8.2f}\n")
                
                f.write("\n")
            
            # MADRID Recommendations
            if include_recommendations and 'madrid_recommendations' in results:
                f.write("MADRID PARAMETER RECOMMENDATIONS\n")
                f.write("-"*40 + "\n")
                
                madrid_recs = results['madrid_recommendations']
                fs = madrid_recs['sampling_rate']
                
                f.write(f"Based on seizure duration analysis at {fs:.0f} Hz sampling rate:\n\n")
                
                for approach_name, approach in madrid_recs.items():
                    if approach_name == 'sampling_rate':
                        continue
                    
                    f.write(f"{approach_name.upper()} Approach:\n")
                    f.write(f"  Description: {approach['description']}\n")
                    f.write(f"  Time range: {approach['min_m_seconds']:.2f}s - {approach['max_m_seconds']:.2f}s\n")
                    f.write(f"  Sample range: {approach['min_m_samples']} - {approach['max_m_samples']} samples\n")
                    f.write(f"  Recommended step: {approach['step_m_samples']} samples\n")
                    f.write(f"\n  MADRID Configuration:\n")
                    f.write(f"    madrid.fit(T=data, min_length={approach['min_m_samples']}, \n")
                    f.write(f"               max_length={approach['max_m_samples']}, \n")
                    f.write(f"               step_size={approach['step_m_samples']}, \n")
                    f.write(f"               train_test_split=len(data)//3)\n\n")
                
                # Specific m-value recommendations
                f.write("SPECIFIC m-VALUE RECOMMENDATIONS:\n")
                f.write("-"*40 + "\n")
                
                mean_duration = results['duration_stats']['mean_duration']
                median_duration = results['duration_stats']['median_duration']
                
                specific_m_values = [
                    ("Mean duration", mean_duration, int(mean_duration * fs)),
                    ("Median duration", median_duration, int(median_duration * fs)),
                    ("Half mean", mean_duration * 0.5, int(mean_duration * 0.5 * fs)),
                    ("Double mean", mean_duration * 2.0, int(mean_duration * 2.0 * fs)),
                    ("25th percentile", results['duration_stats']['percentile_25'], 
                     int(results['duration_stats']['percentile_25'] * fs)),
                    ("75th percentile", results['duration_stats']['percentile_75'], 
                     int(results['duration_stats']['percentile_75'] * fs))
                ]
                
                for name, duration_sec, m_samples in specific_m_values:
                    f.write(f"  {name:<18}: m = {m_samples:4d} samples ({duration_sec:5.2f}s)\n")
                
                f.write("\n")
            
            # Detailed per-file results
            if include_detailed and results['file_results']:
                f.write("DETAILED PER-FILE RESULTS\n")
                f.write("-"*30 + "\n")
                f.write(f"{'File':<25} {'Subject':<10} {'N':<3} {'Mean':<8} {'Median':<8} {'Min':<8} {'Max':<8} {'Coverage%':<10}\n")
                f.write("-"*85 + "\n")
                
                for file_info in results['file_results']:
                    if file_info['n_seizures'] > 0:
                        file_name = file_info['file_name']
                        if len(file_name) > 24:
                            file_name = file_name[:21] + "..."
                        
                        f.write(f"{file_name:<25} {file_info['subject_id']:<10} {file_info['n_seizures']:<3} "
                               f"{file_info['mean_seizure_duration']:<8.2f} {file_info['median_seizure_duration']:<8.2f} "
                               f"{file_info['min_seizure_duration']:<8.2f} {file_info['max_seizure_duration']:<8.2f} "
                               f"{file_info['seizure_coverage']:<10.1f}\n")
                
                f.write("\n")
            
            # Statistical Insights
            f.write("STATISTICAL INSIGHTS\n")
            f.write("-"*25 + "\n")
            
            # Variability analysis
            cv = results['duration_stats']['coefficient_of_variation']
            if cv < 0.5:
                f.write("✓ Low variability in seizure durations (CV < 0.5)\n")
                f.write("  → Consistent seizure patterns, narrow m-parameter range recommended\n")
            elif cv < 1.0:
                f.write("⚠ Moderate variability in seizure durations (0.5 ≤ CV < 1.0)\n")
                f.write("  → Some variation in seizure patterns, medium m-parameter range recommended\n")
            else:
                f.write("❌ High variability in seizure durations (CV ≥ 1.0)\n")
                f.write("  → Highly variable seizure patterns, wide m-parameter range recommended\n")
            
            # Duration distribution insights
            short_seizures = categories['very_short']['percentage'] + categories['short']['percentage']
            long_seizures = categories['long']['percentage'] + categories['very_long']['percentage']
            
            f.write(f"\n")
            if short_seizures > 50:
                f.write(f"✓ Predominantly short seizures ({short_seizures:.1f}% ≤ 5s)\n")
                f.write(f"  → Focus on small m-values (0.1-5s range)\n")
            elif long_seizures > 50:
                f.write(f"✓ Predominantly long seizures ({long_seizures:.1f}% ≥ 15s)\n")
                f.write(f"  → Focus on large m-values (5-60s range)\n")
            else:
                f.write(f"✓ Mixed seizure durations\n")
                f.write(f"  → Use comprehensive m-parameter range (0.1-30s)\n")
            
            # Sample size adequacy
            n_seizures = results['duration_stats']['n_seizures']
            f.write(f"\n")
            if n_seizures >= 100:
                f.write(f"✓ Large sample size (n={n_seizures}) - results highly reliable\n")
            elif n_seizures >= 50:
                f.write(f"✓ Good sample size (n={n_seizures}) - results reliable\n")
            elif n_seizures >= 20:
                f.write(f"⚠ Moderate sample size (n={n_seizures}) - results reasonably reliable\n")
            else:
                f.write(f"❌ Small sample size (n={n_seizures}) - results may not be representative\n")
            
            f.write("\n")
            
            # Final Recommendations
            f.write("FINAL RECOMMENDATIONS\n")
            f.write("-"*25 + "\n")
            
            # Choose best approach based on data characteristics
            if cv < 0.5:
                recommended_approach = 'aggressive'
            elif cv < 1.0:
                recommended_approach = 'focused'
            else:
                recommended_approach = 'conservative'
            
            f.write(f"1. RECOMMENDED APPROACH: {recommended_approach.upper()}\n")
            
            if 'madrid_recommendations' in results:
                approach = results['madrid_recommendations'][recommended_approach]
                f.write(f"   Based on your data's variability (CV={cv:.2f})\n")
                f.write(f"   Time range: {approach['min_m_seconds']:.2f}s - {approach['max_m_seconds']:.2f}s\n")
                f.write(f"   Sample range: {approach['min_m_samples']} - {approach['max_m_samples']} samples\n\n")
            
            f.write(f"2. OPTIMAL SINGLE m-VALUE:\n")
            mean_m = int(results['duration_stats']['mean_duration'] * fs)
            f.write(f"   m = {mean_m} samples ({results['duration_stats']['mean_duration']:.2f}s)\n")
            f.write(f"   This represents the mean seizure duration\n\n")
            
            f.write(f"3. TESTING STRATEGY:\n")
            f.write(f"   Start with the {recommended_approach} approach\n")
            f.write(f"   If sensitivity is low, expand to conservative range\n")
            f.write(f"   Use madrid_m_analysis.py to test individual m-values\n")
        
        print(f"✓ Seizure duration analysis saved to: {output_file}")
    
    def plot_duration_distribution(self, output_file: str = None, show_plot: bool = True):
        """
        Create a plot of seizure duration distribution.
        
        Args:
            output_file: Optional file to save plot
            show_plot: Whether to display the plot
        """
        if not self.duration_stats or not self.duration_stats['success']:
            print("No data available for plotting")
            return
        
        durations = np.array(self.duration_stats['all_durations'])
        stats = self.duration_stats['duration_stats']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Seizure Duration Analysis', fontsize=16)
        
        # 1. Histogram
        axes[0, 0].hist(durations, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(stats['mean_duration'], color='red', linestyle='--', 
                          label=f'Mean: {stats["mean_duration"]:.2f}s')
        axes[0, 0].axvline(stats['median_duration'], color='orange', linestyle='--', 
                          label=f'Median: {stats["median_duration"]:.2f}s')
        axes[0, 0].set_xlabel('Duration (seconds)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Seizure Duration Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Box plot
        axes[0, 1].boxplot(durations, vert=True, patch_artist=True,
                          boxprops=dict(facecolor='lightblue'))
        axes[0, 1].set_ylabel('Duration (seconds)')
        axes[0, 1].set_title('Seizure Duration Box Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Cumulative distribution
        sorted_durations = np.sort(durations)
        cumulative = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations) * 100
        axes[1, 0].plot(sorted_durations, cumulative, 'b-', linewidth=2)
        axes[1, 0].axvline(stats['mean_duration'], color='red', linestyle='--', alpha=0.7)
        axes[1, 0].axvline(stats['median_duration'], color='orange', linestyle='--', alpha=0.7)
        axes[1, 0].set_xlabel('Duration (seconds)')
        axes[1, 0].set_ylabel('Cumulative Percentage')
        axes[1, 0].set_title('Cumulative Distribution Function')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Category distribution
        categories = self.duration_stats['category_distribution']
        category_names = list(categories.keys())
        category_percentages = [categories[cat]['percentage'] for cat in category_names]
        colors = ['lightcoral', 'lightsalmon', 'lightgreen', 'lightblue', 'plum']
        
        axes[1, 1].pie(category_percentages, labels=category_names, autopct='%1.1f%%', 
                      colors=colors, startangle=90)
        axes[1, 1].set_title('Duration Categories')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to: {output_file}")
        
        if show_plot:
            plt.show()


def main():
    """Main function for seizure duration analysis."""
    parser = argparse.ArgumentParser(description='Seizure Duration Analysis')
    
    # Required arguments
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing seizure preprocessed files')
    
    # File selection
    parser.add_argument('--percentage', type=float, default=100.0,
                       help='Percentage of files to analyze (default: 100)')
    parser.add_argument('--subject', type=str,
                       help='Filter by subject ID (e.g., sub-001)')
    
    # Output options
    parser.add_argument('--output', type=str, default='seizure_durations.txt',
                       help='Output file for results (default: seizure_durations.txt)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate duration distribution plots')
    parser.add_argument('--plot-file', type=str,
                       help='File to save plots (e.g., duration_plots.png)')
    parser.add_argument('--no-detailed', action='store_true',
                       help='Skip detailed per-file results in output')
    parser.add_argument('--no-recommendations', action='store_true',
                       help='Skip MADRID parameter recommendations')
    
    # Other options
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for file selection (default: 42)')
    
    args = parser.parse_args()
    
    # Validate percentage
    if not 0 < args.percentage <= 100:
        print("Percentage must be between 0 and 100")
        return 1
    
    # Set random seed
    random.seed(args.seed)
    
    print("Seizure Duration Analysis")
    print("=" * 30)
    print(f"Data directory: {args.data_dir}")
    print(f"Processing: {args.percentage}% of files")
    print(f"Output file: {args.output}")
    
    try:
        # Initialize analyzer
        analyzer = SeizureDurationAnalyzer()
        
        # Discover files
        seizure_files = analyzer.discover_seizure_files(
            data_dir=args.data_dir,
            percentage=args.percentage,
            subject_filter=args.subject
        )
        
        if not seizure_files:
            print("No seizure files found")
            return 1
        
        # Analyze durations
        results = analyzer.analyze_seizure_durations(seizure_files)
        
        if not results['success']:
            print(f"Analysis failed: {results['message']}")
            return 1
        
        # Print summary
        stats = results['duration_stats']
        print(f"\n{'='*60}")
        print(f"SEIZURE DURATION ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"Files analyzed: {stats['n_files']}")
        print(f"Total seizures: {stats['n_seizures']}")
        print(f"Mean duration: {stats['mean_duration']:.2f}s")
        print(f"Median duration: {stats['median_duration']:.2f}s")
        print(f"Duration range: {stats['min_duration']:.2f}s - {stats['max_duration']:.2f}s")
        print(f"Standard deviation: {stats['std_duration']:.2f}s")
        
        # Save results
        analyzer.save_duration_analysis(
            output_file=args.output,
            include_detailed=not args.no_detailed,
            include_recommendations=not args.no_recommendations
        )
        
        # Generate plots if requested
        if args.plot:
            analyzer.plot_duration_distribution(
                output_file=args.plot_file,
                show_plot=True
            )
        
        print(f"\n✓ Analysis completed successfully!")
        print(f"✓ Results saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())