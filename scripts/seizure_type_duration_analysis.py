#!/usr/bin/env python3
"""
Seizure Type Duration Analysis Script

This script analyzes seizure durations by event type (seizure type) from 
preprocessed seizure data files. It provides detailed statistics for each 
seizure type and compares duration distributions across different types.

Key features:
- Analyze seizure durations by event type/seizure type
- Extract event type information from metadata or file annotations
- Calculate comprehensive statistics per seizure type
- Generate duration distribution analysis by type
- Export results in human-readable format with type-specific MADRID recommendations
- Support for different file formats and sampling rates
- Provide type-specific recommendations for MADRID m-parameter ranges

Usage:
    python seizure_type_duration_analysis.py --data-dir DATA_DIR
    python seizure_type_duration_analysis.py --data-dir DATA_DIR --output type_durations.txt
    python seizure_type_duration_analysis.py --data-dir DATA_DIR --percentage 50 --plot
    python seizure_type_duration_analysis.py --data-dir DATA_DIR --type-filter focal --detailed

Author: Generated for seizure type duration analysis
Date: 2025
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


class SeizureTypeDurationAnalyzer:
    """
    Analyzer for seizure duration statistics by event type from preprocessed data files.
    
    Extracts and analyzes ictal phase durations grouped by seizure type/event type.
    """
    
    def __init__(self):
        """Initialize the seizure type duration analyzer."""
        self.seizure_data = []
        self.type_duration_stats = {}
        self.known_seizure_types = set()
        
        print("Initialized SeizureTypeDurationAnalyzer")
    
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
        Load and analyze a single seizure file with event type information.
        
        Args:
            file_path: Path to seizure file
            
        Returns:
            Dictionary with seizure and event type information or None if failed
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
            
            # Extract event type information
            event_type = self._extract_event_type(data, file_path)
            
            # Extract basic information
            labels = channel['labels']
            data_length = len(channel['data'])
            total_duration = data_length / fs
            
            # Find seizure durations
            seizure_regions = self._find_seizure_regions(labels, fs)
            
            # Add event type to known types
            if event_type:
                self.known_seizure_types.add(event_type)
            
            # Extract file metadata
            file_info = {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'subject_id': data.get('subject_id', 'unknown'),
                'run_id': data.get('run_id', 'unknown'), 
                'seizure_index': data.get('seizure_index', 0),
                'event_type': event_type,
                'seizure_type': event_type,  # Alias for compatibility
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
    
    def _extract_event_type(self, data: Dict[str, Any], file_path: Path) -> Optional[str]:
        """
        Extract event type from seizure data.
        
        Args:
            data: Loaded seizure data
            file_path: Path to the file
            
        Returns:
            Event type string or None if not found
        """
        # Method 1: Check if event_type is directly in metadata
        if 'event_type' in data:
            return data['event_type']
        
        if 'metadata' in data and 'event_type' in data['metadata']:
            return data['metadata']['event_type']
        
        # Method 2: Check if seizure_type is in metadata
        if 'seizure_type' in data:
            return data['seizure_type']
        
        if 'metadata' in data and 'seizure_type' in data['metadata']:
            return data['metadata']['seizure_type']
        
        # Method 3: Try to extract from annotations if available
        if 'annotations' in data:
            annotations = data['annotations']
            if hasattr(annotations, 'types') and annotations.types:
                # Return the first type (assuming single seizure per file)
                return annotations.types[0] if annotations.types else None
            elif isinstance(annotations, dict) and 'types' in annotations:
                types = annotations['types']
                return types[0] if types else None
        
        # Method 4: Try to infer from filename patterns
        filename = file_path.name.lower()
        
        # Common seizure type patterns in filenames
        seizure_type_patterns = {
            'focal': ['focal', 'foc', 'partial'],
            'generalized': ['generalized', 'gen', 'gtcs', 'tonic', 'clonic'],
            'absence': ['absence', 'abs', 'petit'],
            'myoclonic': ['myoclonic', 'myo'],
            'tonic': ['tonic'],
            'clonic': ['clonic'],
            'atonic': ['atonic', 'drop'],
            'complex': ['complex', 'cps'],
            'simple': ['simple', 'sps']
        }
        
        for seizure_type, patterns in seizure_type_patterns.items():
            if any(pattern in filename for pattern in patterns):
                return seizure_type
        
        # Method 5: Try to extract from subject metadata in original SeizeIT2 format
        if 'metadata' in data:
            metadata = data['metadata']
            # Look for common metadata fields
            for field in ['type', 'eventType', 'seizureType', 'event_class']:
                if field in metadata:
                    return metadata[field]
        
        # Method 6: Default fallback - try to determine from other indicators
        # If we have seizure index, we might infer some patterns
        seizure_idx = data.get('seizure_index', 0)
        
        # Default to 'unknown' if we can't determine the type
        return 'unknown'
    
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
    
    def analyze_seizure_durations_by_type(self, file_paths: List[Path], 
                                        type_filter: str = None) -> Dict[str, Any]:
        """
        Analyze seizure durations by event type across multiple files.
        
        Args:
            file_paths: List of seizure file paths
            type_filter: Optional filter for specific seizure types
            
        Returns:
            Comprehensive duration analysis results by type
        """
        print(f"\n{'='*70}")
        print(f"SEIZURE TYPE DURATION ANALYSIS")
        print(f"{'='*70}")
        print(f"Analyzing {len(file_paths)} files...")
        if type_filter:
            print(f"Filtering for seizure type: {type_filter}")
        
        file_results = []
        type_durations = defaultdict(list)
        type_file_count = defaultdict(int)
        subject_type_durations = defaultdict(lambda: defaultdict(list))
        sampling_rates = []
        
        successful_files = 0
        total_seizures = 0
        
        # Process each file
        for i, file_path in enumerate(file_paths):
            print(f"\nProcessing file {i+1}/{len(file_paths)}: {file_path.name}")
            
            file_info = self.load_seizure_file(file_path)
            if file_info is None:
                continue
            
            # Apply type filter if specified
            if type_filter and file_info['event_type'] != type_filter:
                print(f"  ⚠️  Skipped: Event type '{file_info['event_type']}' doesn't match filter '{type_filter}'")
                continue
            
            file_results.append(file_info)
            successful_files += 1
            
            # Collect data for type-based analysis
            event_type = file_info['event_type'] or 'unknown'
            if file_info['seizure_durations']:
                type_durations[event_type].extend(file_info['seizure_durations'])
                type_file_count[event_type] += 1
                subject_type_durations[file_info['subject_id']][event_type].extend(file_info['seizure_durations'])
                total_seizures += len(file_info['seizure_durations'])
                sampling_rates.append(file_info['sampling_rate'])
                
                print(f"  ✓ Type: {event_type}")
                print(f"  ✓ Found {file_info['n_seizures']} seizures")
                print(f"  ✓ Durations: {file_info['min_seizure_duration']:.2f}s - {file_info['max_seizure_duration']:.2f}s")
                print(f"  ✓ Mean: {file_info['mean_seizure_duration']:.2f}s")
            else:
                print(f"  ⚠️  No seizures found (Type: {event_type})")
        
        print(f"\n✓ Successfully processed {successful_files}/{len(file_paths)} files")
        print(f"✓ Total seizures found: {total_seizures}")
        print(f"✓ Seizure types found: {list(type_durations.keys())}")
        
        if not any(type_durations.values()):
            print("❌ No seizure durations found in any files")
            return {
                'success': False,
                'message': 'No seizure durations found',
                'file_results': file_results,
                'seizure_types': list(self.known_seizure_types)
            }
        
        # Calculate type-specific statistics
        type_stats = {}
        overall_durations = []
        
        for seizure_type, durations in type_durations.items():
            if not durations:
                continue
                
            durations_array = np.array(durations)
            overall_durations.extend(durations)
            
            # Basic statistics for this type
            type_stat = {
                'n_files': type_file_count[seizure_type],
                'n_seizures': len(durations),
                'mean_duration': float(np.mean(durations_array)),
                'median_duration': float(np.median(durations_array)),
                'std_duration': float(np.std(durations_array)),
                'min_duration': float(np.min(durations_array)),
                'max_duration': float(np.max(durations_array)),
                'range': float(np.max(durations_array) - np.min(durations_array)),
                'coefficient_of_variation': float(np.std(durations_array) / np.mean(durations_array)),
            }
            
            # Add all percentiles in 5% steps
            percentile_values = range(5, 100, 5)  # 5%, 10%, 15%, ..., 95%
            for p in percentile_values:
                type_stat[f'percentile_{p}'] = float(np.percentile(durations_array, p))
            
            type_stats[seizure_type] = type_stat
        
        # Calculate overall statistics
        if overall_durations:
            overall_durations = np.array(overall_durations)
            overall_stats = {
                'n_files': successful_files,
                'n_seizures': total_seizures,
                'n_types': len(type_durations),
                'n_subjects': len(subject_type_durations),
                'mean_sampling_rate': np.mean(sampling_rates) if sampling_rates else 125,
                'mean_duration': float(np.mean(overall_durations)),
                'median_duration': float(np.median(overall_durations)),
                'std_duration': float(np.std(overall_durations)),
                'min_duration': float(np.min(overall_durations)),
                'max_duration': float(np.max(overall_durations)),
                'range': float(np.max(overall_durations) - np.min(overall_durations)),
                'coefficient_of_variation': float(np.std(overall_durations) / np.mean(overall_durations)),
            }
            
            # Add overall percentiles
            for p in range(5, 100, 5):
                overall_stats[f'percentile_{p}'] = float(np.percentile(overall_durations, p))
        else:
            overall_stats = {}
        
        # Generate type-specific MADRID recommendations
        type_recommendations = {}
        for seizure_type, stats in type_stats.items():
            type_recommendations[seizure_type] = self._generate_madrid_recommendations_for_type(
                stats, seizure_type
            )
        
        # Subject-type analysis
        subject_type_stats = {}
        for subject_id, type_data in subject_type_durations.items():
            subject_type_stats[subject_id] = {}
            for seizure_type, durations in type_data.items():
                if durations:
                    subject_type_stats[subject_id][seizure_type] = {
                        'n_seizures': len(durations),
                        'mean_duration': float(np.mean(durations)),
                        'median_duration': float(np.median(durations)),
                        'std_duration': float(np.std(durations)),
                        'min_duration': float(np.min(durations)),
                        'max_duration': float(np.max(durations))
                    }
        
        results = {
            'success': True,
            'overall_stats': overall_stats,
            'type_stats': type_stats,
            'type_recommendations': type_recommendations,
            'subject_type_stats': subject_type_stats,
            'seizure_types': list(type_durations.keys()),
            'all_seizure_types': list(self.known_seizure_types),
            'file_results': file_results,
            'type_filter': type_filter
        }
        
        self.seizure_data = file_results
        self.type_duration_stats = results
        
        return results
    
    def _generate_madrid_recommendations_for_type(self, stats: Dict[str, float], 
                                                seizure_type: str) -> Dict[str, Any]:
        """Generate MADRID parameter recommendations for a specific seizure type."""
        fs = 125  # Default sampling rate, will be updated with actual values
        
        # Type-specific recommendations based on seizure characteristics
        if seizure_type.lower() in ['focal', 'partial', 'simple', 'complex']:
            # Focal seizures tend to be more variable in duration
            safety_factor = 1.5
        elif seizure_type.lower() in ['generalized', 'gtcs', 'tonic', 'clonic']:
            # Generalized seizures tend to be longer
            safety_factor = 2.0
        elif seizure_type.lower() in ['absence', 'petit']:
            # Absence seizures are typically very short
            safety_factor = 1.2
        elif seizure_type.lower() in ['myoclonic']:
            # Myoclonic seizures are very brief
            safety_factor = 1.1
        else:
            # Unknown or other types
            safety_factor = 1.5
        
        # Calculate optimal m-parameter ranges
        recommendations = {
            'seizure_type': seizure_type,
            'sampling_rate': fs,
            
            # Conservative approach - cover most seizures of this type
            'conservative': {
                'min_m_seconds': max(0.1, stats['percentile_5'] * 0.5),
                'max_m_seconds': min(600.0, stats['percentile_95'] * safety_factor),
                'description': f'Covers 90% of {seizure_type} seizures with type-specific safety margin'
            },
            
            # Focused approach - target typical seizures of this type
            'focused': {
                'min_m_seconds': max(0.2, stats['percentile_25'] * 0.8),
                'max_m_seconds': min(300.0, stats['percentile_75'] * 1.2),
                'description': f'Targets central 50% of {seizure_type} seizures'
            },
            
            # Aggressive approach - tight around mean for this type
            'aggressive': {
                'min_m_seconds': max(0.1, stats['mean_duration'] * 0.5),
                'max_m_seconds': min(200.0, stats['mean_duration'] * 2.0),
                'description': f'Tight range around mean {seizure_type} seizure duration'
            }
        }
        
        # Convert to samples and add practical parameters
        for approach_name, approach in recommendations.items():
            if approach_name in ['seizure_type', 'sampling_rate']:
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
    
    def save_type_duration_analysis(self, output_file: str, include_detailed: bool = True,
                                  include_recommendations: bool = True):
        """
        Save seizure type duration analysis results in human-readable format.
        
        Args:
            output_file: Output file path
            include_detailed: Whether to include detailed per-file results
            include_recommendations: Whether to include type-specific MADRID recommendations
        """
        if not self.type_duration_stats:
            print("No results to save")
            return
        
        results = self.type_duration_stats
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # Header
            f.write("="*80 + "\n")
            f.write("SEIZURE TYPE DURATION ANALYSIS RESULTS\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            if results['overall_stats']:
                overall = results['overall_stats']
                f.write(f"Files analyzed: {overall['n_files']}\n")
                f.write(f"Total seizures: {overall['n_seizures']}\n")
                f.write(f"Seizure types: {overall['n_types']}\n")
                f.write(f"Subjects: {overall['n_subjects']}\n")
                f.write(f"Mean sampling rate: {overall['mean_sampling_rate']:.1f} Hz\n")
            
            if results['type_filter']:
                f.write(f"Type filter applied: {results['type_filter']}\n")
            
            f.write(f"Available seizure types: {', '.join(results['seizure_types'])}\n\n")
            
            # Executive Summary by Type
            f.write("EXECUTIVE SUMMARY BY SEIZURE TYPE\n")
            f.write("-"*40 + "\n")
            
            type_stats = results['type_stats']
            if type_stats:
                f.write(f"{'Type':<15} {'N Files':<8} {'N Seizures':<10} {'Mean(s)':<8} {'Median(s)':<10} {'Std(s)':<8} {'Range(s)':<12}\n")
                f.write("-"*75 + "\n")
                
                for seizure_type, stats in sorted(type_stats.items()):
                    range_str = f"{stats['min_duration']:.1f}-{stats['max_duration']:.1f}"
                    f.write(f"{seizure_type:<15} {stats['n_files']:<8} {stats['n_seizures']:<10} "
                           f"{stats['mean_duration']:<8.2f} {stats['median_duration']:<10.2f} "
                           f"{stats['std_duration']:<8.2f} {range_str:<12}\n")
                
                f.write("\n")
            
            # Overall Statistics
            if results['overall_stats']:
                f.write("OVERALL STATISTICS (ALL TYPES)\n")
                f.write("-"*35 + "\n")
                overall = results['overall_stats']
                f.write(f"Mean:                {overall['mean_duration']:8.2f} seconds\n")
                f.write(f"Median:              {overall['median_duration']:8.2f} seconds\n")
                f.write(f"Standard deviation:  {overall['std_duration']:8.2f} seconds\n")
                f.write(f"Minimum:             {overall['min_duration']:8.2f} seconds\n")
                f.write(f"Maximum:             {overall['max_duration']:8.2f} seconds\n")
                f.write(f"Range:               {overall['range']:8.2f} seconds\n")
                f.write(f"Coefficient of variation: {overall['coefficient_of_variation']:8.2f}\n\n")
            
            # Detailed Type Statistics
            f.write("DETAILED TYPE-SPECIFIC STATISTICS\n")
            f.write("-"*40 + "\n")
            
            for seizure_type, stats in sorted(type_stats.items()):
                f.write(f"\nSEIZURE TYPE: {seizure_type.upper()}\n")
                f.write("="*30 + "\n")
                f.write(f"Files:               {stats['n_files']:8d}\n")
                f.write(f"Seizures:            {stats['n_seizures']:8d}\n")
                f.write(f"Mean:                {stats['mean_duration']:8.2f} seconds\n")
                f.write(f"Median:              {stats['median_duration']:8.2f} seconds\n")
                f.write(f"Standard deviation:  {stats['std_duration']:8.2f} seconds\n")
                f.write(f"Minimum:             {stats['min_duration']:8.2f} seconds\n")
                f.write(f"Maximum:             {stats['max_duration']:8.2f} seconds\n")
                f.write(f"Range:               {stats['range']:8.2f} seconds\n")
                f.write(f"Coefficient of variation: {stats['coefficient_of_variation']:8.2f}\n\n")
                
                f.write("Percentiles (in 5% steps):\n")
                percentile_values = range(5, 100, 5)
                for i, p in enumerate(percentile_values):
                    percentile_key = f'percentile_{p}'
                    if percentile_key in stats:
                        f.write(f"  {p:2d}th percentile:   {stats[percentile_key]:8.2f} seconds\n")
                        if (i + 1) % 5 == 0:
                            f.write("\n")
                
                if len(percentile_values) % 5 != 0:
                    f.write("\n")
            
            # Type Comparison
            if len(type_stats) > 1:
                f.write("TYPE COMPARISON\n")
                f.write("-"*20 + "\n")
                
                # Sort types by mean duration
                sorted_types = sorted(type_stats.items(), key=lambda x: x[1]['mean_duration'])
                
                f.write("Types ordered by mean duration (shortest to longest):\n")
                for i, (seizure_type, stats) in enumerate(sorted_types, 1):
                    f.write(f"  {i}. {seizure_type:<15}: {stats['mean_duration']:6.2f}s (n={stats['n_seizures']})\n")
                
                f.write("\nVariability comparison (coefficient of variation):\n")
                sorted_by_cv = sorted(type_stats.items(), key=lambda x: x[1]['coefficient_of_variation'])
                for i, (seizure_type, stats) in enumerate(sorted_by_cv, 1):
                    variability = "Low" if stats['coefficient_of_variation'] < 0.5 else "Moderate" if stats['coefficient_of_variation'] < 1.0 else "High"
                    f.write(f"  {i}. {seizure_type:<15}: CV={stats['coefficient_of_variation']:5.2f} ({variability})\n")
                
                f.write("\n")
            
            # Type-Specific MADRID Recommendations
            if include_recommendations and results['type_recommendations']:
                f.write("TYPE-SPECIFIC MADRID RECOMMENDATIONS\n")
                f.write("-"*45 + "\n")
                
                for seizure_type, recommendations in results['type_recommendations'].items():
                    f.write(f"\nSEIZURE TYPE: {seizure_type.upper()}\n")
                    f.write("="*30 + "\n")
                    
                    fs = recommendations['sampling_rate']
                    f.write(f"Recommendations for {seizure_type} seizures at {fs:.0f} Hz:\n\n")
                    
                    for approach_name, approach in recommendations.items():
                        if approach_name in ['seizure_type', 'sampling_rate']:
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
                    
                    # Type-specific m-value recommendations
                    f.write(f"SPECIFIC m-VALUES for {seizure_type.upper()}:\n")
                    f.write("-"*30 + "\n")
                    
                    type_stat = type_stats[seizure_type]
                    specific_m_values = [
                        ("Mean duration", type_stat['mean_duration'], int(type_stat['mean_duration'] * fs)),
                        ("Median duration", type_stat['median_duration'], int(type_stat['median_duration'] * fs)),
                    ]
                    
                    # Add percentiles
                    percentile_values = [25, 50, 75, 90, 95]  # Key percentiles
                    for p in percentile_values:
                        percentile_key = f'percentile_{p}'
                        if percentile_key in type_stat:
                            duration_sec = type_stat[percentile_key]
                            m_samples = int(duration_sec * fs)
                            specific_m_values.append((f"{p}th percentile", duration_sec, m_samples))
                    
                    f.write(f"{'Description':<18} {'Duration (s)':<12} {'m (samples)':<12}\n")
                    f.write("-"*45 + "\n")
                    for name, duration_sec, m_samples in specific_m_values:
                        f.write(f"{name:<18} {duration_sec:>10.2f}s {m_samples:>10d}\n")
                    
                    f.write("\n")
            
            # Subject-Type Analysis
            if results['subject_type_stats'] and include_detailed:
                f.write("SUBJECT-TYPE ANALYSIS\n")
                f.write("-"*25 + "\n")
                
                for subject_id, type_data in sorted(results['subject_type_stats'].items()):
                    f.write(f"\nSubject: {subject_id}\n")
                    f.write("-"*15 + "\n")
                    f.write(f"{'Type':<15} {'N':<4} {'Mean':<8} {'Median':<8} {'Std':<8} {'Min':<8} {'Max':<8}\n")
                    f.write("-"*60 + "\n")
                    
                    for seizure_type, stats in sorted(type_data.items()):
                        f.write(f"{seizure_type:<15} {stats['n_seizures']:<4} {stats['mean_duration']:<8.2f} "
                               f"{stats['median_duration']:<8.2f} {stats['std_duration']:<8.2f} "
                               f"{stats['min_duration']:<8.2f} {stats['max_duration']:<8.2f}\n")
                
                f.write("\n")
            
            # Detailed per-file results
            if include_detailed and results['file_results']:
                f.write("DETAILED PER-FILE RESULTS\n")
                f.write("-"*30 + "\n")
                f.write(f"{'File':<25} {'Subject':<10} {'Type':<12} {'N':<3} {'Mean':<8} {'Median':<8} {'Min':<8} {'Max':<8}\n")
                f.write("-"*90 + "\n")
                
                for file_info in results['file_results']:
                    if file_info['n_seizures'] > 0:
                        file_name = file_info['file_name']
                        if len(file_name) > 24:
                            file_name = file_name[:21] + "..."
                        
                        event_type = file_info['event_type'] or 'unknown'
                        if len(event_type) > 11:
                            event_type = event_type[:8] + "..."
                        
                        f.write(f"{file_name:<25} {file_info['subject_id']:<10} {event_type:<12} "
                               f"{file_info['n_seizures']:<3} {file_info['mean_seizure_duration']:<8.2f} "
                               f"{file_info['median_seizure_duration']:<8.2f} {file_info['min_seizure_duration']:<8.2f} "
                               f"{file_info['max_seizure_duration']:<8.2f}\n")
                
                f.write("\n")
            
            # Final Recommendations
            f.write("FINAL RECOMMENDATIONS\n")
            f.write("-"*25 + "\n")
            
            if type_stats:
                # Find most common type
                most_common_type = max(type_stats.items(), key=lambda x: x[1]['n_seizures'])
                f.write(f"1. MOST COMMON TYPE: {most_common_type[0].upper()}\n")
                f.write(f"   {most_common_type[1]['n_seizures']} seizures, mean duration: {most_common_type[1]['mean_duration']:.2f}s\n")
                f.write(f"   Use type-specific parameters for best results\n\n")
                
                # Find shortest and longest types
                shortest_type = min(type_stats.items(), key=lambda x: x[1]['mean_duration'])
                longest_type = max(type_stats.items(), key=lambda x: x[1]['mean_duration'])
                
                if len(type_stats) > 1:
                    f.write(f"2. TYPE RANGE:\n")
                    f.write(f"   Shortest: {shortest_type[0]} ({shortest_type[1]['mean_duration']:.2f}s)\n")
                    f.write(f"   Longest:  {longest_type[0]} ({longest_type[1]['mean_duration']:.2f}s)\n")
                    f.write(f"   Consider type-specific MADRID configurations\n\n")
                
                f.write(f"3. ANALYSIS STRATEGY:\n")
                f.write(f"   - Use separate models for each seizure type if possible\n")
                f.write(f"   - Focus on the most common type first\n")
                f.write(f"   - Use madrid_m_analysis.py with type-specific parameters\n")
                f.write(f"   - Consider ensemble approaches for mixed types\n")
        
        print(f"✓ Seizure type duration analysis saved to: {output_file}")
    
    def plot_type_duration_distributions(self, output_file: str = None, show_plot: bool = True):
        """
        Create plots of seizure duration distributions by type.
        
        Args:
            output_file: Optional file to save plot
            show_plot: Whether to display the plot
        """
        if not self.type_duration_stats or not self.type_duration_stats['success']:
            print("No data available for plotting")
            return
        
        type_stats = self.type_duration_stats['type_stats']
        if not type_stats:
            print("No type-specific data available for plotting")
            return
        
        # Prepare data for plotting
        type_durations = {}
        for file_info in self.type_duration_stats['file_results']:
            if file_info['seizure_durations']:
                event_type = file_info['event_type'] or 'unknown'
                if event_type not in type_durations:
                    type_durations[event_type] = []
                type_durations[event_type].extend(file_info['seizure_durations'])
        
        n_types = len(type_durations)
        if n_types == 0:
            print("No duration data available for plotting")
            return
        
        # Create figure with subplots
        fig_height = max(8, n_types * 2 + 4)
        fig, axes = plt.subplots(2, 2, figsize=(15, fig_height))
        fig.suptitle('Seizure Duration Analysis by Event Type', fontsize=16)
        
        # Prepare colors for each type
        colors = plt.cm.Set3(np.linspace(0, 1, n_types))
        
        # 1. Histograms by type
        ax1 = axes[0, 0]
        for i, (seizure_type, durations) in enumerate(type_durations.items()):
            ax1.hist(durations, bins=20, alpha=0.6, label=seizure_type, 
                    color=colors[i], density=True)
        ax1.set_xlabel('Duration (seconds)')
        ax1.set_ylabel('Density')
        ax1.set_title('Duration Distributions by Type')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Box plots by type
        ax2 = axes[0, 1]
        type_names = list(type_durations.keys())
        duration_lists = [type_durations[t] for t in type_names]
        box_plot = ax2.boxplot(duration_lists, labels=type_names, patch_artist=True)
        
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        ax2.set_ylabel('Duration (seconds)')
        ax2.set_title('Duration Box Plots by Type')
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # 3. Mean comparison
        ax3 = axes[1, 0]
        means = [type_stats[t]['mean_duration'] for t in type_names]
        stds = [type_stats[t]['std_duration'] for t in type_names]
        
        bars = ax3.bar(type_names, means, yerr=stds, capsize=5, 
                      color=colors, alpha=0.7)
        ax3.set_ylabel('Mean Duration (seconds)')
        ax3.set_title('Mean Duration by Type (with std)')
        ax3.grid(True, alpha=0.3, axis='y')
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mean:.1f}s', ha='center', va='bottom')
        
        # 4. Cumulative distributions
        ax4 = axes[1, 1]
        for i, (seizure_type, durations) in enumerate(type_durations.items()):
            sorted_durations = np.sort(durations)
            cumulative = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations) * 100
            ax4.plot(sorted_durations, cumulative, 'o-', 
                    label=seizure_type, color=colors[i], alpha=0.8)
        
        ax4.set_xlabel('Duration (seconds)')
        ax4.set_ylabel('Cumulative Percentage')
        ax4.set_title('Cumulative Distribution by Type')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ Type distribution plots saved to: {output_file}")
        
        if show_plot:
            plt.show()


def main():
    """Main function for seizure type duration analysis."""
    parser = argparse.ArgumentParser(description='Seizure Type Duration Analysis')
    
    # Required arguments
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing seizure preprocessed files')
    
    # File selection
    parser.add_argument('--percentage', type=float, default=100.0,
                       help='Percentage of files to analyze (default: 100)')
    parser.add_argument('--subject', type=str,
                       help='Filter by subject ID (e.g., sub-001)')
    parser.add_argument('--type-filter', type=str,
                       help='Filter by seizure type (e.g., focal, generalized)')
    
    # Output options
    parser.add_argument('--output', type=str, default='seizure_type_durations.txt',
                       help='Output file for results (default: seizure_type_durations.txt)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate duration distribution plots by type')
    parser.add_argument('--plot-file', type=str,
                       help='File to save plots (e.g., type_duration_plots.png)')
    parser.add_argument('--no-detailed', action='store_true',
                       help='Skip detailed per-file results in output')
    parser.add_argument('--no-recommendations', action='store_true',
                       help='Skip type-specific MADRID parameter recommendations')
    
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
    
    print("Seizure Type Duration Analysis")
    print("=" * 35)
    print(f"Data directory: {args.data_dir}")
    print(f"Processing: {args.percentage}% of files")
    if args.type_filter:
        print(f"Type filter: {args.type_filter}")
    print(f"Output file: {args.output}")
    
    try:
        # Initialize analyzer
        analyzer = SeizureTypeDurationAnalyzer()
        
        # Discover files
        seizure_files = analyzer.discover_seizure_files(
            data_dir=args.data_dir,
            percentage=args.percentage,
            subject_filter=args.subject
        )
        
        if not seizure_files:
            print("No seizure files found")
            return 1
        
        # Analyze durations by type
        results = analyzer.analyze_seizure_durations_by_type(
            seizure_files, 
            type_filter=args.type_filter
        )
        
        if not results['success']:
            print(f"Analysis failed: {results['message']}")
            return 1
        
        # Print summary
        if results['overall_stats']:
            overall = results['overall_stats']
            print(f"\n{'='*70}")
            print(f"SEIZURE TYPE DURATION ANALYSIS SUMMARY")
            print(f"{'='*70}")
            print(f"Files analyzed: {overall['n_files']}")
            print(f"Total seizures: {overall['n_seizures']}")
            print(f"Seizure types: {overall['n_types']}")
            print(f"Overall mean duration: {overall['mean_duration']:.2f}s")
            print(f"Overall range: {overall['min_duration']:.2f}s - {overall['max_duration']:.2f}s")
            
            # Show type breakdown
            type_stats = results['type_stats']
            if type_stats:
                print(f"\nType breakdown:")
                for seizure_type, stats in sorted(type_stats.items(), 
                                                key=lambda x: x[1]['n_seizures'], reverse=True):
                    print(f"  {seizure_type:<15}: {stats['n_seizures']:3d} seizures, "
                          f"mean {stats['mean_duration']:5.2f}s")
        
        # Save results
        analyzer.save_type_duration_analysis(
            output_file=args.output,
            include_detailed=not args.no_detailed,
            include_recommendations=not args.no_recommendations
        )
        
        # Generate plots if requested
        if args.plot:
            analyzer.plot_type_duration_distributions(
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