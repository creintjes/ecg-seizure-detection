#!/usr/bin/env python3
"""
SeizeIT2 Event Duration Analysis Script

This script analyzes seizure durations by event type directly from 
SeizeIT2 original TSV events files. It reads the events.tsv files
to extract eventType information and calculate duration statistics
for each seizure type.

Key features:
- Read original SeizeIT2 events.tsv files directly
- Extract eventType, lateralization, localization, vigilance information
- Calculate duration statistics by eventType
- Generate comprehensive reports with type-specific MADRID recommendations
- Support for BIDS dataset structure
- Filter by specific event types or subjects

Usage:
    python seizeit2_event_duration_analysis.py --data-dir /path/to/seizeit2/bids
    python seizeit2_event_duration_analysis.py --data-dir /path/to/seizeit2/bids --event-filter focal
    python seizeit2_event_duration_analysis.py --data-dir /path/to/seizeit2/bids --subject sub-001
    python seizeit2_event_duration_analysis.py --data-dir /path/to/seizeit2/bids --percentage 50 --plot

Author: Generated for SeizeIT2 event analysis
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import argparse
from pathlib import Path
import warnings
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import random

warnings.filterwarnings('ignore')


class SeizeIT2EventAnalyzer:
    """
    Analyzer for seizure events directly from SeizeIT2 TSV files.
    
    Reads original SeizeIT2 events.tsv files and analyzes durations by eventType.
    """
    
    def __init__(self, sampling_rate: float = 250.0):
        """
        Initialize the SeizeIT2 event analyzer.
        
        Args:
            sampling_rate: Default sampling rate for MADRID recommendations
        """
        self.sampling_rate = sampling_rate
        self.event_data = []
        self.event_stats = {}
        
        print(f"Initialized SeizeIT2EventAnalyzer with {sampling_rate} Hz sampling rate")
    
    def discover_event_files(self, data_dir: str, percentage: float = 100.0, 
                            subject_filter: str = None) -> List[Path]:
        """
        Discover SeizeIT2 events.tsv files in BIDS structure.
        
        Args:
            data_dir: Path to SeizeIT2 BIDS dataset root
            percentage: Percentage of files to analyze (0-100)
            subject_filter: Optional subject filter (e.g., 'sub-001')
            
        Returns:
            List of events.tsv file paths
        """
        data_path = Path(data_dir)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_path}")
        
        # Find all events.tsv files in BIDS structure
        # Pattern: sub-*/ses-*/eeg/*_events.tsv
        event_files = list(data_path.glob("sub-*/ses-*/eeg/*_events.tsv"))
        
        print(f"Found {len(event_files)} events.tsv files in {data_path}")
        
        # Apply subject filter
        if subject_filter:
            event_files = [f for f in event_files if subject_filter in str(f)]
            print(f"After subject filter '{subject_filter}': {len(event_files)} files")
        
        if not event_files:
            raise ValueError("No events.tsv files found matching criteria")
        
        # Calculate number of files to process
        n_files_to_process = max(1, int(len(event_files) * percentage / 100))
        
        # Randomly sample files for representative results
        random.seed(42)  # For reproducible results
        selected_files = random.sample(event_files, n_files_to_process)
        
        print(f"Selected {len(selected_files)} files ({percentage}% of {len(event_files)} total)")
        
        return sorted(selected_files)
    
    def load_events_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load and parse a single events.tsv file.
        
        Args:
            file_path: Path to events.tsv file
            
        Returns:
            Dictionary with event information or None if failed
        """
        try:
            # Read TSV file
            df = pd.read_csv(file_path, delimiter="\t")
            
            # Extract subject and run information from path
            path_parts = file_path.parts
            subject_id = None
            session_id = None
            run_id = None
            
            for part in path_parts:
                if part.startswith('sub-'):
                    subject_id = part
                elif part.startswith('ses-'):
                    session_id = part
            
            # Extract run from filename
            filename = file_path.name
            if '_run-' in filename:
                run_part = filename.split('_run-')[1].split('_')[0]
                run_id = f"run-{run_part}"
            
            # Filter out background and impedance events
            seizure_events = df[(df['eventType'] != 'bckg') & (df['eventType'] != 'impd')].copy()
            
            if len(seizure_events) == 0:
                print(f"  Warning: No seizure events found in {file_path.name}")
                return None
            
            # Extract events information
            events = []
            for _, event in seizure_events.iterrows():
                event_info = {
                    'onset': float(event['onset']),
                    'duration': float(event['duration']),
                    'offset': float(event['onset']) + float(event['duration']),
                    'eventType': str(event['eventType']),
                    'lateralization': str(event.get('lateralization', 'n/a')),
                    'localization': str(event.get('localization', 'n/a')),
                    'vigilance': str(event.get('vigilance', 'n/a'))
                }
                events.append(event_info)
            
            # Get recording duration
            recording_duration = df['recordingDuration'].iloc[0] if 'recordingDuration' in df.columns else None
            
            file_info = {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'subject_id': subject_id or 'unknown',
                'session_id': session_id or 'ses-01',
                'run_id': run_id or 'unknown',
                'recording_duration': recording_duration,
                'n_events': len(events),
                'events': events
            }
            
            return file_info
            
        except Exception as e:
            print(f"  Error loading {file_path.name}: {e}")
            return None
    
    def analyze_events_by_type(self, file_paths: List[Path], 
                             event_filter: str = None) -> Dict[str, Any]:
        """
        Analyze seizure events by eventType across multiple files.
        
        Args:
            file_paths: List of events.tsv file paths
            event_filter: Optional filter for specific event types
            
        Returns:
            Comprehensive event analysis results by type
        """
        print(f"\n{'='*70}")
        print(f"SEIZEIT2 EVENT DURATION ANALYSIS")
        print(f"{'='*70}")
        print(f"Analyzing {len(file_paths)} events.tsv files...")
        if event_filter:
            print(f"Event type filter: {event_filter}")
        
        file_results = []
        event_type_durations = defaultdict(list)
        event_type_counts = defaultdict(int)
        subject_event_data = defaultdict(lambda: defaultdict(list))
        lateralization_data = defaultdict(lambda: defaultdict(list))
        localization_data = defaultdict(lambda: defaultdict(list))
        vigilance_data = defaultdict(lambda: defaultdict(list))
        
        successful_files = 0
        total_events = 0
        all_event_types = set()
        
        # Process each file
        for i, file_path in enumerate(file_paths):
            print(f"\nProcessing file {i+1}/{len(file_paths)}: {file_path.name}")
            
            file_info = self.load_events_file(file_path)
            if file_info is None:
                continue
            
            file_results.append(file_info)
            successful_files += 1
            
            # Process events from this file
            file_event_types = set()
            for event in file_info['events']:
                event_type = event['eventType']
                duration = event['duration']
                subject_id = file_info['subject_id']
                
                # Apply event filter if specified
                if event_filter and event_type != event_filter:
                    continue
                
                all_event_types.add(event_type)
                file_event_types.add(event_type)
                
                # Collect duration data
                event_type_durations[event_type].append(duration)
                event_type_counts[event_type] += 1
                subject_event_data[subject_id][event_type].append(duration)
                
                # Collect metadata
                lateralization_data[event_type][event['lateralization']].append(duration)
                localization_data[event_type][event['localization']].append(duration)
                vigilance_data[event_type][event['vigilance']].append(duration)
                
                total_events += 1
            
            print(f"  ✓ Subject: {file_info['subject_id']}")
            print(f"  ✓ Found {file_info['n_events']} events")
            print(f"  ✓ Event types: {', '.join(sorted(file_event_types))}")
            
            if file_info['events']:
                durations = [e['duration'] for e in file_info['events']]
                print(f"  ✓ Duration range: {min(durations):.2f}s - {max(durations):.2f}s")
        
        print(f"\n✓ Successfully processed {successful_files}/{len(file_paths)} files")
        print(f"✓ Total events found: {total_events}")
        print(f"✓ Event types found: {sorted(all_event_types)}")
        
        if not any(event_type_durations.values()):
            print("❌ No event durations found")
            return {
                'success': False,
                'message': 'No event durations found',
                'file_results': file_results,
                'event_types': list(all_event_types)
            }
        
        # Calculate type-specific statistics
        type_stats = {}
        overall_durations = []
        
        for event_type, durations in event_type_durations.items():
            if not durations:
                continue
                
            durations_array = np.array(durations)
            overall_durations.extend(durations)
            
            # Basic statistics for this type
            type_stat = {
                'n_events': len(durations),
                'n_files': len(set(file_info['subject_id'] for file_info in file_results 
                                 for event in file_info['events'] 
                                 if event['eventType'] == event_type)),
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
            
            type_stats[event_type] = type_stat
        
        # Calculate overall statistics
        if overall_durations:
            overall_durations = np.array(overall_durations)
            overall_stats = {
                'n_files': successful_files,
                'n_events': total_events,
                'n_types': len(event_type_durations),
                'n_subjects': len(subject_event_data),
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
        for event_type, stats in type_stats.items():
            type_recommendations[event_type] = self._generate_madrid_recommendations_for_type(
                stats, event_type
            )
        
        # Subject-type analysis
        subject_type_stats = {}
        for subject_id, type_data in subject_event_data.items():
            subject_type_stats[subject_id] = {}
            for event_type, durations in type_data.items():
                if durations:
                    subject_type_stats[subject_id][event_type] = {
                        'n_events': len(durations),
                        'mean_duration': float(np.mean(durations)),
                        'median_duration': float(np.median(durations)),
                        'std_duration': float(np.std(durations)),
                        'min_duration': float(np.min(durations)),
                        'max_duration': float(np.max(durations))
                    }
        
        # Metadata analysis
        metadata_stats = {
            'lateralization': self._analyze_metadata_distribution(lateralization_data),
            'localization': self._analyze_metadata_distribution(localization_data),
            'vigilance': self._analyze_metadata_distribution(vigilance_data)
        }
        
        results = {
            'success': True,
            'overall_stats': overall_stats,
            'type_stats': type_stats,
            'type_recommendations': type_recommendations,
            'subject_type_stats': subject_type_stats,
            'metadata_stats': metadata_stats,
            'event_types': list(event_type_durations.keys()),
            'all_event_types': list(all_event_types),
            'file_results': file_results,
            'event_filter': event_filter,
            'sampling_rate': self.sampling_rate
        }
        
        self.event_data = file_results
        self.event_stats = results
        
        return results
    
    def _analyze_metadata_distribution(self, metadata_data: Dict[str, Dict[str, List]]) -> Dict[str, Any]:
        """Analyze distribution of metadata (lateralization, localization, vigilance) by event type."""
        analysis = {}
        
        for event_type, metadata_values in metadata_data.items():
            type_analysis = {}
            total_events = sum(len(durations) for durations in metadata_values.values())
            
            for metadata_value, durations in metadata_values.items():
                if durations:
                    type_analysis[metadata_value] = {
                        'count': len(durations),
                        'percentage': len(durations) / total_events * 100 if total_events > 0 else 0,
                        'mean_duration': float(np.mean(durations)),
                        'std_duration': float(np.std(durations))
                    }
            
            analysis[event_type] = type_analysis
        
        return analysis
    
    def _generate_madrid_recommendations_for_type(self, stats: Dict[str, float], 
                                                event_type: str) -> Dict[str, Any]:
        """Generate MADRID parameter recommendations for a specific event type."""
        fs = self.sampling_rate
        
        # Type-specific recommendations based on known seizure characteristics
        type_lower = event_type.lower()
        
        if 'focal' in type_lower or 'partial' in type_lower:
            safety_factor = 1.5
            description_suffix = "focal seizures (variable duration)"
        elif 'generalized' in type_lower or 'gtcs' in type_lower or 'tonic' in type_lower:
            safety_factor = 2.0
            description_suffix = "generalized seizures (longer duration)"
        elif 'absence' in type_lower or 'petit' in type_lower:
            safety_factor = 1.2
            description_suffix = "absence seizures (typically short)"
        elif 'myoclonic' in type_lower:
            safety_factor = 1.1
            description_suffix = "myoclonic seizures (very brief)"
        elif 'clonic' in type_lower:
            safety_factor = 1.3
            description_suffix = "clonic seizures"
        elif 'atonic' in type_lower:
            safety_factor = 1.2
            description_suffix = "atonic seizures (drop attacks)"
        else:
            safety_factor = 1.5
            description_suffix = f"{event_type} seizures"
        
        # Calculate optimal m-parameter ranges
        recommendations = {
            'event_type': event_type,
            'sampling_rate': fs,
            
            # Conservative approach - cover most seizures of this type
            'conservative': {
                'min_m_seconds': max(0.1, stats['percentile_5'] * 0.5),
                'max_m_seconds': min(600.0, stats['percentile_95'] * safety_factor),
                'description': f'Covers 90% of {description_suffix} with safety margin'
            },
            
            # Focused approach - target typical seizures of this type
            'focused': {
                'min_m_seconds': max(0.2, stats['percentile_25'] * 0.8),
                'max_m_seconds': min(300.0, stats['percentile_75'] * 1.2),
                'description': f'Targets central 50% of {description_suffix}'
            },
            
            # Aggressive approach - tight around mean for this type
            'aggressive': {
                'min_m_seconds': max(0.1, stats['mean_duration'] * 0.5),
                'max_m_seconds': min(200.0, stats['mean_duration'] * 2.0),
                'description': f'Tight range around mean {description_suffix} duration'
            }
        }
        
        # Convert to samples and add practical parameters
        for approach_name, approach in recommendations.items():
            if approach_name in ['event_type', 'sampling_rate']:
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
    
    def save_event_analysis(self, output_file: str, include_detailed: bool = True,
                          include_recommendations: bool = True,
                          include_metadata: bool = True):
        """
        Save SeizeIT2 event analysis results in human-readable format.
        
        Args:
            output_file: Output file path
            include_detailed: Whether to include detailed per-file results
            include_recommendations: Whether to include type-specific MADRID recommendations
            include_metadata: Whether to include metadata analysis
        """
        if not self.event_stats:
            print("No results to save")
            return
        
        results = self.event_stats
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # Header
            f.write("="*80 + "\n")
            f.write("SEIZEIT2 EVENT TYPE DURATION ANALYSIS RESULTS\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Analysis based on original SeizeIT2 events.tsv files\n")
            
            if results['overall_stats']:
                overall = results['overall_stats']
                f.write(f"Files analyzed: {overall['n_files']}\n")
                f.write(f"Total events: {overall['n_events']}\n")
                f.write(f"Event types: {overall['n_types']}\n")
                f.write(f"Subjects: {overall['n_subjects']}\n")
            
            if results['event_filter']:
                f.write(f"Event type filter applied: {results['event_filter']}\n")
            
            f.write(f"Available event types: {', '.join(results['event_types'])}\n")
            f.write(f"Sampling rate for recommendations: {results['sampling_rate']} Hz\n\n")
            
            # Executive Summary by Event Type
            f.write("EXECUTIVE SUMMARY BY EVENT TYPE\n")
            f.write("-"*40 + "\n")
            
            type_stats = results['type_stats']
            if type_stats:
                f.write(f"{'Event Type':<15} {'N Files':<8} {'N Events':<9} {'Mean(s)':<8} {'Median(s)':<10} {'Std(s)':<8} {'Range(s)':<12}\n")
                f.write("-"*75 + "\n")
                
                for event_type, stats in sorted(type_stats.items()):
                    range_str = f"{stats['min_duration']:.1f}-{stats['max_duration']:.1f}"
                    f.write(f"{event_type:<15} {stats['n_files']:<8} {stats['n_events']:<9} "
                           f"{stats['mean_duration']:<8.2f} {stats['median_duration']:<10.2f} "
                           f"{stats['std_duration']:<8.2f} {range_str:<12}\n")
                
                f.write("\n")
            
            # Overall Statistics
            if results['overall_stats']:
                f.write("OVERALL STATISTICS (ALL EVENT TYPES)\n")
                f.write("-"*38 + "\n")
                overall = results['overall_stats']
                f.write(f"Mean:                {overall['mean_duration']:8.2f} seconds\n")
                f.write(f"Median:              {overall['median_duration']:8.2f} seconds\n")
                f.write(f"Standard deviation:  {overall['std_duration']:8.2f} seconds\n")
                f.write(f"Minimum:             {overall['min_duration']:8.2f} seconds\n")
                f.write(f"Maximum:             {overall['max_duration']:8.2f} seconds\n")
                f.write(f"Range:               {overall['range']:8.2f} seconds\n")
                f.write(f"Coefficient of variation: {overall['coefficient_of_variation']:8.2f}\n\n")
            
            # Detailed Event Type Statistics
            f.write("DETAILED EVENT TYPE STATISTICS\n")
            f.write("-"*35 + "\n")
            
            for event_type, stats in sorted(type_stats.items()):
                f.write(f"\nEVENT TYPE: {event_type.upper()}\n")
                f.write("="*25 + "\n")
                f.write(f"Files:               {stats['n_files']:8d}\n")
                f.write(f"Events:              {stats['n_events']:8d}\n")
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
            
            # Metadata Analysis
            if include_metadata and results['metadata_stats']:
                f.write("METADATA ANALYSIS\n")
                f.write("-"*20 + "\n")
                
                for metadata_type, metadata_analysis in results['metadata_stats'].items():
                    f.write(f"\n{metadata_type.upper()} by Event Type:\n")
                    f.write("-" * (len(metadata_type) + 16) + "\n")
                    
                    for event_type, values in metadata_analysis.items():
                        if values:
                            f.write(f"\n{event_type}:\n")
                            for value, stats in sorted(values.items(), key=lambda x: x[1]['count'], reverse=True):
                                f.write(f"  {value:<15}: {stats['count']:3d} events ({stats['percentage']:5.1f}%), "
                                       f"mean {stats['mean_duration']:5.2f}s\n")
                
                f.write("\n")
            
            # Event Type Comparison
            if len(type_stats) > 1:
                f.write("EVENT TYPE COMPARISON\n")
                f.write("-"*25 + "\n")
                
                # Sort types by mean duration
                sorted_types = sorted(type_stats.items(), key=lambda x: x[1]['mean_duration'])
                
                f.write("Event types ordered by mean duration (shortest to longest):\n")
                for i, (event_type, stats) in enumerate(sorted_types, 1):
                    f.write(f"  {i}. {event_type:<15}: {stats['mean_duration']:6.2f}s (n={stats['n_events']})\n")
                
                f.write("\nVariability comparison (coefficient of variation):\n")
                sorted_by_cv = sorted(type_stats.items(), key=lambda x: x[1]['coefficient_of_variation'])
                for i, (event_type, stats) in enumerate(sorted_by_cv, 1):
                    variability = "Low" if stats['coefficient_of_variation'] < 0.5 else "Moderate" if stats['coefficient_of_variation'] < 1.0 else "High"
                    f.write(f"  {i}. {event_type:<15}: CV={stats['coefficient_of_variation']:5.2f} ({variability})\n")
                
                f.write("\n")
            
            # Type-Specific MADRID Recommendations
            if include_recommendations and results['type_recommendations']:
                f.write("EVENT TYPE-SPECIFIC MADRID RECOMMENDATIONS\n")
                f.write("-"*50 + "\n")
                
                for event_type, recommendations in results['type_recommendations'].items():
                    f.write(f"\nEVENT TYPE: {event_type.upper()}\n")
                    f.write("="*25 + "\n")
                    
                    fs = recommendations['sampling_rate']
                    f.write(f"Recommendations for {event_type} events at {fs:.0f} Hz:\n\n")
                    
                    for approach_name, approach in recommendations.items():
                        if approach_name in ['event_type', 'sampling_rate']:
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
                    f.write(f"SPECIFIC m-VALUES for {event_type.upper()}:\n")
                    f.write("-"*25 + "\n")
                    
                    type_stat = type_stats[event_type]
                    specific_m_values = [
                        ("Mean duration", type_stat['mean_duration'], int(type_stat['mean_duration'] * fs)),
                        ("Median duration", type_stat['median_duration'], int(type_stat['median_duration'] * fs)),
                    ]
                    
                    # Add key percentiles
                    percentile_values = [25, 50, 75, 90, 95]
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
            
            # Subject-Event Type Analysis
            if results['subject_type_stats'] and include_detailed:
                f.write("SUBJECT-EVENT TYPE ANALYSIS\n")
                f.write("-"*30 + "\n")
                
                for subject_id, type_data in sorted(results['subject_type_stats'].items()):
                    f.write(f"\nSubject: {subject_id}\n")
                    f.write("-"*15 + "\n")
                    f.write(f"{'Event Type':<15} {'N':<4} {'Mean':<8} {'Median':<8} {'Std':<8} {'Min':<8} {'Max':<8}\n")
                    f.write("-"*65 + "\n")
                    
                    for event_type, stats in sorted(type_data.items()):
                        f.write(f"{event_type:<15} {stats['n_events']:<4} {stats['mean_duration']:<8.2f} "
                               f"{stats['median_duration']:<8.2f} {stats['std_duration']:<8.2f} "
                               f"{stats['min_duration']:<8.2f} {stats['max_duration']:<8.2f}\n")
                
                f.write("\n")
            
            # Final Recommendations
            f.write("FINAL RECOMMENDATIONS\n")
            f.write("-"*25 + "\n")
            
            if type_stats:
                # Find most common type
                most_common_type = max(type_stats.items(), key=lambda x: x[1]['n_events'])
                f.write(f"1. MOST COMMON EVENT TYPE: {most_common_type[0].upper()}\n")
                f.write(f"   {most_common_type[1]['n_events']} events, mean duration: {most_common_type[1]['mean_duration']:.2f}s\n")
                f.write(f"   Use event type-specific parameters for best results\n\n")
                
                # Find shortest and longest types
                shortest_type = min(type_stats.items(), key=lambda x: x[1]['mean_duration'])
                longest_type = max(type_stats.items(), key=lambda x: x[1]['mean_duration'])
                
                if len(type_stats) > 1:
                    f.write(f"2. EVENT TYPE RANGE:\n")
                    f.write(f"   Shortest: {shortest_type[0]} ({shortest_type[1]['mean_duration']:.2f}s)\n")
                    f.write(f"   Longest:  {longest_type[0]} ({longest_type[1]['mean_duration']:.2f}s)\n")
                    f.write(f"   Consider event type-specific MADRID configurations\n\n")
                
                f.write(f"3. ANALYSIS STRATEGY:\n")
                f.write(f"   - Use separate models for each event type if possible\n")
                f.write(f"   - Focus on the most common event type first\n")
                f.write(f"   - Use madrid_m_analysis.py with event type-specific parameters\n")
                f.write(f"   - Consider ensemble approaches for mixed event types\n")
                f.write(f"   - Leverage metadata (lateralization, localization) for further refinement\n")
        
        print(f"✓ SeizeIT2 event analysis saved to: {output_file}")
    
    def plot_event_type_distributions(self, output_file: str = None, show_plot: bool = True):
        """
        Create plots of event duration distributions by type.
        
        Args:
            output_file: Optional file to save plot
            show_plot: Whether to display the plot
        """
        if not self.event_stats or not self.event_stats['success']:
            print("No data available for plotting")
            return
        
        # Prepare data for plotting
        event_type_durations = defaultdict(list)
        for file_info in self.event_stats['file_results']:
            for event in file_info['events']:
                event_type_durations[event['eventType']].append(event['duration'])
        
        type_stats = self.event_stats['type_stats']
        if not type_stats:
            print("No event type data available for plotting")
            return
        
        n_types = len(event_type_durations)
        if n_types == 0:
            print("No duration data available for plotting")
            return
        
        # Create figure with subplots
        fig_height = max(10, n_types * 2 + 4)
        fig, axes = plt.subplots(2, 2, figsize=(15, fig_height))
        fig.suptitle('SeizeIT2 Event Duration Analysis by Event Type', fontsize=16)
        
        # Prepare colors for each type
        colors = plt.cm.Set3(np.linspace(0, 1, n_types))
        
        # 1. Histograms by type
        ax1 = axes[0, 0]
        for i, (event_type, durations) in enumerate(event_type_durations.items()):
            ax1.hist(durations, bins=20, alpha=0.6, label=event_type, 
                    color=colors[i], density=True)
        ax1.set_xlabel('Duration (seconds)')
        ax1.set_ylabel('Density')
        ax1.set_title('Event Duration Distributions by Type')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Box plots by type
        ax2 = axes[0, 1]
        type_names = list(event_type_durations.keys())
        duration_lists = [event_type_durations[t] for t in type_names]
        box_plot = ax2.boxplot(duration_lists, labels=type_names, patch_artist=True)
        
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        ax2.set_ylabel('Duration (seconds)')
        ax2.set_title('Event Duration Box Plots by Type')
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # 3. Mean comparison with event counts
        ax3 = axes[1, 0]
        means = [type_stats[t]['mean_duration'] for t in type_names]
        stds = [type_stats[t]['std_duration'] for t in type_names]
        
        bars = ax3.bar(type_names, means, yerr=stds, capsize=5, 
                      color=colors, alpha=0.7)
        ax3.set_ylabel('Mean Duration (seconds)')
        ax3.set_title('Mean Duration by Event Type (with std)')
        ax3.grid(True, alpha=0.3, axis='y')
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, mean, event_type in zip(bars, means, type_names):
            height = bar.get_height()
            n_events = type_stats[event_type]['n_events']
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mean:.1f}s\n(n={n_events})', ha='center', va='bottom')
        
        # 4. Event type frequency
        ax4 = axes[1, 1]
        event_counts = [type_stats[t]['n_events'] for t in type_names]
        
        bars = ax4.bar(type_names, event_counts, color=colors, alpha=0.7)
        ax4.set_ylabel('Number of Events')
        ax4.set_title('Event Frequency by Type')
        ax4.grid(True, alpha=0.3, axis='y')
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars, event_counts):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ Event type distribution plots saved to: {output_file}")
        
        if show_plot:
            plt.show()


def main():
    """Main function for SeizeIT2 event analysis."""
    parser = argparse.ArgumentParser(description='SeizeIT2 Event Type Duration Analysis')
    
    # Required arguments
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to SeizeIT2 BIDS dataset root directory')
    
    # File selection
    parser.add_argument('--percentage', type=float, default=100.0,
                       help='Percentage of files to analyze (default: 100)')
    parser.add_argument('--subject', type=str,
                       help='Filter by subject ID (e.g., sub-001)')
    parser.add_argument('--event-filter', type=str,
                       help='Filter by event type (e.g., focal, generalized)')
    
    # Configuration
    parser.add_argument('--sampling-rate', type=float, default=250.0,
                       help='Sampling rate for MADRID recommendations (default: 250)')
    
    # Output options
    parser.add_argument('--output', type=str, default='seizeit2_event_analysis.txt',
                       help='Output file for results (default: seizeit2_event_analysis.txt)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate event duration distribution plots by type')
    parser.add_argument('--plot-file', type=str,
                       help='File to save plots (e.g., event_plots.png)')
    parser.add_argument('--no-detailed', action='store_true',
                       help='Skip detailed per-file results in output')
    parser.add_argument('--no-recommendations', action='store_true',
                       help='Skip event type-specific MADRID parameter recommendations')
    parser.add_argument('--no-metadata', action='store_true',
                       help='Skip metadata analysis (lateralization, localization, vigilance)')
    
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
    
    print("SeizeIT2 Event Type Duration Analysis")
    print("=" * 40)
    print(f"Data directory: {args.data_dir}")
    print(f"Processing: {args.percentage}% of files")
    if args.event_filter:
        print(f"Event type filter: {args.event_filter}")
    print(f"Sampling rate: {args.sampling_rate} Hz")
    print(f"Output file: {args.output}")
    
    try:
        # Initialize analyzer
        analyzer = SeizeIT2EventAnalyzer(sampling_rate=args.sampling_rate)
        
        # Discover event files
        event_files = analyzer.discover_event_files(
            data_dir=args.data_dir,
            percentage=args.percentage,
            subject_filter=args.subject
        )
        
        if not event_files:
            print("No events.tsv files found")
            return 1
        
        # Analyze events by type
        results = analyzer.analyze_events_by_type(
            event_files, 
            event_filter=args.event_filter
        )
        
        if not results['success']:
            print(f"Analysis failed: {results['message']}")
            return 1
        
        # Print summary
        if results['overall_stats']:
            overall = results['overall_stats']
            print(f"\n{'='*70}")
            print(f"SEIZEIT2 EVENT ANALYSIS SUMMARY")
            print(f"{'='*70}")
            print(f"Files analyzed: {overall['n_files']}")
            print(f"Total events: {overall['n_events']}")
            print(f"Event types: {overall['n_types']}")
            print(f"Overall mean duration: {overall['mean_duration']:.2f}s")
            print(f"Overall range: {overall['min_duration']:.2f}s - {overall['max_duration']:.2f}s")
            
            # Show event type breakdown
            type_stats = results['type_stats']
            if type_stats:
                print(f"\nEvent type breakdown:")
                for event_type, stats in sorted(type_stats.items(), 
                                               key=lambda x: x[1]['n_events'], reverse=True):
                    print(f"  {event_type:<15}: {stats['n_events']:3d} events, "
                          f"mean {stats['mean_duration']:5.2f}s")
        
        # Save results
        analyzer.save_event_analysis(
            output_file=args.output,
            include_detailed=not args.no_detailed,
            include_recommendations=not args.no_recommendations,
            include_metadata=not args.no_metadata
        )
        
        # Generate plots if requested
        if args.plot:
            analyzer.plot_event_type_distributions(
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