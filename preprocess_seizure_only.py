#!/usr/bin/env python3
"""
Script to preprocess only seizure segments from SeizeIT2 dataset.
Extracts seizure events ± 30 minutes for efficient algorithm testing.

This script implements a targeted preprocessing approach that:
- Discovers all seizure events across the dataset
- Extracts only seizure segments with surrounding context
- Reduces data volume by ~13x while preserving all seizures
- Enables rapid parameter optimization and algorithm testing
"""

from seizure_only_preprocessing import SeizureOnlyECGPreprocessor
from pathlib import Path
import pandas as pd
import time
import csv
import argparse


# Configuration presets for different algorithms and experiments
PREPROCESSING_CONFIGS = {
    'default': {
        'downsample_freq': 125,
        'filter_range': (0.5, 40),
        'context_minutes': 30
    },
    'merlin_125hz': {
        'downsample_freq': 125,
        'filter_range': (0.5, 40),
        'context_minutes': 30
    },
    'merlin_32hz': {
        'downsample_freq': 32,
        'filter_range': (0.5, 40),
        'context_minutes': 30
    },
    'merlin_8hz': {
        'downsample_freq': 8,
        'filter_range': (0.5, 40),
        'context_minutes': 30
    },
    'timevqvae': {
        'downsample_freq': 100,
        'filter_range': (0.5, 40),
        'context_minutes': 30
    },
    'matrix_profile': {
        'downsample_freq': 50,
        'filter_range': (0.5, 40),
        'context_minutes': 30
    },
    'multi_algorithm': {
        'downsample_freq': 125,  # High resolution for flexibility
        'filter_range': (0.5, 40),
        'context_minutes': 30
    }
}


def main():
    parser = argparse.ArgumentParser(description='Preprocess seizure-only segments from SeizeIT2')
    parser.add_argument('--config', choices=list(PREPROCESSING_CONFIGS.keys()), 
                       default='default', help='Preprocessing configuration preset')
    parser.add_argument('--data-path', type=str, 
                       default="/home/swolf/asim_shared/raw_data/ds005873-1.1.0",
                       help='Path to SeizeIT2 dataset')
    parser.add_argument('--output-path', type=str,
                       default="/home/swolf/asim_shared/preprocessed_data/seizure_only",
                       help='Output directory for processed segments')
    parser.add_argument('--context-minutes', type=int, default=None,
                       help='Minutes of context before/after seizure (overrides config)')
    
    args = parser.parse_args()
    
    print("SEIZURE-ONLY ECG PREPROCESSING")
    print("=" * 50)
    print(f"Configuration: {args.config}")
    print(f"Data path: {args.data_path}")
    print(f"Output path: {args.output_path}")
    
    # Get configuration
    config = PREPROCESSING_CONFIGS[args.config]
    if args.context_minutes:
        config['context_minutes'] = args.context_minutes
    
    print(f"Settings:")
    print(f"  • Downsample frequency: {config['downsample_freq']} Hz")
    print(f"  • Filter range: {config['filter_range'][0]}-{config['filter_range'][1]} Hz") 
    print(f"  • Context: ±{config['context_minutes']} minutes")
    
    # Initialize preprocessor
    preprocessor = SeizureOnlyECGPreprocessor(
        filter_params={
            'low_freq': config['filter_range'][0],
            'high_freq': config['filter_range'][1],
            'order': 4
        },
        downsample_freq=config['downsample_freq'],
        context_minutes=config['context_minutes']
    )
    
    # Check data path exists
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data path {data_path} does not exist!")
        print("Please ensure the SeizeIT2 dataset is downloaded and extracted.")
        return
    
    # Create output directory
    output_path = Path(args.output_path)
    config_suffix = f"downsample_{config['downsample_freq']}hz_context_{config['context_minutes']}min"
    results_path = output_path / config_suffix
    results_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {results_path}")
    
    # Phase 1: Discover all seizure segments
    print("\n" + "=" * 50)
    print("PHASE 1: SEIZURE DISCOVERY")
    print("=" * 50)
    
    start_time = time.time()
    seizure_segments = preprocessor.discover_seizure_segments(str(data_path))
    
    if not seizure_segments:
        print("No seizure segments found!")
        return
    
    discovery_time = time.time() - start_time
    print(f"\nDiscovery complete: {len(seizure_segments)} seizure segments found")
    print(f"Discovery time: {discovery_time:.1f} seconds")
    
    # Calculate expected data reduction
    total_segments = len(seizure_segments)
    avg_segment_duration = config['context_minutes'] * 2 + 5  # Rough estimate
    total_hours = total_segments * avg_segment_duration / 60
    print(f"Estimated output: ~{total_hours:.0f} hours of seizure-focused data")
    
    # Phase 2: Process seizure segments
    print("\n" + "=" * 50) 
    print("PHASE 2: SEGMENT PROCESSING")
    print("=" * 50)
    
    successful_segments = []
    failed_segments = []
    processing_start = time.time()
    
    for i, (subject_id, run_id, seizure_idx, seizure_start, seizure_end, 
            extract_start, extract_end) in enumerate(seizure_segments):
        
        # Check if already processed
        filename = f"{subject_id}_{run_id}_seizure_{seizure_idx:02d}_preprocessed.pkl"
        filepath = results_path / filename
        
        if filepath.exists():
            print(f"[{i+1:3d}/{total_segments:3d}] Skipping {subject_id} {run_id} #{seizure_idx}: already processed")
            successful_segments.append((subject_id, run_id, seizure_idx))
            continue
        
        print(f"[{i+1:3d}/{total_segments:3d}] Processing {subject_id} {run_id} seizure #{seizure_idx}...")
        
        try:
            result = preprocessor.preprocess_seizure_segment(
                str(data_path), subject_id, run_id, seizure_idx,
                seizure_start, seizure_end, extract_start, extract_end
            )
            
            if result is not None:
                # Save result
                pd.to_pickle(result, filepath)
                successful_segments.append((subject_id, run_id, seizure_idx))
                
                # Print summary
                total_samples = sum(ch['n_samples'] for ch in result['channels'])
                ictal_samples = sum(ch['n_ictal_samples'] for ch in result['channels'])
                duration = result['metadata']['total_duration']
                
                print(f"    Success: {duration:.1f}s segment, {total_samples:,} samples "
                      f"({ictal_samples:,} ictal)")
                
            else:
                failed_segments.append((subject_id, run_id, seizure_idx))
                print(f"    Failed: No data or processing error")
                
        except Exception as e:
            failed_segments.append((subject_id, run_id, seizure_idx))
            print(f"    Error: {str(e)}")
        
        # Progress update every 10 segments
        if (i + 1) % 10 == 0:
            elapsed = time.time() - processing_start
            processed = len(successful_segments) + len(failed_segments)
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total_segments - processed) / rate if rate > 0 else 0
            
            print(f"Progress: {processed}/{total_segments} segments "
                  f"({len(successful_segments)}/{processed} successful, {rate:.1f}/min, "
                  f"ETA: {eta/60:.1f} min)")
    
    # Final summary
    total_time = time.time() - start_time
    processing_time = time.time() - processing_start
    
    print("\n" + "=" * 50)
    print("PROCESSING COMPLETE!")
    print("=" * 50)
    
    print(f"\nFINAL STATISTICS:")
    print(f"  • Total seizure segments found:  {len(seizure_segments)}")
    print(f"  • Successfully processed:        {len(successful_segments)}")
    print(f"  • Failed:                        {len(failed_segments)}")
    print(f"  • Success rate:                  {len(successful_segments)/len(seizure_segments)*100:.1f}%")
    print(f"  • Total time:                    {total_time/60:.1f} minutes")
    print(f"  • Processing time:               {processing_time/60:.1f} minutes")
    print(f"  • Average time per segment:      {processing_time/len(seizure_segments):.1f} seconds")
    
    # Calculate data statistics
    if successful_segments:
        print(f"\nDATA SUMMARY:")
        
        total_duration = 0
        total_samples = 0
        total_ictal_samples = 0
        total_pre_samples = 0
        total_post_samples = 0
        
        for subject_id, run_id, seizure_idx in successful_segments:
            filename = f"{subject_id}_{run_id}_seizure_{seizure_idx:02d}_preprocessed.pkl"
            filepath = results_path / filename
            
            try:
                result = pd.read_pickle(filepath)
                total_duration += result['metadata']['total_duration']
                
                for channel in result['channels']:
                    total_samples += channel['n_samples']
                    total_ictal_samples += channel['n_ictal_samples']
                    total_pre_samples += channel['n_pre_seizure_samples']
                    total_post_samples += channel['n_post_seizure_samples']
            except:
                continue
        
        ictal_percentage = (total_ictal_samples / total_samples * 100) if total_samples else 0
        pre_percentage = (total_pre_samples / total_samples * 100) if total_samples else 0
        post_percentage = (total_post_samples / total_samples * 100) if total_samples else 0
        
        print(f"  • Total processed duration:      {total_duration/3600:.1f} hours")
        print(f"  • Total samples:                 {total_samples:,}")
        print(f"  • Ictal samples:                 {total_ictal_samples:,} ({ictal_percentage:.1f}%)")
        print(f"  • Pre-seizure samples:           {total_pre_samples:,} ({pre_percentage:.1f}%)")
        print(f"  • Post-seizure samples:          {total_post_samples:,} ({post_percentage:.1f}%)")
        
        # Calculate storage size
        total_size = sum(f.stat().st_size for f in results_path.glob("*.pkl"))
        print(f"  • Storage size:                  {total_size/(1024**3):.2f} GB")
        
        # Data reduction factor (rough estimate)
        original_estimate = 11640  # hours in full dataset  
        reduction_factor = original_estimate / (total_duration/3600)
        print(f"  • Data reduction factor:         ~{reduction_factor:.1f}x")
        
        # Export summary statistics
        stats = {
            "Total seizure segments found": len(seizure_segments),
            "Successfully processed": len(successful_segments),
            "Failed": len(failed_segments),
            "Success rate (%)": (len(successful_segments)/len(seizure_segments)*100) if seizure_segments else 0,
            "Total time (minutes)": total_time / 60,
            "Processing time (minutes)": processing_time / 60,
            "Average time per segment (seconds)": (processing_time / len(seizure_segments)) if seizure_segments else 0,
            "Total processed duration (hours)": total_duration / 3600,
            "Total samples": total_samples,
            "Ictal samples": total_ictal_samples,
            "Ictal percentage (%)": ictal_percentage,
            "Pre-seizure samples": total_pre_samples,
            "Post-seizure samples": total_post_samples,
            "Storage size (GB)": total_size / (1024 ** 3),
            "Data reduction factor": reduction_factor
        }
        
        df_stats = pd.DataFrame(list(stats.items()), columns=["Metric", "Value"])
        excel_path = results_path / "seizure_preprocessing_summary.xlsx"
        df_stats.to_excel(excel_path, index=False)
        print(f"  • Summary saved to:              {excel_path}")
    
    # Save failed segments list
    if failed_segments:
        print(f"\nFAILED SEGMENTS ({len(failed_segments)}):")
        csv_filename = results_path / "failed_seizure_segments.csv"
        with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["subject_id", "run_id", "seizure_index"])
            for subject_id, run_id, seizure_idx in failed_segments:
                writer.writerow([subject_id, run_id, seizure_idx])
                print(f"  • {subject_id} {run_id} #{seizure_idx}")
        print(f"Failed segments saved to: {csv_filename}")
    
    print(f"\nResults saved to: {results_path}")
    print("\nNext steps:")
    print("  1. Use processed seizure segments for parameter optimization")
    print("  2. Test different algorithms on this focused dataset")
    print("  3. Once parameters are optimized, run full dataset preprocessing")


if __name__ == "__main__":
    main()