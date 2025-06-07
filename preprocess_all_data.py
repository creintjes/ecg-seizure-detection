#!/usr/bin/env python3
"""
Script to preprocess ALL ECG data from the SeizeIT2 dataset.
"""

from preprocessing import ECGPreprocessor
from pathlib import Path
import os
import time
import pandas as pd

def discover_all_recordings(data_path):
    """
    Discover all available recordings in the SeizeIT2 dataset.
    
    Args:
        data_path: Path to SeizeIT2 dataset
        
    Returns:
        List of (subject_id, run_id) tuples
    """
    data_path = Path(data_path)
    recordings = []
    
    # Find all subjects
    subjects = [x for x in data_path.glob("sub-*") if x.is_dir()]
    print(f"Found {len(subjects)} subjects")
    
    for subject_dir in subjects:
        subject_id = subject_dir.name
        
        # Look for ECG sessions
        ecg_dir = subject_dir / 'ses-01' / 'ecg'
        if ecg_dir.exists():
            # Find all runs for this subject
            edf_files = list(ecg_dir.glob("*_ecg.edf"))
            
            for edf_file in edf_files:
                # Extract run ID from filename
                # Format: sub-XXX_ses-01_task-szMonitoring_run-XX_ecg.edf
                parts = edf_file.stem.split('_')
                run_part = [p for p in parts if p.startswith('run-')]
                
                if run_part:
                    run_id = run_part[0]
                    recordings.append((subject_id, run_id))
                    print(f"  Found: {subject_id} {run_id}")
    
    return recordings

def main():
    print("🔄 PREPROCESSING ALL SEIZEIT2 ECG DATA")
    print("=" * 50)
    
    # Initialize preprocessor
    preprocessor = ECGPreprocessor(
        filter_params={
            'low_freq': 0.5,     # Remove baseline drift
            'high_freq': 40.0,   # Remove high-frequency noise
            'order': 4           # Filter order
        },
        downsample_freq=125,     # Target sampling rate
        window_size=30.0,        # 30 second windows
        stride=15.0              # 50% overlap
    )
    
    # Set data path
    data_path = "ds005873-download"
    
    if not Path(data_path).exists():
        print(f"❌ Error: Data path {data_path} does not exist!")
        print("Please ensure the SeizeIT2 dataset is downloaded and extracted.")
        return
    
    # Discover all recordings
    print("\n📂 Discovering recordings...")
    recordings = discover_all_recordings(data_path)
    
    if not recordings:
        print("❌ No recordings found!")
        return
    
    print(f"✅ Found {len(recordings)} recordings to process")
    
    # Create output directory
    results_path = Path("./results/preprocessed_all")
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Process in batches to avoid memory issues
    batch_size = 20
    start_time = time.time()
    
    successful_recordings = []
    failed_recordings = []
    
    for i in range(0, len(recordings), batch_size):
        batch = recordings[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(recordings) + batch_size - 1) // batch_size
        
        print(f"\n🔄 Processing batch {batch_num}/{total_batches} ({len(batch)} recordings)")
        print("-" * 40)
        
        for subject_id, run_id in batch:
            try:
                print(f"Processing {subject_id} {run_id}...")
                
                result = preprocessor.preprocess_pipeline(data_path, subject_id, run_id)
                
                if result is not None:
                    # Save individual result
                    filename = f"{subject_id}_{run_id}_preprocessed.pkl"
                    filepath = results_path / filename
                    pd.to_pickle(result, filepath)
                    
                    successful_recordings.append((subject_id, run_id))
                    
                    # Print summary
                    total_windows = sum(ch['n_windows'] for ch in result['channels'])
                    seizure_windows = sum(ch['n_seizure_windows'] for ch in result['channels'])
                    
                    print(f"  ✅ Success: {total_windows} windows ({seizure_windows} seizure)")
                else:
                    failed_recordings.append((subject_id, run_id))
                    print(f"  ❌ Failed: No data or processing error")
                    
            except Exception as e:
                failed_recordings.append((subject_id, run_id))
                print(f"  ❌ Error: {str(e)}")
        
        # Progress update
        elapsed = time.time() - start_time
        processed = len(successful_recordings) + len(failed_recordings)
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (len(recordings) - processed) / rate if rate > 0 else 0
        
        print(f"\n📊 Progress: {processed}/{len(recordings)} recordings")
        print(f"   Success rate: {len(successful_recordings)}/{processed} ({len(successful_recordings)/processed*100:.1f}%)")
        print(f"   Processing rate: {rate:.1f} recordings/min")
        print(f"   ETA: {eta/60:.1f} minutes")
    
    # Final summary
    total_time = time.time() - start_time
    
    print("\n" + "="*50)
    print("🎯 PREPROCESSING COMPLETE!")
    print("="*50)
    
    print(f"\n📊 FINAL STATISTICS:")
    print(f"  • Total recordings found:     {len(recordings)}")
    print(f"  • Successfully processed:     {len(successful_recordings)}")
    print(f"  • Failed:                     {len(failed_recordings)}")
    print(f"  • Success rate:               {len(successful_recordings)/len(recordings)*100:.1f}%")
    print(f"  • Total processing time:      {total_time/60:.1f} minutes")
    print(f"  • Average time per recording: {total_time/len(recordings):.1f} seconds")
    
    # Calculate total data statistics
    if successful_recordings:
        print(f"\n💾 DATA SUMMARY:")
        
        total_windows = 0
        total_seizure_windows = 0
        total_duration = 0
        
        for subject_id, run_id in successful_recordings:
            filename = f"{subject_id}_{run_id}_preprocessed.pkl"
            filepath = results_path / filename
            
            try:
                result = pd.read_pickle(filepath)
                total_duration += result['recording_duration']
                
                for channel in result['channels']:
                    total_windows += channel['n_windows']
                    total_seizure_windows += channel['n_seizure_windows']
            except:
                continue
        
        print(f"  • Total recording duration:   {total_duration/3600:.1f} hours")
        print(f"  • Total windows created:      {total_windows:,}")
        print(f"  • Seizure windows:            {total_seizure_windows:,}")
        print(f"  • Seizure percentage:         {total_seizure_windows/total_windows*100:.2f}%")
        
        # Calculate storage size
        total_size = sum(f.stat().st_size for f in results_path.glob("*.pkl"))
        print(f"  • Total storage size:         {total_size/(1024**3):.2f} GB")
    
    # List failed recordings
    if failed_recordings:
        print(f"\n❌ FAILED RECORDINGS ({len(failed_recordings)}):")
        for subject_id, run_id in failed_recordings:
            print(f"  • {subject_id} {run_id}")
    
    print(f"\n💾 Results saved to: {results_path}")
    print("✅ Ready for anomaly detection model training!")

if __name__ == "__main__":
    main()