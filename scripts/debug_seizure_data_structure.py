#!/usr/bin/env python3
"""
Debug script to examine the structure of seizure-only preprocessed data
"""

import pickle
import sys
from pathlib import Path

def debug_seizure_data(file_path):
    """Debug seizure-only data structure"""
    print(f"🔍 Debugging: {file_path}")
    print("=" * 50)
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print("📋 TOP-LEVEL KEYS:")
        for key in data.keys():
            value = data[key]
            if isinstance(value, (int, float, str)):
                print(f"  {key}: {value} ({type(value).__name__})")
            elif isinstance(value, (list, tuple)):
                print(f"  {key}: {type(value).__name__} with {len(value)} items")
            elif isinstance(value, dict):
                print(f"  {key}: dict with keys: {list(value.keys())}")
            else:
                print(f"  {key}: {type(value)} - {str(value)[:50]}...")
        
        print("\n📊 CHANNELS STRUCTURE:")
        if 'channels' in data and isinstance(data['channels'], list):
            for i, channel in enumerate(data['channels']):
                print(f"  Channel {i}:")
                for key, value in channel.items():
                    if key == 'data' and hasattr(value, '__len__'):
                        print(f"    {key}: array with {len(value)} samples")
                    elif key == 'labels' and hasattr(value, '__len__'):
                        unique_labels = set(value) if hasattr(value, '__iter__') else {value}
                        print(f"    {key}: array with {len(value)} labels, unique: {unique_labels}")
                    elif isinstance(value, (int, float, str)):
                        print(f"    {key}: {value}")
                    else:
                        print(f"    {key}: {type(value)} - {str(value)[:30]}...")
        
        print("\n🔧 PROCESSING PARAMS:")
        if 'processing_params' in data:
            params = data['processing_params']
            for key, value in params.items():
                print(f"  {key}: {value}")
        
        # Try to determine sampling rate
        print("\n📡 SAMPLING RATE DETECTION:")
        sampling_rate = None
        
        # Method 1: processing_params
        if 'processing_params' in data:
            sampling_rate = data['processing_params'].get('downsample_freq')
            if sampling_rate:
                print(f"  From processing_params.downsample_freq: {sampling_rate}Hz")
        
        # Method 2: direct key
        if sampling_rate is None and 'downsample_freq' in data:
            sampling_rate = data['downsample_freq']
            print(f"  From downsample_freq: {sampling_rate}Hz")
        
        # Method 3: channel metadata
        if sampling_rate is None and 'channels' in data:
            channel = data['channels'][0]
            if 'fs' in channel:
                sampling_rate = channel['fs']
                print(f"  From channel.fs: {sampling_rate}Hz")
        
        # Method 4: filename inference
        if sampling_rate is None:
            file_path_lower = str(file_path).lower()
            if '8hz' in file_path_lower:
                sampling_rate = 8
                print(f"  From filename (8hz): {sampling_rate}Hz")
            elif '32hz' in file_path_lower:
                sampling_rate = 32
                print(f"  From filename (32hz): {sampling_rate}Hz")
            elif '125hz' in file_path_lower:
                sampling_rate = 125
                print(f"  From filename (125hz): {sampling_rate}Hz")
        
        if sampling_rate is None:
            print("  ❌ Could not determine sampling rate!")
        else:
            print(f"  ✅ Final sampling rate: {sampling_rate}Hz")
        
        # Calculate actual sampling rate from data
        if 'channels' in data and len(data['channels']) > 0:
            channel = data['channels'][0]
            if 'data' in channel and 'timestamps' in channel:
                data_array = channel['data']
                timestamps = channel['timestamps']
                if len(timestamps) > 1 and len(data_array) > 1:
                    duration = timestamps[-1] - timestamps[0]
                    actual_fs = len(data_array) / duration
                    print(f"  📏 Calculated from data: {actual_fs:.1f}Hz")
        
        print("\n" + "=" * 50)
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug_seizure_data_structure.py <path_to_pkl_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        sys.exit(1)
    
    debug_seizure_data(file_path)