#!/usr/bin/env python3
"""
Debug script to examine Madrid results structure and identify why metrics are 0.
"""

import json
from pathlib import Path

def debug_madrid_results(results_dir: str):
    """Debug Madrid results structure."""
    results_path = Path(results_dir)
    
    print(f"Checking directory: {results_path}")
    print(f"Directory exists: {results_path.exists()}")
    
    if not results_path.exists():
        print("Directory not found!")
        return
    
    # Find JSON files
    json_files = list(results_path.glob("madrid_results_*.json"))
    print(f"Found {len(json_files)} JSON files")
    
    if len(json_files) == 0:
        print("No madrid_results_*.json files found!")
        return
    
    # Examine first file structure
    first_file = json_files[0]
    print(f"\nExamining file: {first_file.name}")
    
    try:
        with open(first_file, 'r') as f:
            data = json.load(f)
        
        print("\nTop-level keys:")
        for key in data.keys():
            print(f"  - {key}")
        
        if 'analysis_results' in data:
            analysis = data['analysis_results']
            print(f"\nanalysis_results keys:")
            for key in analysis.keys():
                print(f"  - {key}")
            
            # Check for anomalies
            anomaly_fields = ['anomalies', 'ranked_anomalies', 'results', 'detections']
            for field in anomaly_fields:
                if field in analysis:
                    anomalies = analysis[field]
                    print(f"\nFound {field}: {len(anomalies)} items")
                    if len(anomalies) > 0:
                        print(f"First anomaly keys: {list(anomalies[0].keys())}")
                        print(f"First anomaly: {anomalies[0]}")
                    break
            else:
                print(f"\nNo anomaly field found! Available keys: {list(analysis.keys())}")
        
        # Check validation data
        if 'validation_data' in data:
            validation = data['validation_data']
            print(f"\nvalidation_data keys:")
            for key in validation.keys():
                print(f"  - {key}")
            
            if 'seizure_overlap_info' in validation:
                overlap_info = validation['seizure_overlap_info']
                print(f"\nFound seizure_overlap_info: {len(overlap_info)} items")
                if len(overlap_info) > 0:
                    print(f"First overlap item: {overlap_info[0]}")
        
    except Exception as e:
        print(f"Error reading file: {e}")

def main():
    """Test different possible paths."""
    possible_paths = [
        "Madrid/madrid_results/madrid_seizure_results_parallel_400/tolerance_adjusted",
        "Madrid/madrid_results/madrid_seizure_results_parallel_400/tolerance_adjusted_smart_clustered",
        "madrid_results/tolerance_adjusted_smart_clustered"
    ]
    
    for path in possible_paths:
        print("=" * 60)
        debug_madrid_results(path)
        print()

if __name__ == "__main__":
    main()