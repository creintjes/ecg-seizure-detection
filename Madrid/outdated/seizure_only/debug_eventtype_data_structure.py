#!/usr/bin/env python3
"""
Debug script to analyze the data structure of Madrid results and clustered data.
This will help identify the correct structure for loading clustered results.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any
import pprint

def analyze_directory_structure(base_dir: str):
    """Analyze the directory structure to find relevant files."""
    base_path = Path(base_dir)
    print(f"🔍 ANALYZING DIRECTORY STRUCTURE: {base_path}")
    print("=" * 80)
    
    if not base_path.exists():
        print(f"❌ Directory does not exist: {base_path}")
        return
    
    # Look for relevant files
    files_found = {
        'madrid_results': list(base_path.glob("madrid_results_*.json")),
        'clustered_files': list(base_path.glob("**/best_representatives.json")),
        'clusters_dir': list(base_path.glob("**/clusters/")),
        'json_files': list(base_path.glob("**/*.json")),
        'subdirs': [d for d in base_path.iterdir() if d.is_dir()]
    }
    
    for category, files in files_found.items():
        print(f"\n📁 {category.upper()}:")
        if files:
            for file in files[:5]:  # Show first 5
                print(f"  - {file}")
            if len(files) > 5:
                print(f"  ... and {len(files) - 5} more")
        else:
            print("  (none found)")
    
    return files_found

def analyze_madrid_results_structure(file_path: str):
    """Analyze the structure of a Madrid results file."""
    print(f"\n🔬 ANALYZING MADRID RESULTS FILE: {file_path}")
    print("=" * 80)
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print("📋 TOP-LEVEL KEYS:")
        for key in data.keys():
            print(f"  - {key}")
        
        print("\n📊 INPUT_DATA STRUCTURE:")
        if 'input_data' in data:
            pprint.pprint(data['input_data'], depth=2, width=120)
        else:
            print("  ❌ No 'input_data' found")
        
        print("\n🎯 ANALYSIS_RESULTS STRUCTURE:")
        if 'analysis_results' in data:
            analysis = data['analysis_results']
            print(f"  Keys: {list(analysis.keys())}")
            
            # Check for anomalies
            anomalies_keys = ['anomalies', 'ranked_anomalies', 'representatives']
            for key in anomalies_keys:
                if key in analysis:
                    anomalies = analysis[key]
                    print(f"  📈 {key}: {len(anomalies)} items")
                    if len(anomalies) > 0:
                        print(f"    Sample item keys: {list(anomalies[0].keys())}")
                        if 'seizure_hit' in anomalies[0]:
                            seizure_hits = sum(1 for a in anomalies if a.get('seizure_hit', False))
                            print(f"    Seizure hits: {seizure_hits}/{len(anomalies)}")
        else:
            print("  ❌ No 'analysis_results' found")
            
    except Exception as e:
        print(f"❌ Error analyzing file: {e}")

def analyze_clustered_results_structure(file_path: str):
    """Analyze the structure of clustered results file."""
    print(f"\n🧩 ANALYZING CLUSTERED RESULTS FILE: {file_path}")
    print("=" * 80)
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print("📋 TOP-LEVEL KEYS:")
        for key in data.keys():
            print(f"  - {key}")
        
        if 'representatives' in data:
            representatives = data['representatives']
            print(f"\n👥 REPRESENTATIVES: {len(representatives)} items")
            
            if len(representatives) > 0:
                print("📝 Sample representative structure:")
                sample = representatives[0]
                pprint.pprint(sample, depth=2, width=120)
                
                # Analyze file_id distribution
                file_ids = set()
                seizure_hits = 0
                for rep in representatives:
                    if 'file_id' in rep:
                        file_ids.add(rep['file_id'])
                    if rep.get('seizure_hit', False):
                        seizure_hits += 1
                
                print(f"\n📊 STATISTICS:")
                print(f"  Unique file_ids: {len(file_ids)}")
                print(f"  Seizure hits: {seizure_hits}/{len(representatives)}")
                print(f"  Sample file_ids: {list(file_ids)[:5]}")
                
        else:
            print("  ❌ No 'representatives' found")
            
    except Exception as e:
        print(f"❌ Error analyzing clustered file: {e}")

def compare_original_vs_clustered(madrid_file: str, clustered_file: str):
    """Compare original Madrid results with clustered results."""
    print(f"\n⚖️  COMPARING ORIGINAL VS CLUSTERED")
    print("=" * 80)
    
    try:
        # Load both files
        with open(madrid_file, 'r') as f:
            original = json.load(f)
        with open(clustered_file, 'r') as f:
            clustered = json.load(f)
        
        # Extract file info from original
        orig_subject = original['input_data']['subject_id']
        orig_run = original['input_data']['run_id'] 
        orig_seizure = original['input_data']['seizure_id']
        orig_file_id = f"{orig_subject}_{orig_run}_{orig_seizure}"
        
        print(f"🔍 Looking for file_id: {orig_file_id}")
        
        # Find matching representatives
        matching_reps = []
        if 'representatives' in clustered:
            for rep in clustered['representatives']:
                if rep.get('file_id') == orig_file_id:
                    matching_reps.append(rep)
        
        print(f"📊 COMPARISON RESULTS:")
        
        # Original anomalies
        orig_anomalies = []
        if 'analysis_results' in original:
            orig_anomalies = original['analysis_results'].get('anomalies', []) or \
                           original['analysis_results'].get('ranked_anomalies', [])
        
        print(f"  Original anomalies: {len(orig_anomalies)}")
        print(f"  Clustered representatives: {len(matching_reps)}")
        
        if orig_anomalies:
            orig_seizure_hits = sum(1 for a in orig_anomalies if a.get('seizure_hit', False))
            print(f"  Original seizure hits: {orig_seizure_hits}")
        
        if matching_reps:
            clustered_seizure_hits = sum(1 for r in matching_reps if r.get('seizure_hit', False))
            print(f"  Clustered seizure hits: {clustered_seizure_hits}")
            
            print(f"\n📝 Sample clustered representative:")
            pprint.pprint(matching_reps[0], depth=1, width=120)
        
    except Exception as e:
        print(f"❌ Error in comparison: {e}")

def main():
    """Main debug function."""
    print("🐛 MADRID EVENTTYPE DATA STRUCTURE DEBUG")
    print("=" * 80)
    
    # Configuration - adjust these paths
    base_dirs = [
        "Madrid/madrid_results/madrid_seizure_results_parallel_400/tolerance_adjusted",
        "Madrid/madrid_results/madrid_seizure_results_parallel_400/tolerance_adjusted_smart_clustered",
        # Add more potential paths
    ]
    
    results_files = []
    clustered_files = []
    
    # Analyze each directory
    for base_dir in base_dirs:
        if Path(base_dir).exists():
            print(f"\n🔍 ANALYZING: {base_dir}")
            files_found = analyze_directory_structure(base_dir)
            
            if files_found['madrid_results']:
                results_files.extend(files_found['madrid_results'])
            if files_found['clustered_files']:
                clustered_files.extend(files_found['clustered_files'])
    
    # Analyze sample Madrid results file
    if results_files:
        sample_madrid = results_files[0]
        analyze_madrid_results_structure(str(sample_madrid))
    else:
        print("\n❌ No Madrid results files found!")
    
    # Analyze clustered results file
    if clustered_files:
        sample_clustered = clustered_files[0]
        analyze_clustered_results_structure(str(sample_clustered))
    else:
        print("\n❌ No clustered results files found!")
    
    # Compare if both exist
    if results_files and clustered_files:
        compare_original_vs_clustered(str(results_files[0]), str(clustered_files[0]))
    
    print("\n" + "=" * 80)
    print("🎯 DEBUG COMPLETE - Please share this output!")
    print("=" * 80)

if __name__ == "__main__":
    main()