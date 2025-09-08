#!/usr/bin/env python3
"""
Madrid Tolerance Re-evaluation Script

This script re-evaluates Madrid results with tolerance windows:
- Anomalies detected 5 minutes before seizure onset are considered TP
- Anomalies detected 3 minutes after seizure offset are considered TP
- Maintains original JSON format but updates TP/FP classifications
"""

import json
import os
import glob
from typing import Dict, List, Any
import argparse
from pathlib import Path


def calculate_time_tolerance_samples(sampling_rate: float, pre_minutes: float = 5.0, post_minutes: float = 3.0):
    """Calculate tolerance windows in samples"""
    pre_samples = int(pre_minutes * 60 * sampling_rate)
    post_samples = int(post_minutes * 60 * sampling_rate)
    return pre_samples, post_samples


def is_anomaly_in_tolerance_window(anomaly_location: int, seizure_start: int, seizure_end: int, 
                                 pre_tolerance: int, post_tolerance: int) -> bool:
    """Check if anomaly is within tolerance window around seizure"""
    tolerance_start = seizure_start - pre_tolerance
    tolerance_end = seizure_end + post_tolerance
    
    return tolerance_start <= anomaly_location <= tolerance_end


def reclassify_anomalies_with_tolerance(result_data: Dict[str, Any]) -> Dict[str, Any]:
    """Reclassify anomalies using tolerance windows"""
    
    # Get sampling rate
    sampling_rate = result_data["input_data"]["signal_metadata"]["sampling_rate"]
    
    # Calculate tolerance in samples (5 min before, 3 min after)
    pre_tolerance, post_tolerance = calculate_time_tolerance_samples(sampling_rate, 5.0, 3.0)
    
    # Get seizure regions
    seizure_regions = result_data["validation_data"]["ground_truth"]["seizure_regions"]
    
    # Initialize counters
    new_tp_count = 0
    new_fp_count = 0
    
    # Process each anomaly
    for i, anomaly in enumerate(result_data["analysis_results"]["anomalies"]):
        anomaly_location = anomaly["location_sample"]
        original_classification = result_data["validation_data"]["seizure_overlap_info"][i]["classification"]
        
        # Check if anomaly is within tolerance window of any seizure
        is_tp = False
        for seizure in seizure_regions:
            if is_anomaly_in_tolerance_window(
                anomaly_location, 
                seizure["onset_sample"], 
                seizure["offset_sample"],
                pre_tolerance, 
                post_tolerance
            ):
                is_tp = True
                break
        
        # Update classification
        new_classification = "true_positive" if is_tp else "false_positive"
        
        # Update the anomaly's seizure_hit field
        result_data["analysis_results"]["anomalies"][i]["seizure_hit"] = is_tp
        
        # Update the overlap info
        overlap_info = result_data["validation_data"]["seizure_overlap_info"][i]
        overlap_info["classification"] = new_classification
        
        # If classification changed, update additional fields
        if new_classification != original_classification:
            if is_tp:
                # Find the closest seizure for distance calculation
                min_distance = float('inf')
                closest_seizure = None
                for seizure in seizure_regions:
                    distance = min(
                        abs(anomaly_location - seizure["onset_sample"]),
                        abs(anomaly_location - seizure["offset_sample"])
                    )
                    if distance < min_distance:
                        min_distance = distance
                        closest_seizure = seizure
                
                overlap_info["distance_to_onset"] = abs(anomaly_location - closest_seizure["onset_sample"])
                
                # Calculate overlap ratio (simplified - just check if in tolerance)
                overlap_info["overlap_ratio"] = 1.0 if (
                    closest_seizure["onset_sample"] <= anomaly_location <= closest_seizure["offset_sample"]
                ) else 0.0
                
                overlap_info["overlap_sample_start"] = closest_seizure["onset_sample"]
                overlap_info["overlap_sample_end"] = closest_seizure["offset_sample"]
            else:
                overlap_info["distance_to_onset"] = None
                overlap_info["overlap_ratio"] = 0.0
                overlap_info["overlap_sample_start"] = None
                overlap_info["overlap_sample_end"] = None
        
        # Count new classifications
        if new_classification == "true_positive":
            new_tp_count += 1
        else:
            new_fp_count += 1
    
    # Update performance metrics
    total_anomalies = len(result_data["analysis_results"]["anomalies"])
    result_data["analysis_results"]["performance_metrics"]["true_positives"] = new_tp_count
    result_data["analysis_results"]["performance_metrics"]["false_positives"] = new_fp_count
    
    # Update derived metrics
    if new_tp_count > 0:
        result_data["analysis_results"]["performance_metrics"]["sensitivity"] = float(new_tp_count)
        result_data["analysis_results"]["performance_metrics"]["precision"] = new_tp_count / total_anomalies
    else:
        result_data["analysis_results"]["performance_metrics"]["sensitivity"] = 0.0
        result_data["analysis_results"]["performance_metrics"]["precision"] = 0.0
    
    # Add tolerance info to metadata
    result_data["analysis_metadata"]["tolerance_applied"] = {
        "pre_seizure_minutes": 5.0,
        "post_seizure_minutes": 3.0,
        "pre_seizure_samples": pre_tolerance,
        "post_seizure_samples": post_tolerance,
        "reprocessed_timestamp": "2025-07-01"
    }
    
    return result_data


def process_single_file(input_file: str, output_file: str = None) -> None:
    """Process a single result file"""
    
    # Load original results
    with open(input_file, 'r') as f:
        result_data = json.load(f)
    
    # Reclassify with tolerance
    updated_data = reclassify_anomalies_with_tolerance(result_data)
    
    # Determine output file
    if output_file is None:
        # Create output filename with tolerance suffix
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_tolerance_5min_3min{input_path.suffix}"
    
    # Save updated results
    with open(output_file, 'w') as f:
        json.dump(updated_data, f, indent=2)
    
    print(f"Processed: {input_file} -> {output_file}")
    
    # Print summary
    orig_tp = sum(1 for info in result_data["validation_data"]["seizure_overlap_info"] 
                  if info["classification"] == "true_positive")
    new_tp = updated_data["analysis_results"]["performance_metrics"]["true_positives"]
    new_fp = updated_data["analysis_results"]["performance_metrics"]["false_positives"]
    
    print(f"  Original: TP={orig_tp}, FP={len(result_data['analysis_results']['anomalies'])-orig_tp}")
    print(f"  Updated:  TP={new_tp}, FP={new_fp}")
    print(f"  Change:   TP diff={new_tp-orig_tp}")


def process_directory(input_dir: str, output_dir: str = None, pattern: str = "*.json") -> None:
    """Process all JSON files in a directory"""
    
    input_path = Path(input_dir)
    if output_dir is None:
        output_path = input_path / "tolerance_adjusted"
    else:
        output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(exist_ok=True)
    
    # Find all JSON files
    json_files = list(input_path.glob(pattern))
    
    if not json_files:
        print(f"No files found matching pattern '{pattern}' in {input_dir}")
        return
    
    print(f"Found {len(json_files)} files to process")
    
    total_orig_tp = 0
    total_new_tp = 0
    total_orig_fp = 0
    total_new_fp = 0
    
    for json_file in json_files:
        # Skip processing summary files
        if "processing_summary" in json_file.name:
            continue
            
        try:
            # Load original data for stats
            with open(json_file, 'r') as f:
                orig_data = json.load(f)
            
            # Process file
            output_file = output_path / json_file.name
            process_single_file(str(json_file), str(output_file))
            
            # Load processed data for stats
            with open(output_file, 'r') as f:
                new_data = json.load(f)
            
            # Accumulate statistics
            orig_tp = sum(1 for info in orig_data["validation_data"]["seizure_overlap_info"] 
                         if info["classification"] == "true_positive")
            orig_fp = len(orig_data["analysis_results"]["anomalies"]) - orig_tp
            
            new_tp = new_data["analysis_results"]["performance_metrics"]["true_positives"]
            new_fp = new_data["analysis_results"]["performance_metrics"]["false_positives"]
            
            total_orig_tp += orig_tp
            total_orig_fp += orig_fp
            total_new_tp += new_tp
            total_new_fp += new_fp
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    # Print overall summary
    print("\n" + "="*50)
    print("OVERALL SUMMARY")
    print("="*50)
    print(f"Files processed: {len([f for f in json_files if 'processing_summary' not in f.name])}")
    print(f"Original Total:  TP={total_orig_tp}, FP={total_orig_fp}")
    print(f"Tolerance Total: TP={total_new_tp}, FP={total_new_fp}")
    print(f"Net Change:      TP diff={total_new_tp-total_orig_tp}")
    
    if total_orig_tp + total_orig_fp > 0:
        orig_precision = total_orig_tp / (total_orig_tp + total_orig_fp)
        new_precision = total_new_tp / (total_new_tp + total_new_fp)
        print(f"Precision:       Original={orig_precision:.3f}, New={new_precision:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate Madrid results with tolerance windows")
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("-o", "--output", help="Output file or directory")
    parser.add_argument("-p", "--pattern", default="madrid_results_*.json", 
                       help="File pattern for directory processing (default: madrid_results_*.json)")
    parser.add_argument("--pre-minutes", type=float, default=5.0,
                       help="Minutes before seizure onset to consider as TP (default: 5.0)")
    parser.add_argument("--post-minutes", type=float, default=3.0,
                       help="Minutes after seizure offset to consider as TP (default: 3.0)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Process single file
        process_single_file(args.input, args.output)
    elif input_path.is_dir():
        # Process directory
        process_directory(args.input, args.output, args.pattern)
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())