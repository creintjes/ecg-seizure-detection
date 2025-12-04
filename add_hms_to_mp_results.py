#!/usr/bin/env python3
"""
Add HMS (Harmonic Mean Score) column to Matrix Profile confidence results.

HMS = sensitivity (%) - 0.4 * false_alarms_per_hour

Processes all .xlsx files in confidence/MP directory:
- Reads subject, sensitivity, false_alarms_per_hour
- Calculates HMS
- Adds HMS column to the original file
"""

import sys
import pandas as pd
from pathlib import Path
from typing import List


def calculate_hms(sensitivity: float, false_alarms_per_hour: float) -> float:
    """
    Calculate HMS score.

    Args:
        sensitivity: Sensitivity as decimal (0-1)
        false_alarms_per_hour: False alarms per hour

    Returns:
        HMS score
    """
    # Convert sensitivity to percentage and apply formula
    sensitivity_percent = sensitivity * 100
    hms = sensitivity_percent - 0.4 * false_alarms_per_hour
    return hms


def process_excel_file(file_path: Path) -> bool:
    """
    Process a single Excel file: add HMS column.

    Args:
        file_path: Path to Excel file

    Returns:
        True if successful, False otherwise
    """
    try:
        # Read Excel file
        df = pd.read_excel(file_path)

        # Check if required columns exist
        required_columns = ['subject', 'sensitivity', 'false_alarms_per_hour']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"  WARNING: Missing columns: {missing_columns}")
            print(f"  Available columns: {list(df.columns)}")
            print(f"  Skipping file.")
            return False

        # Check if HMS column already exists
        if 'HMS' in df.columns:
            print(f"  HMS column already exists. Recalculating...")
            # Remove old HMS column
            df = df.drop(columns=['HMS'])

        # Calculate HMS for each row
        df['HMS'] = df.apply(
            lambda row: calculate_hms(row['sensitivity'], row['false_alarms_per_hour']),
            axis=1
        )

        # Save back to Excel (overwrite original)
        df.to_excel(file_path, index=False)

        print(f"  ✓ Added HMS column ({len(df)} rows)")
        print(f"  HMS range: [{df['HMS'].min():.2f}, {df['HMS'].max():.2f}]")

        return True

    except Exception as e:
        print(f"  ERROR processing {file_path.name}: {e}")
        return False


def process_directory(directory: Path) -> dict:
    """
    Process all Excel files in directory and subdirectories recursively.

    Args:
        directory: Path to directory containing Excel files

    Returns:
        Dictionary with statistics
    """
    if not directory.exists():
        print(f"ERROR: Directory not found: {directory}")
        return {'success': 0, 'failed': 0, 'total': 0}

    # Find all Excel files recursively
    excel_files = sorted(directory.rglob("*.xlsx"))

    # Filter out temporary Excel files (start with ~$)
    excel_files = [f for f in excel_files if not f.name.startswith("~$")]

    if not excel_files:
        print(f"No Excel files found in {directory} or its subdirectories")
        return {'success': 0, 'failed': 0, 'total': 0}

    print("="*60)
    print(f"Found {len(excel_files)} Excel file(s) in {directory} and subdirectories")
    print("="*60)
    print()

    success_count = 0
    failed_count = 0

    for excel_file in excel_files:
        # Show relative path for better readability
        relative_path = excel_file.relative_to(directory)
        print(f"File: {relative_path}")

        if process_excel_file(excel_file):
            success_count += 1
        else:
            failed_count += 1
        print()

    return {
        'success': success_count,
        'failed': failed_count,
        'total': len(excel_files)
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Add HMS column to Matrix Profile confidence results'
    )
    parser.add_argument(
        '-d', '--directory',
        type=str,
        default='confidence/MP',
        help='Directory containing Excel files (default: confidence/MP)'
    )

    args = parser.parse_args()

    # Convert to Path
    directory = Path(args.directory)

    print("="*60)
    print("ADD HMS TO MATRIX PROFILE RESULTS")
    print("="*60)
    print(f"Directory: {directory.absolute()}")
    print(f"Formula: HMS = sensitivity (%) - 0.4 * false_alarms_per_hour")
    print()

    # Process all files
    stats = process_directory(directory)

    # Print summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total files: {stats['total']}")
    print(f"Successfully processed: {stats['success']}")
    print(f"Failed: {stats['failed']}")

    if stats['success'] > 0:
        print(f"\n✓ HMS column added to {stats['success']} file(s)")

    if stats['failed'] > 0:
        print(f"\n⚠ {stats['failed']} file(s) could not be processed")

    return 0 if stats['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
