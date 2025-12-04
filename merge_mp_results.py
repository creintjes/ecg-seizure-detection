#!/usr/bin/env python3
"""
Merge all Matrix Profile Excel results into a single CSV file.

Processes all Excel files in confidence/MP subdirectories and merges them into one CSV.
Adds two columns:
- config: Extracted from parent directory (FAR, HMS, Sensitivity)
- window_type: Extracted from grandparent directory (window, no_window)

Output: confidence/MP/merged_mp_results.csv
"""

import sys
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional


def extract_metadata_from_path(file_path: Path, base_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract config and window_type from file path.

    Args:
        file_path: Path to Excel file
        base_dir: Base directory (confidence/MP)

    Returns:
        Tuple of (window_type, config) or (None, None) if extraction fails
    """
    try:
        # Get relative path from base directory
        relative_path = file_path.relative_to(base_dir)
        parts = relative_path.parts

        # Expected structure: window_type/config/filename.xlsx
        # e.g., no_window/FAR/subject_level_metrics_FAR & no SDW.xlsx
        if len(parts) >= 3:
            window_type = parts[0]  # "window" or "no_window"
            config = parts[1]       # "FAR", "HMS", or "Sensitivity"

            # Validate values
            if window_type in ['window', 'no_window'] and config in ['FAR', 'HMS', 'Sensitivity']:
                return window_type, config

        print(f"  WARNING: Could not extract metadata from path: {relative_path}")
        return None, None

    except Exception as e:
        print(f"  ERROR extracting metadata: {e}")
        return None, None


def load_excel_with_metadata(file_path: Path, base_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load Excel file and add config and window_type columns.

    Args:
        file_path: Path to Excel file
        base_dir: Base directory

    Returns:
        DataFrame with added columns, or None if loading fails
    """
    try:
        # Extract metadata from path
        window_type, config = extract_metadata_from_path(file_path, base_dir)

        if window_type is None or config is None:
            print(f"  Skipping file due to metadata extraction failure")
            return None

        # Load Excel file
        df = pd.read_excel(file_path)

        if df.empty:
            print(f"  WARNING: File is empty")
            return None

        # Add metadata columns
        df['config'] = config
        df['window_type'] = window_type

        print(f"  ✓ Loaded {len(df)} rows (config={config}, window_type={window_type})")

        return df

    except Exception as e:
        print(f"  ERROR loading file: {e}")
        return None


def merge_mp_results(base_dir: Path, output_file: Path) -> bool:
    """
    Merge all Excel files in subdirectories into a single CSV.

    Args:
        base_dir: Base directory containing Excel files
        output_file: Output CSV file path

    Returns:
        True if successful, False otherwise
    """
    if not base_dir.exists():
        print(f"ERROR: Directory not found: {base_dir}")
        return False

    print("="*60)
    print("MERGE MATRIX PROFILE RESULTS")
    print("="*60)
    print(f"Base directory: {base_dir.absolute()}")
    print(f"Output file: {output_file.absolute()}")
    print()

    # Find all Excel files recursively
    excel_files = sorted(base_dir.rglob("*.xlsx"))

    # Filter out temporary Excel files (start with ~$)
    excel_files = [f for f in excel_files if not f.name.startswith("~$")]

    if not excel_files:
        print(f"ERROR: No Excel files found in {base_dir}")
        return False

    print(f"Found {len(excel_files)} Excel file(s)")
    print()

    # Load all files
    all_dataframes = []
    loaded_count = 0
    failed_count = 0

    for excel_file in excel_files:
        relative_path = excel_file.relative_to(base_dir)
        print(f"Processing: {relative_path}")

        df = load_excel_with_metadata(excel_file, base_dir)

        if df is not None:
            all_dataframes.append(df)
            loaded_count += 1
        else:
            failed_count += 1

        print()

    # Check if we have any data
    if not all_dataframes:
        print("ERROR: No data loaded from any files!")
        return False

    # Merge all dataframes
    print("="*60)
    print("MERGING DATA")
    print("="*60)

    merged_df = pd.concat(all_dataframes, ignore_index=True)

    print(f"Total rows in merged data: {len(merged_df)}")
    print(f"Total columns: {len(merged_df.columns)}")
    print(f"Columns: {list(merged_df.columns)}")
    print()

    # Show summary statistics
    print("Summary by config and window_type:")
    print("-"*60)
    summary = merged_df.groupby(['config', 'window_type']).size().reset_index(name='count')
    for _, row in summary.iterrows():
        print(f"  {row['config']:<15} {row['window_type']:<12} {row['count']:>4} rows")
    print()

    # Check for duplicate subjects (same subject in multiple configs)
    if 'subject' in merged_df.columns:
        unique_subjects = merged_df['subject'].nunique()
        total_rows = len(merged_df)
        print(f"Unique subjects: {unique_subjects}")
        print(f"Total rows: {total_rows}")
        if total_rows > unique_subjects:
            print(f"Note: Some subjects appear in multiple configs/window_types (expected)")
    print()

    # Save to CSV
    try:
        merged_df.to_csv(output_file, index=False)
        print("="*60)
        print(f"✓ Successfully saved merged CSV to: {output_file}")
        print("="*60)
        return True

    except Exception as e:
        print(f"ERROR saving CSV: {e}")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Merge all Matrix Profile Excel results into a single CSV'
    )
    parser.add_argument(
        '-d', '--directory',
        type=str,
        default='confidence/MP',
        help='Base directory containing Excel files (default: confidence/MP)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output CSV file path (default: <directory>/merged_mp_results.csv)'
    )

    args = parser.parse_args()

    # Setup paths
    base_dir = Path(args.directory)

    if args.output:
        output_file = Path(args.output)
    else:
        output_file = base_dir / "merged_mp_results.csv"

    # Merge files
    success = merge_mp_results(base_dir, output_file)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
