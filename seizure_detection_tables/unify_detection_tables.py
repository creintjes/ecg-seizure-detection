#!/usr/bin/env python3
"""
Simple script to unify seizure detection tables into standard format.

Output: One CSV per model with columns: subject, run, seizure_idx, detected, config, window_type
"""

import pandas as pd
from pathlib import Path


def process_timevqvae():
    """Process all TimeVQVAE-AD files and combine into one dataframe."""
    base_dir = Path('seizure_detection_tables_TIME')
    all_data = []

    for window_type in ['window', 'no_window']:
        for config in ['FAR', 'HMS', 'Sensitivity']:
            csv_files = list(base_dir.glob(f'{window_type}/{config}/*.csv'))

            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                # Rename columns to match standard
                df = df.rename(columns={
                    'subject_id': 'subject',
                    'run_id': 'run',
                    'seizure_index': 'seizure_idx'
                })
                # Convert global seizure_idx to per-run index (TimeVQVAE uses global index)
                # Matrix Profile uses per-run index, so we need to match that
                df = df.sort_values(['subject', 'run', 'seizure_idx'])
                df['seizure_idx'] = df.groupby(['subject', 'run']).cumcount()

                # Add config and window_type columns
                df['config'] = config
                df['window_type'] = window_type
                # Keep only required columns
                df = df[['subject', 'run', 'seizure_idx', 'detected', 'config', 'window_type']]
                all_data.append(df)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        # Remove duplicates
        combined = combined.drop_duplicates()
        return combined
    return None


def process_matrix_profile():
    """Process all Matrix Profile files and combine into one dataframe."""
    base_dir = Path('seizure_detection_tables_MP')
    all_data = []

    for window_type in ['window', 'no_window']:
        for config in ['FAR', 'HMS', 'Sensitivity']:
            xlsx_files = list(base_dir.glob(f'{window_type}/{config}/*.xlsx'))

            for xlsx_file in xlsx_files:
                df = pd.read_excel(xlsx_file)
                # Rename columns to match standard
                df = df.rename(columns={
                    'sub': 'subject',
                    'seizure_index': 'seizure_idx'
                })
                # Convert seizure_idx from 1-based to 0-based (Matrix Profile starts at 1)
                df['seizure_idx'] = df['seizure_idx'] - 1
                # Add config and window_type columns
                df['config'] = config
                df['window_type'] = window_type
                # Keep only required columns
                df = df[['subject', 'run', 'seizure_idx', 'detected', 'config', 'window_type']]
                all_data.append(df)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        # Remove duplicates
        combined = combined.drop_duplicates()
        return combined
    return None


def main():
    """Main function."""
    print("Processing TimeVQVAE-AD data...")
    timevqvae_df = process_timevqvae()
    if timevqvae_df is not None:
        output_file = 'timevqvae_unified.csv'
        timevqvae_df.to_csv(output_file, index=False)
        print(f"  Saved {len(timevqvae_df)} records to {output_file}")
        print(f"  Configs: {timevqvae_df['config'].unique().tolist()}")
        print(f"  Window types: {timevqvae_df['window_type'].unique().tolist()}")

    print("\nProcessing Matrix Profile data...")
    mp_df = process_matrix_profile()
    if mp_df is not None:
        output_file = 'matrix_profile_unified.csv'
        mp_df.to_csv(output_file, index=False)
        print(f"  Saved {len(mp_df)} records to {output_file}")
        print(f"  Configs: {mp_df['config'].unique().tolist()}")
        print(f"  Window types: {mp_df['window_type'].unique().tolist()}")

    print("\nDone!")


if __name__ == '__main__':
    main()
