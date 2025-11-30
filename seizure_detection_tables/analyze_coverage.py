#!/usr/bin/env python3
"""
Analyze differences in seizure coverage between TimeVQVAE-AD and Matrix Profile.
"""

import pandas as pd


def analyze_coverage():
    """Compare seizure coverage between both models."""

    # Load data
    print("Loading data...")
    timevqvae = pd.read_csv('timevqvae_unified.csv')
    mp = pd.read_csv('matrix_profile_unified.csv')

    # Create unique seizure identifiers
    timevqvae['seizure_id'] = timevqvae['subject'] + '_' + timevqvae['run'] + '_sz' + timevqvae['seizure_idx'].astype(str)
    mp['seizure_id'] = mp['subject'] + '_' + mp['run'] + '_sz' + mp['seizure_idx'].astype(str)

    # Get unique seizures per model
    timevqvae_seizures = set(timevqvae['seizure_id'].unique())
    mp_seizures = set(mp['seizure_id'].unique())

    # Find differences
    only_timevqvae = timevqvae_seizures - mp_seizures
    only_mp = mp_seizures - timevqvae_seizures
    common = timevqvae_seizures & mp_seizures

    # Print summary
    print("\n" + "="*80)
    print("SEIZURE COVERAGE ANALYSIS")
    print("="*80)
    print(f"\nTimeVQVAE-AD total seizures: {len(timevqvae_seizures)}")
    print(f"Matrix Profile total seizures: {len(mp_seizures)}")
    print(f"Common seizures: {len(common)}")
    print(f"\nOnly in TimeVQVAE-AD: {len(only_timevqvae)}")
    print(f"Only in Matrix Profile: {len(only_mp)}")

    # Analyze missing seizures in TimeVQVAE
    if only_mp:
        print("\n" + "="*80)
        print(f"SEIZURES MISSING IN TIMEVQVAE-AD ({len(only_mp)} seizures)")
        print("="*80)

        # Get details for missing seizures
        missing_df = mp[mp['seizure_id'].isin(only_mp)][['subject', 'run', 'seizure_idx']].drop_duplicates()
        missing_df = missing_df.sort_values(['subject', 'run', 'seizure_idx'])

        # Group by subject
        for subject in missing_df['subject'].unique():
            subject_missing = missing_df[missing_df['subject'] == subject]
            runs = subject_missing['run'].unique()
            print(f"\n{subject}: {len(subject_missing)} seizures missing")
            for run in sorted(runs):
                run_seizures = subject_missing[subject_missing['run'] == run]['seizure_idx'].tolist()
                print(f"  {run}: seizures {run_seizures}")

        # Save to file
        missing_df.to_csv('missing_in_timevqvae.csv', index=False)
        print(f"\nSaved details to: missing_in_timevqvae.csv")

    # Analyze missing seizures in Matrix Profile
    if only_timevqvae:
        print("\n" + "="*80)
        print(f"SEIZURES MISSING IN MATRIX PROFILE ({len(only_timevqvae)} seizures)")
        print("="*80)

        # Get details for missing seizures
        missing_df = timevqvae[timevqvae['seizure_id'].isin(only_timevqvae)][['subject', 'run', 'seizure_idx']].drop_duplicates()
        missing_df = missing_df.sort_values(['subject', 'run', 'seizure_idx'])

        # Group by subject
        for subject in missing_df['subject'].unique():
            subject_missing = missing_df[missing_df['subject'] == subject]
            runs = subject_missing['run'].unique()
            print(f"\n{subject}: {len(subject_missing)} seizures missing")
            for run in sorted(runs):
                run_seizures = subject_missing[subject_missing['run'] == run]['seizure_idx'].tolist()
                print(f"  {run}: seizures {run_seizures}")

        # Save to file
        missing_df.to_csv('missing_in_matrix_profile.csv', index=False)
        print(f"\nSaved details to: missing_in_matrix_profile.csv")

    # Analyze by configuration
    print("\n" + "="*80)
    print("SEIZURE COUNTS PER CONFIGURATION")
    print("="*80)

    print("\nTimeVQVAE-AD:")
    timevqvae_by_config = timevqvae.groupby(['config', 'window_type'])['seizure_id'].nunique().reset_index()
    timevqvae_by_config.columns = ['config', 'window_type', 'unique_seizures']
    print(timevqvae_by_config.to_string(index=False))

    print("\nMatrix Profile:")
    mp_by_config = mp.groupby(['config', 'window_type'])['seizure_id'].nunique().reset_index()
    mp_by_config.columns = ['config', 'window_type', 'unique_seizures']
    print(mp_by_config.to_string(index=False))

    # Check if counts vary by configuration
    print("\n" + "="*80)
    print("CONFIGURATION CONSISTENCY CHECK")
    print("="*80)

    timevqvae_counts = timevqvae_by_config['unique_seizures'].unique()
    mp_counts = mp_by_config['unique_seizures'].unique()

    if len(timevqvae_counts) > 1:
        print("\n⚠️  WARNING: TimeVQVAE-AD has different seizure counts per configuration!")
        print("    This should not happen - all configs should analyze the same seizures.")
    else:
        print(f"\n✓ TimeVQVAE-AD: Consistent across all configurations ({timevqvae_counts[0]} seizures)")

    if len(mp_counts) > 1:
        print("\n⚠️  WARNING: Matrix Profile has different seizure counts per configuration!")
        print("    This should not happen - all configs should analyze the same seizures.")
    else:
        print(f"\n✓ Matrix Profile: Consistent across all configurations ({mp_counts[0]} seizures)")

    print("\n" + "="*80)


if __name__ == '__main__':
    analyze_coverage()
