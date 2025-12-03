#!/usr/bin/env python3
"""
Analyze differences in seizure coverage between TimeVQVAE-AD, Matrix Profile, and Madrid.
"""

import pandas as pd
from pathlib import Path


def analyze_coverage():
    """Compare seizure coverage between all three models."""

    # Load data
    print("Loading data...")
    models = {}

    if Path('timevqvae_unified.csv').exists():
        models['TimeVQVAE-AD'] = pd.read_csv('timevqvae_unified.csv')
        # Convert detected column to boolean (handle string 'True'/'False')
        models['TimeVQVAE-AD']['detected'] = models['TimeVQVAE-AD']['detected'].map(
            lambda x: str(x).lower() == 'true' if isinstance(x, str) else bool(x)
        )
        print(f"  Loaded TimeVQVAE-AD: {len(models['TimeVQVAE-AD'])} records")
    else:
        print("  Warning: timevqvae_unified.csv not found")

    if Path('matrix_profile_unified.csv').exists():
        models['Matrix Profile'] = pd.read_csv('matrix_profile_unified.csv')
        # Convert detected column to boolean (handle string 'True'/'False')
        models['Matrix Profile']['detected'] = models['Matrix Profile']['detected'].map(
            lambda x: str(x).lower() == 'true' if isinstance(x, str) else bool(x)
        )
        print(f"  Loaded Matrix Profile: {len(models['Matrix Profile'])} records")
    else:
        print("  Warning: matrix_profile_unified.csv not found")

    if Path('madrid_unified.csv').exists():
        models['Madrid'] = pd.read_csv('madrid_unified.csv')
        # Convert detected column to boolean (handle string 'True'/'False')
        models['Madrid']['detected'] = models['Madrid']['detected'].map(
            lambda x: str(x).lower() == 'true' if isinstance(x, str) else bool(x)
        )
        print(f"  Loaded Madrid: {len(models['Madrid'])} records")
    else:
        print("  Warning: madrid_unified.csv not found")

    if len(models) < 2:
        print("\nError: Need at least 2 models to compare!")
        return

    # Create unique seizure identifiers for each model
    for model_name, df in models.items():
        df['seizure_id'] = df['subject'] + '_' + df['run'] + '_sz' + df['seizure_idx'].astype(str)

    # Get unique seizures per model
    model_seizures = {name: set(df['seizure_id'].unique()) for name, df in models.items()}

    # Print overall summary
    print("\n" + "="*80)
    print("OVERALL SEIZURE COVERAGE")
    print("="*80)
    for model_name, seizures in model_seizures.items():
        print(f"{model_name}: {len(seizures)} unique seizures")

    # Find seizures present in all models
    if len(models) >= 2:
        all_models = list(model_seizures.values())
        common_all = set.intersection(*all_models)
        print(f"\nCommon to ALL models: {len(common_all)} seizures")

    # Pairwise comparisons
    print("\n" + "="*80)
    print("PAIRWISE COMPARISONS")
    print("="*80)

    model_names = list(models.keys())
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            model_a = model_names[i]
            model_b = model_names[j]

            seizures_a = model_seizures[model_a]
            seizures_b = model_seizures[model_b]

            common = seizures_a & seizures_b
            only_a = seizures_a - seizures_b
            only_b = seizures_b - seizures_a

            print(f"\n{model_a} vs {model_b}:")
            print(f"  Common: {len(common)}")
            print(f"  Only in {model_a}: {len(only_a)}")
            print(f"  Only in {model_b}: {len(only_b)}")

    # Detailed analysis of missing seizures
    for model_name, df in models.items():
        other_models = [name for name in models.keys() if name != model_name]

        if not other_models:
            continue

        # Find seizures in other models but not in this one
        other_seizures = set.union(*[model_seizures[name] for name in other_models])
        missing = other_seizures - model_seizures[model_name]

        if missing:
            print("\n" + "="*80)
            print(f"SEIZURES MISSING IN {model_name.upper()} ({len(missing)} seizures)")
            print("="*80)

            # Get details from one of the other models
            reference_model = other_models[0]
            reference_df = models[reference_model]

            missing_df = reference_df[reference_df['seizure_id'].isin(missing)][['subject', 'run', 'seizure_idx']].drop_duplicates()
            missing_df = missing_df.sort_values(['subject', 'run', 'seizure_idx'])

            # Group by subject
            for subject in sorted(missing_df['subject'].unique()):
                subject_missing = missing_df[missing_df['subject'] == subject]
                runs = subject_missing['run'].unique()
                print(f"\n{subject}: {len(subject_missing)} seizures missing")
                for run in sorted(runs):
                    run_seizures = sorted(subject_missing[subject_missing['run'] == run]['seizure_idx'].tolist())
                    print(f"  {run}: seizures {run_seizures}")

            # Save to file
            safe_name = model_name.lower().replace(' ', '_').replace('-', '_')
            output_file = f'missing_in_{safe_name}.csv'
            missing_df.to_csv(output_file, index=False)
            print(f"\nSaved details to: {output_file}")

    # Analyze by configuration
    print("\n" + "="*80)
    print("SEIZURE COUNTS PER CONFIGURATION")
    print("="*80)

    for model_name, df in models.items():
        print(f"\n{model_name}:")
        by_config = df.groupby(['config', 'window_type'])['seizure_id'].nunique().reset_index()
        by_config.columns = ['config', 'window_type', 'unique_seizures']
        print(by_config.to_string(index=False))

    # Configuration consistency check
    print("\n" + "="*80)
    print("CONFIGURATION CONSISTENCY CHECK")
    print("="*80)

    for model_name, df in models.items():
        by_config = df.groupby(['config', 'window_type'])['seizure_id'].nunique()
        unique_counts = by_config.unique()

        if len(unique_counts) > 1:
            print(f"\n⚠️  WARNING: {model_name} has different seizure counts per configuration!")
            print(f"    Counts: {unique_counts}")
            print("    This should not happen - all configs should analyze the same seizures.")
        else:
            print(f"\n✓ {model_name}: Consistent across all configurations ({unique_counts[0]} seizures)")

    # Cross-model detection rate analysis
    if len(models) >= 2:
        print("\n" + "="*80)
        print("CROSS-MODEL DETECTION RATE ANALYSIS")
        print("="*80)

        # Use seizures that are common to all models
        if common_all:
            print(f"\nAnalyzing {len(common_all)} seizures that are present in all models...")

            for model_name, df in models.items():
                # Filter to common seizures
                common_df = df[df['seizure_id'].isin(common_all)]

                # Calculate detection rate per config
                detection_rates = common_df.groupby(['config', 'window_type']).agg({
                    'detected': ['sum', 'count', 'mean']
                }).round(4)
                detection_rates.columns = ['detected', 'total', 'detection_rate']
                detection_rates = detection_rates.reset_index()

                print(f"\n{model_name}:")
                print(detection_rates.to_string(index=False))

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    analyze_coverage()
