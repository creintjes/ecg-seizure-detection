#!/usr/bin/env python3
"""
McNemar Test for comparing seizure detection models.

Performs pairwise McNemar tests between models to determine if there are
statistically significant differences in their detection performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import chi2
from itertools import combinations


def mcnemar_test(n01, n10):
    """
    Perform McNemar test using the chi-square approximation.

    Args:
        n01: Number of cases where model A detected but model B didn't
        n10: Number of cases where model B detected but model A didn't

    Returns:
        Dictionary with test statistic, p-value, and interpretation
    """
    # McNemar test statistic with continuity correction
    if (n01 + n10) == 0:
        return {
            'statistic': 0.0,
            'p_value': 1.0,
            'significant': False,
            'note': 'No discordant pairs'
        }

    # Chi-square statistic with Yates continuity correction
    chi_square = ((abs(n01 - n10) - 1) ** 2) / (n01 + n10)

    # P-value (one degree of freedom)
    p_value = 1 - chi2.cdf(chi_square, df=1)

    # Significance at alpha=0.05
    significant = p_value < 0.05

    return {
        'statistic': chi_square,
        'p_value': p_value,
        'significant': significant,
        'note': 'Significant' if significant else 'Not significant'
    }


def load_models():
    """Load all available model data."""
    models = {}

    files = {
        'TimeVQVAE-AD': 'timevqvae_unified.csv',
        'Matrix Profile': 'matrix_profile_unified.csv',
        'Madrid': 'madrid_unified.csv'
    }

    for model_name, filename in files.items():
        if Path(filename).exists():
            df = pd.read_csv(filename)
            # Convert detected column to boolean
            df['detected'] = df['detected'].map(
                lambda x: str(x).lower() == 'true' if isinstance(x, str) else bool(x)
            )
            models[model_name] = df
            print(f"Loaded {model_name}: {len(df)} records")
        else:
            print(f"Warning: {filename} not found")

    return models


def create_contingency_table(model_a_detections, model_b_detections):
    """
    Create 2x2 contingency table for McNemar test.

    Args:
        model_a_detections: Boolean array of model A detections
        model_b_detections: Boolean array of model B detections

    Returns:
        Dictionary with contingency table values
    """
    # Both detected
    n11 = np.sum(model_a_detections & model_b_detections)

    # Model A detected, Model B missed
    n10 = np.sum(model_a_detections & ~model_b_detections)

    # Model A missed, Model B detected
    n01 = np.sum(~model_a_detections & model_b_detections)

    # Both missed
    n00 = np.sum(~model_a_detections & ~model_b_detections)

    return {
        'both_detected': n11,
        'a_only': n10,
        'b_only': n01,
        'both_missed': n00,
        'total': n11 + n10 + n01 + n00
    }


def perform_mcnemar_tests(models):
    """
    Perform McNemar tests for all model pairs and configurations.

    Args:
        models: Dictionary of model dataframes

    Returns:
        List of test results
    """
    if len(models) < 2:
        print("Error: Need at least 2 models for comparison")
        return []

    results = []

    # Get all model pairs
    model_pairs = list(combinations(models.keys(), 2))

    print(f"\nPerforming McNemar tests for {len(model_pairs)} model pairs...")

    for model_a_name, model_b_name in model_pairs:
        model_a = models[model_a_name]
        model_b = models[model_b_name]

        # Add seizure_id for matching
        model_a['seizure_id'] = model_a['subject'] + '_' + model_a['run'] + '_sz' + model_a['seizure_idx'].astype(str)
        model_b['seizure_id'] = model_b['subject'] + '_' + model_b['run'] + '_sz' + model_b['seizure_idx'].astype(str)

        # Find common seizures
        common_seizures = set(model_a['seizure_id']) & set(model_b['seizure_id'])

        if not common_seizures:
            print(f"  Warning: No common seizures between {model_a_name} and {model_b_name}")
            continue

        print(f"\n{model_a_name} vs {model_b_name}:")
        print(f"  Common seizures: {len(common_seizures)}")

        # Get unique configs and window types
        configs_a = set(model_a['config'].unique())
        configs_b = set(model_b['config'].unique())
        common_configs = configs_a & configs_b

        window_types_a = set(model_a['window_type'].unique())
        window_types_b = set(model_b['window_type'].unique())
        common_window_types = window_types_a & window_types_b

        # Compare same configs and window types
        for config in common_configs:
            for window_type in common_window_types:
                # Filter to config and window type
                a_subset = model_a[
                    (model_a['config'] == config) &
                    (model_a['window_type'] == window_type) &
                    (model_a['seizure_id'].isin(common_seizures))
                ][['seizure_id', 'detected']].drop_duplicates()

                b_subset = model_b[
                    (model_b['config'] == config) &
                    (model_b['window_type'] == window_type) &
                    (model_b['seizure_id'].isin(common_seizures))
                ][['seizure_id', 'detected']].drop_duplicates()

                if len(a_subset) == 0 or len(b_subset) == 0:
                    continue

                # Merge on seizure_id to ensure alignment
                merged = a_subset.merge(
                    b_subset,
                    on='seizure_id',
                    how='inner',
                    suffixes=('_a', '_b')
                )

                if len(merged) == 0:
                    print(f"    Warning: No common seizures after merge for {config}/{window_type}")
                    continue

                # Get detections from merged dataframe
                a_detected = merged['detected_a'].values
                b_detected = merged['detected_b'].values

                # Create contingency table
                contingency = create_contingency_table(a_detected, b_detected)

                # Perform McNemar test
                test_result = mcnemar_test(contingency['a_only'], contingency['b_only'])

                # Calculate detection rates
                a_rate = np.mean(a_detected)
                b_rate = np.mean(b_detected)

                # Store result
                result = {
                    'model_a': model_a_name,
                    'model_b': model_b_name,
                    'config': config,
                    'window_type': window_type,
                    'n_seizures': contingency['total'],
                    'both_detected': contingency['both_detected'],
                    'a_only': contingency['a_only'],
                    'b_only': contingency['b_only'],
                    'both_missed': contingency['both_missed'],
                    'a_detection_rate': a_rate,
                    'b_detection_rate': b_rate,
                    'chi_square': test_result['statistic'],
                    'p_value': test_result['p_value'],
                    'significant': test_result['significant'],
                    'note': test_result['note']
                }

                results.append(result)

                print(f"  {config}/{window_type}:")
                print(f"    Detection rates: {model_a_name}={a_rate:.4f}, {model_b_name}={b_rate:.4f}")
                print(f"    McNemar p-value: {test_result['p_value']:.4f} ({test_result['note']})")

    return results


def print_contingency_tables(results):
    """Print detailed contingency tables for all comparisons."""
    print("\n" + "="*80)
    print("DETAILED CONTINGENCY TABLES")
    print("="*80)

    for result in results:
        print(f"\n{result['model_a']} vs {result['model_b']} | {result['config']} | {result['window_type']}")
        print("-" * 60)
        print(f"                    {result['model_b']} Detected   {result['model_b']} Missed")
        print(f"{result['model_a']} Detected        {result['both_detected']:>8}            {result['a_only']:>8}")
        print(f"{result['model_a']} Missed          {result['b_only']:>8}            {result['both_missed']:>8}")
        print(f"\nTotal: {result['n_seizures']} | Chi²={result['chi_square']:.4f} | p={result['p_value']:.4f} | {result['note']}")


def save_results(results, output_file='mcnemar_test_results.csv'):
    """Save results to CSV file."""
    if not results:
        print("No results to save")
        return

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")


def print_summary(results):
    """Print summary of all McNemar tests."""
    if not results:
        return

    print("\n" + "="*80)
    print("MCNEMAR TEST SUMMARY")
    print("="*80)

    df = pd.DataFrame(results)

    # Overall summary
    total_tests = len(results)
    significant_tests = df['significant'].sum()

    print(f"\nTotal comparisons: {total_tests}")
    print(f"Significant differences (p < 0.05): {significant_tests}")
    print(f"No significant difference: {total_tests - significant_tests}")

    # Summary by model pair
    print("\n" + "-"*80)
    print("SUMMARY BY MODEL PAIR")
    print("-"*80)

    for (model_a, model_b), group in df.groupby(['model_a', 'model_b']):
        sig_count = group['significant'].sum()
        total_count = len(group)

        print(f"\n{model_a} vs {model_b}:")
        print(f"  Total comparisons: {total_count}")
        print(f"  Significant: {sig_count}")

        # Show which configs/windows are significant
        sig_configs = group[group['significant']]
        if len(sig_configs) > 0:
            print("  Significant in:")
            for _, row in sig_configs.iterrows():
                print(f"    - {row['config']}/{row['window_type']} (p={row['p_value']:.4f})")


def main():
    """Main execution function."""
    print("="*80)
    print("MCNEMAR TEST FOR SEIZURE DETECTION MODELS")
    print("="*80)

    # Load models
    models = load_models()

    if len(models) < 2:
        print("\nError: Need at least 2 models for comparison!")
        return

    # Perform McNemar tests
    results = perform_mcnemar_tests(models)

    if not results:
        print("\nNo comparisons could be performed!")
        return

    # Print summary
    print_summary(results)

    # Print contingency tables
    print_contingency_tables(results)

    # Save results
    save_results(results)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
