#!/usr/bin/env python3
"""
Extract Best Clustering Strategies from Strategy Comparison File
Analyzes large strategy comparison JSON files and extracts key information about best strategies.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict, Counter


def load_strategy_comparison(filepath: Path) -> Dict[str, Any]:
    """Load strategy comparison JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def extract_best_strategy_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract summary of best strategies across all files."""
    
    individual_results = data.get('individual_results', {})
    
    # Collect best strategy info
    best_strategies = []
    strategy_counts = Counter()
    score_stats = []
    
    files_with_clustering = 0
    files_without_clustering = 0
    
    for filename, file_result in individual_results.items():
        file_info = file_result.get('file_info', {})
        base_metrics = file_result.get('base_metrics', {})
        best_strategy = file_result.get('best_strategy')
        
        if best_strategy is None:
            files_without_clustering += 1
            continue
        
        files_with_clustering += 1
        
        # Extract key info about best strategy
        strategy_name = best_strategy.get('name', 'unknown')
        strategy_counts[strategy_name] += 1
        
        metrics = best_strategy.get('metrics', {})
        improvements = best_strategy.get('improvements', {})
        
        strategy_info = {
            'file': filename,
            'subject_id': file_info.get('subject_id'),
            'run_id': file_info.get('run_id'),
            'duration_hours': file_info.get('total_duration_hours', 0),
            'strategy_name': strategy_name,
            'time_threshold_seconds': best_strategy.get('time_threshold_seconds'),
            'score': improvements.get('score', 0),
            'base_anomalies': base_metrics.get('total_anomalies', 0),
            'base_false_positives': base_metrics.get('false_positives', 0),
            'base_sensitivity': base_metrics.get('sensitivity'),
            'clustered_anomalies': metrics.get('total_anomalies', 0),
            'clustered_false_positives': metrics.get('false_positives', 0),
            'clustered_sensitivity': metrics.get('sensitivity'),
            'anomaly_reduction': improvements.get('anomaly_reduction', 0),
            'fp_reduction': improvements.get('fp_reduction', 0),
            'sensitivity_change': improvements.get('sensitivity_change', 0),
            'num_clusters': best_strategy.get('num_clusters', 0)
        }
        
        best_strategies.append(strategy_info)
        score_stats.append(improvements.get('score', 0))
    
    # Calculate summary statistics
    if score_stats:
        avg_score = sum(score_stats) / len(score_stats)
        max_score = max(score_stats)
        min_score = min(score_stats)
    else:
        avg_score = max_score = min_score = 0
    
    # Calculate overall improvements
    total_base_anomalies = sum(s['base_anomalies'] for s in best_strategies)
    total_clustered_anomalies = sum(s['clustered_anomalies'] for s in best_strategies)
    total_base_fp = sum(s['base_false_positives'] for s in best_strategies)
    total_clustered_fp = sum(s['clustered_false_positives'] for s in best_strategies)
    
    overall_anomaly_reduction = ((total_base_anomalies - total_clustered_anomalies) / 
                               total_base_anomalies if total_base_anomalies > 0 else 0)
    overall_fp_reduction = ((total_base_fp - total_clustered_fp) / 
                          total_base_fp if total_base_fp > 0 else 0)
    
    return {
        'summary_stats': {
            'total_files': len(individual_results),
            'files_with_clustering': files_with_clustering,
            'files_without_clustering': files_without_clustering,
            'avg_score': round(avg_score, 4),
            'max_score': round(max_score, 4),
            'min_score': round(min_score, 4),
            'overall_anomaly_reduction': round(overall_anomaly_reduction, 4),
            'overall_fp_reduction': round(overall_fp_reduction, 4)
        },
        'strategy_frequency': dict(strategy_counts.most_common()),
        'best_strategies_per_file': best_strategies
    }


def extract_strategy_performance_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze performance of different time thresholds across all files."""
    
    individual_results = data.get('individual_results', {})
    
    # Collect performance data for each strategy
    strategy_performance = defaultdict(list)
    
    for filename, file_result in individual_results.items():
        strategy_results = file_result.get('strategy_results', {})
        
        for strategy_name, strategy_data in strategy_results.items():
            improvements = strategy_data.get('improvements', {})
            metrics = strategy_data.get('metrics', {})
            
            performance_info = {
                'file': filename,
                'score': improvements.get('score', 0),
                'anomaly_reduction': improvements.get('anomaly_reduction', 0),
                'fp_reduction': improvements.get('fp_reduction', 0),
                'sensitivity_change': improvements.get('sensitivity_change', 0),
                'num_clusters': strategy_data.get('num_clusters', 0),
                'clustered_anomalies': metrics.get('total_anomalies', 0),
                'clustered_false_positives': metrics.get('false_positives', 0)
            }
            
            strategy_performance[strategy_name].append(performance_info)
    
    # Calculate statistics for each strategy
    strategy_stats = {}
    for strategy_name, performances in strategy_performance.items():
        scores = [p['score'] for p in performances]
        anomaly_reductions = [p['anomaly_reduction'] for p in performances]
        fp_reductions = [p['fp_reduction'] for p in performances]
        sensitivity_changes = [p['sensitivity_change'] for p in performances]
        
        if scores:
            strategy_stats[strategy_name] = {
                'files_count': len(performances),
                'avg_score': round(sum(scores) / len(scores), 4),
                'max_score': round(max(scores), 4),
                'min_score': round(min(scores), 4),
                'avg_anomaly_reduction': round(sum(anomaly_reductions) / len(anomaly_reductions), 4),
                'avg_fp_reduction': round(sum(fp_reductions) / len(fp_reductions), 4),
                'avg_sensitivity_change': round(sum(sensitivity_changes) / len(sensitivity_changes), 4),
                'time_threshold_seconds': performances[0].get('file', '').replace('time_', '').replace('s', '') if 'time_' in strategy_name else None
            }
    
    # Sort by average score
    sorted_strategies = sorted(strategy_stats.items(), key=lambda x: x[1]['avg_score'], reverse=True)
    
    return {
        'strategy_count': len(strategy_stats),
        'strategies_by_performance': dict(sorted_strategies[:10]),  # Top 10 strategies
        'all_strategy_stats': strategy_stats
    }


def extract_top_performing_files(data: Dict[str, Any], top_n: int = 10) -> Dict[str, Any]:
    """Extract top performing files based on clustering score."""
    
    individual_results = data.get('individual_results', {})
    
    file_scores = []
    
    for filename, file_result in individual_results.items():
        best_strategy = file_result.get('best_strategy')
        if best_strategy is None:
            continue
        
        file_info = file_result.get('file_info', {})
        base_metrics = file_result.get('base_metrics', {})
        improvements = best_strategy.get('improvements', {})
        metrics = best_strategy.get('metrics', {})
        
        file_score_info = {
            'file': filename,
            'subject_id': file_info.get('subject_id'),
            'run_id': file_info.get('run_id'),
            'score': improvements.get('score', 0),
            'strategy': best_strategy.get('name'),
            'base_anomalies': base_metrics.get('total_anomalies', 0),
            'clustered_anomalies': metrics.get('total_anomalies', 0),
            'anomaly_reduction': improvements.get('anomaly_reduction', 0),
            'fp_reduction': improvements.get('fp_reduction', 0),
            'sensitivity_change': improvements.get('sensitivity_change', 0)
        }
        
        file_scores.append(file_score_info)
    
    # Sort by score
    file_scores.sort(key=lambda x: x['score'], reverse=True)
    
    return {
        'top_performers': file_scores[:top_n],
        'bottom_performers': file_scores[-top_n:] if len(file_scores) > top_n else [],
        'all_file_scores': file_scores
    }


def save_extracted_summary(data: Dict[str, Any], output_path: Path):
    """Save extracted summary to JSON file."""
    
    print("Extracting best strategy summary...")
    best_summary = extract_best_strategy_summary(data)
    
    print("Analyzing strategy performance...")
    strategy_analysis = extract_strategy_performance_analysis(data)
    
    print("Identifying top performing files...")
    top_files = extract_top_performing_files(data)
    
    # Combine all analyses
    summary = {
        'extraction_metadata': {
            'source_analysis': data.get('analysis_metadata', {}),
            'extraction_timestamp': data.get('analysis_metadata', {}).get('calculation_timestamp', 'unknown')
        },
        'overall_results': data.get('overall_results', {}),
        'best_strategy_summary': best_summary,
        'strategy_performance_analysis': strategy_analysis,
        'top_performing_files': top_files
    }
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Extracted summary saved to: {output_path}")
    
    return summary


def print_console_summary(summary: Dict[str, Any]):
    """Print key findings to console."""
    
    best_summary = summary['best_strategy_summary']
    strategy_analysis = summary['strategy_performance_analysis']
    top_files = summary['top_performing_files']
    overall = summary.get('overall_results', {})
    
    print(f"\n{'='*70}")
    print("BEST CLUSTERING STRATEGIES SUMMARY")
    print(f"{'='*70}")
    
    # Overall stats
    stats = best_summary['summary_stats']
    print(f"Total files analyzed: {stats['total_files']}")
    print(f"Files with clustering applied: {stats['files_with_clustering']}")
    print(f"Files without clustering: {stats['files_without_clustering']}")
    
    if overall:
        base = overall.get('base_metrics', {})
        best = overall.get('best_metrics', {})
        improvements = overall.get('improvements', {})
        
        print(f"\nOVERALL IMPROVEMENTS:")
        print(f"  Anomaly reduction: {improvements.get('anomaly_reduction', 0):.4f} ({improvements.get('anomaly_reduction', 0)*100:.2f}%)")
        print(f"  FP reduction: {improvements.get('fp_reduction', 0):.4f} ({improvements.get('fp_reduction', 0)*100:.2f}%)")
        print(f"  Sensitivity change: {improvements.get('sensitivity_change', 0):+.4f}")
        
        print(f"\nBEFORE vs AFTER:")
        print(f"  Total anomalies: {base.get('total_anomalies', 0)} → {best.get('total_anomalies', 0)}")
        print(f"  False positives: {base.get('total_false_positives', 0)} → {best.get('total_false_positives', 0)}")
        print(f"  False alarms/hour: {base.get('false_alarms_per_hour', 0):.4f} → {best.get('false_alarms_per_hour', 0):.4f}")
    
    # Strategy frequency
    print(f"\nMOST FREQUENTLY CHOSEN STRATEGIES:")
    strategy_freq = best_summary['strategy_frequency']
    for strategy, count in list(strategy_freq.items())[:5]:
        print(f"  {strategy}: {count} files")
    
    # Top performing strategies
    print(f"\nTOP PERFORMING CLUSTERING STRATEGIES:")
    top_strategies = strategy_analysis['strategies_by_performance']
    for i, (strategy, stats) in enumerate(list(top_strategies.items())[:5], 1):
        print(f"  {i}. {strategy}")
        print(f"     Avg Score: {stats['avg_score']:.4f}, Files: {stats['files_count']}")
        print(f"     Avg Anomaly Reduction: {stats['avg_anomaly_reduction']:.4f}")
        print(f"     Avg FP Reduction: {stats['avg_fp_reduction']:.4f}")
    
    # Top files
    print(f"\nTOP PERFORMING FILES:")
    for i, file_info in enumerate(top_files['top_performers'][:5], 1):
        print(f"  {i}. {file_info['subject_id']}-{file_info['run_id']} (Score: {file_info['score']:.4f})")
        print(f"     Strategy: {file_info['strategy']}")
        print(f"     Anomalies: {file_info['base_anomalies']} → {file_info['clustered_anomalies']}")
    
    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract best clustering strategies from large strategy comparison files"
    )
    parser.add_argument(
        "strategy_comparison_file",
        help="Path to strategy comparison JSON file"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file for extracted summary (default: auto-generated)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top/bottom performing files to include (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Load strategy comparison file
    input_path = Path(args.strategy_comparison_file)
    if not input_path.exists():
        print(f"Error: File {input_path} does not exist")
        return
    
    print(f"Loading strategy comparison file: {input_path}")
    data = load_strategy_comparison(input_path)
    
    if data is None:
        print("Failed to load strategy comparison file")
        return
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"best_strategies_summary_{input_path.stem}.json"
    
    # Extract and save summary
    summary = save_extracted_summary(data, output_path)
    
    # Print console summary
    print_console_summary(summary)


if __name__ == "__main__":
    main()