#!/usr/bin/env python3

import argparse
import json
import os
import glob
import re
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict

def parse_model_name(model_name: str) -> str:
    """Convert model name to slug format."""
    return model_name.replace('/', '--')

def find_result_files(results_path: str, model_slug: str) -> List[str]:
    """Find all result files matching the model slug."""
    pattern = os.path.join(results_path, f"*{model_slug}*.json")
    files = glob.glob(pattern, recursive=False)
    # Filter out analysis files
    filtered_files = [f for f in files if not f.endswith('_analysis.json')]
    return sorted(filtered_files)

def load_result_file(filepath: str) -> Dict:
    """Load and parse a result JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def load_swebench_dataset(dataset_name: str = "princeton-nlp/SWE-bench_Verified", split: str = "test") -> Dict[str, str]:
    """Load SWE-bench dataset and return instance_id to repo mapping."""
    try:
        from swebench.harness.utils import load_swebench_dataset
        dataset = load_swebench_dataset(dataset_name, split)
        
        # Create mapping from instance_id to repo
        instance_to_repo = {}
        for item in dataset:
            instance_to_repo[item['instance_id']] = item['repo']
        
        return instance_to_repo
    except ImportError:
        print("Warning: swebench not available. Using instance_id for repo mapping.")
        return {}

def extract_repo_from_instance_id(instance_id: str) -> str:
    """Extract repo name from instance_id as fallback."""
    if '__' in instance_id:
        return instance_id.split('__')[0]
    return instance_id

def calculate_repo_level_resolved_rates(resolved_ids: List[str], completed_ids: List[str], instance_to_repo: Dict[str, str]) -> Dict:
    """Calculate repository-level resolved rates properly."""
    # Group instances by repo
    repo_to_completed = defaultdict(set)
    repo_to_resolved = defaultdict(set)
    
    # Process completed instances
    for instance_id in completed_ids:
        if instance_to_repo:
            repo = instance_to_repo.get(instance_id)
        else:
            repo = extract_repo_from_instance_id(instance_id)
        
        if repo:
            repo_to_completed[repo].add(instance_id)
    
    # Process resolved instances
    for instance_id in resolved_ids:
        if instance_to_repo:
            repo = instance_to_repo.get(instance_id)
        else:
            repo = extract_repo_from_instance_id(instance_id)
        
        if repo:
            repo_to_resolved[repo].add(instance_id)
    
    # Calculate per-repo resolved rates
    repo_resolved_rates = {}
    all_repos = set(repo_to_completed.keys()) | set(repo_to_resolved.keys())
    
    for repo in all_repos:
        completed_count = len(repo_to_completed[repo])
        resolved_count = len(repo_to_resolved[repo])
        
        if completed_count > 0:
            repo_resolved_rates[repo] = resolved_count / completed_count
        else:
            repo_resolved_rates[repo] = 0.0
    
    # Calculate overall statistics
    if repo_resolved_rates:
        average_repo_resolved_rate = np.mean(list(repo_resolved_rates.values()))
        std_repo_resolved_rate = np.std(list(repo_resolved_rates.values()))
    else:
        average_repo_resolved_rate = 0.0
        std_repo_resolved_rate = 0.0
    
    return {
        'repo_resolved_rates': repo_resolved_rates,
        'average_repo_resolved_rate': average_repo_resolved_rate,
        'std_repo_resolved_rate': std_repo_resolved_rate,
        'total_repos': len(all_repos),
        'repo_names': sorted(list(all_repos))
    }

def analyze_single_run(result_data: Dict, instance_to_repo: Dict[str, str]) -> Dict:
    """Analyze a single run's results."""
    # Basic metrics
    metrics = {
        'total_instances': result_data.get('total_instances', 0),
        'submitted_instances': result_data.get('submitted_instances', 0),
        'completed_instances': result_data.get('completed_instances', 0),
        'resolved_instances': result_data.get('resolved_instances', 0),
        'unresolved_instances': result_data.get('unresolved_instances', 0),
        'empty_patch_instances': result_data.get('empty_patch_instances', 0),
        'error_instances': result_data.get('error_instances', 0),
    }
    
    # Calculate resolved rate
    if metrics['total_instances'] > 0:
        metrics['resolved_rate'] = metrics['resolved_instances'] / metrics['total_instances']
    else:
        metrics['resolved_rate'] = 0.0
    
    # Repository-level analysis
    resolved_ids = result_data.get('resolved_ids', [])
    completed_ids = result_data.get('completed_ids', [])
    
    repo_stats = calculate_repo_level_resolved_rates(resolved_ids, completed_ids, instance_to_repo)
    metrics['repo_level'] = repo_stats
    
    return metrics

def compute_aggregated_stats(all_metrics: List[Dict]) -> Dict:
    """Compute mean and std across multiple runs."""
    if not all_metrics:
        return {}
    
    # Metrics to aggregate
    keys_to_aggregate = [
        'total_instances', 'submitted_instances', 'completed_instances',
        'resolved_instances', 'unresolved_instances', 'empty_patch_instances',
        'error_instances', 'resolved_rate'
    ]
    
    aggregated = {}
    
    # Aggregate basic metrics
    for key in keys_to_aggregate:
        values = [m[key] for m in all_metrics if key in m]
        if values:
            aggregated[f'{key}_mean'] = float(np.mean(values))
            aggregated[f'{key}_std'] = float(np.std(values))
    
    # Aggregate repo-level metrics
    repo_resolved_rates = [m['repo_level']['average_repo_resolved_rate'] for m in all_metrics if 'repo_level' in m]
    if repo_resolved_rates:
        aggregated['repo_resolved_rate_mean'] = float(np.mean(repo_resolved_rates))
        aggregated['repo_resolved_rate_std'] = float(np.std(repo_resolved_rates))
    
    # Collect all unique repos across runs
    all_repo_names = set()
    for metrics in all_metrics:
        if 'repo_level' in metrics and 'repo_names' in metrics['repo_level']:
            all_repo_names.update(metrics['repo_level']['repo_names'])
    
    aggregated['all_repo_names'] = sorted(list(all_repo_names))
    aggregated['total_unique_repos'] = len(all_repo_names)
    
    # Compute per-repo resolved rate statistics across all runs
    repo_rate_collections = defaultdict(list)
    
    # Collect resolved rates for each repo across all runs
    for metrics in all_metrics:
        if 'repo_level' in metrics and 'repo_resolved_rates' in metrics['repo_level']:
            for repo, rate in metrics['repo_level']['repo_resolved_rates'].items():
                repo_rate_collections[repo].append(rate)
    
    # Compute mean and std for each repo
    per_repo_stats = {}
    for repo in sorted(repo_rate_collections.keys()):
        rates = repo_rate_collections[repo]
        per_repo_stats[repo] = {
            'mean': float(np.mean(rates)),
            'std': float(np.std(rates)),
            'num_runs': len(rates)
        }
    
    aggregated['per_repo_resolved_rate_stats'] = per_repo_stats
    
    return aggregated

def main():
    parser = argparse.ArgumentParser(description='Analyze SWE-agent evaluation results')
    parser.add_argument('--model_name', required=True, 
                       help='Model name (e.g., openai/SWE-smith-32B-Agent_qwen32B_bs1x8_lr5e-5_ep3)')
    parser.add_argument('--results_path', default='evaluation_results_1r1m',
                       help='Path to evaluation results directory')
    parser.add_argument('--dataset_name', default='princeton-nlp/SWE-bench_Verified',
                       help='SWE-bench dataset name')
    parser.add_argument('--split', default='test',
                       help='Dataset split to use')
    
    args = parser.parse_args()
    
    # Convert model name to slug
    model_slug = parse_model_name(args.model_name)
    print(f"Analyzing results for model: {args.model_name}")
    print(f"Model slug: {model_slug}")
    
    # Find result files
    result_files = find_result_files(args.results_path, model_slug)
    if not result_files:
        print(f"No result files found for model slug: {model_slug}")
        return
    
    print(f"Found {len(result_files)} result files:")
    for f in result_files:
        print(f"  {f}")
    
    # Load SWE-bench dataset for repo mapping
    print("Loading SWE-bench dataset...")
    instance_to_repo = load_swebench_dataset(args.dataset_name, args.split)
    print(f"Loaded {len(instance_to_repo)} instance-to-repo mappings")
    
    # Analyze each run
    all_metrics = []
    individual_results = {}
    
    for i, filepath in enumerate(result_files, 1):
        print(f"\nAnalyzing run {i}: {os.path.basename(filepath)}")
        
        try:
            result_data = load_result_file(filepath)
            metrics = analyze_single_run(result_data, instance_to_repo)
            all_metrics.append(metrics)
            individual_results[f'run_{i}'] = metrics
            
            # Print run summary
            print(f"  Total instances: {metrics['total_instances']}")
            print(f"  Resolved instances: {metrics['resolved_instances']}")
            print(f"  Instance-level resolved rate: {metrics['resolved_rate']:.3f}")
            print(f"  Repo-level average resolved rate: {metrics['repo_level']['average_repo_resolved_rate']:.3f}")
            print(f"  Total repos: {metrics['repo_level']['total_repos']}")
            
        except Exception as e:
            print(f"Error analyzing {filepath}: {e}")
            continue
    
    if not all_metrics:
        print("No valid results to analyze")
        return
    
    # Compute aggregated statistics
    print(f"\nComputing aggregated statistics across {len(all_metrics)} runs...")
    aggregated_stats = compute_aggregated_stats(all_metrics)
    
    # Print summary
    print(f"\nSUMMARY:")
    print(f"  Average instance-level resolved rate: {aggregated_stats.get('resolved_rate_mean', 0):.3f} ± {aggregated_stats.get('resolved_rate_std', 0):.3f}")
    print(f"  Average repo-level resolved rate: {aggregated_stats.get('repo_resolved_rate_mean', 0):.3f} ± {aggregated_stats.get('repo_resolved_rate_std', 0):.3f}")
    print(f"  Total unique repositories: {aggregated_stats.get('total_unique_repos', 0)}")
    
    # Print detailed repo information
    if 'all_repo_names' in aggregated_stats:
        print(f"\nRepository names ({len(aggregated_stats['all_repo_names'])}):")
        for repo in aggregated_stats['all_repo_names']:
            print(f"  - {repo}")
    
    # Prepare final results
    final_results = {
        'model_name': args.model_name,
        'model_slug': model_slug,
        'num_runs': len(all_metrics),
        'aggregated_stats': aggregated_stats,
        'individual_runs': individual_results,
        'analysis_config': {
            'results_path': args.results_path,
            'dataset_name': args.dataset_name,
            'split': args.split
        }
    }
    
    # Save results
    output_dir = os.path.dirname(result_files[0]) if result_files else args.results_path
    output_file = os.path.join(output_dir, f"{model_slug}_analysis.json")
    
    print(f"\nSaving results to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("Analysis complete!")

if __name__ == '__main__':
    main()
