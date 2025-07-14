#!/usr/bin/env python3
"""
Analyze trajectory files by applying message filters to each example.

This script:
1. Loads all .traj files from subdirectories in the trajectories folder
2. Applies count_messages logic to each example in the dataset
3. Supports flexible filter functions from counter.py
"""

import argparse
import os
import sys
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from datetime import datetime

# Import required functions from existing modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from process_trajectories_smith import transform_traj_xml, extract_agent_config
from counter import FILTER_FUNCTIONS, filter_is_malformed_function_call


def load_all_trajectories(trajectories_folder):
    """
    Load all .traj files from subdirectories in the trajectories folder.
    
    Args:
        trajectories_folder (str): Path to trajectories folder
        
    Returns:
        dict: Dataset-like dictionary with loaded trajectories
    """
    dataset_data = {
        'instance_id': [],
        'agent_config': [],
        'messages': []
    }
    
    # Extract agent config from folder path
    agent_config = extract_agent_config(trajectories_folder)
    
    # Find all .traj files in subdirectories
    traj_files = []
    for root, dirs, files in os.walk(trajectories_folder):
        for file in files:
            if file.endswith('.traj'):
                traj_files.append(os.path.join(root, file))
    
    if not traj_files:
        print(f"No .traj files found in {trajectories_folder}")
        return None
    
    print(f"Found {len(traj_files)} .traj files")
    
    # Load each trajectory file
    for traj_path in traj_files:
        try:
            # Extract instance_id from filename (remove .traj extension)
            instance_id = os.path.basename(traj_path).replace('.traj', '')
            
            # Load and transform trajectory
            with open(traj_path, 'r') as f:
                traj_data = json.load(f)
            
            transformed_traj = transform_traj_xml(traj_data)
            
            dataset_data['instance_id'].append(instance_id)
            dataset_data['agent_config'].append(agent_config)
            dataset_data['messages'].append(transformed_traj['messages'])
            
        except Exception as e:
            print(f"Error loading {traj_path}: {e}")
            continue
    
    if not dataset_data['instance_id']:
        print("No valid trajectory data found")
        return None
    
    # Convert to dataset-like format for compatibility
    class SimpleDataset:
        def __init__(self, data):
            self.data = data
            self._length = len(data['instance_id'])
        
        def __len__(self):
            return self._length
        
        def __getitem__(self, idx):
            return {
                'instance_id': self.data['instance_id'][idx],
                'agent_config': self.data['agent_config'][idx],
                'messages': self.data['messages'][idx]
            }
    
    return SimpleDataset(dataset_data)


def analyze_dataset_messages(dataset, filter_func):
    """
    Analyze messages in a dataset based on a filter function.
    
    Args:
        dataset: Dataset with trajectory data
        filter_func (callable): Function to filter messages
        
    Returns:
        dict: Analysis results
    """
    total_examples = len(dataset)
    total_assistant_messages = 0
    total_filtered_messages = 0
    examples_with_filter = 0
    
    # Per-example statistics
    example_stats = []
    
    console = Console()
    
    # Process each example with progress bar
    for idx in track(range(total_examples), description="Analyzing examples..."):
        example = dataset[idx]
        messages = example['messages']
        
        example_assistant_count = 0
        example_filtered_count = 0
        example_has_filter = False
        
        # Special handling for filter_submit_function - only check last assistant message
        if filter_func.__name__ == 'filter_submit_function':
            # Find the last assistant message
            last_assistant_message = None
            for msg in reversed(messages):
                if msg['role'] == 'assistant':
                    last_assistant_message = msg
                    break
            
            if last_assistant_message:
                example_assistant_count = 1
                if filter_func(last_assistant_message):
                    example_filtered_count = 1
                    example_has_filter = True
        else:
            # Original behavior: check all assistant messages
            for msg in messages:
                if msg['role'] == 'assistant':
                    example_assistant_count += 1
                    if filter_func(msg):
                        example_filtered_count += 1
                        example_has_filter = True
        
        # Update totals
        total_assistant_messages += example_assistant_count
        total_filtered_messages += example_filtered_count
        if example_has_filter:
            examples_with_filter += 1
        
        # Store per-example stats
        example_stats.append({
            'instance_id': example['instance_id'],
            'assistant_messages': example_assistant_count,
            'filtered_messages': example_filtered_count,
            'has_filter': example_has_filter
        })
    
    return {
        'total_examples': total_examples,
        'total_assistant_messages': total_assistant_messages,
        'total_filtered_messages': total_filtered_messages,
        'examples_with_filter': examples_with_filter,
        'example_stats': example_stats
    }


def display_results(results, filter_func, dataset_info):
    """
    Display analysis results in a formatted table.
    
    Args:
        results (dict): Analysis results
        filter_func (callable): The filter function used
        dataset_info (dict): Information about the dataset
    """
    console = Console()
    
    # Create main results table
    table = Table(title="Dataset Analysis Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", no_wrap=True, width=30)
    table.add_column("Count", justify="right", style="green", width=10)
    table.add_column("Percentage", justify="right", style="yellow", width=12)
    
    total_examples = results['total_examples']
    total_assistant_messages = results['total_assistant_messages']
    total_filtered_messages = results['total_filtered_messages']
    examples_with_filter = results['examples_with_filter']
    
    # Add training examples section
    table.add_section()
    table.add_row("Training Examples", "", "", style="bold blue")
    table.add_row("├─ Total examples", f"{total_examples:,}", "100.00%")
    table.add_row("└─ With filter criterion", f"{examples_with_filter:,}", 
                  f"{examples_with_filter / total_examples * 100:.2f}%")
    
    # Add assistant messages section
    table.add_section()
    table.add_row("Assistant Messages", "", "", style="bold blue")
    if filter_func.__name__ == 'filter_submit_function':
        table.add_row("├─ Total last assistant msgs", f"{total_assistant_messages:,}", "100.00%")
    else:
        table.add_row("├─ Total assistant messages", f"{total_assistant_messages:,}", "100.00%")
    table.add_row("└─ Matching filter", f"{total_filtered_messages:,}", 
                  f"{total_filtered_messages / total_assistant_messages * 100:.2f}%" if total_assistant_messages > 0 else "N/A")
    
    console.print()
    console.print(table)
    
    # Add filter info panel
    mode_info = "Last assistant message only" if filter_func.__name__ == 'filter_submit_function' else "All assistant messages"
    filter_info = Panel(
        f"Filter Function: [bold cyan]{filter_func.__name__}[/bold cyan]\n"
        f"Trajectories Folder: [dim]{dataset_info['trajectories_folder']}[/dim]\n"
        f"Agent Config: [dim]{dataset_info['agent_config']}[/dim]\n"
        f"Mode: [yellow]{mode_info}[/yellow]\n"
        f"[dim]Note: Only assistant messages are analyzed[/dim]",
        title="Configuration",
        border_style="blue"
    )
    console.print()
    console.print(filter_info)
    
    # Show examples with highest filter matches
    if results['example_stats']:
        sorted_examples = sorted(results['example_stats'], 
                               key=lambda x: x['filtered_messages'], 
                               reverse=True)[:5]
        
        if any(ex['filtered_messages'] > 0 for ex in sorted_examples):
            console.print("\n[bold yellow]Top 5 Examples with Most Filter Matches:[/bold yellow]")
            for ex in sorted_examples:
                if ex['filtered_messages'] > 0:
                    console.print(f"  {ex['instance_id']}: {ex['filtered_messages']} matches")


def load_cached_results(cache_file):
    """Load cached analysis results from JSON file."""
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_cached_results(cache_file, results):
    """Save analysis results to JSON file."""
    with open(cache_file, 'w') as f:
        json.dump(results, f, indent=2)


def analyze_single_model(trajectories_folder, filter_func, filter_name):
    """Analyze a single model folder and return results."""
    dataset = load_all_trajectories(trajectories_folder)
    
    if dataset is None:
        return None
    
    # Extract agent config
    agent_config = dataset[0]['agent_config'] if len(dataset) > 0 else 'unknown'
    
    # Analyze messages
    results = analyze_dataset_messages(dataset, filter_func)
    
    # Add metadata
    results['trajectories_folder'] = trajectories_folder
    results['agent_config'] = agent_config
    results['filter_name'] = filter_name
    results['timestamp'] = datetime.now().isoformat()
    
    return results


def display_multi_model_summary(all_results, filter_name):
    """Display a summary table for multiple models."""
    console = Console()
    
    # Create summary table
    table = Table(
        title=f"Multi-Model Analysis Summary - Filter: {filter_name}",
        show_header=True,
        header_style="bold magenta"
    )
    
    # Increase width and remove no_wrap to allow full model names to display
    table.add_column("Model", style="cyan", no_wrap=False, width=None)  # Let Rich auto-size
    table.add_column("Examples", justify="right", style="green", width=8)
    table.add_column("With Filter", justify="right", style="yellow", width=10)
    table.add_column("Filter %", justify="right", style="yellow", width=8)
    table.add_column("Asst Msgs", justify="right", style="blue", width=10)
    table.add_column("Filtered", justify="right", style="blue", width=8)
    table.add_column("Filtered %", justify="right", style="blue", width=10)
    
    # Sort models by name
    sorted_models = sorted(all_results.items())
    
    for model_name, result in sorted_models:
        if result is None:
            continue
            
        total_examples = result['total_examples']
        examples_with_filter = result['examples_with_filter']
        total_assistant_messages = result['total_assistant_messages']
        total_filtered_messages = result['total_filtered_messages']
        
        examples_percentage = f"{examples_with_filter / total_examples * 100:.1f}%" if total_examples > 0 else "N/A"
        filtered_percentage = f"{total_filtered_messages / total_assistant_messages * 100:.1f}%" if total_assistant_messages > 0 else "N/A"
        
        table.add_row(
            model_name,
            f"{total_examples:,}",
            f"{examples_with_filter:,}",
            examples_percentage,
            f"{total_assistant_messages:,}",
            f"{total_filtered_messages:,}",
            filtered_percentage
        )
    
    console.print()
    console.print(table)
    
    # Add summary statistics
    total_models = len([r for r in all_results.values() if r is not None])
    total_all_examples = sum(r['total_examples'] for r in all_results.values() if r is not None)
    total_all_filtered = sum(r['examples_with_filter'] for r in all_results.values() if r is not None)
    
    summary_panel = Panel(
        f"[bold]Summary Statistics[/bold]\n"
        f"Total Models Analyzed: [cyan]{total_models}[/cyan]\n"
        f"Total Examples: [green]{total_all_examples:,}[/green]\n"
        f"Total Examples with Filter: [yellow]{total_all_filtered:,}[/yellow]\n"
        f"Overall Percentage: [red]{total_all_filtered / total_all_examples * 100:.2f}%[/red]",
        title="Overall Summary",
        border_style="blue"
    )
    console.print()
    console.print(summary_panel)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze trajectory files by applying message filters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  # Analyze a single model
  python analyze_trajectories.py \\
    --trajectories-folder trajectories/user/experiment_name \\
    --filter-function filter_is_malformed_function_call
    
  # Analyze all models in trajectories/zhengyanshi@microsoft.com
  python analyze_trajectories.py \\
    --filter-function filter_is_malformed_function_call
    
  # Force re-analysis of all models (ignore cache)
  python analyze_trajectories.py \\
    --filter-function filter_is_malformed_function_call \\
    --rerun
        """
    )
    
    parser.add_argument(
        '--trajectories-folder',
        default=None,
        help='Path to trajectories folder containing .traj files in subdirectories. If not specified, analyzes all models in trajectories/zhengyanshi@microsoft.com'
    )
    
    parser.add_argument(
        '--filter-function',
        type=str,
        default='filter_is_malformed_function_call',
        choices=list(FILTER_FUNCTIONS.keys()),
        help='Name of the filter function to use (default: filter_is_malformed_function_call)'
    )
    
    parser.add_argument(
        '--run-all-filters',
        action='store_true',
        help='Run all available filters instead of just one'
    )
    
    parser.add_argument(
        '--rerun',
        action='store_true',
        help='Force re-analysis of all models, ignoring cached results'
    )
    
    args = parser.parse_args()
    
    console = Console()
    
    # Determine if we're analyzing multiple models
    if args.trajectories_folder is None:
        # Auto-load all models
        parent_folder = "trajectories/zhengyanshi@microsoft.com"
        cache_file = os.path.join(parent_folder, "analysis_results.json")
        
        if not os.path.exists(parent_folder):
            console.print(f"[red]Error: Parent folder does not exist: {parent_folder}[/red]")
            sys.exit(1)
        
        # Load cached results
        cached_results = load_cached_results(cache_file) if not args.rerun else {}
        
        # Find all subdirectories
        model_folders = []
        for item in os.listdir(parent_folder):
            item_path = os.path.join(parent_folder, item)
            if os.path.isdir(item_path) and item != "analysis_results.json":
                model_folders.append(item_path)
        
        if not model_folders:
            console.print(f"[red]No model folders found in {parent_folder}[/red]")
            sys.exit(1)
        
        console.print(f"[green]Found {len(model_folders)} model folders to analyze[/green]")
        
        # Analyze each model with each filter
        if args.run_all_filters:
            for filter_name, filter_func in sorted(FILTER_FUNCTIONS.items()):
                console.print(f"\n[bold cyan]{'='*80}[/bold cyan]")
                console.print(f"[bold yellow]Running filter: {filter_name}[/bold yellow]")
                console.print(f"[bold cyan]{'='*80}[/bold cyan]")
                
                all_results = {}
                
                for model_folder in model_folders:
                    model_name = os.path.basename(model_folder)
                    cache_key = f"{model_name}_{filter_name}"
                    
                    # Check cache
                    if cache_key in cached_results and not args.rerun:
                        console.print(f"[dim]Loading cached results for {model_name}...[/dim]")
                        all_results[model_name] = cached_results[cache_key]
                    else:
                        console.print(f"\n[cyan]Analyzing model: {model_name}[/cyan]")
                        result = analyze_single_model(model_folder, filter_func, filter_name)
                        all_results[model_name] = result
                        
                        # Update cache
                        if result is not None:
                            cached_results[cache_key] = result
                            save_cached_results(cache_file, cached_results)
                
                # ── NEW: add cached-only models ────────────────────────────────
                for cache_key, cached_result in cached_results.items():
                    if cache_key.endswith(f"_{filter_name}"):
                        model_name = cache_key[:-(len(filter_name) + 1)]
                        # If the model was not just analyzed (folder missing) add it
                        if model_name not in all_results:
                            all_results[model_name] = cached_result
                # ───────────────────────────────────────────────────────────────

                # Display summary for this filter
                display_multi_model_summary(all_results, filter_name)
        else:
            # Single filter
            filter_func = FILTER_FUNCTIONS[args.filter_function]
            all_results = {}
            
            for model_folder in model_folders:
                model_name = os.path.basename(model_folder)
                cache_key = f"{model_name}_{args.filter_function}"
                
                # Check cache
                if cache_key in cached_results and not args.rerun:
                    console.print(f"[dim]Loading cached results for {model_name}...[/dim]")
                    all_results[model_name] = cached_results[cache_key]
                else:
                    console.print(f"\n[cyan]Analyzing model: {model_name}[/cyan]")
                    result = analyze_single_model(model_folder, filter_func, args.filter_function)
                    all_results[model_name] = result
                    
                    # Update cache
                    if result is not None:
                        cached_results[cache_key] = result
                        save_cached_results(cache_file, cached_results)
            
            # ── NEW: include cached-only models for single filter ─────────────
            for cache_key, cached_result in cached_results.items():
                if cache_key.endswith(f"_{args.filter_function}"):
                    model_name = cache_key[:-(len(args.filter_function) + 1)]
                    if model_name not in all_results:
                        all_results[model_name] = cached_result
            # ────────────────────────────────────────────────────────────────
            
            # Display summary
            display_multi_model_summary(all_results, args.filter_function)
    
    else:
        # Single model analysis (original behavior)
        if not os.path.exists(args.trajectories_folder):
            console.print(f"[red]Error: Trajectories folder does not exist: {args.trajectories_folder}[/red]")
            sys.exit(1)
        
        # Load all trajectory files
        console.print(f"Loading trajectories from: {args.trajectories_folder}")
        dataset = load_all_trajectories(args.trajectories_folder)
        
        if dataset is None:
            console.print("[red]No dataset created due to errors or no data[/red]")
            sys.exit(1)
        
        # Extract agent config from first example
        agent_config = dataset[0]['agent_config'] if len(dataset) > 0 else 'unknown'
        
        dataset_info = {
            'trajectories_folder': args.trajectories_folder,
            'agent_config': agent_config
        }
        
        console.print(f"\n[bold green]Successfully loaded dataset with {len(dataset)} examples[/bold green]")
        
        if args.run_all_filters:
            # Run all filters
            console.print("\n[bold magenta]Running all available filters...[/bold magenta]\n")
            
            for filter_name, filter_func in sorted(FILTER_FUNCTIONS.items()):
                console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
                console.print(f"[bold yellow]Filter: {filter_name}[/bold yellow]")
                console.print(f"[bold cyan]{'='*60}[/bold cyan]")
                
                results = analyze_dataset_messages(dataset, filter_func)
                display_results(results, filter_func, dataset_info)
                console.print()
        else:
            # Run single filter
            filter_func = FILTER_FUNCTIONS[args.filter_function]
            console.print(f"\n[bold magenta]Running filter: {args.filter_function}[/bold magenta]\n")
            
            results = analyze_dataset_messages(dataset, filter_func)
            display_results(results, filter_func, dataset_info)


if __name__ == "__main__":
    main()
