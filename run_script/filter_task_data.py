#!/usr/bin/env python3
"""
Script to filter task data based on instance IDs from another file.

Usage:
    python filter_task_data.py <task_file> <instance_ids_file> [--output <output_path>]

Example:
    python filter_task_data.py logs/automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10/astropy__astropy.26d14786/task_insts/astropy__astropy.26d14786_ps.json data/automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10/swesmith_gen_claude__claude-sonnet-4_gpt4.1_gpt-4o__t-0.00__p-1.00__c-2.00___patch_swesmith_astropy__astropy.26d14786_ps/astropy__astropy.26d14786_submit.json
"""

import argparse
import json
import os
from pathlib import Path


def collect_instance_ids(instance_ids_file):
    """
    Collect unique instance IDs from the given file.
    Each line should be a JSON object with an 'instance_id' field.
    """
    instance_ids = set()
    
    with open(instance_ids_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    if 'instance_id' in data:
                        instance_ids.add(data['instance_id'])
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line: {line}")
                    continue
    
    return instance_ids


def filter_task_data(task_file, instance_ids):
    """
    Filter task data based on the given instance IDs.
    """
    with open(task_file, 'r') as f:
        tasks = json.load(f)
    
    filtered_tasks = []
    for task in tasks:
        if task.get('instance_id') in instance_ids:
            filtered_tasks.append(task)
    
    return filtered_tasks


def main():
    parser = argparse.ArgumentParser(
        description="Filter task data based on instance IDs from another file"
    )
    parser.add_argument(
        "task_file", 
        help="Path to the task file (JSON array with tasks)"
    )
    parser.add_argument(
        "instance_ids_file", 
        help="Path to the file containing instance IDs (JSONL format)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output path for filtered data (default: parent folder of instance_ids_file with name 'filtered_swesmith_task.json')"
    )
    
    args = parser.parse_args()
    
    # Collect instance IDs
    print(f"Collecting instance IDs from {args.instance_ids_file}...")
    instance_ids = collect_instance_ids(args.instance_ids_file)
    print(f"Found {len(instance_ids)} unique instance IDs")
    
    # Filter task data
    print(f"Filtering task data from {args.task_file}...")
    filtered_tasks = filter_task_data(args.task_file, instance_ids)
    print(f"Filtered to {len(filtered_tasks)} tasks")
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        parent_dir = Path(args.instance_ids_file).parent
        output_path = parent_dir / "filtered_swesmith_task.json"
    
    # Save filtered data
    print(f"Saving filtered data to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(filtered_tasks, f, indent=2)
    
    print(f"Done! Filtered {len(filtered_tasks)} tasks saved to {output_path}")


if __name__ == "__main__":
    main()