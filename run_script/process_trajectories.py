#!/usr/bin/env python3
"""
Process evaluation results and extract trajectory history for resolved instances.

This script takes inputs to:
1. Load evaluation results JSON file containing resolved_ids
2. Load trajectories from folder containing trajectory files for each instance
3. Save results in HuggingFace dataset format

For each resolved instance ID, it loads the trajectory file and extracts the history.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datasets import Dataset


def load_evaluation_results(eval_file_path):
    """
    Load evaluation results and extract resolved instance IDs.
    
    Args:
        eval_file_path (str): Path to the evaluation results JSON file
        
    Returns:
        list: List of resolved instance IDs
    """
    try:
        with open(eval_file_path, 'r') as f:
            eval_data = json.load(f)
        
        resolved_ids = eval_data.get('resolved_ids', [])
        print(f"Found {len(resolved_ids)} resolved instances in evaluation results")
        return resolved_ids
    
    except FileNotFoundError:
        print(f"Error: Evaluation file not found: {eval_file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in evaluation file: {e}")
        return []
    except Exception as e:
        print(f"Error loading evaluation results: {e}")
        return []


def load_trajectory_history(trajectories_folder, instance_id):
    """
    Load trajectory history for a specific instance ID.
    
    Args:
        trajectories_folder (str): Path to the trajectories folder
        instance_id (str): Instance ID to load trajectory for
        
    Returns:
        list: List of history entries from the trajectory, or empty list if not found
    """
    # Construct the trajectory file path
    trajectory_path = os.path.join(trajectories_folder, instance_id, f"{instance_id}.traj")
    
    try:
        with open(trajectory_path, 'r') as f:
            trajectory_data = json.load(f)
        
        # Extract history from trajectory data
        history = trajectory_data.get('history', [])
        return history
    
    except FileNotFoundError:
        print(f"  - {instance_id}: Trajectory file not found: {trajectory_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"  - {instance_id}: Invalid JSON in trajectory file: {e}")
        return []
    except Exception as e:
        print(f"  - {instance_id}: Error loading trajectory: {e}")
        return []


def extract_agent_config(trajectories_folder):
    """
    Extract agent config from the trajectories folder path.
    
    Args:
        trajectories_folder (str): Path to trajectories folder
        
    Returns:
        str: Extracted agent config
    """
    # Extract config from path like: 
    # trajectories/zhengyanshi@microsoft.com/default__o3__t-0.00__p-1.00__c-2.00___patch_swesmith_astropy__astropy.26d14786_ps
    path_parts = trajectories_folder.strip('/').split('/')
    
    # Find the config part (should contain the pattern with __)
    for part in path_parts:
        if '__' in part and ('default' in part or 'anthropic' in part):
            # Extract the config part before the last ___
            if '___' in part:
                config = part.split('___')[0]
            else:
                config = part
            return config
    
    # Fallback - use the last directory name
    return os.path.basename(trajectories_folder.rstrip('/'))


def sanitize_history(history):
    """
    Flatten 'content' lists produced by the tooling layer.

    Each entry of `history` may look like
        {"role": "...", "content": [{"type": "text", "text": "some text"}], ...}
    This function replaces the list with the first item's "text" field so that
    `"content"` becomes a plain string.
    """
    for entry in history:
        if isinstance(entry, dict):
            content = entry.get("content")
            if isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict) and "text" in first:
                    entry["content"] = first["text"]
    return history


def remove_last_nonassistant(history):
    """
    If the last history entry is not from the assistant, drop it.
    """
    if history and isinstance(history[-1], dict) and history[-1].get("role") != "assistant":
        return history[:-1]
    return history


def process_trajectories(eval_file_path, trajectories_folder, dataset_name):
    """
    Main function to process evaluation results and extract trajectory histories.
    
    Args:
        eval_file_path (str): Path to evaluation results JSON file
        trajectories_folder (str): Path to trajectories folder
        dataset_name (str): Name for the output dataset
        
    Returns:
        Dataset: HuggingFace dataset with trajectory data
    """
    print(f"Processing evaluation results from: {eval_file_path}")
    print(f"Processing trajectories from: {trajectories_folder}")
    print(f"Dataset name: {dataset_name}")
    print("-" * 80)
    
    # Extract agent config from trajectories folder path
    agent_config = extract_agent_config(trajectories_folder)
    print(f"Extracted agent config: {agent_config}")
    
    # Load resolved instance IDs from evaluation results
    resolved_ids = load_evaluation_results(eval_file_path)
    if not resolved_ids:
        print("No resolved instances found or error loading evaluation results")
        return None
    
    # Process each resolved instance
    print(f"\nProcessing trajectories for {len(resolved_ids)} resolved instances:")
    
    # Prepare data for HuggingFace dataset
    dataset_data = {
        'instance_id': [],
        'agent_config': [],
        'messages': []
    }

    for instance_id in resolved_ids:
        history = load_trajectory_history(trajectories_folder, instance_id)
        history = sanitize_history(history)
        history = remove_last_nonassistant(history)        # <- new trimming step
        if history:
            dataset_data['instance_id'].append(instance_id)
            dataset_data['agent_config'].append(agent_config)
            dataset_data['messages'].append(history)
            assert type(history) is list, f"History for {instance_id} is not a list"
    
    print("-" * 80)
    print(f"Successfully processed {len(dataset_data['instance_id'])} trajectory histories")
    
    # Create HuggingFace dataset
    if dataset_data['instance_id']:
        dataset = Dataset.from_dict(dataset_data)
        return dataset
    else:
        print("No valid trajectory data found")
        return None


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Process evaluation results and extract trajectory history for resolved instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python process_trajectories.py \\
    --eval-file default__o3__t-0.00__p-1.00__c-2.00___patch_swesmith_astropy__astropy.26d14786_ps.1r1m_eval.json \\
    --trajectories-folder trajectories/zhengyanshi@microsoft.com/default__o3__t-0.00__p-1.00__c-2.00___patch_swesmith_astropy__astropy.26d14786_ps \\
    --dataset-name my_experiment
        """
    )

    parser.add_argument(
        '--eval-file',
        required=True,
        help='Path to evaluation results JSON file containing resolved_ids'
    )
    
    parser.add_argument(
        '--trajectories-folder',
        required=True,
        help='Path to trajectories folder containing trajectory files for each instance'
    )
    
    parser.add_argument(
        '--dataset-name',
        required=True,
        help='Name for the output dataset (will have "_train_traj" suffix added)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.eval_file):
        print(f"Error: Evaluation file does not exist: {args.eval_file}")
        sys.exit(1)
    
    if not os.path.exists(args.trajectories_folder):
        print(f"Error: Trajectories folder does not exist: {args.trajectories_folder}")
        sys.exit(1)
    
    # Process trajectories
    dataset = process_trajectories(args.eval_file, args.trajectories_folder, args.dataset_name)
    
    if dataset is None:
        print("No dataset created due to errors or no data")
        sys.exit(1)
    
    # Print dataset info
    print("\nDataset Information:")
    print("=" * 80)
    print(f"Number of examples: {len(dataset)}")
    print(f"Columns: {list(dataset.column_names)}")

    # Show sample data
    if len(dataset) > 0:
        print(f"\nSample data:")
        sample = dataset[0]
        print(f"Instance ID: {sample['instance_id']}")
        print(f"Agent Config: {sample['agent_config']}")
        print(f"Message length: {len(sample['messages'])} history entries")
        
        if sample['messages']:
            first_entry = sample['messages'][0]
            if isinstance(first_entry, dict) and 'messages' in first_entry:
                content_preview = str(first_entry['messages'])[:200]
                print(f"First messages preview: {content_preview}...")
    
    # Save dataset
    output_name = f"{args.dataset_name}_train_traj"

    # --- new: ensure everything goes into ./data ---------------------------------
    output_dir = Path("data")           # default output root
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / output_name
    # ------------------------------------------------------------------------------

    try:
        dataset.save_to_disk(str(output_path))
        print(f"\nDataset saved to: {output_path}")

        # Also save as JSON for inspection
        json_output = output_path.with_suffix('.json')
        dataset.to_json(str(json_output))
        print(f"Dataset also saved as JSON: {json_output}")

    except Exception as e:
        print(f"\nError saving dataset: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
