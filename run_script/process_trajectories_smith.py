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
import yaml


XML_STR_REPLACES = ["old_str", "new_str", "file_text"]


# TODO: Fix this, this is hardcoded, so will break if not called from root of a directory
SYSTEM_PROMPT = yaml.safe_load(open("agent/swesmith_infer.yaml", "r"))["agent"][
    "templates"
]["system_template"]


def transform_traj_xml(traj: dict) -> dict:
    def tool_call_to_action(tool_calls):
        actions = []
        if tool_calls is None:
            return []
        for tool_call in tool_calls:
            action = [f"<function={tool_call['function']['name']}>"]
            arguments = json.loads(tool_call["function"]["arguments"])
            for k, v in arguments.items():
                a = f"<parameter={k}>{v}</parameter>"
                if k in XML_STR_REPLACES:
                    a = f"<parameter={k}>\n{v}\n</parameter>"
                action.append(a)
            action.append("</function>")
            actions.append("\n".join(action))
        return actions

    new_traj = []
    messages = traj["history"][:-1]
    for message in messages:
        role = message["role"] if message["role"] != "tool" else "user"
        if message["role"] == "assistant":
            if message["content"] == "Exit due to cost limit":
                content = (
                    "Since we have successfully fixed the issue and verified it works, "
                    + "let's submit the changes:\n\n"
                    + "<function=submit>\n</function>"
                )
            else:
                action = "\n".join(tool_call_to_action(message["tool_calls"]))
                content = f"{message['thought']}\n\n{action}"
        elif message["role"] == "system":
            content = SYSTEM_PROMPT
        else:
            if isinstance(message["content"], list):
                assert len(message["content"]) == 1
                content = message["content"][0]["text"]
            elif isinstance(message["content"], str):
                content = message["content"]
            else:
                raise ValueError(f"Message type not recognized: {type(message)}")
        new_traj.append({"role": role, "content": content})
    return {"messages": new_traj}


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


def process_trajectories(eval_file_path, trajectories_folder, folder_path, repo_name):
    """
    Main function to process evaluation results and extract trajectory histories.
    
    Args:
        eval_file_path (str): Path to evaluation results JSON file
        trajectories_folder (str): Path to trajectories folder
        folder_path (str): Folder path for output
        repo_name (str): Repository name for output file
        
    Returns:
        Dataset: HuggingFace dataset with trajectory data
    """
    print(f"Processing evaluation results from: {eval_file_path}")
    print(f"Processing trajectories from: {trajectories_folder}")
    print(f"Output folder: {folder_path}")
    print(f"Repository name: {repo_name}")
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
        trajectory_path = os.path.join(trajectories_folder, instance_id, f"{instance_id}.traj")
        traj = transform_traj_xml(json.load(open(trajectory_path, "r")))

        dataset_data['instance_id'].append(instance_id)
        dataset_data['agent_config'].append(agent_config)
        dataset_data['messages'].append(traj["messages"])
    
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
    --trajectories-folder trajectories/zhengyanshi@microsoft.com/default__o3__t-0.00__p-1.00__c-2.00___patch_swesmith_astropy__astropy.26d14786_ps \\
    --folder-path my_experiment \\
    --repo-name astropy
        """
    )

    parser.add_argument(
        '--eval-file',
        default=None,
        help='Path to evaluation results JSON file containing resolved_ids (DEPRECATED: if not provided, will use trajectories-folder + /results.json)'
    )
    
    parser.add_argument(
        '--trajectories-folder',
        required=True,
        help='Path to trajectories folder containing trajectory files for each instance'
    )
    
    parser.add_argument(
        '--folder-path',
        required=True,
        help='Folder path for the output dataset'
    )
    
    parser.add_argument(
        '--repo-name',
        required=True,
        help='Repository name for the output file'
    )
    
    args = parser.parse_args()
    
    # Auto-derive eval file path if not provided
    if args.eval_file is None:
        args.eval_file = os.path.join(args.trajectories_folder, 'results.json')
        print(f"No eval file specified, using: {args.eval_file}")
    
    # Validate inputs
    if not os.path.exists(args.eval_file):
        print(f"Error: Evaluation file does not exist: {args.eval_file}")
        sys.exit(1)
    
    if not os.path.exists(args.trajectories_folder):
        print(f"Error: Trajectories folder does not exist: {args.trajectories_folder}")
        sys.exit(1)
    
    # Process trajectories
    dataset = process_trajectories(args.eval_file, args.trajectories_folder, args.folder_path, args.repo_name)
    
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
    # Create output directory structure: data/folder_path/repo_name
    output_dir = Path("data") / args.folder_path
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / args.repo_name
    
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
