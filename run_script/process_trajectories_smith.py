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
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
import re


XML_STR_REPLACES = ["old_str", "new_str", "file_text"]

def count_tokens_approximate(text: str) -> int:
    """Approximate token count using character count / 4 heuristic."""
    return len(text) // 4

def truncate_user_message(content: str, max_tokens: int = 2000) -> str:
    """Truncate user message if it exceeds token limit, keeping start and end."""
    if not content:
        return content
    
    token_count = count_tokens_approximate(content)
    if token_count <= max_tokens:
        return content
    
    # Keep roughly 40% from start, 40% from end, with (...) in between
    target_tokens = max_tokens - 20  # Reserve tokens for truncation marker
    start_tokens = int(target_tokens * 0.4)
    end_tokens = int(target_tokens * 0.4)
    
    # Convert back to character positions (approximate)
    start_chars = start_tokens * 4
    end_chars = end_tokens * 4
    
    if start_chars + end_chars >= len(content):
        return content
    
    start_part = content[:start_chars].rstrip()
    end_part = content[-end_chars:].lstrip()
    
    return f"{start_part}\n\n(...)\n\n{end_part}"


# TODO: Fix this, this is hardcoded, so will break if not called from root of a directory
SYSTEM_PROMPT = yaml.safe_load(open("agent/swesmith_infer.yaml", "r"))["agent"][
    "templates"
]["system_template"]


def filter_code_blocks(text):
    """
    Remove text between triple backticks (```) from the input text.
    
    Args:
        text (str): Input text that may contain code blocks
        
    Returns:
        str: Text with code blocks removed
    """
    # Pattern to match content between triple backticks (including the backticks)
    pattern = r'```.*?```'
    # Remove the matched patterns (code blocks) from the text
    filtered_text = re.sub(pattern, '', text, flags=re.DOTALL)
    # Clean up any extra whitespace that might be left
    filtered_text = re.sub(r'\n\s*\n\s*\n', '\n\n', filtered_text)
    return filtered_text.strip()


def transform_traj_xml(traj: dict, max_user_tokens: int = 2000) -> dict:
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
                # Filter out code blocks from the thought content
                filtered_thought = filter_code_blocks(message['thought'])
                content = f"{filtered_thought}\n\n{action}"
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
            
            # Truncate user messages if they're too long
            content = truncate_user_message(content, max_user_tokens)
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


def process_trajectories(eval_file_path, trajectories_folder, max_user_tokens=2000):
    """
    Main function to process evaluation results and extract trajectory histories.
    
    Args:
        eval_file_path (str): Path to evaluation results JSON file
        trajectories_folder (str): Path to trajectories folder
        
    Returns:
        Dataset: HuggingFace dataset with trajectory data
    """
    print(f"Processing evaluation results from: {eval_file_path}")
    print(f"Processing trajectories from: {trajectories_folder}")
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
        traj = transform_traj_xml(json.load(open(trajectory_path, "r")), max_user_tokens)

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


def filter_submit_function(message):
    """
    Filter for messages containing submit function call pattern.
    
    Checks for:
    <function=submit>
    </function>
    
    Note: This filter is special - it only checks the last assistant message in each example.
    """
    content = message['content']
    # Check for submit function pattern (with or without parameters)
    submit_pattern = re.compile(r'<function=submit>.*?</function>', re.DOTALL)
    return bool(submit_pattern.search(content))


def count_tokens_with_template(dataset, model_name, output_dir, repo_name, max_length=32768):
    output_plot = os.path.join(output_dir, f"{model_name.replace('/', '_')}_token_histogram.png")
    
    print(f"Loading tokenizer for model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None
    
    print(f"Processing {len(dataset)} samples...")
    
    token_counts = []
    failed_count = 0
    valid_indices = []  # Track which samples are valid
    
    for i, example in enumerate(dataset):
        try:
            messages = example.get('messages', [])
            tools = example.get('tools', [])
            
            # Apply chat template and tokenize in one step
            model_input = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=False
            )
            
            # Tokenize and count - do this only once
            tokens = tokenizer.encode(model_input)
            token_count = len(tokens)
            token_counts.append(token_count)
            valid_indices.append(i)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(dataset)} samples")
                
        except Exception as e:
            failed_count += 1
            print(f"Failed to process sample {i}: {e}")
            continue
    
    if not token_counts:
        print("No samples were successfully processed!")
        return None
    
    # Create filtered datasets
    valid_dataset = dataset.select(valid_indices)
    
    # Find indices of samples that don't exceed max_length
    truncated_indices = [i for i, count in enumerate(token_counts) if count <= max_length]
    truncated_dataset = valid_dataset.select(truncated_indices)
    
    # Filter for examples ending with submit function
    submit_indices = []
    for i in range(len(valid_dataset)):
        messages = valid_dataset[i]['messages']
        # Find the last assistant message
        last_assistant_msg = None
        for msg in reversed(messages):
            if msg['role'] == 'assistant':
                last_assistant_msg = msg
                break
        
        # Check if it contains submit function
        if last_assistant_msg and filter_submit_function(last_assistant_msg):
            submit_indices.append(i)
    
    submit_dataset = valid_dataset.select(submit_indices)
    
    # Save JSON files
    full_json_path = os.path.join(output_dir, f"{repo_name}_full.json")
    truncated_json_path = os.path.join(output_dir, f"{repo_name}_ml{max_length}.json")
    submit_json_path = os.path.join(output_dir, f"{repo_name}_submit.json")
    
    print(f"\nSaving datasets:")
    print(f"Full dataset ({len(valid_dataset)} samples) -> {full_json_path}")
    print(f"Truncated dataset ({len(truncated_dataset)} samples) -> {truncated_json_path}")
    print(f"Submit-filtered dataset ({len(submit_dataset)} samples) -> {submit_json_path}")
    
    valid_dataset.to_json(full_json_path)
    truncated_dataset.to_json(truncated_json_path)
    submit_dataset.to_json(submit_json_path)
    
    # Statistical analysis
    token_counts = np.array(token_counts)
    min_length = np.min(token_counts)
    max_length_actual = np.max(token_counts)
    avg_length = np.mean(token_counts)
    median_length = np.median(token_counts)
    std_length = np.std(token_counts)
    
    # Count examples exceeding max_length
    exceeding_count = np.sum(token_counts > max_length)
    exceeding_percentage = (exceeding_count / len(token_counts)) * 100
    
    print("\n=== Token Count Statistics ===")
    print(f"Total samples processed: {len(token_counts)}")
    print(f"Failed samples: {failed_count}")
    print(f"Minimum length: {min_length}")
    print(f"Maximum length: {max_length_actual}")
    print(f"Average length: {avg_length:.2f}")
    print(f"Median length: {median_length:.2f}")
    print(f"Standard deviation: {std_length:.2f}")
    print(f"Examples exceeding max_length ({max_length}): {exceeding_count} ({exceeding_percentage:.1f}%)")
    
    # Percentile analysis
    percentiles = [25, 50, 75, 90, 95, 99]
    print("\n=== Percentiles ===")
    for p in percentiles:
        value = np.percentile(token_counts, p)
        print(f"{p}th percentile: {value:.2f}")
    
    # Create histogram
    plt.figure(figsize=(12, 8))
    plt.hist(token_counts, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(min_length, color='red', linestyle='--', label=f'Min: {min_length}')
    plt.axvline(max_length_actual, color='red', linestyle='--', label=f'Max: {max_length_actual}')
    plt.axvline(avg_length, color='green', linestyle='-', linewidth=2, label=f'Mean: {avg_length:.2f}')
    plt.axvline(median_length, color='orange', linestyle='-', linewidth=2, label=f'Median: {median_length:.2f}')
    plt.axvline(max_length, color='purple', linestyle=':', linewidth=2, label=f'Max Length Limit: {max_length}')
    
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.title(f'Token Count Distribution After Chat Template\nModel: {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = f'Samples: {len(token_counts)}\nMean: {avg_length:.1f}\nStd: {std_length:.1f}\nMin: {min_length}\nMax: {max_length_actual}\nExceeding {max_length}: {exceeding_count} ({exceeding_percentage:.1f}%)'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"\nHistogram saved to: {output_plot}")
    
    # Save detailed statistics to JSON
    stats_file = output_plot.replace('.png', '_stats.json')
    detailed_stats = {
        'total_samples': len(token_counts),
        'failed_samples': failed_count,
        'min_length': int(min_length),
        'max_length': int(max_length_actual),
        'avg_length': float(avg_length),
        'median_length': float(median_length),
        'std_length': float(std_length),
        'max_length_limit': max_length,
        'exceeding_max_length': int(exceeding_count),
        'exceeding_percentage': float(exceeding_percentage),
        'percentiles': {f'p{p}': float(np.percentile(token_counts, p)) for p in percentiles},
        'model': model_name,
        'dataset_size': len(dataset),
        'valid_samples': len(valid_dataset),
        'truncated_samples': len(truncated_dataset),
        'submit_samples': len(submit_dataset),
        'full_dataset_path': full_json_path,
        'truncated_dataset_path': truncated_json_path,
        'submit_dataset_path': submit_json_path
    }
    
    with open(stats_file, 'w') as f:
        json.dump(detailed_stats, f, indent=2)
    print(f"Detailed statistics saved to: {stats_file}")
    
    return detailed_stats


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
    
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Model name for tokenizer")
    parser.add_argument("--output_plot", type=str, default=None, help="Output plot filename")
    parser.add_argument("--count_tokens", type=bool, default=True, help="Whether to count tokens after applying chat template")
    parser.add_argument("--max_length", type=int, default=32700, help="Maximum length of tokenized input (default: 32768)")
    parser.add_argument("--max_user_tokens", type=int, default=2000, help="Maximum number of tokens for user messages before truncation (default: 2000)")
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
    dataset = process_trajectories(args.eval_file, args.trajectories_folder, args.max_user_tokens)
    
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
            if isinstance(first_entry, dict) and 'content' in first_entry:
                content_preview = str(first_entry['content'])[:200]
                print(f"First message preview: {content_preview}...")
    
    # Save dataset
    # Create output directory structure: data/folder_path/repo_name
    output_dir = Path("data") / args.folder_path
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / args.repo_name
    
    try:
        # Save as JSON for inspection (original dataset)
        json_output = output_path.parent / (output_path.name + '.json')
        dataset.to_json(str(json_output))
        print(f"Original dataset saved as JSON: {json_output}")

    except Exception as e:
        print(f"\nError saving dataset: {e}")
        sys.exit(1)
    
    # Add token counting functionality
    if args.count_tokens:
        print("\n=== Starting Token Count Analysis ===")
        token_stats = count_tokens_with_template(
            dataset=dataset,
            model_name=args.model,
            output_dir=output_dir,
            repo_name=args.repo_name,
            max_length=args.max_length
        )
        
        if token_stats:
            print(f"\nToken analysis completed successfully!")
            print(f"Average tokens per sample: {token_stats['avg_length']:.2f}")
            print(f"Max tokens: {token_stats['max_length']}")
            print(f"Examples exceeding max_length: {token_stats['exceeding_max_length']} ({token_stats['exceeding_percentage']:.1f}%)")
            print(f"Full dataset saved to: {token_stats['full_dataset_path']}")
            print(f"Truncated dataset saved to: {token_stats['truncated_dataset_path']}")
            print(f"Submit-filtered dataset saved to: {token_stats['submit_dataset_path']}")
        else:
            print("Token analysis failed!")


if __name__ == "__main__":
    main()
