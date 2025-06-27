""" 
python count_tokens_with_template.py --dataset data/automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10_train_traj.json
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
import json
import os


def main():
    parser = argparse.ArgumentParser(description="Count tokens after applying chat template")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name or path")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Model name for tokenizer")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process")
    parser.add_argument("--output_plot", type=str, default=None, help="Output plot filename (default: based on dataset name)")
    
    args = parser.parse_args()
    
    # Generate default output filename if not provided
    if args.output_plot is None:
        # Extract dataset name and sanitize for filename
        dataset_name = os.path.basename(args.dataset)
        if dataset_name.endswith(('.json', '.jsonl')):
            dataset_name = os.path.splitext(dataset_name)[0]
        # Replace problematic characters for filenames
        dataset_name = dataset_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        args.output_plot = f"token_distribution_{dataset_name}.png"
    
    print(f"Loading dataset: {args.dataset} (split: {args.split})")
    
    # Load dataset
    try:
        if args.dataset.endswith('.jsonl') or args.dataset.endswith('.json'):
            dataset = load_dataset("json", data_files=args.dataset, split=args.split)
        else:
            dataset = load_dataset(args.dataset, split=args.split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    print(f"Loading tokenizer for model: {args.model}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
    
    # Limit samples if specified
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    
    print(f"Processing {len(dataset)} samples...")
    
    token_counts = []
    failed_count = 0
    
    for i, example in enumerate(dataset):
        try:
            messages = example.get('messages', [])
            tools = example.get('tools', [])
            
            # Apply chat template
            model_input = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=False
            )
            
            # Tokenize and count
            tokens = tokenizer.encode(model_input)
            token_counts.append(len(tokens))
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(dataset)} samples")
                
        except Exception as e:
            failed_count += 1
            print(f"Failed to process sample {i}: {e}")
            continue
    
    if not token_counts:
        print("No samples were successfully processed!")
        return
    
    # Statistical analysis
    token_counts = np.array(token_counts)
    min_length = np.min(token_counts)
    max_length = np.max(token_counts)
    avg_length = np.mean(token_counts)
    median_length = np.median(token_counts)
    std_length = np.std(token_counts)
    
    print("\n=== Token Count Statistics ===")
    print(f"Total samples processed: {len(token_counts)}")
    print(f"Failed samples: {failed_count}")
    print(f"Minimum length: {min_length}")
    print(f"Maximum length: {max_length}")
    print(f"Average length: {avg_length:.2f}")
    print(f"Median length: {median_length:.2f}")
    print(f"Standard deviation: {std_length:.2f}")
    
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
    plt.axvline(max_length, color='red', linestyle='--', label=f'Max: {max_length}')
    plt.axvline(avg_length, color='green', linestyle='-', linewidth=2, label=f'Mean: {avg_length:.2f}')
    plt.axvline(median_length, color='orange', linestyle='-', linewidth=2, label=f'Median: {median_length:.2f}')
    
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.title(f'Token Count Distribution After Chat Template\nModel: {args.model}\nDataset: {args.dataset}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = f'Samples: {len(token_counts)}\nMean: {avg_length:.1f}\nStd: {std_length:.1f}\nMin: {min_length}\nMax: {max_length}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=300, bbox_inches='tight')
    print(f"\nHistogram saved to: {args.output_plot}")
    
    # Save detailed statistics to JSON
    stats_file = args.output_plot.replace('.png', '_stats.json')
    detailed_stats = {
        'total_samples': len(token_counts),
        'failed_samples': failed_count,
        'min_length': int(min_length),
        'max_length': int(max_length),
        'avg_length': float(avg_length),
        'median_length': float(median_length),
        'std_length': float(std_length),
        'percentiles': {f'p{p}': float(np.percentile(token_counts, p)) for p in percentiles},
        'model': args.model,
        'dataset': args.dataset,
        'split': args.split
    }
    
    with open(stats_file, 'w') as f:
        json.dump(detailed_stats, f, indent=2)
    print(f"Detailed statistics saved to: {stats_file}")

if __name__ == "__main__":
    main()
