#!/usr/bin/env python3
"""
Script to load SWE-smith-trajectories dataset from local cache and save to data folder
"""

import os
import shutil
from datasets import Dataset
import pyarrow as pa
import pyarrow.parquet as pq

def load_local_dataset():
    # Path to the local cached dataset
    cache_path = os.path.expanduser("~/.cache/huggingface/datasets/SWE-bench___swe-smith-trajectories/default/0.0.0/f6b6d7e01f2b2aa2f342ced6306d516412e37981")
    
    # Try to load using datasets library directly from cache path
    print(f"Loading dataset from {cache_path}")
    
    # Use Dataset.from_file for arrow format
    arrow_file = os.path.join(cache_path, "swe-smith-trajectories-train.arrow")
    
    if not os.path.exists(arrow_file):
        raise FileNotFoundError(f"Dataset file not found at {arrow_file}")
    
    try:
        # Try loading with datasets library
        dataset = Dataset.from_file(arrow_file)
    except Exception as e:
        print(f"Failed to load with Dataset.from_file: {e}")
        # Try loading as memory mapped file
        try:
            table = pa.memory_map(arrow_file).read_buffer()
            dataset = Dataset(pa.ipc.open_stream(table).read_all())
        except Exception as e2:
            print(f"Failed with memory map: {e2}")
            # Last resort: try different arrow reading methods
            with open(arrow_file, 'rb') as f:
                table = pa.ipc.open_stream(f).read_all()
                dataset = Dataset(table)
    
    return dataset

def save_dataset(dataset, output_dir="data"):
    import json
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSONL format (each line is a JSON dictionary)
    jsonl_path = os.path.join(output_dir, "swe-smith-trajectories.jsonl")
    
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for example in dataset:
            # Convert each example to a dictionary and write as JSON line
            json.dump(example, f, ensure_ascii=False)
            f.write("\n")
    
    print(f"Dataset saved to {jsonl_path}")
    
    # Also save dataset info
    info_path = os.path.join(output_dir, "dataset_info.txt")
    with open(info_path, "w") as f:
        f.write(f"Dataset: SWE-smith-trajectories\n")
        f.write(f"Number of examples: {len(dataset)}\n")
        f.write(f"Features: {list(dataset.features.keys())}\n")
        f.write(f"Saved as: {jsonl_path} (JSONL format - each line is a JSON dictionary)\n")
    
    print(f"Dataset info saved to {info_path}")
    return jsonl_path

if __name__ == "__main__":
    try:
        # Load the local dataset
        dataset = load_local_dataset()
        print(f"Loaded dataset with {len(dataset)} examples")
        
        # Save to data folder
        output_path = save_dataset(dataset)
        print(f"Successfully copied dataset to data folder: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")