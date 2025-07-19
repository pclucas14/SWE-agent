#!/usr/bin/env python3
"""
Analyze instance IDs from multiple JSON files and count overlaps with resolved instances.
"""

import json
import argparse
from typing import Set, Dict, Any


def load_instance_ids_from_file(file_path: str) -> Set[str]:
    """Load instance IDs from a JSON file where each line contains one example."""
    instance_ids = set()
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    if 'instance_id' in data:
                        instance_ids.add(data['instance_id'])
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
    except json.JSONDecodeError as e:
        print(f"Warning: JSON decode error in {file_path}: {e}")
    
    return instance_ids


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load a JSON file and return its contents."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Warning: JSON decode error in {file_path}: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description='Analyze instance IDs from multiple JSON files')
    parser.add_argument('--claude-sonnet-4', 
                       default='data/automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10/swesmith_gen_claude__claude-sonnet-4__t-0.00__p-1.00__c-2.00___patch_swesmith_astropy__astropy.26d14786_ps__ms50_as1/astropy__astropy.26d14786_ml32700.json',
                       help='Path to Claude Sonnet 4 JSON file')
    parser.add_argument('--gpt-4-1', 
                       default='data/automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10/swesmith_gen_claude__gpt-4.1__t-0.00__p-1.00__c-2.00___patch_swesmith_astropy__astropy.26d14786_ps_ms50_as1/astropy__astropy.26d14786_ml32700.json',
                       help='Path to GPT-4.1 JSON file')
    parser.add_argument('--gpt-4o', 
                       default='data/automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10/swesmith_gen_claude__gpt-4o__t-0.00__p-1.00__c-2.00___patch_swesmith_astropy__astropy.26d14786_ps_ms50_as1/astropy__astropy.26d14786_ml32700.json',
                       help='Path to GPT-4o JSON file')
    parser.add_argument('--submit-file', 
                       default='data/automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10/swesmith_gen_claude__claude-sonnet-4_gpt4.1_gpt-4o__t-0.00__p-1.00__c-2.00___patch_swesmith_astropy__astropy.26d14786_ps/astropy__astropy.26d14786_submit.json',
                       help='Path to submit JSON file')
    parser.add_argument('--results-file', 
                       default='trajectories/zhengyanshi@microsoft.com/swesmith_infer__openai--tt_SWE-agent-LM-32B_cl32768_lr1e-4_ep3_astropy__astropy.26d14786_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o__t-0.00__p-1.00__c-0.00___patch_swesmith_filtered_swesmith_task__ms75_as1_sanity_check/results.json',
                       help='Path to results JSON file')
    
    args = parser.parse_args()
    
    # Step 1: Collect instance IDs from the three files
    print("Step 1: Collecting instance IDs from three files...")
    claude_sonnet_4_ids = load_instance_ids_from_file(args.claude_sonnet_4)
    gpt_4_1_ids = load_instance_ids_from_file(args.gpt_4_1)
    gpt_4o_ids = load_instance_ids_from_file(args.gpt_4o)
    
    print(f"Claude Sonnet 4 IDs: {len(claude_sonnet_4_ids)}")
    print(f"GPT-4.1 IDs: {len(gpt_4_1_ids)}")
    print(f"GPT-4o IDs: {len(gpt_4o_ids)}")
    
    # Combine all IDs
    all_ids = claude_sonnet_4_ids | gpt_4_1_ids | gpt_4o_ids
    print(f"Total unique IDs from all three files: {len(all_ids)}")
    
    # Step 2: Load submit file and find intersection
    print("\nStep 2: Building intersection with submit file...")
    submit_ids = load_instance_ids_from_file(args.submit_file)
    
    print(f"Submit file IDs: {len(submit_ids)}")
    
    # Find intersection
    intersection_ids = all_ids & submit_ids
    print(f"Intersection IDs: {len(intersection_ids)}")
    
    # Step 3: Load results file and count resolved_ids overlap
    print("\nStep 3: Loading results file and counting resolved_ids overlap...")
    results_data = load_json_file(args.results_file)
    
    resolved_ids = set()
    if 'resolved_ids' in results_data:
        resolved_ids = set(results_data['resolved_ids'])
    
    print(f"Resolved IDs: {len(resolved_ids)}")
    
    # Count overlap between resolved_ids and intersection_ids
    overlap_count = len(resolved_ids & intersection_ids)
    print(f"Overlap between resolved_ids and intersection: {overlap_count}")
    
    # Additional statistics
    print(f"\nSummary:")
    print(f"- Total IDs from three files: {len(all_ids)}")
    print(f"- IDs in submit file: {len(submit_ids)}")
    print(f"- Intersection: {len(intersection_ids)}")
    print(f"- Resolved IDs: {len(resolved_ids)}")
    print(f"- Final overlap count: {overlap_count}")
    
    if len(intersection_ids) > 0:
        print(f"- Success rate: {overlap_count}/{len(intersection_ids)} = {overlap_count/len(intersection_ids):.2%}")


if __name__ == "__main__":
    main()