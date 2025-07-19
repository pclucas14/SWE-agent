#!/usr/bin/env python3
"""
Analysis script to understand the overlap distribution of three different models.
Creates visualizations for model overlap and instance ID distributions.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib_venn import venn3
import numpy as np
from typing import Set
import os

def load_instance_ids_from_file(file_path: str) -> Set[str]:
    """Load instance IDs from a JSON file where each line contains one example."""
    instance_ids = set()
    
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return instance_ids
    
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        if 'instance_id' in data:
                            instance_ids.add(data['instance_id'])
                    except json.JSONDecodeError as e:
                        print(f"Warning: JSON decode error in {file_path} at line {line_num}: {e}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return instance_ids

def main():
    # File paths for the three models
    file_paths = {
        'Claude Sonnet 4': 'data/automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10/swesmith_gen_claude__claude-sonnet-4__t-0.00__p-1.00__c-2.00___patch_swesmith_astropy__astropy.26d14786_ps__ms50_as1/astropy__astropy.26d14786_submit.json',
        'GPT-4.1': 'data/automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10/swesmith_gen_claude__gpt-4.1__t-0.00__p-1.00__c-2.00___patch_swesmith_pylint-dev__pylint.1f8c4d9e_ps/pylint-dev__pylint.1f8c4d9e_submit.json',
        'GPT-4o': 'data/automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10/swesmith_gen_claude__gpt-4o__t-0.00__p-1.00__c-2.00___patch_swesmith_astropy__astropy.26d14786_ps_ms50_as1/astropy__astropy.26d14786_submit.json'
    }
    
    # Load instance IDs from each file
    model_ids = {}
    for model_name, file_path in file_paths.items():
        ids = load_instance_ids_from_file(file_path)
        model_ids[model_name] = ids
        print(f"{model_name}: {len(ids)} instance IDs")
    
    # Extract individual sets for easier manipulation
    claude_sonnet_4_ids = model_ids['Claude Sonnet 4']
    gpt_4_1_ids = model_ids['GPT-4.1']
    gpt_4o_ids = model_ids['GPT-4o']
    
    # Calculate overlaps
    all_three = claude_sonnet_4_ids & gpt_4_1_ids & gpt_4o_ids
    claude_gpt41_only = (claude_sonnet_4_ids & gpt_4_1_ids) - gpt_4o_ids
    claude_gpt4o_only = (claude_sonnet_4_ids & gpt_4o_ids) - gpt_4_1_ids
    gpt41_gpt4o_only = (gpt_4_1_ids & gpt_4o_ids) - claude_sonnet_4_ids
    claude_only = claude_sonnet_4_ids - gpt_4_1_ids - gpt_4o_ids
    gpt41_only = gpt_4_1_ids - claude_sonnet_4_ids - gpt_4o_ids
    gpt4o_only = gpt_4o_ids - claude_sonnet_4_ids - gpt_4_1_ids
    
    print(f"\nOverlap Analysis:")
    print(f"All three models: {len(all_three)}")
    print(f"Claude Sonnet 4 & GPT-4.1 only: {len(claude_gpt41_only)}")
    print(f"Claude Sonnet 4 & GPT-4o only: {len(claude_gpt4o_only)}")
    print(f"GPT-4.1 & GPT-4o only: {len(gpt41_gpt4o_only)}")
    print(f"Claude Sonnet 4 only: {len(claude_only)}")
    print(f"GPT-4.1 only: {len(gpt41_only)}")
    print(f"GPT-4o only: {len(gpt4o_only)}")
    
    # Create visualization plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Venn diagram for overlap visualization
    if len(claude_sonnet_4_ids) > 0 or len(gpt_4_1_ids) > 0 or len(gpt_4o_ids) > 0:
        venn = venn3([claude_sonnet_4_ids, gpt_4_1_ids, gpt_4o_ids], 
                     set_labels=('Claude Sonnet 4', 'GPT-4.1', 'GPT-4o'),
                     ax=ax1)
        ax1.set_title('Model Overlap Distribution\n(Instance IDs)', fontsize=14, fontweight='bold')
        
        # Add numbers to venn diagram regions if available
        if venn:
            # Customize colors
            if venn.get_patch_by_id('100'): venn.get_patch_by_id('100').set_color('#ff9999')
            if venn.get_patch_by_id('010'): venn.get_patch_by_id('010').set_color('#99ff99')
            if venn.get_patch_by_id('001'): venn.get_patch_by_id('001').set_color('#9999ff')
            if venn.get_patch_by_id('110'): venn.get_patch_by_id('110').set_color('#ffff99')
            if venn.get_patch_by_id('101'): venn.get_patch_by_id('101').set_color('#ff99ff')
            if venn.get_patch_by_id('011'): venn.get_patch_by_id('011').set_color('#99ffff')
            if venn.get_patch_by_id('111'): venn.get_patch_by_id('111').set_color('#ffcc99')
    else:
        ax1.text(0.5, 0.5, 'No data available for Venn diagram', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Model Overlap Distribution\n(No Data)', fontsize=14, fontweight='bold')
    
    # Plot 2: Bar chart for instance ID distribution
    model_names = list(model_ids.keys())
    model_counts = [len(ids) for ids in model_ids.values()]
    
    bars = ax2.bar(model_names, model_counts, color=['#ff9999', '#99ff99', '#9999ff'], alpha=0.7)
    ax2.set_title('Instance IDs Distribution by Model', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Instance IDs')
    ax2.set_xlabel('Model')
    
    # Add value labels on bars
    for bar, count in zip(bars, model_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Rotate x-axis labels for better readability
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plots
    output_file = 'model_overlap_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as: {output_file}")
    
    # Generate detailed statistics
    total_unique_ids = len(claude_sonnet_4_ids | gpt_4_1_ids | gpt_4o_ids)
    
    print(f"\nDetailed Statistics:")
    print(f"Total unique instance IDs across all models: {total_unique_ids}")
    
    if total_unique_ids > 0:
        print(f"Coverage by individual models:")
        for model_name, ids in model_ids.items():
            coverage = len(ids) / total_unique_ids * 100 if total_unique_ids > 0 else 0
            print(f"  {model_name}: {coverage:.1f}% ({len(ids)}/{total_unique_ids})")
        
        print(f"\nPairwise overlaps:")
        claude_gpt41_overlap = len(claude_sonnet_4_ids & gpt_4_1_ids)
        claude_gpt4o_overlap = len(claude_sonnet_4_ids & gpt_4o_ids)
        gpt41_gpt4o_overlap = len(gpt_4_1_ids & gpt_4o_ids)
        
        print(f"  Claude Sonnet 4 & GPT-4.1: {claude_gpt41_overlap} instances")
        print(f"  Claude Sonnet 4 & GPT-4o: {claude_gpt4o_overlap} instances")
        print(f"  GPT-4.1 & GPT-4o: {gpt41_gpt4o_overlap} instances")
        print(f"  All three models: {len(all_three)} instances")
        
        if len(all_three) > 0:
            print(f"\nConsensus rate (all three models agree): {len(all_three)/total_unique_ids*100:.1f}%")
    
    plt.show()

if __name__ == "__main__":
    main()