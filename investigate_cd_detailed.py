#!/usr/bin/env python3

import json
import sys
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

class CDPatternAnalyzer:
    """Analyze CD command patterns in trajectory data to understand consecutive cd issues."""
    
    def __init__(self):
        pass
        
    def _parse_function_call_from_content(self, content: str) -> List[str]:
        """Parse function calls from assistant message content."""
        actions = []
        
        if not content or not isinstance(content, str):
            return []
        
        # Find all function calls using regex
        function_pattern = r'<function=([^>]+)>(.*?)</function>'
        function_matches = re.findall(function_pattern, content, re.DOTALL)
        
        for function_name, function_content in function_matches:
            if function_name == 'bash':
                # For bash commands, extract the command parameter
                parameter_pattern = r'<parameter=([^>]+)>(.*?)</parameter>'
                parameter_matches = re.findall(parameter_pattern, function_content, re.DOTALL)
                
                for param_name, param_value in parameter_matches:
                    if param_name == 'command':
                        actions.append(param_value.strip())
                        break
                else:
                    actions.append('bash_no_command')
        
        return actions
    
    def analyze_cd_patterns(self, file_path: str):
        """Analyze consecutive cd command patterns in trajectory data"""
        
        print(f"\n=== Analyzing CD patterns in: {Path(file_path).name} ===")
        
        try:
            trajectories = []
            with open(file_path, 'r') as f:
                # Try JSON first
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        trajectories = data
                    else:
                        trajectories = [data]
                except json.JSONDecodeError:
                    # Try JSONL format
                    f.seek(0)
                    lines = f.readlines()
                    if lines:
                        for line in lines:
                            if line.strip():
                                trajectories.append(json.loads(line))
                    else:
                        print("Empty file")
                        return None
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
        
        print(f"Found {len(trajectories)} trajectories to analyze")
        
        # Aggregate analysis across all trajectories
        all_bash_commands = []
        cd_sequences = []
        consecutive_cd_pairs = []
        cd_context = []  # Track context around cd commands
        global_command_index = 0
        
        for traj_idx, trajectory in enumerate(trajectories):
            if (traj_idx + 1) % 500 == 0 or traj_idx < 10:
                print(f"Processing trajectory {traj_idx + 1}/{len(trajectories)}")
            
            # Handle different data structures
            messages = trajectory.get('messages', [])
            if not messages and 'history' in trajectory:
                messages = trajectory.get('history', [])
            
            current_cd_sequence = []
            prev_command = None
            traj_bash_commands = []
            
            for message in messages:
                if message.get("role") == "assistant":
                    content = message.get("content", "")
                    
                    # Extract bash commands
                    bash_commands = self._parse_function_call_from_content(content)
                    
                    for bash_cmd in bash_commands:
                        all_bash_commands.append(bash_cmd)
                        traj_bash_commands.append(bash_cmd)
                        global_command_index += 1
                        
                        # Check if it's a cd command
                        if bash_cmd.strip().startswith('cd '):
                            current_cd_sequence.append(bash_cmd.strip())
                            
                            # Track context around cd commands
                            context_start = max(0, len(traj_bash_commands) - 6)
                            context_end = min(len(traj_bash_commands), len(traj_bash_commands) + 1)
                            context = traj_bash_commands[context_start:context_end]
                            
                            cd_context.append({
                                'cd_command': bash_cmd.strip(),
                                'command_index': global_command_index,
                                'trajectory': traj_idx,
                                'context': context,
                                'prev_command': prev_command
                            })
                            
                            # If previous was also cd, record the consecutive pair
                            if prev_command and prev_command.strip().startswith('cd '):
                                consecutive_cd_pairs.append((prev_command.strip(), bash_cmd.strip()))
                        else:
                            # End of cd sequence, record it if length > 1
                            if len(current_cd_sequence) > 1:
                                cd_sequences.append(current_cd_sequence.copy())
                            current_cd_sequence = []
                        
                        prev_command = bash_cmd
            
            # Add final sequence if it exists for this trajectory
            if len(current_cd_sequence) > 1:
                cd_sequences.append(current_cd_sequence)
        
        print(f"Total bash commands found: {len(all_bash_commands)}")
        print(f"Total cd commands: {sum(1 for cmd in all_bash_commands if cmd.strip().startswith('cd '))}")
        print(f"Total consecutive cd pairs: {len(consecutive_cd_pairs)}")
        print(f"CD sequences of length > 1: {len(cd_sequences)}")
        
        # Analyze consecutive cd patterns
        if consecutive_cd_pairs:
            print(f"\n=== CONSECUTIVE CD ANALYSIS ===")
            cd_counter = Counter(consecutive_cd_pairs)
            
            print("Top 20 most common consecutive CD pairs:")
            for i, ((cd1, cd2), count) in enumerate(cd_counter.most_common(20)):
                print(f"  {i+1:2d}. {count:4d}x: {cd1} → {cd2}")
                
                # Show some context for the most common patterns
                if i < 3:
                    print("      Sample contexts:")
                    contexts_shown = 0
                    for ctx in cd_context:
                        if (ctx['prev_command'] and 
                            ctx['prev_command'].strip() == cd1 and 
                            ctx['cd_command'] == cd2 and 
                            contexts_shown < 3):
                            print(f"        Context {contexts_shown + 1}: ...{' → '.join(ctx['context'][-3:])}...")
                            contexts_shown += 1
                    print()
        
        # Analyze cd sequence patterns
        if cd_sequences:
            print(f"\n=== CD SEQUENCE ANALYSIS ===")
            sequence_lengths = Counter(len(seq) for seq in cd_sequences)
            print("CD sequence length distribution:")
            for length, count in sorted(sequence_lengths.items()):
                print(f"  Length {length}: {count} sequences")
            
            # Show some examples of longer sequences
            print("\nExample longer CD sequences:")
            long_sequences = [seq for seq in cd_sequences if len(seq) >= 4][:5]
            for i, seq in enumerate(long_sequences):
                print(f"  {i+1}. {' → '.join(seq)}")
        
        # Analyze unique cd targets
        cd_commands = [cmd for cmd in all_bash_commands if cmd.strip().startswith('cd ')]
        cd_targets = [cmd.strip()[3:].strip() for cmd in cd_commands if len(cmd.strip()) > 3]
        
        print(f"\n=== CD TARGET ANALYSIS ===")
        print(f"Total cd commands: {len(cd_commands)}")
        print(f"Unique cd targets: {len(set(cd_targets))}")
        
        target_counter = Counter(cd_targets)
        print("Top 15 most visited directories:")
        for i, (target, count) in enumerate(target_counter.most_common(15)):
            print(f"  {i+1:2d}. {count:4d}x: cd {target}")
        
        # Check for patterns that might indicate inefficiency
        print(f"\n=== INEFFICIENCY ANALYSIS ===")
        
        # Check for immediate back-and-forth patterns
        back_forth_patterns = []
        for i, (cd1, cd2) in enumerate(consecutive_cd_pairs):
            cd1_target = cd1[3:].strip() if len(cd1) > 3 else ""
            cd2_target = cd2[3:].strip() if len(cd2) > 3 else ""
            
            # Look for patterns like: cd dir1 → cd dir2 → cd dir1
            if i < len(consecutive_cd_pairs) - 1:
                cd3 = consecutive_cd_pairs[i + 1][1]
                cd3_target = cd3[3:].strip() if len(cd3) > 3 else ""
                if cd1_target == cd3_target and cd1_target != cd2_target:
                    back_forth_patterns.append((cd1, cd2, cd3))
        
        if back_forth_patterns:
            print(f"Back-and-forth patterns found: {len(back_forth_patterns)}")
            print("Examples:")
            for i, (cd1, cd2, cd3) in enumerate(back_forth_patterns[:5]):
                print(f"  {i+1}. {cd1} → {cd2} → {cd3}")
        
        # Check for redundant cd commands (cd to same directory consecutively)
        redundant_cds = []
        for cd1, cd2 in consecutive_cd_pairs:
            cd1_target = cd1[3:].strip() if len(cd1) > 3 else ""
            cd2_target = cd2[3:].strip() if len(cd2) > 3 else ""
            if cd1_target == cd2_target:
                redundant_cds.append((cd1, cd2))
        
        if redundant_cds:
            print(f"\nRedundant cd commands (same target): {len(redundant_cds)}")
            redundant_counter = Counter(redundant_cds)
            print("Top redundant patterns:")
            for (cd1, cd2), count in redundant_counter.most_common(10):
                print(f"  {count:3d}x: {cd1} → {cd2}")
        
        return {
            'total_bash': len(all_bash_commands),
            'total_cd': len(cd_commands),
            'consecutive_pairs': len(consecutive_cd_pairs),
            'cd_sequences': len(cd_sequences),
            'unique_targets': len(set(cd_targets)),
            'back_forth': len(back_forth_patterns),
            'redundant': len(redundant_cds)
        }


def main():
    analyzer = CDPatternAnalyzer()
    
    # List of files to analyze
    files_to_analyze = [
        {
            "name": "Claude Sonnet-4 (Astropy)",
            "path": "/home/zhengyanshi/project/SWE-agent/data/automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10/swesmith_gen_claude__claude-sonnet-4__t-0.00__p-1.00__c-2.00___patch_swesmith_astropy__astropy.26d14786_ps__ms50_as1/astropy__astropy.26d14786_submit.json"
        },
        {
            "name": "SWE-Smith Trajectories",
            "path": "/home/zhengyanshi/project/SWE-agent/data/swe-smith-trajectories.jsonl"
        }
    ]
    
    all_results = {}
    
    for file_info in files_to_analyze:
        print(f"\n{'='*80}")
        print(f"ANALYZING: {file_info['name']}")
        print(f"{'='*80}")
        
        result = analyzer.analyze_cd_patterns(file_info['path'])
        
        if result is None:
            print(f"Analysis failed for {file_info['name']}.")
            continue
            
        all_results[file_info['name']] = result
        
        print(f"\n=== SUMMARY FOR {file_info['name'].upper()} ===")
        print(f"Total bash commands: {result['total_bash']}")
        print(f"Total cd commands: {result['total_cd']}")
        print(f"Consecutive cd pairs: {result['consecutive_pairs']}")
        print(f"CD sequences (length > 1): {result['cd_sequences']}")
        print(f"Unique cd targets: {result['unique_targets']}")
        print(f"Back-and-forth patterns: {result['back_forth']}")
        print(f"Redundant cd commands: {result['redundant']}")
        
        if result['total_cd'] > 0:
            print(f"\nConsecutive cd ratio: {result['consecutive_pairs'] / result['total_cd']:.2%}")
            print(f"Efficiency ratio: {result['unique_targets'] / result['total_cd']:.2%}")
            print(f"CD command percentage: {result['total_cd'] / result['total_bash']:.2%}")
    
    # Comparative analysis
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print("COMPARATIVE ANALYSIS")
        print(f"{'='*80}")
        
        print(f"\n{'Metric':<30} {'Claude Sonnet-4':<20} {'SWE-Smith':<20} {'Difference'}")
        print("-" * 80)
        
        for metric_name, key in [
            ("Total Bash Commands", 'total_bash'),
            ("Total CD Commands", 'total_cd'), 
            ("Consecutive CD Pairs", 'consecutive_pairs'),
            ("Unique CD Targets", 'unique_targets'),
            ("Back-and-forth Patterns", 'back_forth'),
            ("Redundant Commands", 'redundant')
        ]:
            claude_val = all_results.get("Claude Sonnet-4 (Astropy)", {}).get(key, 0)
            smith_val = all_results.get("SWE-Smith Trajectories", {}).get(key, 0)
            diff = claude_val - smith_val
            print(f"{metric_name:<30} {claude_val:<20} {smith_val:<20} {diff:+}")
        
        print("\n" + "-" * 80)
        print("EFFICIENCY RATIOS:")
        print("-" * 80)
        
        for name, result in all_results.items():
            if result['total_cd'] > 0:
                consecutive_ratio = result['consecutive_pairs'] / result['total_cd']
                efficiency_ratio = result['unique_targets'] / result['total_cd'] 
                cd_percentage = result['total_cd'] / result['total_bash']
                
                print(f"\n{name}:")
                print(f"  Consecutive CD Ratio: {consecutive_ratio:.2%}")
                print(f"  Efficiency Ratio:     {efficiency_ratio:.2%}")
                print(f"  CD Command %:         {cd_percentage:.2%}")


if __name__ == "__main__":
    main()