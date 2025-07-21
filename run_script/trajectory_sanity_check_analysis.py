#!/usr/bin/env python3
"""
Enhanced Sanity Check Analysis: Training vs Inference Action Patterns

This script performs comprehensive analysis to verify if the model uses 
actions seen during training when performing inference on training tasks.
Enhanced with proper system message handling, success determination, 
step-by-step analysis, and comprehensive plotting.

Features:
- Action distribution comparison with Jensen-Shannon divergence
- Action coverage analysis (novel vs seen actions)
- Step-by-step exact action matching for first 10 steps
- Trajectory length vs training similarity correlation
- Action diversity statistical analysis (Shannon entropy)
- Missing actions impact on trajectory length
- Enhanced success determination using resolved_ids
- Comprehensive plotting with 12-panel visualization
- System message handling (max steps, timeouts, etc.)

Usage:
    # With default paths (astropy task)
    python enhanced_sanity_check_analysis.py
    
    # With custom paths
    python enhanced_sanity_check_analysis.py --train-data <train_file> --inference-data <inference_folder>
    
    # With output file and custom plot directory
    python enhanced_sanity_check_analysis.py --output results.json --plot-dir my_plots
    
Default paths are configured for the astropy task analysis.
"""

import json
import argparse
import re
from pathlib import Path
from typing import Dict, List
from collections import Counter
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Statistical analysis imports
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import ks_2samp, entropy
    from scipy.stats import mannwhitneyu, spearmanr
    import matplotlib.gridspec as gridspec
    HAS_ANALYSIS_LIBS = True
except ImportError:
    print("Warning: Some analysis libraries not found. Basic analysis only.")
    HAS_ANALYSIS_LIBS = False

@dataclass
class AnalysisResults:
    """Container for all analysis results"""
    distribution_similarity: Dict
    coverage_analysis: Dict
    step_exact_analysis: Dict  # Renamed from position_analysis
    pattern_overlap: Dict
    success_analysis: Dict
    temporal_flow: Dict
    length_similarity_analysis: Dict  # New
    diversity_analysis: Dict  # New
    missing_actions_analysis: Dict  # New
    trajectory_matching_analysis: Dict  # New
    summary_metrics: Dict

class EnhancedSanityCheckAnalyzer:
    """Enhanced analyzer for training vs inference action patterns"""
    
    def __init__(self):
        # Define known tools and commands from both scripts
        self.known_tools = {
            'bash', 'submit', 'str_replace_editor', 'system', 'python'
        }
        
        self.str_replace_commands = {
            'view', 'create', 'str_replace', 'insert', 'undo_edit'
        }
        
        # Success determination
        self.resolved_ids = set()
        
    def _is_system_message(self, message: dict) -> str:
        """Check if message is a system message and return the type (improved from inference script)"""
        if not message:
            return None
        
        content = message.get('content', '').strip().lower()
        thought = message.get('thought', '').strip().lower()
        
        # Check for system message patterns
        if any(pattern in content or pattern in thought for pattern in [
            'reached maximum steps limit of 75',
            'maximum steps limit of 75 reached'
        ]):
            return 'max_step_limit'
        
        if any(pattern in content or pattern in thought for pattern in [
            'exit due to context window',
            'context window exceeded',
            'context limit reached'
        ]):
            return 'context_window_limit'
            
        if any(pattern in content or pattern in thought for pattern in [
            'exit due to total execution time exceeded',
            'total execution time exceeded',
            'execution time limit reached'
        ]):
            return 'execution_time_limit'
            
        if any(pattern in content or pattern in thought for pattern in [
            'exit due to multiple consecutive command timeouts',
            'multiple consecutive command timeouts',
            'command timeout limit reached'
        ]):
            return 'command_timeout_limit'
            
        return None

    def load_success_info(self, inference_folder: str):
        """Load success information from results.json"""
        results_path = Path(inference_folder) / "results.json"
        if results_path.exists():
            try:
                with open(results_path, 'r') as f:
                    results = json.load(f)
                    self.resolved_ids = set(results.get('resolved_ids', []))
                    print(f"Loaded success info: {len(self.resolved_ids)} resolved instances")
            except Exception as e:
                print(f"Warning: Failed to load results.json: {e}")
                self.resolved_ids = set()
        else:
            print("Warning: No results.json found, cannot determine success")
            self.resolved_ids = set()

    def _parse_function_call_from_content(self, content: str) -> List[str]:
        """Parse function calls from assistant message content (training format)
        
        Reverse engineers the tool_call_to_action format:
        <function=str_replace_editor>
        <parameter=command>view</parameter>
        <parameter=path>/testbed</parameter>
        </function>
        
        Into the same format as inference: "str_replace_editor view /testbed"
        """
        actions = []
        
        if not content or not isinstance(content, str):
            return []
        
        # Find all function calls using regex for training format
        function_pattern = r'<function=([^>]+)>(.*?)</function>'
        function_matches = re.findall(function_pattern, content, re.DOTALL)
        
        for function_name, function_content in function_matches:
            # Parse parameters within the function
            parameter_pattern = r'<parameter=([^>]+)>\s*(.*?)\s*</parameter>'
            parameter_matches = re.findall(parameter_pattern, function_content, re.DOTALL)
            
            # Convert to same format as inference actions
            if function_name == 'bash':
                # For bash, the action is just the command parameter
                for param_name, param_value in parameter_matches:
                    if param_name == 'command':
                        actions.append(param_value.strip())
                        break
                else:
                    actions.append(function_name)
                    
            elif function_name == 'str_replace_editor':
                # For str_replace_editor, combine function name with parameters
                # Expected format: "str_replace_editor command path [additional_args]"
                params = {}
                for param_name, param_value in parameter_matches:
                    params[param_name] = param_value.strip()
                
                # Build the action string to match inference format
                action_parts = [function_name]
                
                # Add command parameter
                if 'command' in params:
                    action_parts.append(params['command'])
                
                # Add path parameter  
                if 'path' in params:
                    action_parts.append(params['path'])
                
                # For some commands, we might need additional parameters
                # But for exact matching, we only include command and path
                # as those are the primary identifiers
                
                actions.append(' '.join(action_parts))
                
            elif function_name == 'submit':
                actions.append('submit')
            else:
                # For other functions, just use the function name
                actions.append(function_name)
        
        return actions

    def _normalize_action(self, action: str, message: dict = None) -> str:
        """Normalize action to standard format for comparison (enhanced with system message handling)"""
        if not action or not isinstance(action, str):
            # Check if this is a system message
            system_type = self._is_system_message(message) if message else None
            if system_type:
                return f"system {system_type}"
            return "other empty_action"
        
        action = action.strip()
        
        # Handle bash commands - extract first command
        if not any(action.lower().startswith(tool) for tool in self.known_tools):
            # This is likely a bash command
            command_parts = action.split()
            if command_parts:
                first_cmd = command_parts[0]
                first_cmd = re.split(r'[|;&><]', first_cmd)[0].strip()
                first_cmd = first_cmd.split('/')[-1]  # Get basename
                return f"bash {first_cmd}"
            return "bash empty_bash"
        
        # Handle str_replace_editor commands
        if action.startswith('str_replace_editor '):
            parts = action.split()
            if len(parts) >= 2:
                sub_command = parts[1]
                if sub_command in self.str_replace_commands:
                    return f"str_replace_editor {sub_command}"
                else:
                    return f"str_replace_editor other_{sub_command}"
            return "str_replace_editor no_subcommand"
        
        # Handle submit
        if action.lower().startswith('submit'):
            return "submit"
        
        # Handle other known tools
        for tool in self.known_tools:
            if action.startswith(tool):
                return tool
        
        # Fallback
        first_word = action.split()[0] if action.split() else "empty"
        return f"other {first_word}"

    def load_train_data(self, train_file: str) -> Dict:
        """Load training data from JSON file"""
        print(f"Loading training data from: {train_file}")
        
        try:
            with open(train_file, 'r', encoding='utf-8') as f:
                # Try JSONL format first (multiple JSON objects per line)
                data_list = []
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if line:
                        try:
                            data_list.append(json.loads(line))
                        except json.JSONDecodeError as je:
                            print(f"JSON decode error on line {line_num + 1}: {je}")
                            continue
                
                if len(data_list) == 1:
                    data = data_list[0]
                elif len(data_list) > 1:
                    # Multiple objects, treat as trajectories
                    data = {"trajectories": data_list}
                else:
                    # If JSONL failed, try regular JSON
                    f.seek(0)
                    data = json.load(f)
                        
        except Exception as e:
            print(f"Error loading training data: {e}")
            return {}
        
        # Extract trajectories and actions
        trajectories = []
        all_actions = Counter()
        trajectory_metadata = []  # Store metadata for each trajectory
        
        # Handle different data formats
        if "trajectories" in data:
            raw_trajectories = data["trajectories"]
        elif "history" in data:
            raw_trajectories = [data]
        else:
            raw_trajectories = [data]
        
        for traj_idx, traj_data in enumerate(raw_trajectories):
            messages = traj_data.get("messages", [])
            if not messages:
                messages = traj_data.get("history", [])
            
            trajectory_actions = []
            for message in messages:
                if message.get("role") == "assistant":
                    content = message.get("content", "")
                    # Parse actions from content
                    parsed_actions = self._parse_function_call_from_content(content)
                    for action in parsed_actions:
                        normalized_action = self._normalize_action(action, message)
                        trajectory_actions.append(normalized_action)
                        all_actions[normalized_action] += 1
            
            if trajectory_actions:
                trajectories.append(trajectory_actions)
                trajectory_metadata.append({
                    'id': traj_data.get('instance_id', f'train_{traj_idx}'),
                    'length': len(trajectory_actions),
                    'success': True  # Assume training data is successful
                })
        
        print(f"Loaded {len(trajectories)} training trajectories with {sum(all_actions.values())} total actions")
        print(f"Unique actions in training: {len(all_actions)}")
        
        return {
            'trajectories': trajectories,
            'actions': all_actions,
            'metadata': trajectory_metadata,
            'total_trajectories': len(trajectories),
            'raw_trajectories': raw_trajectories  # Store raw data for prompt comparison
        }

    def load_inference_data(self, inference_folder: str) -> Dict:
        """Load inference data from trajectory folder"""
        print(f"Loading inference data from: {inference_folder}")
        
        # Load success information first
        self.load_success_info(inference_folder)
        
        inference_folder = Path(inference_folder)
        if not inference_folder.exists():
            print(f"Inference folder does not exist: {inference_folder}")
            return {}
        
        # Find all .traj files
        traj_files = list(inference_folder.rglob("*.traj"))
        print(f"Found {len(traj_files)} trajectory files")
        
        trajectories = []
        all_actions = Counter()
        trajectory_metadata = []
        
        for traj_file in traj_files:
            try:
                with open(traj_file, 'r', encoding='utf-8') as f:
                    traj_data = json.load(f)
                
                # Extract instance ID from file path
                instance_id = traj_file.parent.name
                
                messages = traj_data.get("history", [])
                trajectory_actions = []
                
                for message in messages:
                    if message.get("role") == "assistant" and "action" in message:
                        action = message["action"]
                        normalized_action = self._normalize_action(action, message)
                        trajectory_actions.append(normalized_action)
                        all_actions[normalized_action] += 1
                
                if trajectory_actions:
                    trajectories.append(trajectory_actions)
                    trajectory_metadata.append({
                        'id': instance_id,
                        'length': len(trajectory_actions),
                        'success': instance_id in self.resolved_ids,
                        'file_path': traj_file
                    })
                    
            except Exception as e:
                print(f"Warning: Failed to load {traj_file}: {e}")
                continue
        
        print(f"Loaded {len(trajectories)} inference trajectories with {sum(all_actions.values())} total actions")
        print(f"Unique actions in inference: {len(all_actions)}")
        successful_count = sum(1 for meta in trajectory_metadata if meta['success'])
        print(f"Successful trajectories: {successful_count}/{len(trajectories)} ({successful_count/len(trajectories)*100:.1f}%)")
        
        return {
            'trajectories': trajectories,
            'actions': all_actions,
            'metadata': trajectory_metadata,
            'total_trajectories': len(trajectories)
        }

    def compare_action_distributions(self, train_actions: Counter, inference_actions: Counter) -> Dict:
        """Compare action frequency distributions between training and inference"""
        print("\n=== Action Distribution Comparison ===")
        
        # Normalize to probabilities
        train_total = sum(train_actions.values())
        inference_total = sum(inference_actions.values())
        
        train_probs = {action: count/train_total for action, count in train_actions.items()}
        inference_probs = {action: count/inference_total for action, count in inference_actions.items()}
        
        # Get all unique actions
        all_actions = set(train_actions.keys()) | set(inference_actions.keys())
        
        # Create aligned probability vectors
        train_vec = [train_probs.get(action, 0) for action in sorted(all_actions)]
        inference_vec = [inference_probs.get(action, 0) for action in sorted(all_actions)]
        
        results = {
            'total_unique_actions': len(all_actions),
            'train_total_actions': train_total,
            'inference_total_actions': inference_total,
            'train_action_counts': dict(train_actions.most_common(15)),
            'inference_action_counts': dict(inference_actions.most_common(15)),
            'complete_train_actions': dict(train_actions),
            'complete_inference_actions': dict(inference_actions)
        }
        
        if HAS_ANALYSIS_LIBS:
            # Jensen-Shannon distance (bounded 0-1, where 0 means identical)
            js_distance = jensenshannon(train_vec, inference_vec)
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = ks_2samp(train_vec, inference_vec)
            
            results.update({
                'jensen_shannon_distance': float(js_distance),
                'ks_statistic': float(ks_stat),
                'ks_pvalue': float(ks_pvalue),
                'distributions_similar': js_distance < 0.1  # Threshold for similarity
            })
            
            print(f"Jensen-Shannon Distance: {js_distance:.4f} (lower is more similar)")
            print(f"KS Test p-value: {ks_pvalue:.4f}")
            print(f"Distributions similar: {js_distance < 0.1}")
        else:
            # Simple comparison without advanced stats
            common_actions = set(train_actions.keys()) & set(inference_actions.keys())
            similarity_score = len(common_actions) / len(all_actions)
            results.update({
                'basic_similarity_score': similarity_score,
                'distributions_similar': similarity_score > 0.8
            })
            print(f"Basic similarity score: {similarity_score:.4f}")
        
        return results

    def analyze_action_coverage(self, train_actions: Counter, inference_actions: Counter) -> Dict:
        """Analyze what percentage of inference actions were seen during training"""
        print("\n=== Action Coverage Analysis ===")
        
        train_set = set(train_actions.keys())
        inference_set = set(inference_actions.keys())
        
        # Calculate coverage metrics
        shared_actions = inference_set & train_set
        novel_actions = inference_set - train_set
        missing_actions = train_set - inference_set
        
        coverage_rate = len(shared_actions) / len(inference_set) if inference_set else 0
        novel_rate = len(novel_actions) / len(inference_set) if inference_set else 0
        
        # Weight by frequency
        inference_total = sum(inference_actions.values())
        shared_frequency = sum(inference_actions[action] for action in shared_actions)
        novel_frequency = sum(inference_actions[action] for action in novel_actions)
        
        weighted_coverage = shared_frequency / inference_total if inference_total else 0
        weighted_novel_rate = novel_frequency / inference_total if inference_total else 0
        
        results = {
            'coverage_rate': coverage_rate,
            'novel_action_rate': novel_rate,
            'weighted_coverage_rate': weighted_coverage,
            'weighted_novel_rate': weighted_novel_rate,
            'shared_actions': len(shared_actions),
            'novel_actions': len(novel_actions),
            'missing_actions': len(missing_actions),
            'novel_actions_list': list(novel_actions),
            'missing_actions_list': list(missing_actions),
            'coverage_healthy': coverage_rate > 0.9 and weighted_coverage > 0.95
        }
        
        print(f"Action Coverage Rate: {coverage_rate:.3f} ({len(shared_actions)}/{len(inference_set)})")
        print(f"Novel Action Rate: {novel_rate:.3f} ({len(novel_actions)}/{len(inference_set)})")
        print(f"Weighted Coverage Rate: {weighted_coverage:.3f}")
        print(f"Weighted Novel Rate: {weighted_novel_rate:.3f}")
        print(f"Coverage Health: {'HEALTHY' if results['coverage_healthy'] else 'CONCERNING'}")
        
        return results

    def analyze_step_exact_matching(self, train_trajectories: List[List[str]], 
                                  inference_trajectories: List[List[str]]) -> Dict:
        """Analyze exact action matching at each step position"""
        print("\n=== Step-by-Step Exact Action Analysis ===")
        
        # Analyze exact matches at each step
        max_steps = 75  # Analyze up to 75 steps (max trajectory length)
        step_exact_matches = {}
        
        for step in range(max_steps):
            train_actions_at_step = []
            inference_actions_at_step = []
            
            # Collect actions at this step
            for traj in train_trajectories:
                if len(traj) > step:
                    train_actions_at_step.append(traj[step])
            
            for traj in inference_trajectories:
                if len(traj) > step:
                    inference_actions_at_step.append(traj[step])
            
            if train_actions_at_step and inference_actions_at_step:
                # Most common action at this step
                train_most_common = Counter(train_actions_at_step).most_common(1)[0]
                inference_most_common = Counter(inference_actions_at_step).most_common(1)[0]
                
                # Calculate exact match rate (if we had paired trajectories)
                # For now, calculate if the most common actions match
                exact_match = train_most_common[0] == inference_most_common[0]
                # Calculate absolute number of unique actions at this step
                train_unique_actions = len(set(train_actions_at_step))
                inference_unique_actions = len(set(inference_actions_at_step))
                
                # Calculate the ratio for comparison (how concentrated the actions are)
                train_concentration = train_unique_actions / len(train_actions_at_step) if train_actions_at_step else 0
                inference_concentration = inference_unique_actions / len(inference_actions_at_step) if inference_actions_at_step else 0
                
                step_exact_matches[step] = {
                    'train_most_common': train_most_common,
                    'inference_most_common': inference_most_common,
                    'actions_match': exact_match,
                    'train_unique_actions': train_unique_actions,
                    'inference_unique_actions': inference_unique_actions,
                    'train_concentration': train_concentration,
                    'inference_concentration': inference_concentration,
                    'train_count': len(train_actions_at_step),
                    'inference_count': len(inference_actions_at_step)
                }
                
                # Print detailed info for first 10 steps, then summary info for key steps
                if step < 10 or step % 10 == 9 or step == max_steps - 1:
                    print(f"Step {step+1:2d}: Train='{train_most_common[0][:30]:<30}' ({train_most_common[1]:3d}) | "
                          f"Inf='{inference_most_common[0][:30]:<30}' ({inference_most_common[1]:3d}) | "
                          f"Match={exact_match} | Unique: T={train_unique_actions}, I={inference_unique_actions}")
        
        # Calculate overall metrics
        total_matches = sum(1 for data in step_exact_matches.values() if data['actions_match'])
        match_rate = total_matches / len(step_exact_matches) if step_exact_matches else 0
        
        results = {
            'step_exact_matches': step_exact_matches,
            'overall_match_rate': match_rate,
            'analyzed_steps': len(step_exact_matches),
            'step_analysis_healthy': match_rate > 0.5  # At least 50% of early steps should match
        }
        
        print(f"\nOverall Step Match Rate: {match_rate:.3f} ({total_matches}/{len(step_exact_matches)}) across {len(step_exact_matches)} analyzed steps")
        print(f"Step Analysis Health: {'HEALTHY' if results['step_analysis_healthy'] else 'CONCERNING'}")
        
        # Add summary statistics for all steps
        if step_exact_matches:
            all_train_unique = [data['train_unique_actions'] for data in step_exact_matches.values()]
            all_inf_unique = [data['inference_unique_actions'] for data in step_exact_matches.values()]
            print(f"Unique Actions Summary - Training: {min(all_train_unique)}-{max(all_train_unique)} (avg: {sum(all_train_unique)/len(all_train_unique):.1f})")
            print(f"Unique Actions Summary - Inference: {min(all_inf_unique)}-{max(all_inf_unique)} (avg: {sum(all_inf_unique)/len(all_inf_unique):.1f})")
        
        return results

    def analyze_trajectory_length_similarity(self, train_data: Dict, inference_data: Dict) -> Dict:
        """Analyze whether shorter trajectories have higher similarity to training"""
        print("\n=== Trajectory Length vs Training Similarity Analysis ===")
        
        if not HAS_ANALYSIS_LIBS:
            return {'analysis_possible': False}
        
        inference_similarities = []
        inference_lengths = []
        
        # Calculate training action distribution
        train_total = sum(train_data['actions'].values())
        train_dist = {action: count/train_total for action, count in train_data['actions'].items()}
        
        # For each inference trajectory, calculate similarity to training distribution
        for traj, meta in zip(inference_data['trajectories'], inference_data['metadata']):
            if len(traj) > 0:
                # Calculate this trajectory's action distribution
                traj_actions = Counter(traj)
                traj_total = sum(traj_actions.values())
                traj_dist = {action: count/traj_total for action, count in traj_actions.items()}
                
                # Calculate similarity (1 - Jensen-Shannon distance)
                all_actions = set(train_dist.keys()) | set(traj_dist.keys())
                train_vec = [train_dist.get(action, 0) for action in sorted(all_actions)]
                traj_vec = [traj_dist.get(action, 0) for action in sorted(all_actions)]
                
                js_distance = jensenshannon(train_vec, traj_vec)
                similarity = 1 - js_distance
                
                inference_similarities.append(similarity)
                inference_lengths.append(len(traj))
        
        # Calculate correlation
        correlation, p_value = spearmanr(inference_lengths, inference_similarities)
        
        # Bin by length for analysis
        length_bins = [(0, 10), (10, 20), (20, 30), (30, 50), (50, 100)]
        bin_similarities = {f"{low}-{high}": [] for low, high in length_bins}
        
        for length, similarity in zip(inference_lengths, inference_similarities):
            for low, high in length_bins:
                if low <= length < high:
                    bin_similarities[f"{low}-{high}"].append(similarity)
                    break
        
        # Calculate statistics for each bin
        bin_stats = {}
        for bin_name, similarities in bin_similarities.items():
            if similarities:
                bin_stats[bin_name] = {
                    'mean_similarity': np.mean(similarities),
                    'std_similarity': np.std(similarities),
                    'count': len(similarities)
                }
        
        results = {
            'correlation': correlation,
            'p_value': p_value,
            'bin_stats': bin_stats,
            'individual_similarities': inference_similarities,
            'individual_lengths': inference_lengths,
            'analysis_possible': True,
            'length_similarity_healthy': abs(correlation) < 0.3  # Low correlation is healthy
        }
        
        print(f"Length-Similarity Correlation: {correlation:.4f} (p={p_value:.4f})")
        print(f"Length vs Similarity Analysis: {'HEALTHY' if results['length_similarity_healthy'] else 'CONCERNING'}")
        
        for bin_name, stats in bin_stats.items():
            print(f"  Length {bin_name}: Mean similarity = {stats['mean_similarity']:.3f} ± {stats['std_similarity']:.3f} (n={stats['count']})")
        
        return results

    def analyze_action_diversity(self, train_data: Dict, inference_data: Dict) -> Dict:
        """Analyze action diversity statistics"""
        print("\n=== Action Diversity Analysis ===")
        
        # Calculate diversity metrics for each trajectory
        def calculate_diversity_metrics(trajectories):
            metrics = {
                'trajectory_diversities': [],
                'trajectory_entropies': [],
                'unique_actions_per_traj': [],
                'trajectory_lengths': []
            }
            
            for traj in trajectories:
                if len(traj) > 0:
                    # Shannon diversity (normalized entropy)
                    action_counts = Counter(traj)
                    total_actions = len(traj)
                    probs = [count/total_actions for count in action_counts.values()]
                    
                    if HAS_ANALYSIS_LIBS:
                        traj_entropy = entropy(probs, base=2)
                        max_entropy = np.log2(len(action_counts))
                        diversity = traj_entropy / max_entropy if max_entropy > 0 else 0
                    else:
                        diversity = len(set(traj)) / len(traj)
                        traj_entropy = diversity  # Approximate
                    
                    metrics['trajectory_diversities'].append(diversity)
                    metrics['trajectory_entropies'].append(traj_entropy)
                    metrics['unique_actions_per_traj'].append(len(set(traj)))
                    metrics['trajectory_lengths'].append(len(traj))
            
            return metrics
        
        train_metrics = calculate_diversity_metrics(train_data['trajectories'])
        inference_metrics = calculate_diversity_metrics(inference_data['trajectories'])
        
        results = {
            'train_metrics': train_metrics,
            'inference_metrics': inference_metrics
        }
        
        if HAS_ANALYSIS_LIBS:
            results.update({
                'train_diversity_mean': np.mean(train_metrics['trajectory_diversities']),
                'train_diversity_std': np.std(train_metrics['trajectory_diversities']),
                'inference_diversity_mean': np.mean(inference_metrics['trajectory_diversities']),
                'inference_diversity_std': np.std(inference_metrics['trajectory_diversities']),
                'train_entropy_mean': np.mean(train_metrics['trajectory_entropies']),
                'inference_entropy_mean': np.mean(inference_metrics['trajectory_entropies']),
                'train_unique_actions_mean': np.mean(train_metrics['unique_actions_per_traj']),
                'inference_unique_actions_mean': np.mean(inference_metrics['unique_actions_per_traj'])
            })
        else:
            # Basic statistics without numpy
            train_divs = train_metrics['trajectory_diversities']
            inf_divs = inference_metrics['trajectory_diversities']
            train_ents = train_metrics['trajectory_entropies']
            inf_ents = inference_metrics['trajectory_entropies']
            train_uniq = train_metrics['unique_actions_per_traj']
            inf_uniq = inference_metrics['unique_actions_per_traj']
            
            results.update({
                'train_diversity_mean': sum(train_divs) / len(train_divs) if train_divs else 0,
                'train_diversity_std': 0,  # Skip std calculation without numpy
                'inference_diversity_mean': sum(inf_divs) / len(inf_divs) if inf_divs else 0,
                'inference_diversity_std': 0,
                'train_entropy_mean': sum(train_ents) / len(train_ents) if train_ents else 0,
                'inference_entropy_mean': sum(inf_ents) / len(inf_ents) if inf_ents else 0,
                'train_unique_actions_mean': sum(train_uniq) / len(train_uniq) if train_uniq else 0,
                'inference_unique_actions_mean': sum(inf_uniq) / len(inf_uniq) if inf_uniq else 0
            })
        
        if HAS_ANALYSIS_LIBS:
            # Statistical test for diversity difference
            _, p_value = mannwhitneyu(train_metrics['trajectory_diversities'], 
                                         inference_metrics['trajectory_diversities'])
            results['diversity_difference_pvalue'] = p_value
            results['diversity_significantly_different'] = p_value < 0.05
        
        print(f"Training Diversity: {results['train_diversity_mean']:.4f} ± {results['train_diversity_std']:.4f}")
        print(f"Inference Diversity: {results['inference_diversity_mean']:.4f} ± {results['inference_diversity_std']:.4f}")
        print(f"Training Unique Actions per Traj: {results['train_unique_actions_mean']:.2f}")
        print(f"Inference Unique Actions per Traj: {results['inference_unique_actions_mean']:.2f}")
        
        if 'diversity_significantly_different' in results:
            print(f"Diversity Difference Significant: {results['diversity_significantly_different']} (p={results['diversity_difference_pvalue']:.4f})")
        
        results['diversity_analysis_healthy'] = abs(results['train_diversity_mean'] - results['inference_diversity_mean']) < 0.1
        print(f"Diversity Analysis Health: {'HEALTHY' if results['diversity_analysis_healthy'] else 'CONCERNING'}")
        
        return results

    def analyze_trajectory_matching(self, train_data: Dict, inference_data: Dict) -> Dict:
        """Analyze exact trajectory matching by instance ID"""
        print("\n=== Trajectory Matching Analysis ===")
        
        # Create dictionaries for quick lookup by instance ID
        train_by_id = {}
        inference_by_id = {}
        
        # Build training lookup with RAW actions (not normalized)
        for i, meta in enumerate(train_data['metadata']):
            instance_id = meta['id']
            train_by_id[instance_id] = {
                'trajectory_idx': i,
                'metadata': meta,
                'actions': train_data['trajectories'][i],  # These are already normalized - we need raw
                'raw_actions': self._extract_raw_actions_from_training(instance_id, train_data)
            }
        
        # Build inference lookup with RAW actions
        for i, meta in enumerate(inference_data['metadata']):
            instance_id = meta['id']
            inference_by_id[instance_id] = {
                'trajectory_idx': i,
                'metadata': meta,
                'actions': inference_data['trajectories'][i],  # These are already normalized - we need raw  
                'raw_actions': self._extract_raw_actions_from_inference(instance_id, inference_data)
            }
        
        # Find matching instance IDs
        train_ids = set(train_by_id.keys())
        inference_ids = set(inference_by_id.keys())
        matched_ids = train_ids & inference_ids
        
        print(f"Training instances: {len(train_ids)}")
        print(f"Inference instances: {len(inference_ids)}")
        print(f"Matched instances: {len(matched_ids)}")
        
        if not matched_ids:
            return {
                'matching_analysis_possible': False,
                'matched_instances': 0,
                'total_train_instances': len(train_ids),
                'total_inference_instances': len(inference_ids)
            }
        
        # For each matched instance, perform detailed comparison
        system_prompt_matches = 0
        user_prompt_matches = 0
        step_wise_matches = []
        trajectory_exact_matches = 0
        
        detailed_mismatches = []
        
        for instance_id in sorted(matched_ids):
            train_info = train_by_id[instance_id]
            inference_info = inference_by_id[instance_id]
            
            # Load the actual trajectory files to compare prompts
            train_messages = self._load_messages_for_instance(instance_id, train_data, inference_data, 'train')
            inference_messages = self._load_messages_for_instance(instance_id, train_data, inference_data, 'inference')
            
            mismatch_details = {
                'instance_id': instance_id,
                'system_prompt_match': False,
                'user_prompt_match': False,
                'action_matches': [],
                'total_exact_match': False
            }
            
            # Compare system prompts
            train_system = self._extract_system_prompt(train_messages)
            inference_system = self._extract_system_prompt(inference_messages)
            
            if train_system == inference_system:
                system_prompt_matches += 1
                mismatch_details['system_prompt_match'] = True
            else:
                mismatch_details['system_prompt_mismatch'] = {
                    'train_system': train_system[:200] + "..." if train_system and len(train_system) > 200 else train_system,
                    'inference_system': inference_system[:200] + "..." if inference_system and len(inference_system) > 200 else inference_system
                }
            
            # Compare user prompts (first user message)
            train_user = self._extract_user_prompt(train_messages)
            inference_user = self._extract_user_prompt(inference_messages)
            
            if train_user == inference_user:
                user_prompt_matches += 1
                mismatch_details['user_prompt_match'] = True
            else:
                mismatch_details['user_prompt_mismatch'] = {
                    'train_user': train_user[:200] + "..." if train_user and len(train_user) > 200 else train_user,
                    'inference_user': inference_user[:200] + "..." if inference_user and len(inference_user) > 200 else inference_user
                }
            
            # Compare action sequences step by step using RAW actions for exact matching
            train_raw_actions = train_info['raw_actions']
            inference_raw_actions = inference_info['raw_actions']
            
            # Fallback to normalized actions if raw actions not available
            if not train_raw_actions or not inference_raw_actions:
                train_raw_actions = train_info['actions']
                inference_raw_actions = inference_info['actions']
                if len(detailed_mismatches) < 5:
                    print(f"DEBUG: Using fallback normalized actions for {instance_id}")
                    print(f"  Train raw actions: {len(train_raw_actions)} actions")
                    print(f"  Inference raw actions: {len(inference_raw_actions)} actions")
            else:
                if len(detailed_mismatches) < 5:
                    print(f"DEBUG: Using extracted raw actions for {instance_id}")
                    print(f"  Train raw actions: {len(train_raw_actions)} actions") 
                    print(f"  Inference raw actions: {len(inference_raw_actions)} actions")
            
            max_steps = max(len(train_raw_actions), len(inference_raw_actions))
            step_matches = []
            
            for step in range(max_steps):
                train_action = train_raw_actions[step] if step < len(train_raw_actions) else None
                inference_action = inference_raw_actions[step] if step < len(inference_raw_actions) else None
                
                # EXACT string comparison including all parameters
                step_match = train_action == inference_action
                step_matches.append(step_match)
                
                # Log examples for debugging (first 5 instances, first 3 steps)
                if len(detailed_mismatches) < 5 and step < 3:
                    print(f"DEBUG Instance {instance_id} Step {step+1}:")
                    print(f"  Train:     '{train_action}'")
                    print(f"  Inference: '{inference_action}'")
                    print(f"  Match:     {step_match}")
                    print()
                
                if not step_match:
                    mismatch_details['action_matches'].append({
                        'step': step + 1,
                        'train_action': train_action,
                        'inference_action': inference_action
                    })
            
            step_wise_matches.append({
                'instance_id': instance_id,
                'matches': step_matches,
                'match_rate': sum(step_matches) / len(step_matches) if step_matches else 0,
                'train_length': len(train_raw_actions),
                'inference_length': len(inference_raw_actions)
            })
            
            # Check if entire trajectory is exactly the same using raw actions
            if train_raw_actions == inference_raw_actions:
                trajectory_exact_matches += 1
                mismatch_details['total_exact_match'] = True
            
            detailed_mismatches.append(mismatch_details)
        
        # Calculate metrics
        total_matched = len(matched_ids)
        system_prompt_match_rate = system_prompt_matches / total_matched
        user_prompt_match_rate = user_prompt_matches / total_matched
        trajectory_exact_match_rate = trajectory_exact_matches / total_matched
        
        # Calculate average step-wise match rate
        all_step_match_rates = [match['match_rate'] for match in step_wise_matches]
        avg_step_match_rate = sum(all_step_match_rates) / len(all_step_match_rates) if all_step_match_rates else 0
        
        # Calculate step-by-step match rates across all trajectories
        max_trajectory_length = max([match['train_length'] for match in step_wise_matches] + 
                                  [match['inference_length'] for match in step_wise_matches])
        
        step_by_step_rates = []
        for step in range(max_trajectory_length):
            step_matches = 0
            step_total = 0
            for match in step_wise_matches:
                if step < len(match['matches']):
                    if match['matches'][step]:
                        step_matches += 1
                    step_total += 1
            
            if step_total > 0:
                step_by_step_rates.append({
                    'step': step + 1,
                    'match_rate': step_matches / step_total,
                    'total_trajectories': step_total
                })
        
        results = {
            'matching_analysis_possible': True,
            'matched_instances': total_matched,
            'total_train_instances': len(train_ids),
            'total_inference_instances': len(inference_ids),
            'system_prompt_match_rate': system_prompt_match_rate,
            'user_prompt_match_rate': user_prompt_match_rate,
            'trajectory_exact_match_rate': trajectory_exact_match_rate,
            'average_step_match_rate': avg_step_match_rate,
            'step_by_step_rates': step_by_step_rates,
            'step_wise_matches': step_wise_matches,
            'detailed_mismatches': detailed_mismatches[:5],  # Store first 5 for analysis
            'trajectory_matching_healthy': (system_prompt_match_rate > 0.95 and 
                                          user_prompt_match_rate > 0.95 and 
                                          avg_step_match_rate > 0.8)
        }
        
        print(f"System Prompt Match Rate: {system_prompt_match_rate:.3f} ({system_prompt_matches}/{total_matched})")
        print(f"User Prompt Match Rate: {user_prompt_match_rate:.3f} ({user_prompt_matches}/{total_matched})")
        print(f"Trajectory Exact Match Rate: {trajectory_exact_match_rate:.3f} ({trajectory_exact_matches}/{total_matched})")
        print(f"Average Step Match Rate: {avg_step_match_rate:.3f}")
        print(f"Note: Using RAW actions with full parameters for exact matching")
        
        # Print step-by-step breakdown for first 10 steps
        print("\nStep-by-step Match Rates (first 10 steps):")
        for i, step_data in enumerate(step_by_step_rates[:10]):
            print(f"  Step {step_data['step']:2d}: {step_data['match_rate']:.3f} ({step_data['total_trajectories']} trajectories)")
        
        # Additional analysis to understand Step 1 vs Step 2 pattern
        step_analysis = self._analyze_step_patterns(step_wise_matches, step_by_step_rates, train_data, inference_data)
        results.update(step_analysis)
        
        print(f"Trajectory Matching Health: {'HEALTHY' if results['trajectory_matching_healthy'] else 'CONCERNING'}")
        
        return results
    
    def _analyze_step_patterns(self, step_wise_matches: List[Dict], step_by_step_rates: List[Dict], 
                              train_data: Dict, inference_data: Dict) -> Dict:
        """Detailed analysis of step patterns to understand why Step 2 might be higher than Step 1"""
        print("\n=== Detailed Step Pattern Analysis ===")
        
        # 1. Sample size analysis
        trajectory_lengths = [match['train_length'] for match in step_wise_matches]
        inference_lengths = [match['inference_length'] for match in step_wise_matches]
        
        # Count trajectories by length
        length_distribution = Counter(trajectory_lengths)
        inference_length_distribution = Counter(inference_lengths)
        
        print(f"\nTrajectory Length Distribution (Training):")
        for length in sorted(length_distribution.keys())[:10]:
            count = length_distribution[length]
            percentage = (count / len(trajectory_lengths)) * 100
            print(f"  {length:2d} steps: {count:3d} trajectories ({percentage:5.1f}%)")
        
        # 2. Short trajectory analysis
        very_short_train = sum(1 for length in trajectory_lengths if length <= 2)
        very_short_inference = sum(1 for length in inference_lengths if length <= 2)
        
        print(f"\nShort Trajectory Analysis:")
        print(f"  Training trajectories ≤2 steps: {very_short_train}/{len(trajectory_lengths)} ({very_short_train/len(trajectory_lengths)*100:.1f}%)")
        print(f"  Inference trajectories ≤2 steps: {very_short_inference}/{len(inference_lengths)} ({very_short_inference/len(inference_lengths)*100:.1f}%)")
        
        # 3. Action diversity analysis for first few steps
        if len(step_by_step_rates) >= 2:
            step1_count = step_by_step_rates[0]['total_trajectories']
            step2_count = step_by_step_rates[1]['total_trajectories'] if len(step_by_step_rates) > 1 else 0
            dropout_rate = (step1_count - step2_count) / step1_count if step1_count > 0 else 0
            
            print(f"\nSample Size Analysis:")
            print(f"  Step 1 trajectories: {step1_count}")
            print(f"  Step 2 trajectories: {step2_count}")
            print(f"  Dropout after Step 1: {step1_count - step2_count} trajectories ({dropout_rate*100:.1f}%)")
        
        # 4. Analyze specific actions at Step 1 and Step 2 from the original data
        step1_train_actions = []
        step1_inference_actions = []
        step2_train_actions = []
        step2_inference_actions = []
        
        # Create lookup dictionaries for quick access
        train_lookup = {meta['id']: i for i, meta in enumerate(train_data['metadata'])}
        inference_lookup = {meta['id']: i for i, meta in enumerate(inference_data['metadata'])}
        
        # Collect actual actions at each step
        for match in step_wise_matches:
            instance_id = match['instance_id']
            
            # Get trajectory indices
            train_idx = train_lookup.get(instance_id)
            inference_idx = inference_lookup.get(instance_id)
            
            if train_idx is not None and inference_idx is not None:
                train_traj = train_data['trajectories'][train_idx]
                inference_traj = inference_data['trajectories'][inference_idx]
                
                # Step 1 actions
                if len(train_traj) >= 1:
                    step1_train_actions.append(train_traj[0])
                if len(inference_traj) >= 1:
                    step1_inference_actions.append(inference_traj[0])
                
                # Step 2 actions
                if len(train_traj) >= 2:
                    step2_train_actions.append(train_traj[1])
                if len(inference_traj) >= 2:
                    step2_inference_actions.append(inference_traj[1])
        
        # Analyze action diversity at each step
        step1_train_diversity = len(set(step1_train_actions)) / len(step1_train_actions) if step1_train_actions else 0
        step1_inference_diversity = len(set(step1_inference_actions)) / len(step1_inference_actions) if step1_inference_actions else 0
        step2_train_diversity = len(set(step2_train_actions)) / len(step2_train_actions) if step2_train_actions else 0
        step2_inference_diversity = len(set(step2_inference_actions)) / len(step2_inference_actions) if step2_inference_actions else 0
        
        print(f"\nAction Diversity Analysis:")
        print(f"  Step 1 - Training: {len(set(step1_train_actions)):3d} unique actions / {len(step1_train_actions):3d} total = {step1_train_diversity:.3f}")
        print(f"  Step 1 - Inference: {len(set(step1_inference_actions)):3d} unique actions / {len(step1_inference_actions):3d} total = {step1_inference_diversity:.3f}")
        print(f"  Step 2 - Training: {len(set(step2_train_actions)):3d} unique actions / {len(step2_train_actions):3d} total = {step2_train_diversity:.3f}")
        print(f"  Step 2 - Inference: {len(set(step2_inference_actions)):3d} unique actions / {len(step2_inference_actions):3d} total = {step2_inference_diversity:.3f}")
        
        # Show most common actions at each step
        print(f"\nMost Common Actions:")
        
        if step1_train_actions:
            step1_train_counter = Counter(step1_train_actions)
            print(f"  Step 1 Training (top 5):")
            for action, count in step1_train_counter.most_common(5):
                print(f"    '{action[:50]:<50}' {count:3d}x")
        
        if step1_inference_actions:
            step1_inference_counter = Counter(step1_inference_actions)
            print(f"  Step 1 Inference (top 5):")
            for action, count in step1_inference_counter.most_common(5):
                print(f"    '{action[:50]:<50}' {count:3d}x")
        
        if step2_train_actions:
            step2_train_counter = Counter(step2_train_actions)
            print(f"  Step 2 Training (top 5):")
            for action, count in step2_train_counter.most_common(5):
                print(f"    '{action[:50]:<50}' {count:3d}x")
        
        if step2_inference_actions:
            step2_inference_counter = Counter(step2_inference_actions)
            print(f"  Step 2 Inference (top 5):")
            for action, count in step2_inference_counter.most_common(5):
                print(f"    '{action[:50]:<50}' {count:3d}x")
        
        # 5. Match rate progression analysis
        if len(step_by_step_rates) >= 3:
            step1_rate = step_by_step_rates[0]['match_rate']
            step2_rate = step_by_step_rates[1]['match_rate']
            step3_rate = step_by_step_rates[2]['match_rate'] if len(step_by_step_rates) > 2 else None
            
            print(f"\nMatch Rate Progression:")
            print(f"  Step 1: {step1_rate:.3f}")
            print(f"  Step 2: {step2_rate:.3f} ({'+' if step2_rate > step1_rate else ''}{step2_rate - step1_rate:+.3f})")
            if step3_rate is not None:
                print(f"  Step 3: {step3_rate:.3f} ({'+' if step3_rate > step2_rate else ''}{step3_rate - step2_rate:+.3f})")
            
            # Analyze if this is a consistent pattern
            step2_higher = step2_rate > step1_rate
            print(f"  Step 2 higher than Step 1: {'YES' if step2_higher else 'NO'}")
            
            if step2_higher:
                print(f"  Possible explanations:")
                print(f"    - Short trajectory filtering: {dropout_rate*100:.1f}% trajectories end after Step 1")
                print(f"    - Initial action variability: Step 1 might have more valid alternatives")
                print(f"    - Context convergence: Step 2 actions more predictable given Step 1")
        
        # 6. Trajectory ending analysis
        trajectories_ending_at_step = {}
        for match in step_wise_matches:
            train_length = match['train_length']
            inference_length = match['inference_length']
            min_length = min(train_length, inference_length)
            
            if min_length not in trajectories_ending_at_step:
                trajectories_ending_at_step[min_length] = 0
            trajectories_ending_at_step[min_length] += 1
        
        print(f"\nTrajectories Ending Analysis (first 10 lengths):")
        for length in sorted(trajectories_ending_at_step.keys())[:10]:
            count = trajectories_ending_at_step[length]
            percentage = (count / len(step_wise_matches)) * 100
            print(f"  End at step {length:2d}: {count:3d} pairs ({percentage:5.1f}%)")
        
        return {
            'step_pattern_analysis': {
                'trajectory_length_distribution': dict(length_distribution),
                'inference_length_distribution': dict(inference_length_distribution),
                'very_short_train_count': very_short_train,
                'very_short_inference_count': very_short_inference,
                'dropout_after_step1': dropout_rate if len(step_by_step_rates) >= 2 else 0,
                'step1_count': step_by_step_rates[0]['total_trajectories'] if step_by_step_rates else 0,
                'step2_count': step_by_step_rates[1]['total_trajectories'] if len(step_by_step_rates) > 1 else 0,
                'step2_higher_than_step1': step_by_step_rates[1]['match_rate'] > step_by_step_rates[0]['match_rate'] if len(step_by_step_rates) > 1 else False,
                'trajectories_ending_distribution': trajectories_ending_at_step,
                'step1_train_diversity': step1_train_diversity,
                'step1_inference_diversity': step1_inference_diversity,
                'step2_train_diversity': step2_train_diversity,
                'step2_inference_diversity': step2_inference_diversity,
                'step1_train_actions': dict(Counter(step1_train_actions).most_common(10)) if step1_train_actions else {},
                'step1_inference_actions': dict(Counter(step1_inference_actions).most_common(10)) if step1_inference_actions else {},
                'step2_train_actions': dict(Counter(step2_train_actions).most_common(10)) if step2_train_actions else {},
                'step2_inference_actions': dict(Counter(step2_inference_actions).most_common(10)) if step2_inference_actions else {}
            }
        }
    
    def _load_messages_for_instance(self, instance_id: str, train_data: Dict, inference_data: Dict, data_type: str) -> List[Dict]:
        """Load raw messages for a specific instance to compare prompts"""
        if data_type == 'train':
            # For training data, try to load the raw training file and extract the specific trajectory
            try:
                # Try to load from the original training file to get full messages
                # Note: This is a best-effort implementation - may need adjustment based on data format
                return self._extract_train_messages_by_id(instance_id, train_data)
            except Exception as e:
                print(f"Warning: Failed to load training messages for {instance_id}: {e}")
                return []
        else:
            # For inference data, load from .traj file
            for meta in inference_data['metadata']:
                if meta['id'] == instance_id:
                    try:
                        with open(meta['file_path'], 'r', encoding='utf-8') as f:
                            traj_data = json.load(f)
                            return traj_data.get("history", [])
                    except Exception as e:
                        print(f"Warning: Failed to load messages for {instance_id}: {e}")
                        return []
        return []
    
    def _extract_train_messages_by_id(self, instance_id: str, train_data: Dict) -> List[Dict]:
        """Extract messages from training data for a specific instance ID"""
        # This method tries to find the original messages from training data
        # Implementation depends on the structure of your training data
        
        # Check if we have stored raw data somewhere
        if 'raw_trajectories' in train_data:
            for traj in train_data['raw_trajectories']:
                if traj.get('instance_id') == instance_id:
                    return traj.get('messages', [])
        
        # If no raw messages available, we can't compare prompts
        # Return empty list to indicate we couldn't find the messages
        return []
    
    def _extract_raw_actions_from_training(self, instance_id: str, train_data: Dict) -> List[str]:
        """Extract raw actions from training data without normalization"""
        # Try to extract from raw_trajectories if available
        if 'raw_trajectories' in train_data:
            for traj in train_data['raw_trajectories']:
                if traj.get('instance_id') == instance_id:
                    messages = traj.get('messages', [])
                    raw_actions = []
                    for message in messages:
                        if message.get("role") == "assistant":
                            content = message.get("content", "")
                            # Parse raw function calls without normalization
                            parsed_actions = self._parse_function_call_from_content(content)
                            for action in parsed_actions:
                                raw_actions.append(action)  # Keep original action string
                    return raw_actions
        return []
    
    def _extract_raw_actions_from_inference(self, instance_id: str, inference_data: Dict) -> List[str]:
        """Extract raw actions from inference data without normalization"""
        for meta in inference_data['metadata']:
            if meta['id'] == instance_id:
                try:
                    with open(meta['file_path'], 'r', encoding='utf-8') as f:
                        traj_data = json.load(f)
                        messages = traj_data.get("history", [])
                        raw_actions = []
                        for message in messages:
                            if message.get("role") == "assistant" and "action" in message:
                                # Use the raw action directly
                                raw_action = message["action"]
                                raw_actions.append(raw_action)
                        return raw_actions
                except Exception as e:
                    print(f"Warning: Failed to extract raw actions for {instance_id}: {e}")
                    return []
        return []
    
    def _extract_system_prompt(self, messages: List[Dict]) -> str:
        """Extract system prompt from messages"""
        for message in messages:
            if message.get("role") == "system":
                return message.get("content", "")
        return ""
    
    def _extract_user_prompt(self, messages: List[Dict]) -> str:
        """Extract first user prompt from messages"""
        for message in messages:
            if message.get("role") == "user":
                return message.get("content", "")
        return ""

    def analyze_missing_actions_impact(self, train_data: Dict, inference_data: Dict) -> Dict:
        """Analyze which missing actions contribute to shorter trajectories"""
        print("\n=== Missing Actions Impact Analysis ===")
        
        train_actions = set(train_data['actions'].keys())
        inference_actions = set(inference_data['actions'].keys())
        missing_actions = train_actions - inference_actions
        
        # Calculate average trajectory lengths
        train_avg_length = np.mean([len(traj) for traj in train_data['trajectories']])
        inference_avg_length = np.mean([len(traj) for traj in inference_data['trajectories']])
        length_difference = train_avg_length - inference_avg_length
        
        # Analyze missing action frequencies in training
        missing_action_frequencies = {}
        total_train_actions = sum(train_data['actions'].values())
        
        for action in missing_actions:
            frequency = train_data['actions'][action]
            percentage = (frequency / total_train_actions) * 100
            missing_action_frequencies[action] = {
                'frequency': frequency,
                'percentage': percentage
            }
        
        # Sort by frequency
        sorted_missing = sorted(missing_action_frequencies.items(), 
                              key=lambda x: x[1]['frequency'], reverse=True)
        
        # Calculate cumulative impact
        cumulative_missing_percentage = sum(data['percentage'] for _, data in missing_action_frequencies.items())
        
        results = {
            'train_avg_length': train_avg_length,
            'inference_avg_length': inference_avg_length,
            'length_difference': length_difference,
            'missing_actions_count': len(missing_actions),
            'missing_action_frequencies': missing_action_frequencies,
            'sorted_missing_actions': sorted_missing,
            'cumulative_missing_percentage': cumulative_missing_percentage,
            'top_missing_actions': sorted_missing[:10]  # Top 10 missing actions
        }
        
        print(f"Length Difference: {length_difference:.2f} (Train: {train_avg_length:.2f}, Inference: {inference_avg_length:.2f})")
        print(f"Missing Actions Count: {len(missing_actions)}")
        print(f"Cumulative Missing Percentage: {cumulative_missing_percentage:.2f}%")
        
        print("\nTop 10 Missing Actions (by frequency in training):")
        for action, data in sorted_missing[:10]:
            print(f"  {action:30} {data['frequency']:4d} ({data['percentage']:5.2f}%)")
        
        # Hypothesis: missing actions might explain shorter trajectories
        results['missing_actions_explain_length'] = cumulative_missing_percentage > 5.0
        print(f"\nMissing actions likely explain shorter trajectories: {results['missing_actions_explain_length']}")
        
        return results

    def create_comprehensive_plots(self, all_results: Dict, output_dir: str = "plots"):
        """Create comprehensive plots for all analyses"""
        if not HAS_ANALYSIS_LIBS:
            print("Plotting libraries not available. Skipping visualization.")
            return
        
        print("\n=== Creating Comprehensive Plots ===")
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(24, 20))
        # Create custom spacing with larger gap between 1st and 2nd rows
        gs = gridspec.GridSpec(4, 3, 
                              height_ratios=[1, 0.2, 1, 1],  # Add spacing row
                              hspace=0.8, wspace=0.3)
        
        # 1. Action Distribution Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        dist_data = all_results['distribution_similarity']
        
        # Get top 10 actions from training data
        actions = list(dist_data['train_action_counts'].keys())[:10]
        train_counts = list(dist_data['train_action_counts'].values())[:10]
        
        # Get corresponding inference counts for the SAME actions (not top 10 inference actions)
        complete_inference_actions = dist_data['complete_inference_actions']
        inference_counts = [complete_inference_actions.get(action, 0) for action in actions]
        
        x = np.arange(len(actions))
        width = 0.35
        ax1.bar(x - width/2, train_counts, width, label='Training', alpha=0.8)
        ax1.bar(x + width/2, inference_counts, width, label='Inference', alpha=0.8)
        ax1.set_title('Top 10 Action Frequencies')
        ax1.set_ylabel('Count')
        ax1.set_xticks(x)
        ax1.set_xticklabels([a[:15] for a in actions], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. Enhanced Action Frequency Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Get complete action data from the original analysis (not truncated)
        # Use the complete train_actions and inference_actions from distribution analysis
        train_actions = all_results['distribution_similarity']['complete_train_actions']
        inference_actions = all_results['distribution_similarity']['complete_inference_actions']
        
        # Sort actions: train-only actions first, then shared actions, then inference-only actions
        train_only = set(train_actions.keys()) - set(inference_actions.keys())
        inference_only = set(inference_actions.keys()) - set(train_actions.keys())
        shared_actions = set(train_actions.keys()) & set(inference_actions.keys())
        
        # Create ordered action list
        ordered_actions = (sorted(train_only) + sorted(shared_actions) + sorted(inference_only))
        
        # Prepare data for plotting
        train_frequencies = [train_actions.get(action, 0) for action in ordered_actions]
        inference_frequencies = [inference_actions.get(action, 0) for action in ordered_actions]
        
        # Create positions for bars
        x_pos = np.arange(len(ordered_actions))
        width = 0.35
        
        # Plot bars
        ax2.bar(x_pos - width/2, train_frequencies, width, 
                label='Training', color='skyblue', alpha=0.8)
        ax2.bar(x_pos + width/2, inference_frequencies, width,
                label='Inference', color='lightcoral', alpha=0.8)
        
        # Customize the plot
        ax2.set_title('Action Frequency Comparison\n(Left: Train-only, Middle: Shared, Right: Inference-only)', 
                     fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency')
        ax2.set_xlabel('Actions')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(ordered_actions, rotation=45, ha='right', fontsize=8)
        # Position legend in upper right corner
        ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax2.grid(alpha=0.3, axis='y')
        
        # Add dividing lines to separate sections
        if train_only and shared_actions:
            ax2.axvline(x=len(train_only)-0.5, color='gray', linestyle='--', alpha=0.5)
        if shared_actions and inference_only:
            ax2.axvline(x=len(train_only)+len(shared_actions)-0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Add section labels
        if train_only:
            ax2.text(len(train_only)/2-0.5, max(max(train_frequencies), max(inference_frequencies))*0.9,
                    'Train Only', ha='center', fontweight='bold', color='blue', fontsize=10)
        if shared_actions:
            ax2.text(len(train_only)+len(shared_actions)/2-0.5, max(max(train_frequencies), max(inference_frequencies))*0.9,
                    'Shared', ha='center', fontweight='bold', color='green', fontsize=10)
        if inference_only:
            ax2.text(len(train_only)+len(shared_actions)+len(inference_only)/2-0.5, max(max(train_frequencies), max(inference_frequencies))*0.9,
                    'Inference Only', ha='center', fontweight='bold', color='red', fontsize=10)
        
        # 3. Step-by-Step Analysis
        ax3 = fig.add_subplot(gs[0, 2])
        step_data = all_results['step_exact_analysis']
        if 'step_exact_matches' in step_data:
            steps = list(step_data['step_exact_matches'].keys())
            train_unique = [step_data['step_exact_matches'][s]['train_unique_actions'] for s in steps]
            inf_unique = [step_data['step_exact_matches'][s]['inference_unique_actions'] for s in steps]
            
            ax3.plot(steps, train_unique, 'o-', label='Training', linewidth=2, markersize=6)
            ax3.plot(steps, inf_unique, 's-', label='Inference', linewidth=2, markersize=6)
            ax3.set_title('Unique Actions by Step')
            ax3.set_xlabel('Step Number')
            ax3.set_ylabel('Number of Unique Actions')
            ax3.legend()
            ax3.grid(alpha=0.3)
        
        # 4. Length vs Similarity (if available) - moved to row 2 (skipping spacing row 1)
        ax4 = fig.add_subplot(gs[2, 0])
        if 'length_similarity_analysis' in all_results and all_results['length_similarity_analysis']['analysis_possible']:
            length_sim_data = all_results['length_similarity_analysis']
            lengths = length_sim_data['individual_lengths']
            similarities = length_sim_data['individual_similarities']
            
            ax4.scatter(lengths, similarities, alpha=0.6, s=30)
            ax4.set_title(f'Length vs Training Similarity\n(ρ={length_sim_data["correlation"]:.3f})')
            ax4.set_xlabel('Trajectory Length')
            ax4.set_ylabel('Similarity to Training')
            ax4.grid(alpha=0.3)
            
            # Add trend line
            z = np.polyfit(lengths, similarities, 1)
            p = np.poly1d(z)
            ax4.plot(lengths, p(lengths), "r--", alpha=0.8)
        
        # 5. Diversity Analysis
        ax5 = fig.add_subplot(gs[2, 1])
        diversity_data = all_results['diversity_analysis']
        train_divs = diversity_data['train_metrics']['trajectory_diversities']
        inf_divs = diversity_data['inference_metrics']['trajectory_diversities']
        
        ax5.hist(train_divs, bins=20, alpha=0.7, label='Training', density=True)
        ax5.hist(inf_divs, bins=20, alpha=0.7, label='Inference', density=True)
        ax5.set_title('Trajectory Diversity Distribution')
        ax5.set_xlabel('Diversity Score')
        ax5.set_ylabel('Density')
        ax5.legend()
        ax5.grid(alpha=0.3)
        
        # 6. Missing Actions Impact
        ax6 = fig.add_subplot(gs[2, 2])
        missing_data = all_results['missing_actions_analysis']
        if 'top_missing_actions' in missing_data:
            top_missing = missing_data['top_missing_actions'][:10]
            actions = [action[:20] for action, _ in top_missing]
            percentages = [data['percentage'] for _, data in top_missing]
            
            bars = ax6.barh(range(len(actions)), percentages, alpha=0.7)
            ax6.set_title('Top Missing Actions\n(% of Training Data)')
            ax6.set_xlabel('Percentage (%)')
            ax6.set_yticks(range(len(actions)))
            ax6.set_yticklabels(actions)
            ax6.grid(alpha=0.3, axis='x')
        
        # 7. Success Analysis
        ax7 = fig.add_subplot(gs[3, 0])
        success_data = all_results['success_analysis']
        if success_data.get('success_analysis_possible', False):
            categories = ['Training\nSuccess Rate', 'Inference\nSuccess Rate']
            values = [success_data['train_success_rate'], success_data['inference_success_rate']]
            colors = ['blue', 'orange']
            bars = ax7.bar(categories, values, color=colors, alpha=0.7)
            ax7.set_title('Success Rate Comparison')
            ax7.set_ylabel('Success Rate')
            ax7.set_ylim(0, 1)
            # Add value labels
            for bar, value in zip(bars, values):
                ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            ax7.grid(alpha=0.3)
        
        # 8. Trajectory Matching Analysis (if available)
        ax8 = fig.add_subplot(gs[3, 1])
        if 'trajectory_matching_analysis' in all_results and all_results['trajectory_matching_analysis'].get('matching_analysis_possible', False):
            matching_data = all_results['trajectory_matching_analysis']
            
            # Create metrics comparison
            metrics = ['System\nPrompt', 'User\nPrompt', 'Avg Step\nMatch', 'Exact\nTraj Match']
            values = [
                matching_data.get('system_prompt_match_rate', 0),
                matching_data.get('user_prompt_match_rate', 0),
                matching_data.get('average_step_match_rate', 0),
                matching_data.get('trajectory_exact_match_rate', 0)
            ]
            
            colors = ['green' if v > 0.8 else 'orange' if v > 0.6 else 'red' for v in values]
            bars = ax8.bar(metrics, values, color=colors, alpha=0.7)
            ax8.set_title('Trajectory Matching Analysis')
            ax8.set_ylabel('Match Rate')
            ax8.set_ylim(0, 1)
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            ax8.grid(alpha=0.3)
        else:
            ax8.text(0.5, 0.5, 'No Matching\nTrajectories Found', ha='center', va='center', 
                    transform=ax8.transAxes, fontsize=14, fontweight='bold')
            ax8.set_title('Trajectory Matching Analysis')
        
        # 9. Step-by-Step Match Rate (if matching data available)
        ax9 = fig.add_subplot(gs[3, 2])
        if 'trajectory_matching_analysis' in all_results and all_results['trajectory_matching_analysis'].get('matching_analysis_possible', False):
            matching_data = all_results['trajectory_matching_analysis']
            step_data = matching_data.get('step_by_step_rates', [])
            
            if step_data:
                steps = [d['step'] for d in step_data[:15]]  # First 15 steps
                match_rates = [d['match_rate'] for d in step_data[:15]]
                
                ax9.plot(steps, match_rates, 'o-', linewidth=2, markersize=6, color='blue')
                ax9.set_title('Step-by-Step Match Rate')
                ax9.set_xlabel('Step Number')
                ax9.set_ylabel('Match Rate')
                ax9.set_ylim(0, 1)
                ax9.grid(alpha=0.3)
                
                # Add horizontal line at 0.8 threshold
                ax9.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Target (0.8)')
                ax9.legend()
            else:
                ax9.text(0.5, 0.5, 'No Step Data\nAvailable', ha='center', va='center', 
                        transform=ax9.transAxes, fontsize=14, fontweight='bold')
                ax9.set_title('Step-by-Step Match Rate')
        else:
            ax9.text(0.5, 0.5, 'No Matching\nTrajectories Found', ha='center', va='center', 
                    transform=ax9.transAxes, fontsize=14, fontweight='bold')
            ax9.set_title('Step-by-Step Match Rate')
        
        # Note: We now have 9 plots in a layout with spacing row
        # Row 0: Action Distribution, Action Frequency Comparison, Step Analysis
        # Row 1: (spacing row - empty)
        # Row 2: Length-Similarity, Diversity, Missing Actions  
        # Row 3: Success Rate, Trajectory Matching, Step-by-Step Match
        
        plt.suptitle('Comprehensive Sanity Check Analysis Results', fontsize=20, fontweight='bold', y=0.98)
        
        # Save the plot
        output_file = output_dir / "comprehensive_sanity_check_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Comprehensive plot saved to: {output_file}")
        
        plt.show()

    def run_enhanced_analysis(self, train_file: str, inference_folder: str) -> AnalysisResults:
        """Run all enhanced analyses and return comprehensive results"""
        print("="*80)
        print("ENHANCED COMPREHENSIVE SANITY CHECK ANALYSIS")
        print("="*80)
        
        # Load data
        train_data = self.load_train_data(train_file)
        inference_data = self.load_inference_data(inference_folder)
        
        if not train_data or not inference_data:
            print("ERROR: Failed to load data")
            return None
        
        results = {}
        
        # Run all analyses
        # 1. Distribution comparison
        results['distribution_similarity'] = self.compare_action_distributions(
            train_data['actions'], inference_data['actions']
        )
        
        # 2. Coverage analysis
        results['coverage_analysis'] = self.analyze_action_coverage(
            train_data['actions'], inference_data['actions']
        )
        
        # 3. Step-by-step exact analysis (improved position analysis)
        results['step_exact_analysis'] = self.analyze_step_exact_matching(
            train_data['trajectories'], inference_data['trajectories']
        )
        
        # 4. Pattern analysis
        results['pattern_overlap'] = self.analyze_action_patterns(
            train_data['trajectories'], inference_data['trajectories'], n=2
        )
        
        # 5. Enhanced success analysis
        results['success_analysis'] = self.analyze_enhanced_successful_trajectories(
            train_data, inference_data
        )
        
        # 6. Temporal flow analysis
        results['temporal_flow'] = self.analyze_temporal_flow(
            train_data['trajectories'], inference_data['trajectories']
        )
        
        # 7. NEW: Length-similarity analysis
        results['length_similarity_analysis'] = self.analyze_trajectory_length_similarity(
            train_data, inference_data
        )
        
        # 8. NEW: Diversity analysis
        results['diversity_analysis'] = self.analyze_action_diversity(
            train_data, inference_data
        )
        
        # 9. NEW: Missing actions impact analysis
        results['missing_actions_analysis'] = self.analyze_missing_actions_impact(
            train_data, inference_data
        )
        
        # 10. NEW: Trajectory matching analysis
        results['trajectory_matching_analysis'] = self.analyze_trajectory_matching(
            train_data, inference_data
        )
        
        # 11. Summary metrics
        results['summary_metrics'] = self.generate_enhanced_summary_metrics(results)
        
        # Create AnalysisResults with proper fields
        return AnalysisResults(
            distribution_similarity=results['distribution_similarity'],
            coverage_analysis=results['coverage_analysis'],
            step_exact_analysis=results['step_exact_analysis'],
            pattern_overlap=results['pattern_overlap'],
            success_analysis=results['success_analysis'],
            temporal_flow=results['temporal_flow'],
            length_similarity_analysis=results['length_similarity_analysis'],
            diversity_analysis=results['diversity_analysis'],
            missing_actions_analysis=results['missing_actions_analysis'],
            trajectory_matching_analysis=results['trajectory_matching_analysis'],
            summary_metrics=results['summary_metrics']
        )

    def analyze_action_patterns(self, train_trajectories: List[List[str]], 
                              inference_trajectories: List[List[str]], n: int = 2) -> Dict:
        """Analyze n-gram action patterns"""
        print(f"\n=== Action Pattern Analysis (n={n}) ===")
        
        def extract_patterns(trajectories, n):
            patterns = Counter()
            for traj in trajectories:
                for i in range(len(traj) - n + 1):
                    pattern = tuple(traj[i:i+n])
                    patterns[pattern] += 1
            return patterns
        
        train_patterns = extract_patterns(train_trajectories, n)
        inference_patterns = extract_patterns(inference_trajectories, n)
        
        # Calculate pattern overlap
        train_pattern_set = set(train_patterns.keys())
        inference_pattern_set = set(inference_patterns.keys())
        shared_patterns = train_pattern_set & inference_pattern_set
        
        pattern_coverage = len(shared_patterns) / len(inference_pattern_set) if inference_pattern_set else 0
        pattern_overlap_ratio = len(shared_patterns) / len(train_pattern_set | inference_pattern_set)
        
        # Frequency-weighted overlap
        inference_total = sum(inference_patterns.values())
        shared_frequency = sum(inference_patterns[pattern] for pattern in shared_patterns)
        weighted_pattern_coverage = shared_frequency / inference_total if inference_total else 0
        
        results = {
            'pattern_coverage': pattern_coverage,
            'pattern_overlap_ratio': pattern_overlap_ratio,
            'weighted_pattern_coverage': weighted_pattern_coverage,
            'shared_patterns': len(shared_patterns),
            'train_unique_patterns': len(train_pattern_set),
            'inference_unique_patterns': len(inference_pattern_set),
            'pattern_analysis_healthy': pattern_coverage > 0.7 and weighted_pattern_coverage > 0.8
        }
        
        print(f"Pattern Coverage: {pattern_coverage:.3f} ({len(shared_patterns)}/{len(inference_pattern_set)})")
        print(f"Pattern Overlap Ratio: {pattern_overlap_ratio:.3f}")
        print(f"Weighted Pattern Coverage: {weighted_pattern_coverage:.3f}")
        print(f"Pattern Analysis Health: {'HEALTHY' if results['pattern_analysis_healthy'] else 'CONCERNING'}")
        
        return results

    def analyze_enhanced_successful_trajectories(self, train_data: Dict, inference_data: Dict) -> Dict:
        """Enhanced success analysis using resolved_ids"""
        print("\n=== Enhanced Success-Conditioned Analysis ===")
        
        # Count successful trajectories
        train_successful = [i for i, meta in enumerate(train_data['metadata']) if meta['success']]
        inference_successful = [i for i, meta in enumerate(inference_data['metadata']) if meta['success']]
        
        train_success_rate = len(train_successful) / len(train_data['metadata']) if train_data['metadata'] else 0
        inference_success_rate = len(inference_successful) / len(inference_data['metadata']) if inference_data['metadata'] else 0
        
        print(f"Successful trajectories - Training: {len(train_successful)}/{len(train_data['metadata'])} ({train_success_rate:.3f})")
        print(f"Successful trajectories - Inference: {len(inference_successful)}/{len(inference_data['metadata'])} ({inference_success_rate:.3f})")
        
        if not train_successful or not inference_successful:
            return {
                'success_analysis_possible': False,
                'train_success_rate': train_success_rate,
                'inference_success_rate': inference_success_rate
            }
        
        # Extract actions from successful trajectories only
        train_success_actions = Counter()
        inference_success_actions = Counter()
        
        for idx in train_successful:
            for action in train_data['trajectories'][idx]:
                train_success_actions[action] += 1
                
        for idx in inference_successful:
            for action in inference_data['trajectories'][idx]:
                inference_success_actions[action] += 1
        
        # Analyze coverage within successful trajectories
        train_set = set(train_success_actions.keys())
        inference_set = set(inference_success_actions.keys())
        shared = train_set & inference_set
        
        success_coverage = len(shared) / len(inference_set) if inference_set else 0
        
        # Calculate average trajectory lengths for successful trajectories
        train_success_lengths = [len(train_data['trajectories'][i]) for i in train_successful]
        inference_success_lengths = [len(inference_data['trajectories'][i]) for i in inference_successful]
        
        train_avg_length = np.mean(train_success_lengths) if train_success_lengths else 0
        inference_avg_length = np.mean(inference_success_lengths) if inference_success_lengths else 0
        
        results = {
            'success_analysis_possible': True,
            'train_success_rate': train_success_rate,
            'inference_success_rate': inference_success_rate,
            'success_coverage': success_coverage,
            'train_successful_count': len(train_successful),
            'inference_successful_count': len(inference_successful),
            'train_avg_success_length': train_avg_length,
            'inference_avg_success_length': inference_avg_length,
            'success_analysis_healthy': success_coverage > 0.9 and inference_success_rate > 0.5
        }
        
        print(f"Success Coverage: {success_coverage:.3f}")
        print(f"Average successful trajectory length - Training: {train_avg_length:.1f}, Inference: {inference_avg_length:.1f}")
        print(f"Success Analysis Health: {'HEALTHY' if results['success_analysis_healthy'] else 'CONCERNING'}")
        
        return results

    def analyze_temporal_flow(self, train_trajectories: List[List[str]], 
                            inference_trajectories: List[List[str]]) -> Dict:
        """Analyze action transition patterns"""
        print("\n=== Temporal Flow Analysis ===")
        
        def build_transition_graph(trajectories):
            transitions = Counter()
            for traj in trajectories:
                for i in range(len(traj) - 1):
                    current_action = traj[i]
                    next_action = traj[i + 1]
                    transitions[(current_action, next_action)] += 1
            return transitions
        
        train_transitions = build_transition_graph(train_trajectories)
        inference_transitions = build_transition_graph(inference_trajectories)
        
        # Calculate transition overlap
        train_transition_set = set(train_transitions.keys())
        inference_transition_set = set(inference_transitions.keys())
        shared_transitions = train_transition_set & inference_transition_set
        
        transition_coverage = len(shared_transitions) / len(inference_transition_set) if inference_transition_set else 0
        
        results = {
            'transition_coverage': transition_coverage,
            'shared_transitions': len(shared_transitions),
            'train_unique_transitions': len(train_transition_set),
            'inference_unique_transitions': len(inference_transition_set),
            'temporal_flow_healthy': transition_coverage > 0.6
        }
        
        print(f"Transition Coverage: {transition_coverage:.3f} ({len(shared_transitions)}/{len(inference_transition_set)})")
        print(f"Temporal Flow Health: {'HEALTHY' if results['temporal_flow_healthy'] else 'CONCERNING'}")
        
        return results

    def generate_enhanced_summary_metrics(self, all_results: Dict) -> Dict:
        """Generate overall health summary with enhanced metrics"""
        print("\n" + "="*80)
        print("ENHANCED SANITY CHECK SUMMARY")
        print("="*80)
        
        health_checks = []
        
        # Extract health indicators
        checks = [
            ('distribution_similarity', 'distributions_similar'),
            ('coverage_analysis', 'coverage_healthy'),
            ('step_exact_analysis', 'step_analysis_healthy'),
            ('pattern_overlap', 'pattern_analysis_healthy'),
            ('success_analysis', 'success_analysis_healthy'),
            ('temporal_flow', 'temporal_flow_healthy'),
            ('length_similarity_analysis', 'length_similarity_healthy'),
            ('diversity_analysis', 'diversity_analysis_healthy'),
            ('trajectory_matching_analysis', 'trajectory_matching_healthy'),
        ]
        
        for analysis_name, health_key in checks:
            if analysis_name in all_results and health_key in all_results[analysis_name]:
                health_checks.append(all_results[analysis_name][health_key])
        
        overall_health_score = sum(health_checks) / len(health_checks) if health_checks else 0
        overall_healthy = overall_health_score >= 0.6  # Lowered threshold due to more stringent checks
        
        summary = {
            'overall_health_score': overall_health_score,
            'overall_healthy': overall_healthy,
            'passed_checks': sum(health_checks),
            'total_checks': len(health_checks),
            'health_status': 'HEALTHY' if overall_healthy else 'CONCERNING'
        }
        
        print(f"Overall Health Score: {overall_health_score:.3f}")
        print(f"Passed Checks: {sum(health_checks)}/{len(health_checks)}")
        print(f"Health Status: {summary['health_status']}")
        
        # Detailed breakdown
        print("\nDetailed Analysis Breakdown:")
        for i, (analysis_name, health_key) in enumerate(checks):
            if i < len(health_checks):
                status = "✓ PASS" if health_checks[i] else "✗ FAIL"
                print(f"  {analysis_name:25} {status}")
        
        if not overall_healthy:
            print("\nKEY INSIGHTS:")
            
            # Length analysis insight
            if 'missing_actions_analysis' in all_results:
                missing_data = all_results['missing_actions_analysis']
                length_diff = missing_data.get('length_difference', 0)
                print(f"- Inference trajectories are {length_diff:.1f} steps shorter on average")
                
            # Diversity insight
            if 'diversity_analysis' in all_results:
                div_data = all_results['diversity_analysis']
                train_div = div_data.get('train_diversity_mean', 0)
                inf_div = div_data.get('inference_diversity_mean', 0)
                if abs(train_div - inf_div) > 0.05:
                    change = "higher" if inf_div > train_div else "lower"
                    print(f"- Inference trajectories have {change} action diversity ({inf_div:.3f} vs {train_div:.3f})")
            
            # Success rate insight
            if 'success_analysis' in all_results and all_results['success_analysis'].get('success_analysis_possible'):
                success_data = all_results['success_analysis']
                inf_success = success_data.get('inference_success_rate', 0)
                train_success = success_data.get('train_success_rate', 1)
                if inf_success < train_success:
                    print(f"- Success rate dropped from {train_success:.1%} to {inf_success:.1%}")
        else:
            print("\nTRAINING APPEARS EFFECTIVE:")
            print("- Model successfully learned training action patterns")
            print("- Behavioral consistency maintained across analyses")
            print("- No major distributional shifts detected")
        
        return summary

def main():
    # Default file paths
    default_train_data = "data/automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10/swesmith_gen_claude__claude-sonnet-4_gpt4.1_gpt-4o__t-0.00__p-1.00__c-2.00___patch_swesmith_astropy__astropy.26d14786_ps/astropy__astropy.26d14786_submit.json"
    default_inference_data = "trajectories/zhengyanshi@microsoft.com/swesmith_infer__openai--tt_SWE-agent-LM-32B_cl32768_lr1e-4_ep3_astropy__astropy.26d14786_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o__t-0.00__p-1.00__c-0.00___patch_swesmith_filtered_swesmith_task__ms75_as1_sanity_check"
    
    parser = argparse.ArgumentParser(description="Enhanced sanity check analysis: Training vs Inference")
    parser.add_argument("--train-data", default=default_train_data, 
                        help=f"Path to training data JSON file (default: {default_train_data[:50]}...)")
    parser.add_argument("--inference-data", default=default_inference_data,
                        help=f"Path to inference trajectory folder (default: {default_inference_data[:50]}...)")
    parser.add_argument("--output", help="Output file for results (optional)")
    parser.add_argument("--plot-dir", default="plots", help="Directory for plots (default: plots)")
    
    args = parser.parse_args()
    
    analyzer = EnhancedSanityCheckAnalyzer()
    results = analyzer.run_enhanced_analysis(args.train_data, args.inference_data)
    
    if results:
        # Create comprehensive plots
        results_dict = {
            'distribution_similarity': results.distribution_similarity,
            'coverage_analysis': results.coverage_analysis,
            'step_exact_analysis': results.step_exact_analysis,
            'pattern_overlap': results.pattern_overlap,
            'success_analysis': results.success_analysis,
            'temporal_flow': results.temporal_flow,
            'length_similarity_analysis': results.length_similarity_analysis,
            'diversity_analysis': results.diversity_analysis,
            'missing_actions_analysis': results.missing_actions_analysis,
            'trajectory_matching_analysis': results.trajectory_matching_analysis,
            'summary_metrics': results.summary_metrics,
            'train_trajectories': [],  # Will be populated from the analyzer
            'inference_trajectories': []  # Will be populated from the analyzer
        }
        
        analyzer.create_comprehensive_plots(results_dict, args.plot_dir)
        
        if args.output:
            # Save results to file
            def convert_numpy_types(obj):
                """Convert numpy types to Python native types for JSON serialization"""
                if isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                elif hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                elif isinstance(obj, (set,)):
                    return list(obj)
                else:
                    return obj
            
            output_data = convert_numpy_types(results_dict)
            
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main()