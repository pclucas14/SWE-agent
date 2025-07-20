#!/usr/bin/env python3
"""
Trajectory Analysis Tool

This script analyzes trajectory files from SWE-agent runs to extract tool usage frequency.
It processes assistant messages and counts how often different tools and commands are used.

Usage:
    python trajectory_analysis.py folder_path1 [folder_path2 ...]

Features:
- Loads trajectory files (.traj) from multiple folders
- Extracts assistant messages and parses "action" fields  
- Counts tool usage frequency (command level analysis)
- Handles bash commands appropriately
- Creates visualization plots for each folder
- Implements caching to avoid reprocessing
- Processes files efficiently to prevent system crashes
"""

import os
import json
import pickle
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    HAS_PLOTTING = True
except ImportError:
    print("Warning: matplotlib, pandas, or numpy not found. Plotting functionality disabled.")
    HAS_PLOTTING = False

try:
    import seaborn as sns
    if HAS_PLOTTING:
        # Set style for better plots
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')
        sns.set_palette("husl")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, **kwargs):
        return iterable
    HAS_TQDM = False

class TrajectoryAnalyzer:
    """Main class for analyzing trajectory files and extracting tool usage patterns."""
    
    def __init__(self, cache_dir: str = "cache"):
        """Initialize analyzer with cache directory."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Define known tools from swesmith_infer.yaml
        self.known_tools = {
            'bash', 'submit', 'str_replace_editor', 'system'
        }
        
        # Define str_replace_editor sub-commands
        self.str_replace_commands = {
            'view', 'create', 'str_replace', 'insert', 'undo_edit'
        }
        
        # Common bash commands to track
        self.common_bash_commands = {
            'ls', 'cd', 'cat', 'grep', 'find', 'mkdir', 'rm', 'cp', 'mv',
            'echo', 'pwd', 'chmod', 'chown', 'head', 'tail', 'less', 'more',
            'python', 'pip', 'git', 'vim', 'nano', 'wget', 'curl', 'tar',
            'unzip', 'ps', 'kill', 'top', 'df', 'du', 'free', 'man'
        }

    def _get_folder_hash(self, folder_path: str) -> str:
        """Generate hash for folder path to use in cache key."""
        return hashlib.md5(folder_path.encode()).hexdigest()[:12]

    def _get_cache_path(self, folder_path: str) -> Path:
        """Get cache file path for given folder."""
        folder_hash = self._get_folder_hash(folder_path)
        folder_name = Path(folder_path).name
        return self.cache_dir / f"{folder_name}_{folder_hash}.pkl"

    def _is_system_message(self, message: dict) -> str:
        """Check if message is a system message and return the type."""
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

    def _load_cache(self, folder_path: str) -> dict:
        """Load cached analysis results if available."""
        cache_path = self._get_cache_path(folder_path)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Warning: Failed to load cache for {folder_path}: {e}")
        return {}

    def _save_cache(self, folder_path: str, data: dict):
        """Save analysis results to cache."""
        cache_path = self._get_cache_path(folder_path)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Warning: Failed to save cache for {folder_path}: {e}")

    def _find_trajectory_files(self, folder_path: str) -> List[str]:
        """Find all .traj files in the given folder and its subdirectories."""
        traj_files = []
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            print(f"Warning: Folder {folder_path} does not exist")
            return []
        
        # Recursively find .traj files
        for traj_file in folder_path.rglob("*.traj"):
            traj_files.append(str(traj_file))
        
        return traj_files

    def _load_trajectory_file(self, file_path: str) -> dict:
        """Load a single trajectory file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
            return {}

    def _parse_action(self, action: str, message: dict = None) -> Tuple[str, str]:
        """Parse action string to extract tool and command."""
        if not action or not isinstance(action, str):
            # Check if this is a system message
            system_type = self._is_system_message(message) if message else None
            if system_type:
                return "system", system_type
            
            # Debug: Print examples of other empty actions
            if hasattr(self, '_empty_action_count'):
                self._empty_action_count += 1
            else:
                self._empty_action_count = 1
            
            if self._empty_action_count <= 10:  # Print first 10 non-max-step examples
                print(f"\n=== Empty action #{self._empty_action_count} ===")
                print(f"Action: {repr(action)} (type: {type(action)})")
                if message:
                    full_content = message.get('content', 'No content')
                    print(f"Full message content: {repr(full_content)}")
                    print(f"Message keys: {list(message.keys())}")
                    if 'thought' in message:
                        print(f"Thought: {repr(message.get('thought', ''))}")
                print("=" * 50)
            
            return "other", "empty_action"
        
        action = action.strip()
        
        # Handle bash commands
        action_lower = action.lower()
        if action.startswith('bash ') or not any(action_lower.startswith(tool) for tool in self.known_tools):
            # This is likely a bash command
            if action.startswith('bash '):
                bash_command = action[5:].strip()
            else:
                bash_command = action
            
            # Extract the first command from bash
            command_parts = bash_command.split()
            if command_parts:
                first_cmd = command_parts[0]
                # Handle complex commands with pipes, redirects, etc.
                first_cmd = re.split(r'[|;&><]', first_cmd)[0].strip()
                # Remove common prefixes
                first_cmd = first_cmd.split('/')[-1]  # Get basename
                return "bash", first_cmd
            return "bash", "empty_bash"
        
        # Handle str_replace_editor commands
        if action.startswith('str_replace_editor '):
            parts = action.split()
            if len(parts) >= 2:
                sub_command = parts[1]
                if sub_command in self.str_replace_commands:
                    return "str_replace_editor", sub_command
                else:
                    # Try to parse non-standard commands
                    return "str_replace_editor", f"other_{sub_command}"
            return "str_replace_editor", "no_subcommand"
        
        # Handle submit (case insensitive)
        if action.lower().startswith('submit'):
            return "submit", "submit"
        
        # Handle other known tools
        for tool in self.known_tools:
            if action.startswith(tool):
                return tool, tool
        
        # Try to identify other common patterns
        if 'python' in action.lower():
            return "bash", "python"
        if 'git' in action.lower():
            return "bash", "git"
        if 'pip' in action.lower():
            return "bash", "pip"
        if 'cd' in action.lower():
            return "bash", "cd"
        if 'ls' in action.lower():
            return "bash", "ls"
        
        # Last resort - categorize as other with first word
        first_word = action.split()[0] if action.split() else "empty"
        return "other", first_word

    def _analyze_messages(self, messages: List[dict]) -> Dict[str, Counter]:
        """Analyze assistant messages to extract tool usage."""
        tool_counts = Counter()
        command_counts = Counter()
        step_actions = {}  # step_number -> Counter of actions
        final_step_action = None  # Track the final step action for this trajectory
        
        assistant_step = 0
        for message in messages:
            if message.get("role") == "assistant" and "action" in message:
                assistant_step += 1
                action = message["action"]
                tool, command = self._parse_action(action, message)
                
                tool_counts[tool] += 1
                if tool == "bash":
                    command_counts[f"bash {command}"] += 1
                    full_action = f"bash {command}"
                elif tool == "str_replace_editor":
                    command_counts[f"str_replace_editor {command}"] += 1
                    full_action = f"str_replace_editor {command}"
                elif tool == "system":
                    command_counts[f"system {command}"] += 1
                    full_action = f"system {command}"
                else:
                    command_counts[command] += 1
                    full_action = command
                
                # Track action by step
                if assistant_step not in step_actions:
                    step_actions[assistant_step] = Counter()
                step_actions[assistant_step][full_action] += 1
                
                # Update final step action (will be the last one processed)
                final_step_action = full_action
        
        return {
            "tools": tool_counts,
            "commands": command_counts,
            "step_actions": step_actions,
            "final_step_action": final_step_action
        }

    def analyze_folder(self, folder_path: str, use_cache: bool = True) -> Dict[str, Counter]:
        """Analyze all trajectory files in a folder."""
        folder_path = str(Path(folder_path).resolve())
        
        # Check cache first
        if use_cache:
            cached_data = self._load_cache(folder_path)
            if cached_data:
                print(f"Using cached results for {folder_path}")
                return cached_data

        print(f"Analyzing folder: {folder_path}")
        
        # Find trajectory files
        traj_files = self._find_trajectory_files(folder_path)
        if not traj_files:
            print(f"No trajectory files found in {folder_path}")
            return {"tools": Counter(), "commands": Counter(), "step_actions": {}, "final_step_actions": Counter()}
        
        print(f"Found {len(traj_files)} trajectory files")
        
        # Initialize counters
        total_tool_counts = Counter()
        total_command_counts = Counter()
        total_step_actions = {}  # step_number -> Counter of actions across all files
        final_step_actions = Counter()  # Counter of all final step actions across trajectories
        
        # Process files in batches to avoid memory issues
        batch_size = 50
        for i in tqdm(range(0, len(traj_files), batch_size), desc="Processing batches"):
            batch_files = traj_files[i:i+batch_size]
            
            for traj_file in batch_files:
                traj_data = self._load_trajectory_file(traj_file)
                if not traj_data:
                    continue
                
                messages = traj_data.get("history", [])
                if not messages:
                    continue
                
                analysis = self._analyze_messages(messages)
                total_tool_counts.update(analysis["tools"])
                total_command_counts.update(analysis["commands"])
                
                # Aggregate step actions
                for step, actions in analysis["step_actions"].items():
                    if step not in total_step_actions:
                        total_step_actions[step] = Counter()
                    total_step_actions[step].update(actions)
                
                # Aggregate final step actions
                if analysis["final_step_action"]:
                    final_step_actions[analysis["final_step_action"]] += 1
        
        result = {
            "tools": total_tool_counts,
            "commands": total_command_counts,
            "step_actions": total_step_actions,
            "final_step_actions": final_step_actions
        }
        
        # Save to cache
        if use_cache:
            self._save_cache(folder_path, result)
        
        return result

    def _get_short_folder_name(self, folder_path: str) -> str:
        """Extract a short, readable name from the folder path."""
        folder_name = Path(folder_path).name
        
        # Extract key components from the long folder name
        if "SWE-bench" in folder_name:
            return "SWE-bench-32B"
        elif "lr1e-4_ep3" in folder_name:
            if "as1" in folder_name:
                return "Fine-tuned-lr1e4-ep3-as1"
            elif "as2" in folder_name:
                return "Fine-tuned-lr1e4-ep3-as2"
            elif "as3" in folder_name:
                return "Fine-tuned-lr1e4-ep3-as3"
            elif "as4" in folder_name:
                return "Fine-tuned-lr1e4-ep3-as4"
        elif "lr1e-5_ep1" in folder_name:
            return "Fine-tuned-lr1e5-ep1"
        elif "lr1e-5_ep3" in folder_name:
            return "Fine-tuned-lr1e5-ep3"
        elif "lr5e-5_ep1" in folder_name:
            return "Fine-tuned-lr5e5-ep1"
        elif "lr5e-5_ep3" in folder_name:
            return "Fine-tuned-lr5e5-ep3"
        
        # Fallback to last part of path
        return folder_name[:30] + "..." if len(folder_name) > 30 else folder_name

    def _create_step_action_plot(self, ax, step_actions_data, final_step_actions, short_name, action_colors):
        """Create step-wise action distribution plot."""
        if not step_actions_data and not final_step_actions:
            ax.text(0.5, 0.5, 'No step data', ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'Step Actions: {short_name}', fontsize=14, fontweight='bold')
            return
        
        # Get key steps: 1, 2, 3, 4, and final step
        # Use steps 1-4 if they exist, plus final step
        available_steps = list(step_actions_data.keys()) if step_actions_data else []
        key_steps = []
        for step in [1, 2, 3, 4]:
            if step in available_steps:
                key_steps.append(step)
        
        # Add final step if we have final step data
        if final_step_actions:
            key_steps.append("final")
        
        # Prepare data for plotting
        step_labels = []
        action_data = []
        
        for step in key_steps:
            if step == "final":
                # Handle final step data
                step_actions = final_step_actions
                # Get top 5 actions for final step
                top_actions = dict(step_actions.most_common(5))
                total_actions = sum(step_actions.values())
                
                # Calculate percentages
                percentages = {}
                for action, count in top_actions.items():
                    percentages[action] = (count / total_actions) * 100
                
                step_label = "Final Step"
                step_labels.append(step_label)
                action_data.append(percentages)
            elif step in step_actions_data:
                step_actions = step_actions_data[step]
                # Get top 5 actions for this step
                top_actions = dict(step_actions.most_common(5))
                total_actions = sum(step_actions.values())
                
                # Calculate percentages
                percentages = {}
                for action, count in top_actions.items():
                    percentages[action] = (count / total_actions) * 100
                
                step_label = f"Step {step}"
                step_labels.append(step_label)
                action_data.append(percentages)
        
        if not action_data:
            ax.text(0.5, 0.5, 'No valid step data', ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'Step Actions: {short_name}', fontsize=14, fontweight='bold')
            return
        
        # Create stacked bar chart
        all_actions = set()
        for step_data in action_data:
            all_actions.update(step_data.keys())
        all_actions = sorted(list(all_actions))
        
        # Prepare data matrix
        data_matrix = []
        for action in all_actions:
            action_percentages = []
            for step_data in action_data:
                action_percentages.append(step_data.get(action, 0))
            data_matrix.append(action_percentages)
        
        # Plot stacked bars
        bottom = [0] * len(step_labels)
        # Use consistent colors from the global action_colors mapping
        colors = [action_colors.get(action, '#888888') for action in all_actions]
        
        for i, (action, percentages) in enumerate(zip(all_actions, data_matrix)):
            bars = ax.bar(step_labels, percentages, bottom=bottom, label=action, color=colors[i], width=0.6)
            
            # Add percentage labels for significant portions (>10%)
            for j, (bar, percentage) in enumerate(zip(bars, percentages)):
                if percentage > 10:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, bottom[j] + height/2, 
                           f'{percentage:.0f}%', ha='center', va='center', 
                           fontsize=9, fontweight='bold', color='white')
            
            # Update bottom for next stack
            bottom = [b + p for b, p in zip(bottom, percentages)]
        
        ax.set_title(f'Action Distribution by Step: {short_name}', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add legend with smaller font and better positioning
        if len(all_actions) <= 10:  # Show legend for reasonable number of actions
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, 
                     frameon=True, fancybox=True, shadow=True, ncol=1)
        
        # Improve tick labels
        ax.tick_params(axis='both', labelsize=10)

    def create_visualization(self, folder_results: Dict[str, Dict[str, Counter]], output_dir: str = "plots"):
        """Create visualization plots for analysis results."""
        if not HAS_PLOTTING:
            print("Plotting libraries not available. Skipping visualization.")
            return
            
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        num_folders = len(folder_results)
        if num_folders == 0:
            print("No results to plot")
            return
        
        # Create consistent color mappings for tools, commands, and actions across all figures
        all_tools = set()
        all_commands = set()
        all_actions = set()
        for results in folder_results.values():
            all_tools.update(results["tools"].keys())
            all_commands.update(results["commands"].keys())
            # Collect all unique actions from step_actions
            step_actions_data = results.get("step_actions", {})
            for step_actions in step_actions_data.values():
                all_actions.update(step_actions.keys())
        
        # Create color mappings
        tool_colors = {}
        command_colors = {}
        action_colors = {}
        tool_colormap = plt.cm.Set3
        command_colormap = plt.cm.tab20
        
        for i, tool in enumerate(sorted(all_tools)):
            tool_colors[tool] = tool_colormap(i / max(1, len(all_tools)-1))
        
        for i, command in enumerate(sorted(all_commands)):
            command_colors[command] = command_colormap(i / max(1, len(all_commands)-1))
        
        # Use more distinguishable colors - combine multiple colormaps for actions
        color_palettes = [plt.cm.Set1, plt.cm.Set2, plt.cm.Set3, plt.cm.Dark2, plt.cm.Paired]
        for i, action in enumerate(sorted(all_actions)):
            palette_idx = i // 8  # Switch palette every 8 colors
            color_idx = i % 8
            palette = color_palettes[palette_idx % len(color_palettes)]
            action_colors[action] = palette(color_idx / 7)  # Divide by 7 to get good spread
        
        
        # Create subplots with 3 columns: tools, commands, step actions
        fig_height = max(8, 4 * num_folders)
        # Use gridspec to control column widths - make third column narrower
        from matplotlib import gridspec
        fig = plt.figure(figsize=(24, fig_height))
        gs = gridspec.GridSpec(num_folders, 3, width_ratios=[1, 1, 0.8], hspace=0.5, wspace=0.4, top=0.93)
        
        # Create axes manually with gridspec
        axes = []
        for i in range(num_folders):
            row_axes = []
            for j in range(3):
                ax = fig.add_subplot(gs[i, j])
                row_axes.append(ax)
            axes.append(row_axes)
        
        # Convert to array for easier indexing
        axes = np.array(axes)
        if num_folders == 1:
            axes = axes.reshape(1, -1)
        
        # Create a single title for the entire figure with better positioning
        fig.suptitle('Trajectory Analysis Results', fontsize=20, fontweight='bold', y=0.97)
        
        for i, (folder_path, results) in enumerate(folder_results.items()):
            short_name = self._get_short_folder_name(folder_path)
            
            # Plot tool usage
            ax_tools = axes[i][0]
            ax_commands = axes[i][1] 
            ax_steps = axes[i][2]
                
            tools_data = results["tools"]
            if tools_data:
                tools_df = pd.DataFrame(list(tools_data.items()), columns=['Tool', 'Count'])
                tools_df = tools_df.sort_values('Count', ascending=True)
                
                # Use consistent colors across all figures
                colors = [tool_colors[tool] for tool in tools_df['Tool']]
                bars = ax_tools.barh(tools_df['Tool'], tools_df['Count'], color=colors)
                ax_tools.set_title(f'Tool Usage: {short_name}', fontsize=14, fontweight='bold', pad=20)
                ax_tools.set_xlabel('Frequency', fontsize=12)
                ax_tools.grid(axis='x', alpha=0.3)
                
                # Remove top and right spines
                ax_tools.spines['top'].set_visible(False)
                ax_tools.spines['right'].set_visible(False)
                
                # Add value labels on bars
                for bar in bars:
                    width = bar.get_width()
                    ax_tools.text(width + max(tools_df['Count']) * 0.01, bar.get_y() + bar.get_height()/2, 
                                f'{int(width)}', ha='left', va='center', fontsize=11, fontweight='bold')
                
                # Improve tick labels
                ax_tools.tick_params(axis='y', labelsize=11)
                ax_tools.tick_params(axis='x', labelsize=10)
            else:
                ax_tools.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_tools.transAxes, fontsize=14)
                ax_tools.set_title(f'Tool Usage: {short_name}', fontsize=14, fontweight='bold')
            
            # Plot command usage (top 12)
            commands_data = results["commands"]
            if commands_data:
                # Get top 12 commands for better readability
                top_commands = dict(commands_data.most_common(12))
                commands_df = pd.DataFrame(list(top_commands.items()), columns=['Command', 'Count'])
                commands_df = commands_df.sort_values('Count', ascending=True)
                
                # Use consistent colors across all figures
                colors = [command_colors[command] for command in commands_df['Command']]
                bars = ax_commands.barh(commands_df['Command'], commands_df['Count'], color=colors)
                ax_commands.set_title(f'Top Commands: {short_name}', fontsize=14, fontweight='bold', pad=20)
                ax_commands.set_xlabel('Frequency', fontsize=12)
                ax_commands.grid(axis='x', alpha=0.3)
                
                # Remove top and right spines
                ax_commands.spines['top'].set_visible(False)
                ax_commands.spines['right'].set_visible(False)
                
                # Add value labels on bars
                for bar in bars:
                    width = bar.get_width()
                    ax_commands.text(width + max(commands_df['Count']) * 0.01, bar.get_y() + bar.get_height()/2, 
                                   f'{int(width)}', ha='left', va='center', fontsize=10, fontweight='bold')
                
                # Improve tick labels
                ax_commands.tick_params(axis='y', labelsize=10)
                ax_commands.tick_params(axis='x', labelsize=10)
            else:
                ax_commands.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_commands.transAxes, fontsize=14)
                ax_commands.set_title(f'Commands: {short_name}', fontsize=14, fontweight='bold')
            
            # Plot step-wise action distribution
            step_actions_data = results.get("step_actions", {})
            final_step_actions = results.get("final_step_actions", Counter())
            self._create_step_action_plot(ax_steps, step_actions_data, final_step_actions, short_name, action_colors)
        
        # Save plot with higher quality
        output_file = output_dir / "trajectory_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to {output_file}")
        
        plt.show()

    def print_summary(self, folder_results: Dict[str, Dict[str, Counter]]):
        """Print summary statistics."""
        print("\n" + "="*80)
        print("TRAJECTORY ANALYSIS SUMMARY")
        print("="*80)
        
        for folder_path, results in folder_results.items():
            folder_name = Path(folder_path).name
            tools_data = results["tools"]
            commands_data = results["commands"]
            step_actions_data = results.get("step_actions", {})
            final_step_actions = results.get("final_step_actions", Counter())
            
            print(f"\nFolder: {folder_name}")
            print("-" * 60)
            
            if tools_data:
                total_tool_calls = sum(tools_data.values())
                print(f"Total tool calls: {total_tool_calls}")
                print(f"Unique tools used: {len(tools_data)}")
                
                print("\nTool usage breakdown:")
                for tool, count in tools_data.most_common():
                    percentage = (count / total_tool_calls) * 100
                    print(f"  {tool:20} {count:6} ({percentage:5.1f}%)")
                
                if commands_data:
                    print(f"\nTop 10 commands:")
                    for command, count in commands_data.most_common(10):
                        print(f"  {command:30} {count:6}")
                
                if step_actions_data or final_step_actions:
                    print(f"\nStep-wise action analysis:")
                    
                    # Show steps 1-4 if available
                    for step in [1, 2, 3, 4]:
                        if step in step_actions_data:
                            step_actions = step_actions_data[step]
                            total_step_actions = sum(step_actions.values())
                            print(f"  Step {step}: {total_step_actions} total actions")
                            
                            # Show top 3 actions for this step
                            for action, count in step_actions.most_common(3):
                                percentage = (count / total_step_actions) * 100
                                print(f"    {action:25} {count:4} ({percentage:4.1f}%)")
                    
                    # Show final step actions
                    if final_step_actions:
                        total_final_actions = sum(final_step_actions.values())
                        print(f"  Final Step: {total_final_actions} total actions (across all trajectories)")
                        
                        # Show top 3 final actions
                        for action, count in final_step_actions.most_common(3):
                            percentage = (count / total_final_actions) * 100
                            print(f"    {action:25} {count:4} ({percentage:4.1f}%)")
            else:
                print("No tool usage data found")

def main():
    """Main function to run trajectory analysis."""
    # Default folders to analyze
    default_folders = [
        "trajectories/zhengyanshi@microsoft.com/swesmith_infer__openai--SWE-bench--SWE-agent-LM-32B__t-0.00__p-1.00__c-0.00___patch_swesmith_filtered_swesmith_task__ms75_as1_sanity_check",
        "trajectories/zhengyanshi@microsoft.com/swesmith_infer__openai--tt_SWE-agent-LM-32B_cl32768_lr1e-4_ep3_astropy__astropy.26d14786_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o__t-0.00__p-1.00__c-0.00___patch_swesmith_filtered_swesmith_task__ms75_as1_sanity_check",
        "trajectories/zhengyanshi@microsoft.com/swesmith_infer__openai--tt_SWE-agent-LM-32B_cl32768_lr1e-4_ep3_astropy__astropy.26d14786_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o__t-0.00__p-1.00__c-0.00___patch_swesmith_filtered_swesmith_task__ms75_as2_sanity_check",
        "trajectories/zhengyanshi@microsoft.com/swesmith_infer__openai--tt_SWE-agent-LM-32B_cl32768_lr1e-4_ep3_astropy__astropy.26d14786_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o__t-0.00__p-1.00__c-0.00___patch_swesmith_filtered_swesmith_task__ms75_as3_sanity_check",
        "trajectories/zhengyanshi@microsoft.com/swesmith_infer__openai--tt_SWE-agent-LM-32B_cl32768_lr1e-4_ep3_astropy__astropy.26d14786_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o__t-0.00__p-1.00__c-0.00___patch_swesmith_filtered_swesmith_task__ms75_as4_sanity_check",
        "trajectories/zhengyanshi@microsoft.com/swesmith_infer__openai--tt_SWE-agent-LM-32B_cl32768_lr1e-5_ep1_astropy__astropy.26d14786_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o__t-0.00__p-1.00__c-0.00___patch_swesmith_filtered_swesmith_task__ms75_as1_sanity_check",
        "trajectories/zhengyanshi@microsoft.com/swesmith_infer__openai--tt_SWE-agent-LM-32B_cl32768_lr1e-5_ep3_astropy__astropy.26d14786_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o__t-0.00__p-1.00__c-0.00___patch_swesmith_filtered_swesmith_task__ms75_as1_sanity_check",
        "trajectories/zhengyanshi@microsoft.com/swesmith_infer__openai--tt_SWE-agent-LM-32B_cl32768_lr5e-5_ep1_astropy__astropy.26d14786_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o__t-0.00__p-1.00__c-0.00___patch_swesmith_filtered_swesmith_task__ms75_as1_sanity_check",
        "trajectories/zhengyanshi@microsoft.com/swesmith_infer__openai--tt_SWE-agent-LM-32B_cl32768_lr5e-5_ep3_astropy__astropy.26d14786_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o__t-0.00__p-1.00__c-0.00___patch_swesmith_filtered_swesmith_task__ms75_as1_sanity_check"
    ]
    
    parser = argparse.ArgumentParser(description="Analyze trajectory files for tool usage frequency")
    parser.add_argument("folders", nargs="*", default=default_folders, help="Folder paths containing trajectory files")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--cache-dir", default="cache", help="Cache directory")
    parser.add_argument("--output-dir", default="plots", help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = TrajectoryAnalyzer(cache_dir=args.cache_dir)
    
    # Process each folder
    folder_results = {}
    for folder_path in args.folders:
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist")
            continue
        
        results = analyzer.analyze_folder(folder_path, use_cache=not args.no_cache)
        folder_results[folder_path] = results
    
    if not folder_results:
        print("No valid folders to analyze")
        return
    
    # Create visualizations
    analyzer.create_visualization(folder_results, args.output_dir)
    
    # Print summary
    analyzer.print_summary(folder_results)

if __name__ == "__main__":
    main()