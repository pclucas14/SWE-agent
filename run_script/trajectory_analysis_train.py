#!/usr/bin/env python3
"""
Trajectory Analysis Tool for New Format

This script analyzes trajectory files from SWE-agent runs to extract tool usage frequency.
It processes assistant messages and parses actions from message content rather than explicit action fields.

Usage:
    python trajectory_analysis_train.py [folder_path1 folder_path2 ...]

Features:
- Loads trajectory files (.traj) and JSON files from multiple folders
- Extracts assistant messages and parses actions from content using function call format
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
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
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

# tqdm removed - not needed for this script

# XML string replacement patterns (reverse engineered from tool_call_to_action)
XML_STR_REPLACES = {
    'command', 'code', 'content', 'file_text', 'new_str', 'old_str'
}

class TrainTrajectoryAnalyzer:
    """Main class for analyzing new format trajectory files and extracting tool usage patterns."""
    
    def __init__(self, cache_dir: str = "cache"):
        """Initialize analyzer with cache directory."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Define known tools from function calls
        self.known_tools = {
            'bash', 'submit', 'str_replace_editor', 'system', 'python'
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
        return self.cache_dir / f"train_{folder_name}_{folder_hash}.pkl"

    def _is_system_message(self, message: dict) -> str:
        """Check if message is a system message and return the type."""
        if not message:
            return None
        
        content = message.get('content', '').strip().lower()
        
        # Check for system message patterns
        if any(pattern in content for pattern in [
            'reached maximum steps limit of 75',
            'maximum steps limit of 75 reached'
        ]):
            return 'max_step_limit'
        
        if any(pattern in content for pattern in [
            'exit due to context window',
            'context window exceeded',
            'context limit reached'
        ]):
            return 'context_window_limit'
            
        if any(pattern in content for pattern in [
            'exit due to total execution time exceeded',
            'total execution time exceeded',
            'execution time limit reached'
        ]):
            return 'execution_time_limit'
            
        if any(pattern in content for pattern in [
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
                    cached_data = pickle.load(f)
                    
                    # Backward compatibility: add missing fields if not present
                    missing_fields = []
                    if "action_transitions" not in cached_data:
                        missing_fields.append("action_transitions")
                    if "combined_commands" not in cached_data:
                        missing_fields.append("combined_commands")
                    
                    if missing_fields:
                        print(f"Note: Cache for {folder_path} is missing {missing_fields}, will regenerate")
                        return {}  # Force regeneration for complete data
                    
                    return cached_data
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


    def _load_trajectory_file(self, file_path: str) -> dict:
        """Load a single trajectory file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Check if it's a JSONL file (each line is a JSON object)
                if file_path.endswith('.jsonl') or file_path.endswith('.json'):
                    first_line = f.readline().strip()
                    if first_line:
                        # Reset file pointer
                        f.seek(0)
                        # Try to parse as JSONL first
                        try:
                            # Read all lines and parse each as JSON
                            trajectories = []
                            for line in f:
                                line = line.strip()
                                if line:
                                    trajectory_data = json.loads(line)
                                    trajectories.append(trajectory_data)
                            
                            if trajectories:
                                return {"trajectories": trajectories}
                        except json.JSONDecodeError:
                            # Not JSONL format, try regular JSON
                            f.seek(0)
                            data = json.load(f)
                            
                            # Handle different file formats
                            if isinstance(data, list):
                                # Handle JSON list of trajectories
                                return {"trajectories": data}
                            elif "history" in data:
                                # Standard trajectory format
                                return data
                            elif "messages" in data:
                                # Dataset format with messages
                                return {"trajectories": [data]}
                            else:
                                # Unknown format, assume it's a single trajectory
                                return {"trajectories": [data]}
                    else:
                        # Empty file
                        return {}
                else:
                    # Regular JSON file
                    data = json.load(f)
                    
                    # Handle different file formats
                    if isinstance(data, list):
                        # Handle JSON list of trajectories
                        return {"trajectories": data}
                    elif "history" in data:
                        # Standard trajectory format
                        return data
                    elif "messages" in data:
                        # Dataset format with messages
                        return {"trajectories": [data]}
                    else:
                        # Unknown format, assume it's a single trajectory
                        return {"trajectories": [data]}
                        
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
            return {}

    def _parse_function_call_from_content(self, content: str) -> List[str]:
        """
        Parse function calls from assistant message content.
        Reverse engineers the tool_call_to_action function to extract function calls.
        
        Returns:
            List of action strings in the original format
        """
        actions = []
        
        if not content or not isinstance(content, str):
            return []
        
        # Find all function calls using regex
        # Pattern: <function=name>...</function>
        function_pattern = r'<function=([^>]+)>(.*?)</function>'
        function_matches = re.findall(function_pattern, content, re.DOTALL)
        
        for function_name, function_content in function_matches:
            # Parse parameters within the function
            parameter_pattern = r'<parameter=([^>]+)>(.*?)</parameter>'
            parameter_matches = re.findall(parameter_pattern, function_content, re.DOTALL)
            
            # Build action string
            if function_name == 'bash':
                # For bash commands, extract the command parameter
                for param_name, param_value in parameter_matches:
                    if param_name == 'command':
                        actions.append(param_value.strip())
                        break
                else:
                    # No command parameter found, use function name
                    actions.append(function_name)
            elif function_name == 'str_replace_editor':
                # For str_replace_editor, extract the command parameter
                for param_name, param_value in parameter_matches:
                    if param_name == 'command':
                        actions.append(f"str_replace_editor {param_value.strip()}")
                        break
                else:
                    # No command parameter found, use function name
                    actions.append(function_name)
            elif function_name == 'submit':
                actions.append('submit')
            else:
                # For other functions, use the function name
                actions.append(function_name)
        
        return actions

    def _parse_action(self, content: str, message: dict = None) -> Tuple[str, str]:
        """Parse action from message content to extract tool and command."""
        # First try to parse function calls from content
        actions = self._parse_function_call_from_content(content)
        
        if not actions:
            # Check if this is a system message
            system_type = self._is_system_message(message) if message else None
            if system_type:
                return "system", system_type
            
            return "other", "no_function_call"
        
        # Take the first action (usually there's only one per message)
        action = actions[0]
        
        if not action or not isinstance(action, str):
            return "other", "empty_action"
        
        action = action.strip()
        
        # Handle bash commands
        action_lower = action.lower()
        if not any(action_lower.startswith(tool) for tool in self.known_tools):
            # This is likely a bash command
            # Extract the first command from bash
            command_parts = action.split()
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

    def _parse_actions_with_multi_commands(self, content: str, message: dict = None) -> List[Tuple[str, str, str]]:
        """
        Parse action from message content and return both individual commands and combined action type.
        
        Returns:
            List of tuples: (tool, command, action_type)
            - action_type can be 'single' for single commands or 'multi' for multi-command actions
        """
        # First try to parse function calls from content
        actions = self._parse_function_call_from_content(content)
        
        if not actions:
            # Check if this is a system message
            system_type = self._is_system_message(message) if message else None
            if system_type:
                return [("system", system_type, "single")]
            
            return [("other", "no_function_call", "single")]
        
        # Take the first action (usually there's only one per message)
        action = actions[0]
        
        if not action or not isinstance(action, str):
            return [("other", "empty_action", "single")]
        
        action = action.strip()
        
        # Check if this is a bash command with multiple '&&' operators
        if '&&' in action and action.count('&&') >= 1:
            # This is a multi-command action
            individual_commands = []
            
            # Split by && and process each command
            commands = [cmd.strip() for cmd in action.split('&&')]
            
            for cmd in commands:
                if cmd:
                    # Parse each individual command
                    cmd_parts = cmd.split()
                    if cmd_parts:
                        first_cmd = cmd_parts[0]
                        # Handle complex commands with pipes, redirects, etc.
                        first_cmd = re.split(r'[|;&><]', first_cmd)[0].strip()
                        # Remove common prefixes
                        first_cmd = first_cmd.split('/')[-1]  # Get basename
                        individual_commands.append(("bash", first_cmd, "multi"))
            
            return individual_commands if individual_commands else [("bash", "empty_multi", "multi")]
        
        # Handle single commands (existing logic)
        action_lower = action.lower()
        if not any(action_lower.startswith(tool) for tool in self.known_tools):
            # This is likely a bash command
            # Extract the first command from bash
            command_parts = action.split()
            if command_parts:
                first_cmd = command_parts[0]
                # Handle complex commands with pipes, redirects, etc.
                first_cmd = re.split(r'[|;&><]', first_cmd)[0].strip()
                return [("bash", first_cmd, "single")]
            return [("bash", "empty_bash", "single")]
        
        # Handle str_replace_editor commands
        if action.startswith('str_replace_editor '):
            parts = action.split()
            if len(parts) >= 2:
                sub_command = parts[1]
                if sub_command in self.str_replace_commands:
                    return [("str_replace_editor", sub_command, "single")]
                else:
                    # Try to parse non-standard commands
                    return [("str_replace_editor", f"other_{sub_command}", "single")]
            return [("str_replace_editor", "no_subcommand", "single")]
        
        # Handle submit (case insensitive)
        if action.lower().startswith('submit'):
            return [("submit", "submit", "single")]
        
        # Handle other known tools
        for tool in self.known_tools:
            if action.startswith(tool):
                return [(tool, tool, "single")]
        
        # Try to identify other common patterns
        if 'python' in action.lower():
            return [("bash", "python", "single")]
        if 'git' in action.lower():
            return [("bash", "git", "single")]
        if 'pip' in action.lower():
            return [("bash", "pip", "single")]
        if 'cd' in action.lower():
            return [("bash", "cd", "single")]
        if 'ls' in action.lower():
            return [("bash", "ls", "single")]
        
        # Last resort - categorize as other with first word
        first_word = action.split()[0] if action.split() else "empty"
        return [("other", first_word, "single")]

    def _analyze_messages(self, messages: List[dict]) -> Dict[str, Counter]:
        """Analyze assistant messages to extract tool usage."""
        tool_counts = Counter()
        command_counts = Counter()
        combined_command_counts = Counter()  # New: Track combined multi-command actions
        step_actions = {}  # step_number -> Counter of actions
        action_transitions = Counter()  # (action1, action2) -> count
        final_step_action = None  # Track the final step action for this trajectory
        
        assistant_step = 0
        previous_action = None
        
        for message in messages:
            if message.get("role") == "assistant":
                assistant_step += 1
                content = message.get("content", "")
                
                # Parse actions from content (new method for multi-command support)
                parsed_actions = self._parse_actions_with_multi_commands(content, message)
                
                # Process each parsed action
                for tool, command, action_type in parsed_actions:
                    # Count individual tools and commands for the first figure
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
                    
                    # Update final step action
                    final_step_action = full_action
                
                # For multi-command actions, also track the combined action type
                if len(parsed_actions) > 1 and all(action_type == "multi" for _, _, action_type in parsed_actions):
                    # This is a multi-command action - create a combined action description
                    command_names = [command for _, command, _ in parsed_actions]
                    combined_action = f"multi: {' && '.join(command_names)}"
                    combined_command_counts[combined_action] += 1
                
                # Track action transitions (using the first action for transitions)
                if parsed_actions and previous_action is not None:
                    first_full_action = parsed_actions[0]
                    tool, command, _ = first_full_action
                    if tool == "bash":
                        current_action = f"bash {command}"
                    elif tool == "str_replace_editor":
                        current_action = f"str_replace_editor {command}"
                    elif tool == "system":
                        current_action = f"system {command}"
                    else:
                        current_action = command
                    
                    transition = (previous_action, current_action)
                    action_transitions[transition] += 1
                    previous_action = current_action
                elif parsed_actions:
                    # Set previous action for first message
                    tool, command, _ = parsed_actions[0]
                    if tool == "bash":
                        previous_action = f"bash {command}"
                    elif tool == "str_replace_editor":
                        previous_action = f"str_replace_editor {command}"
                    elif tool == "system":
                        previous_action = f"system {command}"
                    else:
                        previous_action = command
        
        return {
            "tools": tool_counts,
            "commands": command_counts,
            "combined_commands": combined_command_counts,  # New field
            "step_actions": step_actions,
            "action_transitions": action_transitions,
            "final_step_action": final_step_action
        }

    def _analyze_trajectories_from_data(self, data: dict) -> Dict[str, Counter]:
        """Analyze trajectories from loaded data structure."""
        total_tool_counts = Counter()
        total_command_counts = Counter()
        total_combined_command_counts = Counter()  # New: Track combined commands
        total_step_actions = {}
        total_action_transitions = Counter()
        final_step_actions = Counter()
        
        # Handle different data formats
        if "trajectories" in data:
            # Multiple trajectories in list format
            trajectories = data["trajectories"]
        elif "history" in data:
            # Single trajectory with history
            trajectories = [data]
        else:
            # Assume data is a single trajectory
            trajectories = [data]
        
        for traj_data in trajectories:
            # Get messages from trajectory - simplified access
            messages = traj_data.get("messages", [])
            
            # Fallback to other formats if messages not found
            if not messages:
                messages = traj_data.get("history", [])
            
            if not messages:
                continue
            
            analysis = self._analyze_messages(messages)
            total_tool_counts.update(analysis["tools"])
            total_command_counts.update(analysis["commands"])
            total_combined_command_counts.update(analysis["combined_commands"])  # New
            total_action_transitions.update(analysis["action_transitions"])
            
            # Aggregate step actions
            for step, actions in analysis["step_actions"].items():
                if step not in total_step_actions:
                    total_step_actions[step] = Counter()
                total_step_actions[step].update(actions)
            
            # Aggregate final step actions
            if analysis["final_step_action"]:
                final_step_actions[analysis["final_step_action"]] += 1
        
        return {
            "tools": total_tool_counts,
            "commands": total_command_counts,
            "combined_commands": total_combined_command_counts,  # New field
            "step_actions": total_step_actions,
            "action_transitions": total_action_transitions,
            "final_step_actions": final_step_actions
        }

    def analyze_file(self, file_path: str, use_cache: bool = True) -> Dict[str, Counter]:
        """Analyze a trajectory file."""
        file_path = str(Path(file_path).resolve())
        
        # Check cache first
        if use_cache:
            cached_data = self._load_cache(file_path)
            if cached_data:
                print(f"Using cached results for {file_path}")
                return cached_data

        print(f"Analyzing: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return {"tools": Counter(), "commands": Counter(), "combined_commands": Counter(), "step_actions": {}, "action_transitions": Counter(), "final_step_actions": Counter()}
        
        # Load and analyze the trajectory file
        traj_data = self._load_trajectory_file(file_path)
        if not traj_data:
            print(f"Failed to load trajectory data from {file_path}")
            return {"tools": Counter(), "commands": Counter(), "combined_commands": Counter(), "step_actions": {}, "action_transitions": Counter(), "final_step_actions": Counter()}
        
        analysis = self._analyze_trajectories_from_data(traj_data)
        
        # Save to cache
        if use_cache:
            self._save_cache(file_path, analysis)
        
        return analysis

    def _get_short_folder_name(self, folder_path: str) -> str:
        """Extract a short, readable name from the folder path."""
        path = Path(folder_path)
        file_name = path.name
        
        # Special case for swe-smith-trajectories.jsonl
        if "swe-smith-trajectories" in file_name:
            return "SWE-Smith-Trajectories"
        
        # Extract meaningful info from parent folder path
        parent_folder = path.parent.name
        
        # Extract model information from parent folder
        if "claude-sonnet-4__" in parent_folder:
            return "Claude-Sonnet-4"
        elif "claude-sonnet-4_gpt4.1_gpt-4o" in parent_folder:
            return "Claude-Sonnet-4 + GPT-4.1 + GPT-4o"
        elif "gpt-4.1__" in parent_folder:
            return "GPT-4.1"
        elif "gpt-4o__" in parent_folder:
            return "GPT-4o"
        else:
            # Try to extract from the full path for more context
            full_path = str(folder_path)
            if "claude-sonnet-4_gpt4.1_gpt-4o" in full_path:
                return "Claude-Sonnet-4 + GPT-4.1 + GPT-4o"
            elif "claude-sonnet-4" in full_path:
                return "Claude-Sonnet-4"
            elif "gpt-4.1" in full_path:
                return "GPT-4.1"
            elif "gpt-4o" in full_path:
                return "GPT-4o"
        
        # Fallback - use truncated parent folder name
        return parent_folder[:25] + "..." if len(parent_folder) > 25 else parent_folder

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
        
        ax.set_title(f'Step Dist: {short_name}', fontsize=14, fontweight='bold', pad=20)
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

    def _create_action_transition_plot(self, ax, action_transitions, short_name):
        """Create action transition heatmap/distribution plot."""
        if not action_transitions:
            ax.text(0.5, 0.5, 'No transition data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(f'Action Transitions: {short_name}', fontsize=14, fontweight='bold')
            return
        
        # Get top 15 most frequent transitions for better readability
        top_transitions = dict(action_transitions.most_common(15))
        
        if not top_transitions:
            ax.text(0.5, 0.5, 'No transition data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(f'Action Transitions: {short_name}', fontsize=14, fontweight='bold')
            return
        
        # Prepare data for plotting - sort in descending order
        sorted_items = sorted(top_transitions.items(), key=lambda x: x[1], reverse=True)
        total_transitions = sum(action_transitions.values())
        
        # Create labels showing transition and frequency - DO NOT truncate, show full labels
        labels = []
        percentages = []
        counts = []
        
        for (from_action, to_action), count in sorted_items:
            # Show FULL action names without truncation
            percentage = (count / total_transitions) * 100
            percentages.append(percentage)
            counts.append(count)
            labels.append(f"{from_action} → {to_action}")
        
        # Create horizontal bar chart
        y_pos = range(len(labels))
        colors = plt.cm.viridis([p/max(percentages) for p in percentages])
        bars = ax.barh(y_pos, percentages, color=colors)
        
        # Customize the plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7)  # Smaller font for full labels
        ax.set_xlabel('Percentage of Total Transitions (%)', fontsize=10)
        ax.set_title(f'Top Action Transitions: {short_name}', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3)
        
        # Put percentage labels INSIDE the bars on the right side for better readability
        for bar, percentage, count in zip(bars, percentages, counts):
            width = bar.get_width()
            # Place text inside the bar, towards the right end
            ax.text(width * 0.95, bar.get_y() + bar.get_height()/2, 
                   f'{percentage:.1f}% ({count})', ha='right', va='center', 
                   fontsize=8, fontweight='bold', color='white')
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Improve tick labels
        ax.tick_params(axis='both', labelsize=8)

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
        
        # Ensure consistent color mapping for tools across all figures
        sorted_tools = sorted(all_tools)
        for i, tool in enumerate(sorted_tools):
            tool_colors[tool] = tool_colormap(i / max(1, len(sorted_tools)-1))
        
        # Ensure consistent color mapping for commands across all figures
        sorted_commands = sorted(all_commands)
        for i, command in enumerate(sorted_commands):
            command_colors[command] = command_colormap(i / max(1, len(sorted_commands)-1))
        
        # Use more distinguishable colors - combine multiple colormaps for actions
        color_palettes = [plt.cm.Set1, plt.cm.Set2, plt.cm.Set3, plt.cm.Dark2, plt.cm.Paired]
        sorted_actions = sorted(all_actions)
        for i, action in enumerate(sorted_actions):
            palette_idx = i // 8  # Switch palette every 8 colors
            color_idx = i % 8
            palette = color_palettes[palette_idx % len(color_palettes)]
            action_colors[action] = palette(color_idx / 7)  # Divide by 7 to get good spread
        
        
        # Create subplots with 5 columns: tools, commands, combined commands, step actions, action transitions
        fig_height = max(8, 4 * num_folders)
        # Use gridspec to control column widths and spacing
        from matplotlib import gridspec
        fig = plt.figure(figsize=(55, fig_height))
        # Use 8 columns with spacers for better layout - spacers between major sections
        # Layout: tools | spacer | commands | small_spacer | combined | larger_spacer | step_actions | action_transitions
        gs = gridspec.GridSpec(num_folders, 8, width_ratios=[1, 0.05, 1, 0.005, 1.2, 0.15, 1.4, 2.0], hspace=0.5, wspace=0.4, top=0.93)
        
        # Create axes manually with gridspec, skipping the spacer columns
        axes = []
        for i in range(num_folders):
            row_axes = []
            # Column 0: tools
            ax = fig.add_subplot(gs[i, 0])
            row_axes.append(ax)
            # Column 2: commands (skipping column 1 which is spacer)
            ax = fig.add_subplot(gs[i, 2])
            row_axes.append(ax)
            # Column 4: combined commands (skipping column 3 which is spacer)
            ax = fig.add_subplot(gs[i, 4])
            row_axes.append(ax)
            # Column 6: step actions (skipping column 5 which is spacer)
            ax = fig.add_subplot(gs[i, 6])
            row_axes.append(ax)
            # Column 7: action transitions
            ax = fig.add_subplot(gs[i, 7])
            row_axes.append(ax)
            axes.append(row_axes)
        
        # Convert to array for easier indexing
        axes = np.array(axes)
        if num_folders == 1:
            axes = axes.reshape(1, -1)
        
        # Create a single title for the entire figure with better positioning
        fig.suptitle('New Format Trajectory Analysis with Action Transitions', fontsize=20, fontweight='bold', y=0.97)
        
        for i, (folder_path, results) in enumerate(folder_results.items()):
            short_name = self._get_short_folder_name(folder_path)
            
            # Plot tool usage
            ax_tools = axes[i][0]
            ax_commands = axes[i][1] 
            ax_combined = axes[i][2]
            ax_steps = axes[i][3]
            ax_transitions = axes[i][4]
                
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
            
            # Plot combined command usage (top 10)
            combined_commands_data = results.get("combined_commands", Counter())
            if combined_commands_data:
                # Get top 10 combined commands for better readability
                top_combined = dict(combined_commands_data.most_common(10))
                combined_df = pd.DataFrame(list(top_combined.items()), columns=['Combined Command', 'Count'])
                combined_df = combined_df.sort_values('Count', ascending=True)
                
                # Use distinct colors for combined commands
                colors = plt.cm.tab20([i/len(combined_df) for i in range(len(combined_df))])
                bars = ax_combined.barh(combined_df['Combined Command'], combined_df['Count'], color=colors)
                ax_combined.set_title(f'Multi-Commands: {short_name}', fontsize=14, fontweight='bold', pad=20)
                ax_combined.set_xlabel('Frequency', fontsize=12)
                ax_combined.grid(axis='x', alpha=0.3)
                
                # Remove top and right spines
                ax_combined.spines['top'].set_visible(False)
                ax_combined.spines['right'].set_visible(False)
                
                # Add value labels on bars
                for bar in bars:
                    width = bar.get_width()
                    ax_combined.text(width + max(combined_df['Count']) * 0.01, bar.get_y() + bar.get_height()/2, 
                                   f'{int(width)}', ha='left', va='center', fontsize=10, fontweight='bold')
                
                # Improve tick labels - smaller font for longer labels
                ax_combined.tick_params(axis='y', labelsize=8)
                ax_combined.tick_params(axis='x', labelsize=10)
            else:
                ax_combined.text(0.5, 0.5, 'No multi-commands', ha='center', va='center', transform=ax_combined.transAxes, fontsize=14)
                ax_combined.set_title(f'Multi-Commands: {short_name}', fontsize=14, fontweight='bold')
            
            # Plot step-wise action distribution
            step_actions_data = results.get("step_actions", {})
            final_step_actions = results.get("final_step_actions", Counter())
            self._create_step_action_plot(ax_steps, step_actions_data, final_step_actions, short_name, action_colors)
            
            # Plot action transitions
            action_transitions = results.get("action_transitions", Counter())
            self._create_action_transition_plot(ax_transitions, action_transitions, short_name)
        
        # --- NEW: fine-tune horizontal spacing between 4th (step) and 5th (transition) columns ---
        # separate shifts allow asymmetric tweaking:
        left_shift  = 0.040  # move Step-Actions left  (reduce gap with 3rd col)
        right_shift = 0.030  # move Transitions  right (enlarge gap with 4th col)
        for row_axes in axes:
            ax_steps = row_axes[3]        # 4th column (Step Actions)
            ax_trans = row_axes[4]        # 5th column (Action Transitions)

            # Current positions
            steps_pos = ax_steps.get_position()
            trans_pos = ax_trans.get_position()

            # Move Step Actions left, Transition plot right
            ax_steps.set_position([steps_pos.x0 - left_shift,
                                steps_pos.y0,
                                steps_pos.width,
                                steps_pos.height])

            ax_trans.set_position([trans_pos.x0 + right_shift,
                                trans_pos.y0,
                                trans_pos.width,
                                trans_pos.height])
         # ------------------------------------------------------------------------------

        # Save plot with higher quality
        output_file = output_dir / "train_trajectory_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to {output_file}")
        
        plt.show()

    def print_summary(self, file_results: Dict[str, Dict[str, Counter]]):
        """Print summary statistics."""
        print("\n" + "="*80)
        print("NEW FORMAT TRAJECTORY ANALYSIS SUMMARY")
        print("="*80)
        
        for file_path, results in file_results.items():
            file_name = Path(file_path).name
            tools_data = results["tools"]
            commands_data = results["commands"]
            combined_commands_data = results.get("combined_commands", Counter())
            step_actions_data = results.get("step_actions", {})
            action_transitions = results.get("action_transitions", Counter())
            final_step_actions = results.get("final_step_actions", Counter())
            
            print(f"\nFile: {file_name}")
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
                
                # Show combined commands (multi-command actions)
                if combined_commands_data:
                    total_combined = sum(combined_commands_data.values())
                    print(f"\nMulti-command actions: {total_combined} total")
                    print(f"Top 5 multi-command combinations:")
                    for combined_action, count in combined_commands_data.most_common(5):
                        percentage = (count / total_combined) * 100
                        print(f"  {combined_action:40} {count:4} ({percentage:4.1f}%)")
                
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
                
                # Show action transition analysis
                if action_transitions:
                    total_transitions = sum(action_transitions.values())
                    print(f"\nTop 10 Action Transitions:")
                    print(f"  Total transitions: {total_transitions}")
                    
                    for (from_action, to_action), count in action_transitions.most_common(10):
                        percentage = (count / total_transitions) * 100
                        # Truncate long action names for display
                        from_short = from_action[:15] + "..." if len(from_action) > 15 else from_action
                        to_short = to_action[:15] + "..." if len(to_action) > 15 else to_action
                        print(f"    {from_short} → {to_short:18} {count:4} ({percentage:4.1f}%)")
            else:
                print("No tool usage data found")

def main():
    """Main function to run trajectory analysis."""
    # Default trajectory files to analyze
    default_traj_files = [
        "data/automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10/swesmith_gen_claude__claude-sonnet-4__t-0.00__p-1.00__c-2.00___patch_swesmith_astropy__astropy.26d14786_ps__ms50_as1/astropy__astropy.26d14786_submit.json",
        "data/automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10/swesmith_gen_claude__gpt-4.1__t-0.00__p-1.00__c-2.00___patch_swesmith_astropy__astropy.26d14786_ps_ms50_as1/astropy__astropy.26d14786_submit.json",
        "data/automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10/swesmith_gen_claude__gpt-4o__t-0.00__p-1.00__c-2.00___patch_swesmith_astropy__astropy.26d14786_ps_ms50_as1/astropy__astropy.26d14786_submit.json",
        "data/automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10/swesmith_gen_claude__claude-sonnet-4_gpt4.1_gpt-4o__t-0.00__p-1.00__c-2.00___patch_swesmith_astropy__astropy.26d14786_ps/astropy__astropy.26d14786_submit.json",
        "data/swe-smith-trajectories.jsonl"
    ]
    
    parser = argparse.ArgumentParser(description="Training trajectory files for tool usage frequency")
    parser.add_argument("traj_files", nargs="*", default=default_traj_files, help="Trajectory file paths to analyze")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--cache-dir", default="cache", help="Cache directory")
    parser.add_argument("--output-dir", default="plots", help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = TrainTrajectoryAnalyzer(cache_dir=args.cache_dir)
    
    # Process each trajectory file
    file_results = {}
    for traj_file in args.traj_files:
        if not os.path.exists(traj_file):
            print(f"Warning: File {traj_file} does not exist")
            continue
        
        results = analyzer.analyze_file(traj_file, use_cache=not args.no_cache)
        file_results[traj_file] = results
    
    if not file_results:
        print("No valid trajectory files to analyze")
        return
    
    # Create visualizations
    analyzer.create_visualization(file_results, args.output_dir)
    
    # Print summary
    analyzer.print_summary(file_results)

if __name__ == "__main__":
    main()