#!/usr/bin/env python3
"""
Analyze SWE-bench trajectories and determine task success/failure using LLM API.

This script loads trajectories, maps them to SWE-bench instances, checks resolution status,
and uses an LLM to analyze why tasks succeeded or failed.

python run_script/run_error_analysis.py \
    --folder-path trajectories/zhengyanshi@microsoft.com/swesmith_infer__openai--tt_SWE-agent-LM-32B_cl32768_lr1e-4_ep1_astropy__astropy.26d14786_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o__t-0.00__p-1.00__c-0.00___swe_bench_verified_test__ms75_mitNone_as1_astropy__astropy \
    --max-workers 32 \
    --model-name "claude-sonnet-4"
"""

import argparse
import json
import os
import shutil
import sys
import yaml
from typing import Optional, Tuple
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

# Rich progress bar imports
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console

# Import SWE-agent model classes
from sweagent.agent.models import get_model, GenericAPIModelConfig
from sweagent.tools.tools import ToolConfig
from sweagent.types import History


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


def load_analysis_prompts(prompts_file: str = "run_script/analysis_prompts.yaml") -> dict:
    """Load analysis prompts from YAML configuration file."""
    try:
        with open(prompts_file, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Fallback to default prompts if file not found
        return {
            "system_prompt": "You are an expert software engineer analyzing debugging trajectories.",
            "analysis_prompt": "Analyze this trajectory and provide insights."
        }


def transform_traj_xml(traj: dict, max_user_tokens: int = 2000) -> str:
    """Transform trajectory to text format."""
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

    trajectory_text = []
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
            content = "SYSTEM MESSAGE"
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
        
        trajectory_text.append(f"[{role.upper()}]: {content}")
    
    return "\n\n".join(trajectory_text)


def load_swebench_dataset(dataset_name: str = "princeton-nlp/SWE-bench_Verified", split: str = "test") -> dict:
    """Load SWE-bench dataset and create instance_id mapping."""
    print(f"Loading {dataset_name} dataset...")
    dataset = load_dataset(dataset_name, split=split)
    
    # Create mapping from instance_id to dataset example
    instance_mapping = {}
    for example in dataset:
        instance_mapping[example['instance_id']] = example
    
    print(f"Loaded {len(instance_mapping)} instances from SWE-bench dataset")
    return instance_mapping


def extract_instance_id_from_folder(folder_path: str) -> str:
    """Extract instance ID from folder name like astropy__astropy-7166."""
    folder_name = os.path.basename(folder_path.rstrip('/'))
    return folder_name


def load_resolution_data(main_folder_path: str, search_path: Optional[str] = None) -> Tuple[Optional[dict], str]:
    """
    Load resolution data from results.json or external files at the folder level.
    
    Returns:
        Tuple of (results_dict, source_description)
    """
    # First try to find results.json in the main folder
    results_path = os.path.join(main_folder_path, "results.json")
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r') as f:
                results = json.load(f)
            return results, f"Loaded from {results_path}"
        except Exception as e:
            print(f"Error reading {results_path}: {e}")
    
    # If no results.json, search in external path
    if search_path and os.path.exists(search_path):
        # First, look for exact results.json in search_path
        external_results_path = os.path.join(search_path, "results.json")
        if os.path.exists(external_results_path):
            try:
                with open(external_results_path, 'r') as f:
                    results = json.load(f)
                # Copy to main folder path
                shutil.copy2(external_results_path, results_path)
                print(f"📋 Copied results.json from {external_results_path} to {results_path}")
                return results, f"Loaded from {external_results_path} (copied to local)"
            except Exception as e:
                print(f"Error reading or copying {external_results_path}: {e}")
        
        # Extract the experiment name from main folder path
        folder_basename = os.path.basename(main_folder_path.rstrip('/'))
        
        # Look for files matching the pattern
        for filename in os.listdir(search_path):
            if folder_basename in filename and filename.endswith('.json'):
                filepath = os.path.join(search_path, filename)
                try:
                    with open(filepath, 'r') as f:
                        results = json.load(f)
                    # Copy the matching file as results.json in main folder
                    shutil.copy2(filepath, results_path)
                    print(f"📋 Copied {filepath} to {results_path}")
                    return results, f"Loaded from {filepath} (copied as results.json)"
                except Exception as e:
                    print(f"Error reading or copying {filepath}: {e}")
    
    # Default to None if no results found
    return None, "No resolution data found"


def load_predicted_patches(main_folder_path: str) -> dict:
    """
    Load predicted patches from preds.json file.
    
    Returns:
        Dict mapping instance_id to predicted patch
    """
    preds_path = os.path.join(main_folder_path, "preds.json")
    if os.path.exists(preds_path):
        try:
            with open(preds_path, 'r') as f:
                preds_data = json.load(f)
            
            # Create mapping from instance_id to predicted patch
            predicted_patches = {}
            for instance_id, patch_data in preds_data.items():
                try:
                    # Handle different data structures
                    if isinstance(patch_data, dict):
                        # Direct dict with 'patch' or 'model_patch' field
                        if 'model_patch' in patch_data:
                            predicted_patches[instance_id] = patch_data['model_patch']
                        elif 'patch' in patch_data:
                            predicted_patches[instance_id] = patch_data['patch']
                        else:
                            predicted_patches[instance_id] = str(patch_data)
                    elif isinstance(patch_data, str):
                        # Check if it's a string representation of a dict
                        if patch_data.startswith('{') and 'model_patch' in patch_data:
                            try:
                                # Parse the string as a Python literal
                                import ast
                                parsed_data = ast.literal_eval(patch_data)
                                if isinstance(parsed_data, dict) and 'model_patch' in parsed_data:
                                    predicted_patches[instance_id] = parsed_data['model_patch']
                                else:
                                    predicted_patches[instance_id] = patch_data
                            except (ValueError, SyntaxError):
                                # If parsing fails, use the string as-is
                                predicted_patches[instance_id] = patch_data
                        else:
                            predicted_patches[instance_id] = patch_data
                    else:
                        predicted_patches[instance_id] = str(patch_data)
                except Exception as e:
                    print(f"Error processing patch for {instance_id}: {e}")
                    predicted_patches[instance_id] = "Error processing patch"
            
            print(f"Loaded {len(predicted_patches)} predicted patches from {preds_path}")
            return predicted_patches
        except Exception as e:
            print(f"Error reading {preds_path}: {e}")
            return {}
    else:
        print(f"No preds.json found at {preds_path}")
        return {}


def check_task_resolution(instance_id: str, resolution_data: Optional[dict], source_description: str) -> Tuple[Optional[bool], str]:
    """
    Check if a task is resolved using pre-loaded resolution data.
    
    Returns:
        Tuple of (is_resolved, source_description)
    """
    if resolution_data is None:
        return None, source_description

    resolved_ids = resolution_data.get("resolved_ids", [])
    if instance_id in resolved_ids:
        return True, f"Found in resolved_ids ({source_description})"
    else:
        return False, f"Not in unresolved_ids ({source_description})"


def analyze_trajectory_with_llm(
    trajectory_text: str, 
    patch: str, 
    problem_statement: str, 
    test_patch: str, 
    predicted_patch: str,
    is_resolved: Optional[bool],
    resolution_source: str,
    model_name: str = "gpt-4o",
    prompts_config: Optional[dict] = None
) -> str:
    """Use LLM API to analyze why the trajectory succeeded or failed."""
    
    # Load prompts configuration
    if prompts_config is None:
        prompts_config = load_analysis_prompts()
    
    status_text = "UNKNOWN"
    if is_resolved is True:
        status_text = "RESOLVED"
    elif is_resolved is False:
        status_text = "UNRESOLVED"
    
    # Format the analysis prompt with variables
    prompt = prompts_config["analysis_prompt"].format(
        status_text=status_text,
        problem_statement=problem_statement,
        patch=patch,
        test_patch=test_patch,
        predicted_patch=predicted_patch,
        trajectory_text=trajectory_text
    )

    try:
        # Create model config
        model_config = GenericAPIModelConfig(
            name=model_name,
            temperature=0.1,
            per_instance_cost_limit=0.0,  # Disable cost limits for analysis
            total_cost_limit=0.0
        )
        
        # Create tool config (minimal for analysis)
        tool_config = ToolConfig(
            commands=[],
            use_function_calling=False
        )
        
        # Get model instance
        model = get_model(model_config, tool_config)
        
        # Create history for the query
        history: History = [
            {"role": "system", "content": prompts_config["system_prompt"]},
            {"role": "user", "content": prompt}
        ]
        
        # Query the model
        response = model.query(history)
        return response["message"]
    
    except Exception as e:
        return f"Error calling LLM API: {str(e)}"


def generate_readme(folder_path: str, results_data: list) -> str:
    """Generate a comprehensive README file based on the analysis results."""
    
    # Calculate statistics
    total = len(results_data)
    resolved = sum(1 for r in results_data if r.get('is_resolved') is True)
    not_resolved = sum(1 for r in results_data if r.get('is_resolved') is False)
    unknown = sum(1 for r in results_data if r.get('is_resolved') is None)
    
    # Calculate success rate
    success_rate = (resolved / total * 100) if total > 0 else 0
    
    # Get examples
    resolved_examples = [r for r in results_data if r.get('is_resolved') is True]
    failed_examples = [r for r in results_data if r.get('is_resolved') is False]
    
    def format_patch(patch_content: str, patch_type: str) -> str:
        """Format patch content for display."""
        if not patch_content or patch_content == f'No {patch_type.lower()} available':
            return f"*No {patch_type.lower()} available*"
        
        # Limit patch display to first 20 lines for readability
        lines = patch_content.split('\n')
        if len(lines) > 20:
            truncated_patch = '\n'.join(lines[:20])
            return f"```diff\n{truncated_patch}\n... (truncated, showing first 20 lines)\n```"
        else:
            return f"```diff\n{patch_content}\n```"
    
    def format_analysis(analysis: str) -> str:
        """Format LLM analysis for display."""
        if not analysis or analysis == 'No analysis available':
            return "*No analysis available*"
        return analysis
    
    # Start building README content
    readme_content = f"""# 🔍 SWE-bench Trajectory Analysis Results

## 📊 Summary Statistics

| Metric | Count | Percentage |
|--------|-------|------------|
| **Total Instances** | {total} | 100.0% |
| **✅ Resolved** | {resolved} | {success_rate:.1f}% |
| **❌ Not Resolved** | {not_resolved} | {(not_resolved / total * 100) if total > 0 else 0:.1f}% |
| **❓ Unknown Status** | {unknown} | {(unknown / total * 100) if total > 0 else 0:.1f}% |

## 📈 Performance Overview

```
Success Rate: {success_rate:.1f}%
{'█' * int(success_rate // 2)}{'░' * int(50 - success_rate // 2)} {success_rate:.1f}%
```

{'═' * 80}

"""

    # Add resolved examples with full details
    if resolved_examples:
        readme_content += f"""## ✅ Successfully Resolved Instances ({len(resolved_examples)} total)

"""
        for i, example in enumerate(resolved_examples, 1):
            readme_content += f"""
{'▔' * 80}
### 📋 Instance {i}: `{example.get('instance_id', 'Unknown')}`
{'▔' * 80}

**🎯 Status**: ✅ RESOLVED  
**📍 Source**: {example.get('resolution_source', 'N/A')}

#### 🤖 LLM Analysis
{format_analysis(example.get('llm_analysis', 'No analysis available'))}

#### 🔧 Ground Truth Patch
{format_patch(example.get('patch', ''), 'patch')}

#### 🎯 Predicted Patch  
{format_patch(example.get('predicted_patch', ''), 'predicted patch')}

"""
    
    # Add failed examples with full details  
    if failed_examples:
        readme_content += f"""
{'═' * 80}

## ❌ Failed Instances ({len(failed_examples)} total)

"""
        for i, example in enumerate(failed_examples, 1):
            readme_content += f"""
{'▔' * 80}
### 📋 Instance {i}: `{example.get('instance_id', 'Unknown')}`
{'▔' * 80}

**🎯 Status**: ❌ NOT RESOLVED  
**📍 Source**: {example.get('resolution_source', 'N/A')}

#### 🤖 LLM Analysis
{format_analysis(example.get('llm_analysis', 'No analysis available'))}

#### 🔧 Ground Truth Patch
{format_patch(example.get('patch', ''), 'patch')}

#### 🎯 Predicted Patch
{format_patch(example.get('predicted_patch', ''), 'predicted patch')}

"""
    
    # Add final sections
    readme_content += f"""
{'═' * 80}

## 📁 Generated Files

- **`trajectory_analysis_results.json`** - Complete analysis results with all data
- **`README.md`** - This comprehensive summary (you are here! 📍)

## 🔍 How to Use These Results

1. **📖 Review this README** - Get an overview of successes and failures
2. **🗂️ Analyze JSON file** - Dive into detailed data for each instance  
3. **🔍 Compare patches** - Study differences between predicted and ground truth
4. **📊 Identify patterns** - Look for common failure modes in the LLM analyses

## 💡 Analysis Insights

- **Success Rate**: {success_rate:.1f}% of instances were resolved
- **Total Analyzed**: {total} SWE-bench instances processed
- **Resolution Data**: {resolved_examples[0].get('resolution_source', 'N/A').split('(')[0] if resolved_examples else 'N/A'}

{'═' * 80}

*📝 Generated by SWE-agent trajectory analysis tool*  
*⏰ Generated on: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    readme_file = os.path.join(folder_path, "README.md")
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    
    return readme_file


def process_folder(
    folder_path: str, 
    swebench_mapping: dict, 
    resolution_data: Optional[dict],
    resolution_source: str,
    predicted_patches: dict,
    model_name: str = "gpt-4o",
    prompts_config: Optional[dict] = None,
    max_user_tokens: int = 2000
) -> dict:
    """Process a single trajectory folder and return analysis results."""
    
    # Find trajectory file
    trajectory_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.traj'):
                trajectory_files.append(os.path.join(root, file))
    
    if not trajectory_files:
        return {"error": f"No .traj files found in {folder_path}"}
    
    # Use the first trajectory file found
    trajectory_path = trajectory_files[0]
    
    # Extract instance ID from trajectory filename or folder structure
    traj_filename = os.path.basename(trajectory_path)
    instance_id = traj_filename.replace('.traj', '')
    
    # If that doesn't work, try to extract from folder structure
    if instance_id not in swebench_mapping:
        for part in folder_path.split('/'):
            if part in swebench_mapping:
                instance_id = part
                break
    
    if instance_id not in swebench_mapping:
        return {"error": f"Instance ID {instance_id} not found in SWE-bench dataset"}
    
    # Load and transform trajectory
    try:
        with open(trajectory_path, 'r') as f:
            traj_data = json.load(f)
        
        trajectory_text = transform_traj_xml(traj_data, max_user_tokens)
    except Exception as e:
        return {"error": f"Error loading trajectory: {str(e)}"}
    
    # Get SWE-bench data
    swebench_example = swebench_mapping[instance_id]
    patch = swebench_example.get('patch', 'No patch available')
    problem_statement = swebench_example.get('problem_statement', 'No problem statement available')
    test_patch = swebench_example.get('test_patch', 'No test patch available')
    
    # Check resolution status using pre-loaded data
    is_resolved, resolution_desc = check_task_resolution(instance_id, resolution_data, resolution_source)

    # Get predicted patch for this instance
    predicted_patch = predicted_patches.get(instance_id, 'No predicted patch available')

    # Analyze with LLM
    llm_analysis = analyze_trajectory_with_llm(
        trajectory_text, patch, problem_statement, test_patch, predicted_patch, is_resolved, resolution_desc, model_name, prompts_config
    )
    
    return {
        "instance_id": instance_id,
        "trajectory_path": trajectory_path,
        "is_resolved": is_resolved,
        "resolution_source": resolution_desc,
        "problem_statement": problem_statement,
        "patch": patch,
        "test_patch": test_patch,
        "predicted_patch": predicted_patch,
        "llm_analysis": llm_analysis
    }


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Analyze SWE-bench trajectories and determine task success/failure using LLM API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python run_script.py \\
    --folder-path "trajectories/zhengyanshi@microsoft.com/swesmith_infer__openai--tt_SWE-agent-LM-32B_cl32768_lr1e-4_ep1_astropy__astropy.26d14786_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o__t-0.00__p-1.00__c-0.00___swe_bench_verified_test__ms75_mitNone_as1_astropy__astropy" \\
    --search-path "/home/zhengyanshi/project/SWE-agent"
        """
    )
    
    parser.add_argument(
        '--folder-path',
        required=True,
        help='Path to the trajectory folder containing subfolders for each instance'
    )
    
    parser.add_argument(
        '--search-path',
        default=".",
        help='Path to search for external result files (optional)'
    )
    
    parser.add_argument(
        '--dataset-name',
        default='princeton-nlp/SWE-bench_Verified',
        help='SWE-bench dataset name'
    )
    
    parser.add_argument(
        '--split',
        default='test',
        help='Dataset split to use'
    )
    
    parser.add_argument(
        '--model-name',
        default='claude-sonnet-4',
        help='LLM model to use for analysis'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=1,
        help='Maximum number of parallel workers for processing (default: 1)'
    )
    
    parser.add_argument(
        '--prompts-file',
        default='run_script/analysis_prompts.yaml',
        help='YAML file containing analysis prompts (default: analysis_prompts.yaml)'
    )
    
    parser.add_argument(
        '--save-full-data',
        action='store_true',
        help='Save full data including test patches to JSON (creates larger file)'
    )
    
    parser.add_argument(
        '--max-user-tokens',
        type=int,
        default=2000,
        help='Maximum number of tokens for user messages before truncation (default: 2000)'
    )
    
    parser.add_argument(
        '--rerun',
        action='store_true',
        help='Force re-run analysis even if trajectory_analysis_results.json already exists'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.folder_path):
        print(f"Error: Folder path does not exist: {args.folder_path}")
        sys.exit(1)
    
    if args.search_path and not os.path.exists(args.search_path):
        print(f"Warning: Search path does not exist: {args.search_path}")
        args.search_path = None
    
    # Create output file in the same directory as the input folder
    output_file = os.path.join(args.folder_path, "trajectory_analysis_results.json")
    
    # Check if results already exist and --rerun is not set
    if os.path.exists(output_file) and not args.rerun:
        console = Console()
        console.print(f"[yellow]Results file already exists: {output_file}[/yellow]")
        console.print("[yellow]Use --rerun flag to force re-analysis, or delete the file to run again.[/yellow]")
        console.print("[green]Skipping analysis.[/green]")
        sys.exit(0)
    
    # Load SWE-bench dataset
    try:
        swebench_mapping = load_swebench_dataset(args.dataset_name, args.split)
    except Exception as e:
        print(f"Error loading SWE-bench dataset: {e}")
        sys.exit(1)
    
    # Find all instance subfolders
    instance_folders = []
    if os.path.isdir(args.folder_path):
        for item in os.listdir(args.folder_path):
            item_path = os.path.join(args.folder_path, item)
            if os.path.isdir(item_path):
                instance_folders.append(item_path)
    
    if not instance_folders:
        print(f"No subfolders found in {args.folder_path}")
        sys.exit(1)
    
    console = Console()
    console.print(f"Found {len(instance_folders)} instance folders to process")
    
    # Load resolution data once at the folder level
    resolution_data, resolution_source = load_resolution_data(args.folder_path, args.search_path)
    console.print(f"Resolution data: {resolution_source}")
    
    # Load predicted patches once at the folder level
    predicted_patches = load_predicted_patches(args.folder_path)
    console.print(f"Predicted patches: Loaded {len(predicted_patches)} patches")
    
    # Load prompts configuration once
    prompts_config = load_analysis_prompts(args.prompts_file)
    console.print(f"Loaded prompts from: {args.prompts_file}")
    
    # Create a partial function with fixed arguments for parallel processing
    process_func = partial(
        process_folder,
        swebench_mapping=swebench_mapping,
        resolution_data=resolution_data,
        resolution_source=resolution_source,
        predicted_patches=predicted_patches,
        model_name=args.model_name,
        prompts_config=prompts_config,
        max_user_tokens=args.max_user_tokens
    )
    
    # Process folders in parallel with rich progress bar
    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("Processing trajectories...", total=len(instance_folders))
        
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit all tasks
            future_to_folder = {
                executor.submit(process_func, folder): folder 
                for folder in instance_folders
            }
            
            # Process completed tasks
            for future in as_completed(future_to_folder):
                folder = future_to_folder[future]
                try:
                    result = future.result()
                    result['folder_path'] = folder
                    results.append(result)
                    
                    # Update progress
                    if 'error' in result:
                        progress.console.print(f"[red]Error in {os.path.basename(folder)}: {result['error']}")
                    else:
                        status = "RESOLVED" if result['is_resolved'] else "UNRESOLVED" if result['is_resolved'] is False else "UNKNOWN"
                        status_color = "green" if result['is_resolved'] else "red" if result['is_resolved'] is False else "yellow"
                        progress.console.print(f"[{status_color}]{os.path.basename(folder)}: {status}")
                    
                    progress.advance(task)
                    
                except Exception as exc:
                    progress.console.print(f"[red]Exception in {os.path.basename(folder)}: {exc}")
                    results.append({
                        "folder_path": folder,
                        "error": f"Exception during processing: {str(exc)}"
                    })
                    progress.advance(task)
    
    # Determine what to save to JSON
    if args.save_full_data:
        # Save full data including large fields
        json_results = results
        console.print(f"\n✅ Analysis complete. Results saved to: {output_file}")
        console.print(f"📦 Full data saved (includes all patches)")
    else:
        # Create lightweight version for JSON (exclude test_patch but keep patch and predicted_patch)
        json_results = []
        for result in results:
            lightweight_result = {k: v for k, v in result.items() 
                                if k not in ['test_patch']}
            json_results.append(lightweight_result)
        console.print(f"\n✅ Analysis complete. Results saved to: {output_file}")
        console.print(f"📦 JSON file size optimized (excludes test_patch)")
        console.print(f"💡 Use --save-full-data flag to include test_patch in JSON")
    
    # Save results to JSON
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Generate README summary file
    readme_file = generate_readme(args.folder_path, json_results)
    console.print(f"📄 README summary generated: {readme_file}")
    console.print(f"📖 Open README.md to view a summary of analysis results")
    
    # Print summary
    resolved_count = sum(1 for r in results if r.get('is_resolved') is True)
    unresolved_count = sum(1 for r in results if r.get('is_resolved') is False)
    unknown_count = sum(1 for r in results if r.get('is_resolved') is None)
    error_count = sum(1 for r in results if 'error' in r)
    
    console.print(f"\n📊 Summary:")
    console.print(f"  ✅ Resolved: {resolved_count}")
    console.print(f"  ❌ Unresolved: {unresolved_count}")
    console.print(f"  ❓ Unknown: {unknown_count}")
    console.print(f"  🔴 Errors: {error_count}")
    console.print(f"  📈 Total: {len(results)}")
    
    if resolved_count + unresolved_count > 0:
        success_rate = (resolved_count / (resolved_count + unresolved_count)) * 100
        console.print(f"  🎯 Success Rate: {success_rate:.1f}%")


if __name__ == "__main__":
    main()