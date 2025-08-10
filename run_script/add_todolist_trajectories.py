#!/usr/bin/env python3
"""
Enhance SWE-agent trajectories by adding overall checklists and progress annotations.

This script processes trajectories from bug-fixing tasks and:
1. Generates a step-by-step checklist based on the full trajectory using LLM
2. Inserts the checklist into the first assistant message
3. Annotates each subsequent assistant message with progress ticks using LLM

Usage:
    python enhance_trajectories.py input.jsonl output.jsonl
"""

import json
import argparse
import sys
import os
from typing import List, Dict, Any, Tuple
import re
from multiprocessing import Pool, Manager, Lock
from functools import partial

from rich.console import Console
from rich.progress import Progress, TimeElapsedColumn, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
from rich.table import Table

from sweagent.agent.models import CopilotClaudeModel, CopilotClaudeModelConfig
from sweagent.tools.tools import ToolConfig


class TrajectoryEnhancer:
    """Enhances bug-fixing trajectories with checklists and progress annotations."""
    
    def __init__(self, model_config: CopilotClaudeModelConfig = None, 
                 system_msg_limit: int = 2000, 
                 user_msg_limit: int = 1500,
                 assistant_msg_limit: int = -1):
        """Initialize the enhancer with LLM model.
        
        Args:
            model_config: Configuration for the LLM model
            system_msg_limit: Max length for system messages (-1 for no truncation)
            user_msg_limit: Max length for user messages (-1 for no truncation)
            assistant_msg_limit: Max length for assistant messages (-1 for no truncation)
        """
        self.model = None
        self.system_msg_limit = system_msg_limit
        self.user_msg_limit = user_msg_limit
        self.assistant_msg_limit = assistant_msg_limit
        
        if model_config:
            tools_config = ToolConfig()
            try:
                self.model = CopilotClaudeModel(model_config, tools_config)
            except Exception as e:
                print(f"Warning: Could not initialize LLM model: {e}")
                raise RuntimeError("LLM model is required for this enhancer")
        else:
            raise RuntimeError("LLM model config is required for this enhancer")
    
    def extract_checklist_from_trajectory(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Generate a checklist using LLM analysis of the trajectory."""
        trajectory_summary = self._create_trajectory_summary(messages)
        
        prompt = f"""
You are analyzing a completed LLM agent bug-fixing trajectory.
Your task: Derive a concise, factual, high-level checklist summarizing the concrete steps the ASSISTANT actually performed to locate, understand, and fix the bug.

CRITICAL INSTRUCTIONS:
- Focus on ASSISTANT messages (reasoning, tool calls, code edits, test runs). Use USER/SYSTEM only for minimal context.
- Do NOT invent steps that did not occur.
- Each step should represent a distinct phase of the real investigative / fixing workflow (e.g., reproduce, inspect failing test, trace cause, modify code, add/adjust tests, re-run tests, verify success).
- Prefer past-tense actionable summaries of what was done (e.g., "Reproduced failing test", "Inspected stack trace in X", "Updated Y to fix condition", "Ran full test suite to confirm fix").
- Use at most 5 steps. If fewer than 5 distinct real actions occurred, output fewer (do NOT pad).
- Keep steps high-level (no file diffs, no excessive detail) but specific enough to reflect the actual actions.
- Order steps in the sequence they were actually accomplished.
- Omit any step that did not truly happen.

You MAY include brief reasoning BEFORE the checklist (outside the code block). The final checklist MUST appear inside a fenced code block.

FINAL REQUIRED CHECKLIST FORMAT (inside a fenced code block):
```
**Checklist**
1. ...
2. ...
```
(Only as many numbered lines as real steps, max 5.)

Trajectory Summary (conversation with roles; focus on ASSISTANT actions):
{trajectory_summary}

Produce the fenced code block containing the checklist exactly in the required format. Any reasoning must be outside the fenced block.
"""
        
        try:
            response = self.model.query([{"role": "user", "content": prompt}])
            response_text = response.get("message", "") if isinstance(response, dict) else str(response)
            return self._parse_checklist_from_response(response_text)
        except Exception as e:
            print(f"LLM checklist generation failed: {e}")
            return self._get_default_checklist()
    
    def _truncate_message(self, content: str, limit: int) -> str:
        """Truncate message content if it exceeds limit. If limit is -1, no truncation."""
        if limit == -1 or len(content) <= limit:
            return content
        
        truncated_chars = len(content) - limit
        head_len = limit // 2
        tail_len = limit - head_len
        head = content[:head_len]
        tail = content[-tail_len:]
        return f"{head}... (truncated {truncated_chars} chars) ...{tail}"
    
    def _create_trajectory_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Create a complete trajectory summary preserving assistant messages according to limits."""
        summary_parts = []
        
        for msg in messages:
            role = msg['role'].upper()
            content = msg['content']
            
            if role == 'SYSTEM':
                content = self._truncate_message(content, self.system_msg_limit)
                summary_parts.append(f"{role}:\n{content}")
                
            elif role == 'USER':
                content = self._truncate_message(content, self.user_msg_limit)
                summary_parts.append(f"{role}:\n{content}")
                
            elif role == 'ASSISTANT':
                content = self._truncate_message(content, self.assistant_msg_limit)
                summary_parts.append(f"{role}:\n{content}")
        
        return '\n\n\n'.join(summary_parts)
    
    def _parse_checklist_from_response(self, response: str) -> List[str]:
        """Extract numbered checklist items from the first fenced code block containing '**Checklist**'."""
        # Find fenced code blocks
        fenced_blocks = re.findall(r"```(?:[^\n]*)\n(.*?)```", response, re.DOTALL)
        target_block = None
        for block in fenced_blocks:
            if "**Checklist**" in block or "Checklist" in block:
                target_block = block
                break
        items = []
        if target_block:
            for line in target_block.splitlines():
                line = line.strip()
                if not line or line.startswith("**Checklist**"):
                    continue
                if re.match(r'^\d+\.\s+', line):
                    items.append(re.sub(r'^\d+\.\s+', '', line).strip())
        # Fallback: previous simple parsing if fenced extraction failed
        if not items:
            for line in response.strip().splitlines():
                line = line.strip()
                if re.match(r'^\d+\.\s+', line):
                    items.append(re.sub(r'^\d+\.\s+', '', line).strip())
        return items
    
    def _get_default_checklist(self) -> List[str]:
        """Default minimal checklist if generation fails."""
        return ["Reproduced issue", "Identified root cause", "Implemented fix", "Validated tests"]
    
    def _create_context_summary(self, messages: List[Dict[str, Any]], up_to_index: int) -> str:
        """Create a context summary up to a specific message index, preserving messages according to limits."""
        summary_parts = []
        
        for msg in messages[:up_to_index + 1]:
            role = msg['role'].upper()
            content = msg['content']
            
            if role == 'SYSTEM':
                content = self._truncate_message(content, self.system_msg_limit)
                summary_parts.append(f"{role}: {content}")
                
            elif role == 'USER':
                content = self._truncate_message(content, self.user_msg_limit)
                summary_parts.append(f"{role}: {content}")
                
            elif role == 'ASSISTANT':
                content = self._truncate_message(content, self.assistant_msg_limit)
                summary_parts.append(f"{role}: {content}")
        
        return '\n'.join(summary_parts)
    
    def determine_completed_steps(self, messages: List[Dict[str, Any]], current_msg_index: int, checklist: List[str]) -> List[bool]:
        """
        Determine which checklist steps are completed based on messages BEFORE current index using LLM.
        The current message can be used as context to judge if the agent considers previous tasks completed,
        but the actual completion analysis should be based on messages before the current one.
        
        Returns a list of booleans indicating completion status for each step.
        """
        # Create context from messages up to but NOT including current index for completion analysis
        previous_context = self._create_context_summary(messages, current_msg_index - 1) if current_msg_index > 0 else ""
        
        # Get current message as additional context for judging agent's perspective
        current_msg_content = messages[current_msg_index]['content'] if current_msg_index < len(messages) else ""
        
        checklist_text = '\n'.join([f"{i+1}. {step}" for i, step in enumerate(checklist)])
        
        prompt = f"""
Analyze the conversation and determine which checklist items have been COMPLETED based on the actions taken so far.

IMPORTANT EVALUATION CRITERIA:
1. Base your completion assessment ONLY on the work accomplished in the Previous Context below.
2. The Current Agent Message can provide hints about whether the agent considers previous subtasks completed (e.g., if the agent is moving to a next step, it likely considers the previous step done), but do NOT count actions from the current message as completed work.
3. A checklist item is completed only if the concrete actions described in that step have been fully accomplished in the Previous Context.

Previous Context (messages where work was actually completed):
{previous_context}

Current Agent Message (for context about agent's perspective, but do NOT count its actions as completed):
ASSISTANT: {current_msg_content}

Checklist (DO NOT ALTER TEXT; preserve numbering and wording EXACTLY):
{checklist_text}

OUTPUT REQUIREMENTS (MANDATORY):
1. You MAY include brief reasoning BEFORE the fenced code block (optional). Do NOT put reasoning inside the code block.
2. Then output a SINGLE fenced code block (``` on its own line to start and end) containing ONLY the checklist lines.
3. Inside the fenced block, reproduce EACH checklist line in the SAME ORDER, SAME NUMBERING, SAME TEXT.
4. Prepend each line with either:
   ✅ (completed) or
   ❌ (not yet completed)
   followed by a single space, then the original line (starting with the number).
5. Do NOT add, remove, merge, renumber, or paraphrase steps. No extra commentary, blank lines, bullets, or counts inside the block.
6. If unsure, mark ❌.

EXAMPLE FORMAT (illustrative; content will differ):
```
✅ 1. Reproduced failing test
❌ 2. Identified root cause
❌ 3. Implemented fix
```

NOW PRODUCE THE OUTPUT FOLLOWING THE RULES EXACTLY.
"""
        
        try:
            response = self.model.query([{"role": "user", "content": prompt}])
            response_text = response.get("message", "") if isinstance(response, dict) else str(response)

            # Parse new checklist status block
            completed = self._parse_status_block(response_text, checklist)
            if not any(completed) and not all(c is False for c in completed):
                # sanity fallback if parsing produced irregular length
                completed = self._fallback_parse_boolean_list(response_text, checklist)
            elif len(completed) != len(checklist):
                completed = self._fallback_parse_boolean_list(response_text, checklist)
            return completed[:len(checklist)]
            
        except Exception as e:
            print(f"LLM completion determination failed: {e}")
            # Fallback to all False
            return [False] * len(checklist)
    
    def format_checklist_annotation(self, checklist: List[str], completed: List[bool]) -> str:
        """Format checklist with completion status."""
        lines = ["**Overall Checklist**"]
        for i, (step, is_completed) in enumerate(zip(checklist, completed)):
            status = "✅" if is_completed else "❌"
            lines.append(f"{status} {i + 1}. {step}")

        # # Add next step indicator
        # next_steps = [i for i, completed in enumerate(completed) if not completed]
        # if next_steps:
        #     next_step = next_steps[0]
        #     lines.append(f"\nNext: {next_step + 1}. {checklist[next_step]}")
        
        return "\n".join(lines)
    
    def enhance_trajectory(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance a single trajectory with checklist and progress annotations."""
        messages = trajectory.get('messages', [])
        if len(messages) < 2:
            return trajectory
        
        # Find assistant messages
        assistant_messages_indices = [i for i, msg in enumerate(messages) if msg['role'] == 'assistant']
        if not assistant_messages_indices:
            return trajectory
        
        # Generate checklist
        checklist = self.extract_checklist_from_trajectory(messages)
        
        # Create enhanced messages
        enhanced_messages = messages.copy()
        running_completed = [False] * len(checklist)
        
        for i, msg_idx in enumerate(assistant_messages_indices):
            # Skip the first assistant message - no checklist items can be completed yet
            if i == 0:
                # For first assistant message, all items remain incomplete
                new_completions = [False] * len(checklist)
            else:
                # For subsequent messages, check completion based on messages BEFORE current index
                new_completions = self.determine_completed_steps(enhanced_messages, msg_idx, checklist)
            # Accumulate completions (once completed, stays completed)
            running_completed = [old or new for old, new in zip(running_completed, new_completions)]
            
            # Format checklist annotation: TODO: remove the next step annotations
            checklist_annotation = self.format_checklist_annotation(checklist, running_completed)
            
            # Prepend checklist to message content
            original_content = enhanced_messages[msg_idx]['content']
            enhanced_content = f"{checklist_annotation}\n\n{original_content}"
            
            enhanced_messages[msg_idx] = {
                **enhanced_messages[msg_idx],
                'content': enhanced_content
            }
        
        return {
            **trajectory,
            'messages': enhanced_messages
        }
    
    def process_dataset(self, input_path: str, output_path: str, max_trajectories: int = None, num_workers: int = 4):
        """Process the entire dataset with parallel processing and rich progress tracking."""
        console = Console()
        console.print(f"[bold green]Processing trajectories from {input_path}[/bold green]")
        console.print(f"[bold blue]Output will be saved to {output_path}[/bold blue]")
        console.print(f"[bold yellow]Using {num_workers} worker processes[/bold yellow]")
        
        # Load all trajectories first
        trajectories = []
        with open(input_path, 'r') as infile:
            for line_num, line in enumerate(infile, 1):
                if max_trajectories and len(trajectories) >= max_trajectories:
                    break
                try:
                    trajectory = json.loads(line.strip())
                    trajectories.append((line_num, trajectory))
                except json.JSONDecodeError as e:
                    console.print(f"[red]Error parsing line {line_num}: {e}[/red]")
                    continue
        
        total_trajectories = len(trajectories)
        console.print(f"[bold cyan]Loaded {total_trajectories} trajectories for processing[/bold cyan]")
        
        if total_trajectories == 0:
            console.print("[red]No valid trajectories found![/red]")
            return
        
        # Create a manager for shared progress tracking and file writing lock
        manager = Manager()
        progress_dict = manager.dict()
        write_lock = manager.Lock()
        
        # Open output file for writing (in append mode to support incremental writes)
        # First, create/truncate the file to ensure it's empty
        with open(output_path, 'w') as f:
            pass  # Just create/truncate the file
        
        # Initialize progress tracking with transient=True for cleaner output
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,  # This makes progress bars update in place instead of creating new lines
            refresh_per_second=1  # Reduce refresh rate to minimize output
        ) as progress:
            
            # Main progress task
            main_task = progress.add_task("[green]Processing trajectories...", total=total_trajectories)
            
            # Create worker function
            worker_func = partial(
                _process_trajectory_worker,
                enhancer_config={
                    'system_msg_limit': self.system_msg_limit,
                    'user_msg_limit': self.user_msg_limit,
                    'assistant_msg_limit': self.assistant_msg_limit,
                    'model_name': self.model.config.name
                },
                progress_dict=progress_dict,
                output_path=output_path,
                write_lock=write_lock
            )
            
            # Process trajectories in parallel
            with Pool(processes=num_workers) as pool:
                results = []
                
                # Submit all tasks
                for trajectory_data in trajectories:
                    result = pool.apply_async(worker_func, (trajectory_data,))
                    results.append(result)
                
                # Collect results and update progress (trajectories now written incrementally by workers)
                processed_count = 0
                failed_count = 0
                
                for i, result in enumerate(results):
                    try:
                        line_num, success, instance_id, msg_count, current_msg = result.get()
                        if success:
                            processed_count += 1
                        else:
                            failed_count += 1
                            
                        # Update progress less frequently and with consolidated info
                        if i % 10 == 0 or i == len(results) - 1:  # Update every 10 items or on last item
                            status_text = f"[green]Processing trajectories... [cyan]{processed_count} done[/cyan] [yellow]{failed_count} failed[/yellow] [dim](last: {instance_id})[/dim]"
                            progress.update(main_task, completed=processed_count + failed_count, description=status_text)
                        else:
                            progress.advance(main_task, 1)
                            
                    except Exception as e:
                        failed_count += 1
                        console.print(f"[red]Error in worker process: {e}[/red]")
                        progress.advance(main_task, 1)
                        continue
        
        # Results already written incrementally by workers
        console.print(f"[bold green]All {processed_count} enhanced trajectories written incrementally to {output_path}[/bold green]")
        
        # Final summary
        summary_table = Table(title="Processing Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        summary_table.add_row("Total Loaded", str(total_trajectories))
        summary_table.add_row("Successfully Processed", str(processed_count))
        summary_table.add_row("Failed", str(total_trajectories - processed_count))
        summary_table.add_row("Output File", output_path)
        
        console.print(summary_table)


    def _parse_status_block(self, response_text: str, checklist: List[str]) -> List[bool]:
        """
        Parse a fenced code block containing lines like:
        ✅ 1. Original step text
        ❌ 2. Another step
        Returns list of booleans aligned with checklist length.
        """
        import re
        # Find first fenced code block
        block_match = re.search(r"```(?:[^\n]*)\n(.*?)```", response_text, re.DOTALL)
        if not block_match:
            return [False] * len(checklist)
        block = block_match.group(1).strip()
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        status_list = []
        for expected_index in range(len(checklist)):
            if expected_index >= len(lines):
                status_list.append(False)
                continue
            line = lines[expected_index]
            m = re.match(r'^([✅❌])\s+(\d+)\.\s+(.*)$', line)
            if not m:
                status_list.append(False)
                continue
            symbol, _, _ = m.groups()
            # (Optional) could verify text matches checklist[expected_index] but tolerate minor formatting
            status_list.append(symbol == '✅')
        return status_list[:len(checklist)]
    
    def _fallback_parse_boolean_list(self, response_text: str, checklist: List[str]) -> List[bool]:
        """
        Legacy fallback: parse comma-separated true/false tokens (case-insensitive) from the last line.
        """
        line = response_text.strip().split('\n')[-1]
        parts = [p.strip().lower() for p in line.split(',')]
        completed = []
        for i, part in enumerate(parts):
            if i >= len(checklist):
                break
            completed.append(part == 'true')
        while len(completed) < len(checklist):
            completed.append(False)
        return completed
    


def _process_trajectory_worker(trajectory_data: Tuple[int, Dict[str, Any]], enhancer_config: Dict[str, Any], progress_dict, output_path: str, write_lock) -> Tuple[int, bool, str, int, int]:
    """Worker function for processing a single trajectory in parallel."""
    line_num, trajectory = trajectory_data
    
    try:
        # Create a new enhancer instance for this worker
        model_config = CopilotClaudeModelConfig(name=enhancer_config['model_name'])
        enhancer = TrajectoryEnhancer(
            model_config,
            system_msg_limit=enhancer_config['system_msg_limit'],
            user_msg_limit=enhancer_config['user_msg_limit'],
            assistant_msg_limit=enhancer_config['assistant_msg_limit']
        )
        
        # Get trajectory info for progress tracking
        instance_id = trajectory.get('instance_id', f'line_{line_num}')
        messages = trajectory.get('messages', [])
        msg_count = len(messages)
        assistant_msg_count = len([msg for msg in messages if msg.get('role') == 'assistant'])
        
        # Update progress tracking
        progress_dict[f'worker_{os.getpid()}'] = {
            'instance_id': instance_id,
            'current_msg': 0,
            'total_msg': assistant_msg_count
        }
        
        # Process the trajectory
        enhanced_trajectory = enhancer.enhance_trajectory(trajectory)
        
        # Write the enhanced trajectory to file immediately (thread-safe)
        with write_lock:
            with open(output_path, 'a') as outfile:
                outfile.write(json.dumps(enhanced_trajectory) + '\n')
                outfile.flush()  # Ensure data is written to disk
        
        return line_num, True, instance_id, msg_count, assistant_msg_count
        
    except Exception as e:
        print(f"Error processing trajectory {line_num}: {e}")
        return line_num, False, trajectory.get('instance_id', f'line_{line_num}'), 0, 0


def main():
    parser = argparse.ArgumentParser(description="Enhance SWE-agent trajectories with checklists")
    parser.add_argument("--input_file", default="data/swe-smith-trajectories.jsonl", help="Input JSONL file with trajectories")
    parser.add_argument("--output_file", default="data/swe-smith-trajectories-todolist.jsonl", help="Output JSONL file for enhanced trajectories")
    parser.add_argument("--max-trajectories", type=int, default=None, help="Maximum number of trajectories to process")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of worker processes for parallel processing")
    parser.add_argument("--system-msg-limit", type=int, default=2000, help="Max length for system messages (-1 for no truncation)")
    parser.add_argument("--user-msg-limit", type=int, default=1500, help="Max length for user messages (-1 for no truncation)")
    parser.add_argument("--assistant-msg-limit", type=int, default=-1, help="Max length for assistant messages (-1 for no truncation)")
    parser.add_argument("--model", default="o3-2025-04-16", help="Model name to use (e.g., claude-opus-4, o3-2025-04-16)")

    args = parser.parse_args()
    
    # Automatically append model name to output file
    base_name, ext = os.path.splitext(args.output_file)
    args.output_file = f"{base_name}_{args.model}{ext}"
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist")
        sys.exit(1)
    
    # Initialize model config
    try:
        model_config = CopilotClaudeModelConfig(name=args.model)
    except Exception as e:
        print(f"Error: Could not setup LLM: {e}")
        sys.exit(1)
    
    # Create enhancer and process dataset
    try:
        enhancer = TrajectoryEnhancer(
            model_config,
            system_msg_limit=args.system_msg_limit,
            user_msg_limit=args.user_msg_limit, 
            assistant_msg_limit=args.assistant_msg_limit
        )
        enhancer.process_dataset(args.input_file, args.output_file, args.max_trajectories, args.num_workers)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()