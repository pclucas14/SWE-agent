import argparse
from datasets import load_dataset
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import re
import inspect



def filter_is_malformed_function_call(message):
    """
    Returns True if the assistant message content contains malformed or incomplete
    XML-style function call syntax.

    Malformations include:
    - Unclosed <function=...> tags
    - Unclosed <parameter=...> tags
    - Missing or unmatched end tags
    - Function blocks with no parameters (except for submit function)
    """
    content = message['content']

    # Check for unmatched function tag
    function_open_tags = re.findall(r"<function=[^>]+>", content)
    function_close_tags = re.findall(r"</function>", content)
    if len(function_open_tags) != len(function_close_tags):
        print("Unmatched function tags found")
        print(content)
        return True

    # Check for unmatched parameter tags
    parameter_open_tags = re.findall(r"<parameter=[^>]+>", content)
    parameter_close_tags = re.findall(r"</parameter>", content)
    if len(parameter_open_tags) != len(parameter_close_tags):
        print("Unmatched parameter tags found")
        print(content)
        return True

    # Check if any function block is missing a parameter (except submit function)
    function_block_pattern = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
    parameter_pattern = re.compile(r"<parameter=[^>]+>.*?</parameter>", re.DOTALL)
    for match in function_block_pattern.findall(content):
        function_name, function_content = match
        # Allow submit function to have no parameters
        if function_name == "submit":
            continue
        if not parameter_pattern.search(function_content):
            print("Function block without parameters found")
            print(content)
            return True

    return False


def filter_hmac_criterion(message):
    """
    Filter for messages containing 'HMAC timestamp out of range' in content.
    """
    return 'HMAC timestamp out of range' in message['content']


def filter_pytest_criterion(message):
    """
    Filter for messages containing 'pytest' in content.
    """
    return 'pytest' in message['content']


def filter_error_criterion(message):
    """
    Filter for messages containing 'error' in content.
    """
    return 'error' in message['content'].lower()


def filter_code_criterion(message):
    """
    Filter for messages containing code blocks in content.
    """
    return '```' in message['content'] or '`' in message['content']


def filter_submit_function(message):
    """
    Filter for messages containing submit function call pattern.
    
    Checks for:
    <function=submit>
    </function>
    
    Note: This filter is special - it only checks the last assistant message in each example.
    """
    content = message['content']
    # Check for submit function pattern (with or without parameters)
    submit_pattern = re.compile(r'<function=submit>.*?</function>', re.DOTALL)
    return bool(submit_pattern.search(content))


def filter_has_submit_function(message):
    """
    Filter for messages containing submit function call pattern.
    
    Checks for:
    <function=submit>
    </function>
    
    Note: This checks ALL assistant messages, not just the last one.
    """
    content = message['content']
    # Check for submit function pattern (with or without parameters)
    submit_pattern = re.compile(r'<function=submit>.*?</function>', re.DOTALL)
    return bool(submit_pattern.search(content))


# Automatically register filter functions
def _get_filter_functions():
    """Automatically discover and register filter functions."""
    current_module = inspect.getmembers(inspect.getmodule(inspect.currentframe()))
    filter_functions = {}
    
    for name, obj in current_module:
        if (inspect.isfunction(obj) and 
            (name.startswith('filter_') or name.startswith('is_')) and
            name != '_get_filter_functions'):
            filter_functions[name] = obj
    
    return filter_functions


# Available filter functions (automatically populated)
FILTER_FUNCTIONS = _get_filter_functions()


def count_messages(data_file, filter_func):
    """
    Count messages in a dataset based on a filter function.
    
    Args:
        data_file (str): Path to the data file
        filter_func (callable): Function to filter messages
    """
    filtered_count = 0
    total_assistant_calls = 0
    examples_with_filter = 0
    
    # Load dataset
    if "json" in data_file:
        dataset = load_dataset('json', data_files=data_file, split='train')
    else:
        dataset = load_dataset(data_file, split='train')

    for m in dataset['messages']:
        example_has_filter = False
        
        # Special handling for filter_submit_function - only check last assistant message
        if filter_func.__name__ == 'filter_submit_function':
            # Find the last assistant message
            last_assistant_message = None
            for j in reversed(m):
                if j['role'] == 'assistant':
                    last_assistant_message = j
                    break
            
            if last_assistant_message:
                total_assistant_calls += 1
                if filter_func(last_assistant_message):
                    filtered_count += 1
                    example_has_filter = True
        else:
            # Original behavior: check all assistant messages
            for j in m:
                if j['role'] == 'assistant':
                    total_assistant_calls += 1
                    if filter_func(j):
                        filtered_count += 1
                        example_has_filter = True
        
        if example_has_filter:
            examples_with_filter += 1
    
    # Create rich console and table
    console = Console()
    
    # Create main results table
    table = Table(title="Dataset Analysis Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", no_wrap=True, width=30)
    table.add_column("Count", justify="right", style="green", width=10)
    table.add_column("Percentage", justify="right", style="yellow", width=12)
    
    total_examples = len(dataset['messages'])
    
    # Add training examples section
    table.add_section()
    table.add_row("Training Examples", "", "", style="bold blue")
    table.add_row("├─ Total examples", f"{total_examples:,}", "100.00%")
    table.add_row("└─ With filter criterion", f"{examples_with_filter:,}", f"{examples_with_filter / total_examples * 100:.2f}%")
    
    # Add assistant messages section
    table.add_section()
    table.add_row("Assistant Messages", "", "", style="bold blue")
    if filter_func.__name__ == 'filter_submit_function':
        table.add_row("├─ Total last assistant msgs", f"{total_assistant_calls:,}", "100.00%")
    else:
        table.add_row("├─ Total assistant messages", f"{total_assistant_calls:,}", "100.00%")
    table.add_row("└─ Matching filter", f"{filtered_count:,}", f"{filtered_count / total_assistant_calls * 100:.2f}%")
    
    console.print()
    console.print(table)
    
    # Add filter info panel
    mode_info = "Last assistant message only" if filter_func.__name__ == 'filter_submit_function' else "All assistant messages"
    filter_info = Panel(
        f"Filter Function: [bold cyan]{filter_func.__name__}[/bold cyan]\n"
        f"Data File: [dim]{data_file.split('/')[-1]}[/dim]\n"
        f"Mode: [yellow]{mode_info}[/yellow]\n"
        f"[dim]Note: Only assistant messages are analyzed[/dim]",
        title="Configuration",
        border_style="blue"
    )
    console.print()
    console.print(filter_info)


def main():
    parser = argparse.ArgumentParser(description='Count messages in a dataset based on filter criteria')
    parser.add_argument(
        '--data-file', 
        type=str, 
        default='SWE-bench/SWE-smith-trajectories',
        help='Path to the JSON data file'
    )
    parser.add_argument(
        '--filter-function',
        type=str,
        default=None,
        choices=list(FILTER_FUNCTIONS.keys()),
        help='Name of the filter function to use. If not specified, runs all filters.'
    )
    
    args = parser.parse_args()
    print(f"Using data file: {args.data_file}")

    if args.filter_function:
        # Run single filter
        selected_filter = FILTER_FUNCTIONS[args.filter_function]
        count_messages(args.data_file, selected_filter)
    else:
        # Run all filters
        console = Console()
        console.print()
        console.print("[bold magenta]Running all available filters...[/bold magenta]")
        console.print()
        
        for filter_name, filter_func in sorted(FILTER_FUNCTIONS.items()):
            console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
            console.print(f"[bold yellow]Filter: {filter_name}[/bold yellow]")
            console.print(f"[bold cyan]{'='*60}[/bold cyan]")
            count_messages(args.data_file, filter_func)
            console.print()


if __name__ == "__main__":
    main()