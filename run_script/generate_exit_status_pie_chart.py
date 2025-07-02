import argparse
import os
import yaml
import matplotlib.pyplot as plt
from collections import defaultdict
import glob

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate pie chart from exit status data')
    parser.add_argument('--model', help='Model name to search for in folder names')
    parser.add_argument('--folder_path', default='trajectories/zhengyanshi@microsoft.com', 
                       help='Base folder path to search in')
    parser.add_argument('--yaml_file', default='run_batch_exit_statuses.yaml',
                       help='Name of the YAML file to load')
    parser.add_argument('--output_dir', default='evaluation_results_1r1m',
                       help='Output directory for the pie chart')
    return parser.parse_args()

def find_model_folders(base_path, model_name):
    """Find all folders containing the model name"""
    folders = []
    if os.path.exists(base_path):
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path) and model_name in item:
                folders.append(item_path)
    return folders

def normalize_exit_reason(reason):
    """Normalize exit reasons by removing 'skipped' prefix and extracting text from parentheses"""
    if reason.startswith('skipped (') and reason.endswith(')'):
        # Extract text within parentheses
        return reason[9:-1]  # Remove 'skipped (' and ')'
    return reason

def load_yaml_data(yaml_path):
    """Load and parse YAML file to extract exit status counts"""
    try:
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
        
        exit_counts = defaultdict(int)
        instances_by_status = data.get('instances_by_exit_status', {})
        
        for exit_reason, instances in instances_by_status.items():
            normalized_reason = normalize_exit_reason(exit_reason)
            exit_counts[normalized_reason] += len(instances) if instances else 0
        
        return exit_counts
    except Exception as e:
        print(f"Error loading {yaml_path}: {e}")
        return None

def calculate_average_counts(all_counts):
    """Calculate average counts across multiple folders"""
    if not all_counts:
        return {}
    
    # Get all unique exit reasons
    all_reasons = set()
    for counts in all_counts:
        all_reasons.update(counts.keys())
    
    # Calculate averages
    avg_counts = {}
    for reason in all_reasons:
        total = sum(counts.get(reason, 0) for counts in all_counts)
        avg_counts[reason] = total / len(all_counts)
    
    return avg_counts

def generate_pie_chart(exit_counts, model_name, output_dir):
    """Generate pie chart from exit counts"""
    if not exit_counts:
        print("No data to plot")
        return
    
    # Prepare data for pie chart
    labels = list(exit_counts.keys())
    sizes = list(exit_counts.values())
    
    # Filter out zero values
    filtered_data = [(label, size) for label, size in zip(labels, sizes) if size > 0]
    if not filtered_data:
        print("No non-zero data to plot")
        return
    
    # Sort by size (descending order)
    filtered_data.sort(key=lambda x: x[1], reverse=True)
    labels, sizes = zip(*filtered_data)
    total_size = sum(sizes)
    
    # Create subplot with custom layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), gridspec_kw={'width_ratios': [2.5, 1]})
    
    # Generate colors for better distinction
    colors = plt.cm.Set3(range(len(labels)))

    # Need to create a closure to pass labels to autopct function
    def make_autopct(labels, sizes):
        idx = -1  # mutable index counter
        def my_autopct(pct):
            nonlocal idx
            idx += 1
            label = labels[idx]
            if pct >= 5:
                if len(label) > 12:
                    mid_point = len(label) // 2
                    break_chars = [' ', '_', '-']
                    break_pos = mid_point

                    for offset in range(0, mid_point + 1):
                        left = mid_point - offset
                        right = mid_point + offset
                        if left >= 0 and label[left] in break_chars:
                            break_pos = left
                            break
                        elif right < len(label) and label[right] in break_chars:
                            break_pos = right
                            break

                    label = label[:break_pos] + '\n' + label[break_pos+1:]

                return f'{label}\n{pct:.1f}%' if pct >= 2 else ''
        return my_autopct
    
    wedges, texts, autotexts = ax1.pie(sizes, autopct=make_autopct(labels, sizes), startangle=90, 
                                      colors=colors, pctdistance=0.7,
                                      textprops={'fontsize': 16, 'fontweight': 'bold'})
    
    # Customize the chart
    ax1.set_title(f'{model_name}', fontsize=16, fontweight='bold')
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    # Improve percentage text readability
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(16)
        autotext.set_fontweight('bold')
        autotext.set_bbox(dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
        autotext.set_ha('center')
        autotext.set_va('center')
    
    # Improve label text readability
    for text in texts:
        text.set_fontsize(10)
        text.set_fontweight('bold')
        text.set_color('darkblue')
    
    # Create detailed legend in the second subplot
    legend_labels = []
    for label, size in zip(labels, sizes):
        percentage = (size / total_size) * 100
        legend_labels.append(f'{label}: {int(size)} ({percentage:.1f}%)')
    
    # Remove axes from legend subplot
    ax2.axis('off')
    
    # Create legend with larger font
    legend = ax2.legend(wedges, legend_labels, title="Exit Status Details", 
                       loc='center', fontsize=16, title_fontsize=18,
                       frameon=True, fancybox=True, shadow=True)
    
    # Adjust legend appearance
    legend.get_frame().set_facecolor('lightgray')
    legend.get_frame().set_alpha(0.8)
    
    # Adjust layout to minimize white space
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the chart
    output_path = os.path.join(output_dir, f'{model_name}_exit_status_pie_chart.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2)
    plt.show()
    
    print(f"Pie chart saved to: {output_path}")

def main():
    args = parse_arguments()
    
    # Find folders containing the model name
    model_folders = find_model_folders(args.folder_path, args.model)
    
    if not model_folders:
        print(f"No folders found containing model name '{args.model}' in {args.folder_path}")
        return
    
    print(f"Found {len(model_folders)} folder(s) for model '{args.model}':")
    for folder in model_folders:
        print(f"  - {folder}")
    
    # Load data from all folders
    all_counts = []
    for folder in model_folders:
        yaml_path = os.path.join(folder, args.yaml_file)
        if os.path.exists(yaml_path):
            counts = load_yaml_data(yaml_path)
            if counts:
                all_counts.append(counts)
                print(f"Loaded data from: {yaml_path}")
        else:
            print(f"YAML file not found: {yaml_path}")
    
    if not all_counts:
        print("No valid data found to process")
        return
    
    # Calculate average counts if multiple folders
    if len(all_counts) > 1:
        print(f"Calculating averages across {len(all_counts)} folders")
        final_counts = calculate_average_counts(all_counts)
    else:
        final_counts = all_counts[0]
    
    # Display the data
    print(f"\nExit status counts for model '{args.model}':")
    for reason, count in sorted(final_counts.items()):
        print(f"  {reason}: {count:.1f}")
    
    # Generate pie chart
    generate_pie_chart(final_counts, args.model, args.output_dir)

if __name__ == "__main__":
    main()
