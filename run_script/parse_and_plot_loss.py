import re
import matplotlib.pyplot as plt
import numpy as np

def parse_loss_from_log(log_file_path):
    """Parse loss values, steps, and epochs from the log file."""
    epochs = []
    steps = []
    losses = []
    
    # Pattern to match: epoch|step|Loss: value
    pattern = r'(\d+)\|(\d+)\|Loss: ([\d.]+)'
    
    with open(log_file_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                step = int(match.group(2))
                loss = float(match.group(3))
                
                epochs.append(epoch)
                steps.append(step)
                losses.append(loss)
    
    return epochs, steps, losses

def plot_loss_with_epochs(epochs, steps, losses):
    """Plot loss over steps with epoch highlights."""
    plt.figure(figsize=(12, 6))
    
    # Plot the loss curve
    plt.plot(steps, losses, 'b-', linewidth=1.5, alpha=0.7, label='Loss')
    
    # Find epoch boundaries
    epoch_boundaries = []
    current_epoch = epochs[0] if epochs else 0
    
    for i, epoch in enumerate(epochs):
        if epoch != current_epoch:
            epoch_boundaries.append((steps[i-1], current_epoch))
            current_epoch = epoch
    
    # Add the last epoch boundary
    if epochs:
        epoch_boundaries.append((steps[-1], epochs[-1]))
    
    # Highlight epochs with different colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(set(epochs))))
    
    # Add vertical lines at epoch boundaries
    prev_step = 0
    for i, (boundary_step, epoch_num) in enumerate(epoch_boundaries):
        if i < len(colors):
            # Shade the region for this epoch
            epoch_mask = [e == epoch_num for e in epochs]
            epoch_steps = [s for s, m in zip(steps, epoch_mask) if m]
            epoch_losses = [l for l, m in zip(losses, epoch_mask) if m]
            
            if epoch_steps:
                plt.axvspan(min(epoch_steps), max(epoch_steps), 
                           alpha=0.1, color=colors[i], 
                           label=f'Epoch {epoch_num}')
        
        # Add vertical line at epoch boundary
        # if i > 0:
        plt.axvline(x=boundary_step, color='red', linestyle='--', alpha=0.5)
    
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Steps with Epoch Highlights')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save and show the plot
    plt.savefig('loss_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return plt

def main():
    log_file = '/home/zhengyanshi/project/SWE-agent/log.txt'
    
    # Parse the log file
    epochs, steps, losses = parse_loss_from_log(log_file)
    
    print(f"Parsed {len(losses)} loss values")
    print(f"Steps range: {min(steps)} to {max(steps)}")
    print(f"Epochs range: {min(epochs)} to {max(epochs)}")
    print(f"Loss range: {min(losses):.4f} to {max(losses):.4f}")
    
    # Plot the results
    plot_loss_with_epochs(epochs, steps, losses)

if __name__ == "__main__":
    main()