import os
import shutil
import subprocess
import json
import argparse
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
import matplotlib.pyplot as plt


ORG_NAME = "OneRepoOneModel"
REPOS=(
    "astropy__astropy.26d14786",
    "pylint-dev__pylint.1f8c4d9e",
    "sympy__sympy.a36caf5c",
    "pytest-dev__pytest.3c153494",
    "sympy__sympy.360290c4",
    "sphinx-doc__sphinx.6cb783c0",
    "sphinx-doc__sphinx.9bb204dc",
    "pylint-dev__pylint.99589b08",
    "scikit-learn__scikit-learn.586f4318",
    "scikit-learn__scikit-learn.3eacf948",
    "pytest-dev__pytest.58e6a09d",
    "pydata__xarray.41fef6f1",
    "pydata__xarray.7c4e2ac8",
    "matplotlib__matplotlib.3dd06a46",
    "matplotlib__matplotlib.a3e2897b",
    "django__django.4a72da71",
    "django__django.f8fab6f9",
    "astropy__astropy.b16c7d12",
)


def clone_repo(repo: str, dest: str | None = None, org: str = ORG_NAME) -> str | None:
    """Clone a repository from GitHub."""
    if not os.path.exists(dest or repo):
        clone_cmd = (
            f"git clone git@github.com:{org}/{repo}.git"
            if dest is None
            else f"git clone git@github.com:{org}/{repo}.git {dest}"
        )
        subprocess.run(
            clone_cmd,
            check=True,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return repo if dest is None else dest
    return None

def extract_repo_name(repo_url):
    """Extract repository name from GitHub URL."""
    return repo_url.rstrip('/').split('/')[-1].replace('.git', '')

def collect_code_files(repo_path, tokenizer=None):
    """Collect all code files from the repository."""
    code_extensions = {'.py', '.js', '.java', '.cpp', '.c', '.h', '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt'}
    code_files = []
    token_counts = []
    file_names = []
    
    for root, dirs, files in os.walk(repo_path):
        # Skip common non-code directories
        dirs[:] = [d for d in dirs if d not in {'.git', 'node_modules', '__pycache__', '.venv', 'venv'}]
        
        for file in files:
            if Path(file).suffix.lower() in code_extensions:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    relative_path = os.path.relpath(file_path, repo_path)
                    file_data = {
                        'file_path': relative_path,
                        'text': content
                    }
                    
                    # Count tokens if tokenizer is provided
                    if tokenizer:
                        tokens = tokenizer.encode(content)
                        token_count = len(tokens)
                        file_data['token_count'] = token_count
                        token_counts.append(token_count)
                        file_names.append(relative_path)
                    
                    code_files.append(file_data)
                except Exception:
                    continue

    return code_files, token_counts, file_names

def generate_dataset(repo_name, output_dir="data/continued_pretraining", model_name=None):
    """Main function to clone repo, generate dataset, and cleanup."""
    clone_dir = f"temp_{repo_name}"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tokenizer if model name is provided
    tokenizer = None
    if model_name:
        print(f"Loading tokenizer for model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    try:
        # Clone repository
        print(f"Cloning repository: {repo_name}")
        cloned_path = clone_repo(repo_name, clone_dir)
        if cloned_path is None:
            print("Failed to clone repository")
            return False
        
        # Collect code files
        print("Collecting code files...")
        code_files, token_counts, file_names = collect_code_files(clone_dir, tokenizer)
        
        # Statistical analysis if token counting was performed
        if token_counts:
            token_counts = np.array(token_counts)
            min_length = np.min(token_counts)
            max_length = np.max(token_counts)
            avg_length = np.mean(token_counts)
            median_length = np.median(token_counts)
            std_length = np.std(token_counts)
            
            # Find files with min and max token counts
            min_idx = np.argmin(token_counts)
            max_idx = np.argmax(token_counts)
            min_file = file_names[min_idx]
            max_file = file_names[max_idx]
            
            print(f"\nToken Statistics:")
            print(f"  Min tokens: {min_length} (file: {min_file})")
            print(f"  Max tokens: {max_length} (file: {max_file})")
            print(f"  Average tokens: {avg_length:.2f}")
            print(f"  Median tokens: {median_length:.2f}")
            print(f"  Standard deviation: {std_length:.2f}")
            
            # Generate histogram
            plt.figure(figsize=(10, 6))
            plt.hist(token_counts, bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Number of Tokens')
            plt.ylabel('Frequency')
            plt.title(f'Token Distribution for {repo_name}')
            plt.grid(True, alpha=0.3)
            
            # Add statistics as text on the plot
            stats_text = f'Files: {len(token_counts)}\nMean: {avg_length:.0f}\nMedian: {median_length:.0f}\nStd: {std_length:.0f}'
            plt.text(0.7, 0.95, stats_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Save histogram
            histogram_file = os.path.join(output_dir, f"{repo_name}_cp_histogram.png")
            plt.savefig(histogram_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Token distribution histogram saved to: {histogram_file}")
        
        # Save dataset
        output_file = os.path.join(output_dir, f"{repo_name}_cp.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(code_files, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset saved to: {output_file}")
        print(f"Total files processed: {len(code_files)}")
        return True
        
    except Exception as e:
        print(f"Error processing repository: {e}")
        return False
    
    finally:
        # Clean up cloned repository
        if os.path.exists(clone_dir):
            print(f"Removing temporary directory: {clone_dir}")
            shutil.rmtree(clone_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate continued pretraining dataset from a repository")
    parser.add_argument("--repo_name", default="pylint-dev__pylint.1f8c4d9e", help="Name of the repository to process")
    parser.add_argument("--output-dir", default="data/continued_pretraining", 
                       help="Output directory for the dataset (default: data/continued_pretraining)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", 
                       help="Model name for tokenizer")

    args = parser.parse_args()
    generate_dataset(args.repo_name, args.output_dir, args.model)