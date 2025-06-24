from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

# Re-use the dataset loader already shipped with SWE-bench
from swebench.harness.utils import load_swebench_dataset
from swebench.harness.constants import KEY_INSTANCE_ID

DEFAULT_DATASET = "princeton-nlp/SWE-bench_Verified"
DEFAULT_SPLIT = "test"


def sanitize_repo_name(repo: str) -> str:
    """
    Convert a repository name of the form 'org/repo' to the
    SWE-bench instance-id prefix 'org__repo'.
    """
    return repo.replace("/", "__")


def collect_instance_ids(repo: str, dataset_name: str, split: str) -> List[str]:
    repo_prefix = sanitize_repo_name(repo)
    dataset = load_swebench_dataset(dataset_name, split)
    return [
        inst[KEY_INSTANCE_ID]
        for inst in dataset
        if inst[KEY_INSTANCE_ID].startswith(repo_prefix)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List SWE-bench instance IDs belonging to a repository."
    )
    parser.add_argument("repo", help="Repository in the form 'org/repo'")
    parser.add_argument(
        "--dataset_name",
        default=DEFAULT_DATASET,
        help="HF dataset name or local JSON path (default: %(default)s)",
    )
    parser.add_argument(
        "--split", default=DEFAULT_SPLIT, help="Dataset split (default: %(default)s)"
    )
    args = parser.parse_args()

    ids = collect_instance_ids(args.repo, args.dataset_name, args.split)
    if not ids:
        print(f"No instances found for repo '{args.repo}' in {args.dataset_name}:{args.split}")
        return
    # Print space-separated list, suitable for `--instance_ids` argument
    print(f"Found {len(ids)} instances for repo '{args.repo}':")
    print("|".join(ids))


if __name__ == "__main__":
    main()
