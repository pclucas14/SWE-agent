#!/bin/bash


python -m swebench.harness.run_evaluation \
    --dataset_name SWE-bench/SWE-bench_Verified \
    --predictions_path traj/r2egym-32b-agent-swebench-eval-20250602_124544.predictions.json \
    --max_workers 24 \
    --run_id swebench_verified