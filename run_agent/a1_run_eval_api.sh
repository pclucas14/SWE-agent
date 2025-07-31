#!/usr/bin/env bash


# This script runs the SWE-bench agent using a specified local host model and evaluates its performance.
# Here we use the specific template that follows XML format, where XML parsing is needed.
# Usage: ./a1_run_eval_api.sh
set -euo pipefail

MODEL_NAME="gpt-4.1"
USER_RUN_ROOT="trajectories/zhengyanshi@microsoft.com"
MAX_STEPS=75
MAX_INPUT_TOKENS=None
MAX_WORKERS=72
NUM_ITERATIONS=3
CONFIG_FILE="swesmith_gen_claude"
# ------------------------------------------------------------------------------

# 1. Run the agent batch 3 times
for i in $(seq 1 $NUM_ITERATIONS); do
  echo "Running agent batch iteration $i/$NUM_ITERATIONS..."
  sweagent run-batch \
    --num_workers ${MAX_WORKERS} \
    --config agent/${CONFIG_FILE}.yaml \
    --suffix ms${MAX_STEPS}_mit${MAX_INPUT_TOKENS}_as${i} \
    --agent.type max_step \
    --agent.model.name "$MODEL_NAME" \
    --agent.model.per_instance_cost_limit 0 \
    --agent.model.total_cost_limit 0 \
    --agent.max_steps $MAX_STEPS \
    --instances.shuffle True \
    --instances.type swe_bench \
    --instances.subset verified \
    --instances.split test
  
  echo "Completed agent batch iteration $i/$NUM_ITERATIONS"
done

# 2. Run evaluations for all completed runs
for i in $(seq 1 $NUM_ITERATIONS); do
  echo "Running evaluation for iteration $i/$NUM_ITERATIONS..."

  echo "Looking for run directories with pattern: ${USER_RUN_ROOT}/${CONFIG_FILE}__${MODEL_NAME}*ms${MAX_STEPS}_mit${MAX_INPUT_TOKENS}_as${i}*"

  # Locate the run directory that matches this model and suffix
  # Updated to handle different possible directory naming patterns
  RUN_DIR=$(ls -td ${USER_RUN_ROOT}/${CONFIG_FILE}__*${MODEL_NAME}*ms${MAX_STEPS}_mit${MAX_INPUT_TOKENS}_as${i}* 2>/dev/null | head -n1)

  if [[ -z "$RUN_DIR" ]]; then
    echo "Error: no run directory found for model '$MODEL_NAME' iteration $i." >&2
    exit 1
  fi

  echo "Found run directory: $RUN_DIR"

  PREDICTIONS_PATH="${RUN_DIR}/preds.json"

  if [[ ! -f "$PREDICTIONS_PATH" ]]; then
    echo "Error: predictions file not found at '$PREDICTIONS_PATH'." >&2
    echo "Contents of run directory:"
    ls -la "$RUN_DIR"
    exit 1
  fi

  echo "Using predictions from: $PREDICTIONS_PATH"
  echo "File size: $(du -h "$PREDICTIONS_PATH")"

  # 3. Evaluate
  echo "Starting evaluation..."
  if ! python -m swebench.harness.run_evaluation \
      --dataset_name SWE-bench/SWE-bench_Verified \
      --predictions_path "$PREDICTIONS_PATH" \
      --max_workers 72 \
      --run_id swebench_verified_run${i}; then
    echo "Error: Evaluation failed for iteration $i" >&2
    exit 1
  fi
  
  echo "Completed evaluation for iteration $i/$NUM_ITERATIONS"
done