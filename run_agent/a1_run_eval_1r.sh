#!/usr/bin/env bash


# This script runs the SWE-bench agent using a specified local host model and evaluates its performance.
# Here we use the specific template that follows XML format, where XML parsing is needed.
# Usage: ./a1_run_eval.sh MODEL_NAME
set -euo pipefail

# --- user-controlled settings -------------------------------------------------
INSTANCE_IDS="astropy__astropy-12907|astropy__astropy-13033|astropy__astropy-13236|astropy__astropy-13398|astropy__astropy-13453|astropy__astropy-13579|astropy__astropy-13977|astropy__astropy-14096|astropy__astropy-14182|astropy__astropy-14309|astropy__astropy-14365|astropy__astropy-14369|astropy__astropy-14508|astropy__astropy-14539|astropy__astropy-14598|astropy__astropy-14995|astropy__astropy-7166|astropy__astropy-7336|astropy__astropy-7606|astropy__astropy-7671|astropy__astropy-8707|astropy__astropy-8872"
REPO=astropy__astropy
INSTANCE_IDS="pylint-dev__pylint-4551|pylint-dev__pylint-4604|pylint-dev__pylint-4661|pylint-dev__pylint-4970|pylint-dev__pylint-6386|pylint-dev__pylint-6528|pylint-dev__pylint-6903|pylint-dev__pylint-7080|pylint-dev__pylint-7277|pylint-dev__pylint-8898"
REPO=pylint-dev__pylint

# Require MODEL_NAME as command line argument
if [[ $# -eq 0 ]]; then
    echo "Error: MODEL_NAME is required as a command line argument." >&2
    echo "Usage: $0 MODEL_NAME" >&2
    exit 1
fi

MODEL_NAME="openai/$1"
USER_RUN_ROOT="trajectories/zhengyanshi@microsoft.com"
OPENAI_API_BASE=http://127.0.0.1:8000/v1
OPENAI_API_KEY=LOCAL
MAX_STEPS=75
MAX_INPUT_TOKENS=None
MAX_WORKERS=50
NUM_ITERATIONS=1
CONFIG_FILE="swesmith_infer"
# ------------------------------------------------------------------------------

# Compute slug used inside run directory names, e.g. openai--SWE-bench--SWE-agent-LM-32B
MODEL_SLUG=$(echo "$MODEL_NAME" | sed 's|.*/||' | sed 's|/|--|g')

# 1. Run the agent batch 3 times
for i in $(seq 1 $NUM_ITERATIONS); do
  echo "Running agent batch iteration $i/$NUM_ITERATIONS..."
  sweagent run-batch \
    --num_workers ${MAX_WORKERS} \
    --config agent/${CONFIG_FILE}.yaml \
    --suffix ms${MAX_STEPS}_mit${MAX_INPUT_TOKENS}_as${i}_${REPO} \
    --agent.type max_step \
    --agent.model.name "$MODEL_NAME" \
    --agent.model.api_base "$OPENAI_API_BASE" \
    --agent.model.api_key  "$OPENAI_API_KEY" \
    --agent.model.per_instance_cost_limit 0 \
    --agent.model.total_cost_limit 0 \
    --agent.max_steps $MAX_STEPS \
    --instances.shuffle True \
    --instances.type swe_bench \
    --instances.subset verified \
    --instances.split test \
    --instances.filter $INSTANCE_IDS
  
  echo "Completed agent batch iteration $i/$NUM_ITERATIONS"
done

# 2. Run evaluations for all completed runs
for i in $(seq 1 $NUM_ITERATIONS); do
  echo "Running evaluation for iteration $i/$NUM_ITERATIONS..."

  echo "Looking for run directories with pattern: ${USER_RUN_ROOT}/${CONFIG_FILE}__${MODEL_SLUG}*ms${MAX_STEPS}_mit${MAX_INPUT_TOKENS}_as${i}_${REPO}*"

  # Locate the run directory that matches this model and suffix
  # Updated to handle different possible directory naming patterns
  RUN_DIR=$(ls -td ${USER_RUN_ROOT}/${CONFIG_FILE}__*${MODEL_SLUG}*ms${MAX_STEPS}_mit${MAX_INPUT_TOKENS}_as${i}_${REPO}* 2>/dev/null | head -n1)

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
      --max_workers 50 \
      --run_id swebench_verified_run${i}; then
    echo "Error: Evaluation failed for iteration $i" >&2
    exit 1
  fi
  
  echo "Completed evaluation for iteration $i/$NUM_ITERATIONS"
done

python run_script/analyze_results.py \
    --results_path . \
    --model_name $MODEL_SLUG