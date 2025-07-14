#!/usr/bin/env bash


# This script runs the SWE-bench agent using a specified local host model and evaluates its performance.
# Here we use the specific template that follows XML format, where XML parsing is needed.
# Usage: ./a1_run_eval.sh MODEL_NAME
set -euo pipefail

# --- user-controlled settings -------------------------------------------------
INSTANCE_IDS="astropy__astropy-12907|astropy__astropy-13033|astropy__astropy-13236|astropy__astropy-13398|astropy__astropy-13453|astropy__astropy-13579|astropy__astropy-13977|astropy__astropy-14096|astropy__astropy-14182|astropy__astropy-14309|astropy__astropy-14365|astropy__astropy-14369|astropy__astropy-14508|astropy__astropy-14539|astropy__astropy-14598|astropy__astropy-14995|astropy__astropy-7166|astropy__astropy-7336|astropy__astropy-7606|astropy__astropy-7671|astropy__astropy-8707|astropy__astropy-8872"
REPO=astropy__astropy

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
    --instances.filter $INSTANCE_IDS \
    --instances.evaluate=True

  echo "Completed agent batch iteration $i/$NUM_ITERATIONS"
done