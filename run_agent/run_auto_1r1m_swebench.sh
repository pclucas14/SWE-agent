#!/usr/bin/env bash


# This script runs the SWE-bench agent using a specified local host model and evaluates its performance.
# Here we use the specific template that follows XML format, where XML parsing is needed.
set -euo pipefail

# --- user-controlled settings -------------------------------------------------
INSTANCE_IDS="astropy__astropy-12907|astropy__astropy-13033|astropy__astropy-13236|astropy__astropy-13398|astropy__astropy-13453|astropy__astropy-13579|astropy__astropy-13977|astropy__astropy-14096|astropy__astropy-14182|astropy__astropy-14309|astropy__astropy-14365|astropy__astropy-14369|astropy__astropy-14508|astropy__astropy-14539|astropy__astropy-14598|astropy__astropy-14995|astropy__astropy-7166|astropy__astropy-7336|astropy__astropy-7606|astropy__astropy-7671|astropy__astropy-8707|astropy__astropy-8872"

MODEL_NAME="openai/qwen32b_1r1m_astropy_o3_sft_cl32768_bs1x8_lr1e-5_ep2"     # <-- only change here
USER_RUN_ROOT="trajectories/zhengyanshi@microsoft.com"
OPENAI_API_BASE=http://127.0.0.1:8000/v1
OPENAI_API_KEY=LOCAL
# ------------------------------------------------------------------------------

# Compute slug used inside run directory names, e.g. openai--SWE-bench--SWE-agent-LM-32B
MODEL_SLUG=$(echo "$MODEL_NAME" | tr '/' '--')

# 1. Run the agent batch
sweagent run-batch \
  --num_workers 16 \
  --config agent/1r1m.yaml \
  --agent.type max_step \
  --agent.model.name "$MODEL_NAME" \
  --agent.model.api_base "$OPENAI_API_BASE" \
  --agent.model.api_key  "$OPENAI_API_KEY" \
  --agent.model.per_instance_cost_limit 0 \
  --agent.model.total_cost_limit 0 \
  --agent.model.max_input_tokens 24576 \
  --agent.max_steps 100 \
  --instances.shuffle True \
  --instances.type swe_bench \
  --instances.subset verified \
  --instances.split test \
  --instances.filter "$INSTANCE_IDS"

# 2. Locate the newest run directory that matches this model
RUN_DIR=$(ls -td ${USER_RUN_ROOT}/1r1m__${MODEL_SLUG}* 2>/dev/null | head -n1)

if [[ -z "$RUN_DIR" ]]; then
  echo "Error: no run directory found for model '$MODEL_NAME'." >&2
  exit 1
fi

PREDICTIONS_PATH="${RUN_DIR}/preds.json"

if [[ ! -f "$PREDICTIONS_PATH" ]]; then
  echo "Error: predictions file not found at '$PREDICTIONS_PATH'." >&2
  exit 1
fi

echo "Using predictions from: $PREDICTIONS_PATH"

# 3. Evaluate
python -m swebench.harness.run_evaluation \
    --dataset_name SWE-bench/SWE-bench_Verified \
    --predictions_path "$PREDICTIONS_PATH" \
    --max_workers 24 \
    --run_id swebench_verified