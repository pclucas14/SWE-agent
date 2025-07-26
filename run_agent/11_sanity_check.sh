#!/bin/bash

# This script runs the SWE-agent using SWE-smith pipeline and evaluates its performance.
set -euo pipefail

################################################################################
# USER SECTION –- list the models you want to run (one per line).
# Each entry is:  <model_dir>::<served_model_name>
#                └───────────┘└────────────────────┘
MODEL_DIR_PREFIX="/home/zhengyanshi/project/SWE-agent" 
MODELS=(
  "${MODEL_DIR_PREFIX}/amlt/handy-foxhound/lf_SWE-bench--SWE-agent-LM-32B_cl32768_lr1e-6_ep3_astropy__astropy.26d14786_submit::lf_SWE-bench--SWE-agent-LM-32B_cl32768_lr1e-6_ep3_astropy__astropy.26d14786_submit"
  "${MODEL_DIR_PREFIX}/amlt/verified-meerkat/tt_SWE-agent-LM-32B_cl32768_lr1e-6_ep3_astropy__astropy.26d14786_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o/epoch_2::tt_SWE-agent-LM-32B_cl32768_lr1e-6_ep3_astropy__astropy.26d14786_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o"
  # "${MODEL_DIR_PREFIX}/amlt/certain-sunbird/tt_SWE-agent-LM-32B_cl32768_lr1e-4_ep3_astropy__astropy.26d14786_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o/epoch_2::tt_SWE-agent-LM-32B_cl32768_lr1e-4_ep3_astropy__astropy.26d14786_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o"

  # "${MODEL_DIR_PREFIX}/amlt/assured-drake/tt_SWE-agent-LM-32B_cl32768_lr1e-5_ep1_astropy__astropy.26d14786_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o/epoch_0::tt_SWE-agent-LM-32B_cl32768_lr1e-5_ep1_astropy__astropy.26d14786_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o"
  # "${MODEL_DIR_PREFIX}/amlt/normal-primate/tt_SWE-agent-LM-32B_cl32768_lr5e-5_ep1_astropy__astropy.26d14786_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o/epoch_0::tt_SWE-agent-LM-32B_cl32768_lr5e-5_ep1_astropy__astropy.26d14786_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o"
  # "${MODEL_DIR_PREFIX}/amlt/amlt/crack-muskrat/tt_SWE-agent-LM-32B_cl32768_lr1e-4_ep1_astropy__astropy.26d14786_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o/epoch_0::tt_SWE-agent-LM-32B_cl32768_lr1e-4_ep1_astropy__astropy.26d14786_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o"

)
################################################################################

# --- user-controlled settings -------------------------------------------------
HOME_PATH="/home/zhengyanshi/project"
USER_RUN_ROOT="trajectories/zhengyanshi@microsoft.com"
CONFIG_FILE="swesmith_infer"
NUM_WORKERS=32
MAX_STEPS=75
COST_LIMIT=0  # Set to 0 for local vLLM endpoint
NUM_ITERATIONS=1
SWESMITH_TASK_NAME=automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10
TRAJ_PATH=swesmith_gen_claude__claude-sonnet-4_gpt4.1_gpt-4o__t-0.00__p-1.00__c-2.00___patch_swesmith_astropy__astropy.26d14786_ps

# Settings shared by every container/run
PORT=8000
GPU_DEVICES="0,1,2,3"
TP=4
GPU_UTIL=0.95
IMAGE="vllm/vllm-openai:latest"
CACHE_MOUNT="$HOME/.cache/huggingface"
IPC="host"
OPENAI_API_BASE=http://127.0.0.1:8000/v1
OPENAI_API_KEY=LOCAL
# ------------------------------------------------------------------------------

# Dynamic path generation
TASK_DATA_PATH="data/$SWESMITH_TASK_NAME/$TRAJ_PATH/filtered_swesmith_task.json"

# ––– helper: wait for server readiness ––––––––––––––––––––––––––––––––––––––––
wait_for_ready () {
  local retries=60
  local delay=5
  until curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; do
    ((retries--)) || { echo "[ERROR] vLLM did not become ready in time." >&2; return 1; }
    sleep "$delay"
  done
}

# ––– cleanup: stop and remove container ––––––––––––––––––––––––––––––––––––––
cleanup() {
  if [[ -n "${CID:-}" ]]; then
    echo "Stopping container ${CID}..."
    docker stop -t 30 "${CID}" >/dev/null 2>&1 || true
    docker rm   -f   "${CID}" >/dev/null 2>&1 || true
  fi
}

echo "Task data path: $TASK_DATA_PATH"

# ––– main loop ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
for ENTRY in "${MODELS[@]}"; do
  MODEL_DIR=${ENTRY%%::*}              # left of ::
  BASE_MODEL_NAME=${ENTRY##*::}        # right of ::
  MODEL_NAME="openai/$BASE_MODEL_NAME"

  # Compute slug used inside run directory names
  MODEL_SLUG=$(echo "$MODEL_NAME" | sed 's|/|--|g')

  echo
  echo "═══════════════════════════════════════════════════════════════════════"
  echo "Launching vLLM for model: ${BASE_MODEL_NAME}"
  echo "From directory:           ${MODEL_DIR}"
  echo "═══════════════════════════════════════════════════════════════════════"

  # 1) Start container detached and capture container ID
  CID=$(docker run -d --runtime=nvidia --gpus=all \
        -v "${CACHE_MOUNT}:/root/.cache/huggingface" \
        -v "${MODEL_DIR}:/model" \
        -p ${PORT}:8000 \
        --env "CUDA_VISIBLE_DEVICES=${GPU_DEVICES}" \
        --ipc="${IPC}" \
        "${IMAGE}" \
          --model /model \
          --served-model-name "${BASE_MODEL_NAME}" \
          --trust-remote-code \
          --tensor-parallel-size "${TP}" \
          --enable-prefix-caching \
          --disable-log-requests \
          --gpu-memory-utilization "${GPU_UTIL}")
  
  # 2) Ensure we always clean up if something fails
  trap cleanup EXIT INT TERM

  # 3) Wait until the REST endpoint is live
  echo "Waiting for vLLM to be ready ..."
  wait_for_ready
  echo "vLLM is ready!"

  # 4) Run the SWE-agent iterations
  for i in $(seq 1 $NUM_ITERATIONS); do
    echo "Running agent batch iteration $i/$NUM_ITERATIONS..."
    python sweagent/run/run_1r1m_batch.py \
        --config agent/${CONFIG_FILE}.yaml \
        --suffix ms${MAX_STEPS}_as${i}_sanity_check \
        --agent.type max_step \
        --agent.model.name ${MODEL_NAME} \
        --agent.model.api_base ${OPENAI_API_BASE} \
        --agent.model.api_key ${OPENAI_API_KEY} \
        --agent.max_steps ${MAX_STEPS} \
        --num_workers ${NUM_WORKERS} \
        --agent.model.per_instance_cost_limit ${COST_LIMIT} \
        --agent.model.total_cost_limit 0 \
        --instances.type patch_swesmith \
        --instances.path  ${TASK_DATA_PATH} \
        --instances.shuffle=True

    echo "Looking for run directories with pattern: ${USER_RUN_ROOT}/${CONFIG_FILE}__${MODEL_SLUG}*ms${MAX_STEPS}_as${i}_sanity_check*"
    RUN_DIR=$(ls -td ${USER_RUN_ROOT}/${CONFIG_FILE}__${MODEL_SLUG}*ms${MAX_STEPS}_as${i}_sanity_check* 2>/dev/null | head -n1)
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
    if ! python $HOME_PATH/SWE-bench/swebench/harness/run_evaluation_1r1m.py \
        --dataset_name ${TASK_DATA_PATH} \
        --predictions_path  ${PREDICTIONS_PATH} \
        --max_workers ${NUM_WORKERS} \
        --run_id "1r1m_eval_run${i}"; then
        echo "Error: Evaluation failed for iteration $i" >&2
        exit 1
    fi
  done

  echo "Finished job for ${BASE_MODEL_NAME}"

  # 5) Stop & remove container
  cleanup
  trap - EXIT INT TERM   # clear trap so it doesn't run twice
done

echo "All models processed successfully!"