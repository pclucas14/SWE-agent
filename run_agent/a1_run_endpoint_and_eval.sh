#!/usr/bin/env bash
set -Eeuo pipefail

################################################################################
# USER SECTION –- list the models you want to run (one per line).
# Each entry is:  <model_dir>::<served_model_name>
#                └───────────┘└────────────────────┘
MODEL_DIR_PREFIX="/home/zhengyanshi/project/SWE-agent" 
MODELS=(
  "${MODEL_DIR_PREFIX}/amlt/assured-drake/tt_SWE-agent-LM-32B_cl32768_lr1e-5_ep1_astropy__astropy.26d14786_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o/epoch_0::tt_SWE-agent-LM-32B_cl32768_lr1e-5_ep1_astropy__astropy.26d14786_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o"
  "${MODEL_DIR_PREFIX}/amlt/mint-orca/tt_SWE-agent-LM-32B_cl32768_lr1e-5_ep2_astropy__astropy.26d14786_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o/epoch_1::tt_SWE-agent-LM-32B_cl32768_lr1e-5_ep2_astropy__astropy.26d14786_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o"
  "${MODEL_DIR_PREFIX}/amlt/pumped-grizzly/tt_SWE-agent-LM-32B_cl32768_lr1e-5_ep3_astropy__astropy.26d14786_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o/epoch_2::tt_SWE-agent-LM-32B_cl32768_lr1e-5_ep3_astropy__astropy.26d14786_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o"
  # add more here ...
)
################################################################################

# Settings shared by every container/run
PORT=8000
GPU_DEVICES="0,1,2,3"
TP=4
GPU_UTIL=0.95
IMAGE="vllm/vllm-openai:latest"
CACHE_MOUNT="$HOME/.cache/huggingface"
IPC="host"

# ––– helper: wait for server readiness ––––––––––––––––––––––––––––––––––––––––
wait_for_ready () {
  local retries=60
  local delay=5
  until curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; do
    ((retries--)) || { echo "[ERROR] vLLM did not become ready in time." >&2; return 1; }
    sleep "$delay"
  done
}

# ––– main loop ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
for ENTRY in "${MODELS[@]}"; do
  MODEL_DIR=${ENTRY%%::*}              # left of ::
  SERVED_NAME=${ENTRY##*::}            # right of ::

  echo
  echo "═══════════════════════════════════════════════════════════════════════"
  echo "Launching vLLM for model: ${SERVED_NAME}"
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
          --served-model-name "${SERVED_NAME}" \
          --trust-remote-code \
          --tensor-parallel-size "${TP}" \
          --enable-prefix-caching \
          --disable-log-requests \
          --gpu-memory-utilization "${GPU_UTIL}")
  
  # 2) Ensure we always clean up if something fails
  cleanup() {
    echo "Stopping container ${CID}..."
    docker stop -t 30 "${CID}" >/dev/null 2>&1 || true
    docker rm   -f   "${CID}" >/dev/null 2>&1 || true
  }
  trap cleanup EXIT INT TERM

  # 3) Wait until the REST endpoint is live
  echo "Waiting for vLLM to be ready ..."
  wait_for_ready
  echo "vLLM is ready!"

  # 4) Run your SWE-bench batch/eval script
  echo "Running SWE-bench agent + evaluation …"
  OPENAI_API_KEY="LOCAL" \
     run_agent/a1_run_eval.sh "${SERVED_NAME}"

  echo "Finished job for ${SERVED_NAME}"

  # 5) Stop & remove container
  cleanup
  trap - EXIT INT TERM   # clear trap so it doesn’t run twice
done

echo "All models processed successfully!"
