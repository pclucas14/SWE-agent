#!/usr/bin/env bash
set -Eeuo pipefail

################################################################################
# USER SECTION –- list the models you want to run (one per line).
# Each entry is:  <model_dir>::<served_model_name>
#                └───────────┘└────────────────────┘
MODEL_DIR_PREFIX="/home/zhengyanshi/project/SWE-agent" 
MODELS=(
  "${MODEL_DIR_PREFIX}/amlt/driven-mayfly/tt_SWE-agent-LM-32B_cl32768_lr1e-4_ep1_pylint-dev__pylint.1f8c4d9e_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o/epoch_0::tt_SWE-agent-LM-32B_cl32768_lr1e-4_ep1_pylint-dev__pylint.1f8c4d9e_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o"
  "${MODEL_DIR_PREFIX}/amlt/enough-toucan/tt_SWE-agent-LM-32B_cl32768_lr1e-5_ep1_pylint-dev__pylint.1f8c4d9e_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o/epoch_0::tt_SWE-agent-LM-32B_cl32768_lr1e-5_ep1_pylint-dev__pylint.1f8c4d9e_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o"
  "${MODEL_DIR_PREFIX}/amlt/peaceful-turtle/tt_SWE-agent-LM-32B_cl32768_lr5e-5_ep1_pylint-dev__pylint.1f8c4d9e_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o/epoch_0::tt_SWE-agent-LM-32B_cl32768_lr5e-5_ep1_pylint-dev__pylint.1f8c4d9e_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o"
  # "${MODEL_DIR_PREFIX}/amlt/crack-muskrat/tt_SWE-agent-LM-32B_cl32768_lr1e-4_ep1_astropy__astropy.26d14786_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o/epoch_0::tt_SWE-agent-LM-32B_cl32768_lr1e-4_ep1_astropy__astropy.26d14786_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o"
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
INSTANCE_IDS="astropy__astropy-12907|astropy__astropy-13033|astropy__astropy-13236|astropy__astropy-13398|astropy__astropy-13453|astropy__astropy-13579|astropy__astropy-13977|astropy__astropy-14096|astropy__astropy-14182|astropy__astropy-14309|astropy__astropy-14365|astropy__astropy-14369|astropy__astropy-14508|astropy__astropy-14539|astropy__astropy-14598|astropy__astropy-14995|astropy__astropy-7166|astropy__astropy-7336|astropy__astropy-7606|astropy__astropy-7671|astropy__astropy-8707|astropy__astropy-8872"

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
     run_agent/a1_run_eval_1r.sh "${SERVED_NAME}"

  echo "Finished job for ${SERVED_NAME}"

  # 5) Stop & remove container
  cleanup
  trap - EXIT INT TERM   # clear trap so it doesn’t run twice
done

echo "All models processed successfully!"
