#!/bin/bash

# This script runs the SWE-agent using SWE-smith pipeline and evaluates its performance.
set -euo pipefail

# --- user-controlled settings -------------------------------------------------
MODEL_NAME="gpt-4o"
MODEL_NAME="claude-sonnet-4"
HOME_PATH="/home/zhengyanshi/project"
USER_RUN_ROOT="trajectories/zhengyanshi@microsoft.com"
REPO_NAME="pylint-dev__pylint.1f8c4d9e"
REPO_NAME="astropy__astropy.26d14786"
CONFIG_FILE="swesmith_gen_claude"
NUM_WORKERS=64
MAX_STEPS=50
COST_LIMIT=2.00
NUM_ITERATIONS=1
SWESMITH_TASK_NAME=automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10
# ------------------------------------------------------------------------------

# Compute slug used inside run directory names
MODEL_SLUG=$(echo "$MODEL_NAME" | sed 's|/|--|g')

# Dynamic path generation
TASK_DATA_PATH="$HOME_PATH/SWE-smith/logs/$SWESMITH_TASK_NAME/$REPO_NAME/task_insts/${REPO_NAME}_ps.json"

echo "Starting SWE-agent run with model: $MODEL_NAME"
echo "Task data path: $TASK_DATA_PATH"

for i in $(seq 1 $NUM_ITERATIONS); do
  echo "Running agent batch iteration $i/$NUM_ITERATIONS..."
  python sweagent/run/run_1r1m_batch.py \
      --config agent/${CONFIG_FILE}.yaml \
      --suffix ms${MAX_STEPS}_as${i} \
      --agent.type max_step \
      --agent.model.name ${MODEL_NAME} \
      --agent.max_steps ${MAX_STEPS} \
      --num_workers ${NUM_WORKERS} \
      --agent.model.per_instance_cost_limit ${COST_LIMIT} \
      --instances.type patch_swesmith \
      --instances.path  ${TASK_DATA_PATH} \
      --instances.shuffle=True

  echo "Looking for run directories with pattern: ${USER_RUN_ROOT}/${CONFIG_FILE}__${MODEL_SLUG}*${REPO_NAME}*ms${MAX_STEPS}_as${i}*"
  RUN_DIR=$(ls -td ${USER_RUN_ROOT}/${CONFIG_FILE}__${MODEL_SLUG}*${REPO_NAME}*ms${MAX_STEPS}_as${i}* 2>/dev/null | head -n1)
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

  EVAL_RESULT_FILE="${RUN_DIR}/results.json"
  FOLDER_PATH=$(basename "$RUN_DIR")
  if [[ ! -f "$EVAL_RESULT_FILE" ]]; then
      echo "Error: evaluation results file not found at '$EVAL_RESULT_FILE'." >&2
      exit 1
  fi

  python run_script/process_trajectories_smith.py \
      --eval-file $EVAL_RESULT_FILE \
      --trajectories-folder $RUN_DIR \
      --folder-path $SWESMITH_TASK_NAME/$FOLDER_PATH \
      --repo-name $REPO_NAME

  echo "Processing completed for iteration $i/$NUM_ITERATIONS."
done

echo "All processing completed successfully!"