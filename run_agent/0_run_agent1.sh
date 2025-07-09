#!/bin/bash

DATAPATH="/home/zhengyanshi/project/SWE-smith/logs/automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10/astropy__astropy.26d14786/task_insts/astropy__astropy.26d14786_ps.json"
DATAPATH="/home/zhengyanshi/project/SWE-smith/logs/automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10/pylint-dev__pylint.1f8c4d9e/task_insts/pylint-dev__pylint.1f8c4d9e_ps.json"

HOME_PATH="/home/zhengyanshi"
TASK_DATA_PATH=$HOME_PATH/project/SWE-smith/logs/automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10/pylint-dev__pylint.1f8c4d9e/task_insts/pylint-dev__pylint.1f8c4d9e_ps.json
NUM_WORKERS=64

python sweagent/run/run_1r1m_batch.py \
    --config agent/swesmith_gen_claude.yaml \
    --agent.type max_step \
    --agent.model.name gpt-4.1 \
    --agent.max_steps 50 \
    --num_workers ${NUM_WORKERS} \
    --agent.model.per_instance_cost_limit 2.00 \
    --instances.type patch_swesmith \
    --instances.path  ${TASK_DATA_PATH} \
    --instances.shuffle=True


PREDICTIONS_PATH=$HOME_PATH/project/SWE-agent/trajectories/zhengyanshi@microsoft.com/default__o1__t-0.00__p-1.00__c-2.00___patch_swesmith_pylint-dev__pylint.1f8c4d9e_ps/preds.json

python $HOME_PATH/repo/SWE-bench/swebench/harness/run_evaluation_1r1m.py \
    --dataset_name ${TASK_DATA_PATH} \
    --predictions_path  ${PREDICTIONS_PATH} \
    --max_workers ${NUM_WORKERS} \
    --run_id "1r1m_eval"