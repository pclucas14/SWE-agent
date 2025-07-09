#!/bin/bash
# filepath: run_swebench_eval.sh

# SWE-bench evaluation script
# This script runs the evaluation for 1r1m tasks.
# HOME_PATH="/home/zhengyanshi"
# TASK_DATA_PATH=$HOME_PATH/project/SWE-smith/logs/automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10/astropy__astropy.26d14786/task_insts/astropy__astropy.26d14786_ps.json
# PREDICTIONS_PATH=$HOME_PATH/project/SWE-agent/trajectories/zhengyanshi@microsoft.com/default__o3__t-0.00__p-1.00__c-2.00___patch_swesmith_astropy__astropy.26d14786_ps/preds.json
# NUM_WORKERS=64
# python $HOME_PATH/repo/SWE-bench/swebench/harness/run_evaluation_1r1m.py \
#     --dataset_name $TASK_DATA_PATH \
#     --predictions_path  $PREDICTIONS_PATH \
#     --max_workers $NUM_WORKERS \
#     --run_id "1r1m_eval"


HOME_PATH="/home/zhengyanshi/project"
TASK_DATA_PATH=$HOME_PATH/SWE-smith/logs/automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10/pylint-dev__pylint.1f8c4d9e/task_insts/pylint-dev__pylint.1f8c4d9e_ps.json
TASK_DATA_PATH=$HOME_PATH/SWE-smith/logs/automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10/astropy__astropy.26d14786/task_insts/astropy__astropy.26d14786_ps.json
PREDICTIONS_PATH=$HOME_PATH/SWE-agent/trajectories/zhengyanshi@microsoft.com/default__o3__t-0.00__p-1.00__c-2.00___patch_swesmith_pylint-dev__pylint.1f8c4d9e_ps/preds.json
PREDICTIONS_PATH=$HOME_PATH/SWE-agent/trajectories/zhengyanshi@microsoft.com/default__o1__t-0.00__p-1.00__c-2.00___patch_swesmith_pylint-dev__pylint.1f8c4d9e_ps/preds.json
PREDICTIONS_PATH=$HOME_PATH/SWE-agent/trajectories/zhengyanshi@microsoft.com/swesmith_gen_claude__gpt-4o__t-0.00__p-1.00__c-2.00___patch_swesmith_astropy__astropy.26d14786_ps/preds.json
NUM_WORKERS=64
python $HOME_PATH/SWE-bench/swebench/harness/run_evaluation_1r1m.py \
    --dataset_name $TASK_DATA_PATH \
    --predictions_path  $PREDICTIONS_PATH \
    --max_workers $NUM_WORKERS \
    --run_id "1r1m_eval"