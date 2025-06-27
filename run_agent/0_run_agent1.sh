#!/bin/bash

DATAPATH="/home/zhengyanshi/project/SWE-smith/logs/automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10/astropy__astropy.26d14786/task_insts/astropy__astropy.26d14786_ps.json"
DATAPATH="/home/zhengyanshi/project/SWE-smith/logs/automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10/pylint-dev__pylint.1f8c4d9e/task_insts/pylint-dev__pylint.1f8c4d9e_ps.json"
python sweagent/run/run_1r1m_batch.py \
    --config config/default.yaml \
    --agent.type max_step \
    --agent.model.name o3 \
    --agent.max_steps 50 \
    --num_workers 64 \
    --agent.model.per_instance_cost_limit 2.00 \
    --instances.type patch_swesmith \
    --instances.path  ${DATAPATH} \
    --instances.shuffle=True