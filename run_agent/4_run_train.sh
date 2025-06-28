# llamafactory-cli train run_agent/llama_factory_config/train_1r1m_openhands_05B_agent.yaml output_dir=test_1r1m
# amlt workspace add $WORKSPACE_NAME --resource-group $RESOURCE_GROUP_NAME --subscription $SUBSCRIPTION_NAME

DATASET="pylint-dev__pylint.1f8c4d9e"
MODEL="Qwen/Qwen2.5-Coder-32B-instruct"
EPOCH=5
JOB_NAME=":run_1r1m_32B"
EXPERIMENT_NAME="o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10_train_traj_32B_cl32768_bs1x8_lr1e-5_ep$EPOCH"

amlt run run_agent/amlt_config/run_1r1m_32B.yaml $JOB_NAME $EXPERIMENT_NAME \
    -t $VC_NAME \
    -w $WORKSPACE_NAME \
    -x "dataset=$DATASET model_name_or_path=$MODEL num_train_epochs=$EPOCH" \
    -y -d "Train Qwen 2.5 Coder 32B Model on 1R1M dataset"
