DATASET="automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10_train_traj"
MODEL="Qwen/Qwen2.5-Coder-32B-instruct"
TAG_NAME="1r1m_astropy_o3"

amlt run \
    --search run_agent/amlt_config/sweep.yaml \
    qwen32b_1r1m_astropy_o3_sft_sweep_2 \
    -t "$VC_NAME" \
    -w "$WORKSPACE_NAME" \
    -x "dataset=$DATASET model_name_or_path=$MODEL" \
    -d "Qwen-32B sweep cutoff_len/lr/epochs"


