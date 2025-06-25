docker run --runtime=nvidia --gpus=all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/project/SWE-agent/amlt/apparent-perch/automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10_train_traj_32B_cl32768_bs1x8_lr1e-5_ep5/ckpt/automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10_train_traj_32B_cl32768_bs1x8_lr1e-5_ep5:/model \
    -p 8000:8000 \
    --env "CUDA_VISIBLE_DEVICES=0,1,2,3" \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model /model \
    --served-model-name automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10_train_traj_32B_cl32768_bs1x8_lr1e-5_ep5 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --enable-prefix-caching \
    --disable-log-requests \
    --gpu-memory-utilization 0.95
