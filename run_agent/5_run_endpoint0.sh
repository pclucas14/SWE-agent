docker run --runtime=nvidia --gpus=all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/project/SWE-agent/amlt/o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10_train_traj_32B_cl32768_bs1x8_lr1e-5_ep5/1r1m_pylint-dev__pylint.1f8c4d9e/ckpt:/model \
    -p 8000:8000 \
    --env "CUDA_VISIBLE_DEVICES=0,1,2,3" \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model /model \
    --served-model-name qwen32b_1r1m_pylint-dev__pylint.1f8c4d9e_o3_sft_cl32768_lr1e-5_ep5 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --enable-prefix-caching \
    --disable-log-requests \
    --gpu-memory-utilization 0.95
