docker run --runtime=nvidia --gpus=all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/project/SWE-agent/amlt/o3_SWE-bench--SWE-agent-LM-32B_cl32768_lr1e-5_ep2_astropy__astropy.26d14786/run_1r1m_32B_astropy__astropy.26d14786:/model \
    -p 8001:8000 \
    --env "CUDA_VISIBLE_DEVICES=2,3" \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model /model \
    --served-model-name o3_SWE-bench--SWE-agent-LM-32B_cl32768_lr1e-5_ep2_astropy__astropy.26d14786 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --enable-prefix-caching \
    --disable-log-requests \
    --gpu-memory-utilization 0.95
