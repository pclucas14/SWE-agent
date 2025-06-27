docker run --runtime=nvidia --gpus=all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --env "CUDA_VISIBLE_DEVICES=0,1,2,3" \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model all-hands/openhands-lm-32b-v0.1 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --enable-prefix-caching \
    --disable-log-requests \
    --gpu-memory-utilization 0.95
