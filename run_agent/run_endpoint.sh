docker run --runtime=nvidia --gpus=all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --env "CUDA_VISIBLE_DEVICES=0,1,2,3" \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-32B-Instruct \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --disable-log-requests \
    --gpu-memory-utilization 0.95
