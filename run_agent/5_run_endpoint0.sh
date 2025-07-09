docker run --runtime=nvidia --gpus=all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/project/SWE-agent/amlt/equipped-worm/tt_SWE-agent-LM-32B_cl32768_lr5e-5_ep3_astropy__astropy.26d14786_full/epoch_2:/model \
    -p 8000:8000 \
    --env "CUDA_VISIBLE_DEVICES=0,1,2,3" \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model /model \
    --served-model-name tt_SWE-agent-LM-32B_cl32768_lr5e-5_ep3_astropy__astropy.26d14786_full \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --enable-prefix-caching \
    --disable-log-requests \
    --gpu-memory-utilization 0.95
