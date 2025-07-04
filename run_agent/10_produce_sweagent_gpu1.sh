# llamafactory-cli train run_agent/llama_factory_config/train_1r1m_openhands_05B_agent.yaml output_dir=test_1r1m
# amlt workspace add $WORKSPACE_NAME --resource-group $RESOURCE_GROUP_NAME --subscription $SUBSCRIPTION_NAME

# Load environment variables from .env file
set -a  # automatically export all variables
source .env
set +a  # turn off automatic export

export VC_NAME="msrresrchbasicvc"
amlt run run_agent/amlt_config/torch_run_32B_h100_mem.yaml \
    -t $VC_NAME \
    -w $WORKSPACE_NAME \
    -y -d "Reproduce using torch tune on SWE-bench dataset with 2*8 H100 GPUs"

# tune run --nnodes 1 --nproc_per_node 4 \
#       --rdzv_backend c10d \
#       --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
#       full_finetune_distributed \
#       --config run_agent/torchrun_config/full_ft_qwen_05b.yml \
#       output_dir=./ckpt