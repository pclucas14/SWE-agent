# llamafactory-cli train run_agent/llama_factory_config/train_1r1m_openhands_05B_agent.yaml output_dir=test_1r1m
# amlt workspace add $WORKSPACE_NAME --resource-group $RESOURCE_GROUP_NAME --subscription $SUBSCRIPTION_NAME

# Load environment variables from .env file
set -a  # automatically export all variables
source .env
set +a  # turn off automatic export

DATASET="SWE-bench/SWE-smith-trajectories"
MODEL="Qwen/Qwen2.5-Coder-32B-instruct"
MODEL_SLUG=$(echo "$MODEL" | sed 's|.*/||')
DATASET_SLUG=$(echo "$DATASET" | sed 's|.*/||')
EPOCH=3
LEARNING_RATE=5e-5
CONTEXT_LENGTH=32768
WARMUP_STEPS=5
WEIGHT_DECAY=0.01
JOB_NAME=":run_1r1m_32B=${MODEL_SLUG}_cl${CONTEXT_LENGTH}_lr${LEARNING_RATE}_ep${EPOCH}_w${WARMUP_STEPS}_wd${WEIGHT_DECAY}_${DATASET_SLUG}"

amlt run run_agent/amlt_config/run_1r1m_32B.yaml $JOB_NAME \
    -t $VC_NAME \
    -w $WORKSPACE_NAME \
    -x "dataset=$DATASET model_name_or_path=$MODEL num_train_epochs=$EPOCH learning_rate=$LEARNING_RATE cutoff_len=$CONTEXT_LENGTH warmup_steps=$WARMUP_STEPS weight_decay=$WEIGHT_DECAY" \
    -y -d "Train ${MODEL} Agent on SWE-smith dataset"
