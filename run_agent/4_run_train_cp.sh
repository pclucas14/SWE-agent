# llamafactory-cli train run_agent/llama_factory_config/train_1r1m_openhands_05B_agent.yaml output_dir=test_1r1m
# amlt workspace add $WORKSPACE_NAME --resource-group $RESOURCE_GROUP_NAME --subscription $SUBSCRIPTION_NAME

# Load environment variables from .env file
set -a  # automatically export all variables
source .env
set +a  # turn off automatic export

DATASET="pylint-dev__pylint.1f8c4d9e_cp"
# DATASET="astropy__astropy.26d14786_cp"
MODEL="Qwen/Qwen2.5-Coder-32B-instruct"
# MODEL="SWE-bench/SWE-agent-LM-32B"
MODEL_SLUG=$(echo "$MODEL" | sed 's|/|--|g')
EPOCH=1
LEARNING_RATE=1e-5
CONTEXT_LENGTH=32768
JOB_NAME=":run_1r1m_32B=run_cp_32B_$DATASET"
EXPERIMENT_NAME="${MODEL_SLUG}_cl${CONTEXT_LENGTH}_lr${LEARNING_RATE}_ep${EPOCH}_$DATASET"

amlt run run_agent/amlt_config/run_1r1m_32B_cp.yaml $JOB_NAME $EXPERIMENT_NAME \
    -t $VC_NAME \
    -w $WORKSPACE_NAME \
    -x "dataset=$DATASET model_name_or_path=$MODEL num_train_epochs=$EPOCH learning_rate=$LEARNING_RATE cutoff_len=$CONTEXT_LENGTH" \
    -y -d "Train ${MODEL} on 1R1M dataset"
