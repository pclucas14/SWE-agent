# llamafactory-cli train run_agent/llama_factory_config/train_1r1m_openhands_05B_agent.yaml output_dir=test_1r1m
# amlt workspace add $WORKSPACE_NAME --resource-group $RESOURCE_GROUP_NAME --subscription $SUBSCRIPTION_NAME

# Load environment variables from .env file
set -a  # automatically export all variables
source .env
set +a  # turn off automatic export

# Define arrays for parameters
EPOCHS=(2 5)
DATASETS=("astropy__astropy.26d14786" "pylint-dev__pylint.1f8c4d9e")
MODELS=("SWE-bench/SWE-agent-LM-32B" "Qwen/Qwen2.5-Coder-32B-instruct")

LEARNING_RATE=1e-5
CONTEXT_LENGTH=32768

# Loop through all combinations
for EPOCH in "${EPOCHS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            MODEL_SLUG=$(echo "$MODEL" | sed 's|/|--|g')
            JOB_NAME=":run_1r1m_32B_from_cp=from_cp_${MODEL_SLUG}_cl${CONTEXT_LENGTH}_lr${LEARNING_RATE}_ep${EPOCH}_${DATASET}"

            # Construct separate args for checkpoint and main training
            CP_ARGS="dataset=${DATASET}_cp model_name_or_path=$MODEL num_train_epochs=1 learning_rate=$LEARNING_RATE cutoff_len=$CONTEXT_LENGTH"
            MAIN_ARGS="dataset=$DATASET num_train_epochs=$EPOCH learning_rate=$LEARNING_RATE cutoff_len=$CONTEXT_LENGTH"

            amlt run run_agent/amlt_config/run_1r1m_32B_from_cp.yaml $JOB_NAME \
                -t $VC_NAME \
                -w $WORKSPACE_NAME \
                -x "CP_ARGS=\"$CP_ARGS\" MAIN_ARGS=\"$MAIN_ARGS\"" \
                -y -d "(Testing Run) Train ${MODEL} from checkpoint on ${DATASET}_cp to ${DATASET}"
        done
    done
done
