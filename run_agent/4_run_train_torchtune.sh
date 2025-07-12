# llamafactory-cli train run_agent/llama_factory_config/train_1r1m_openhands_05B_agent.yaml output_dir=test_1r1m
# amlt workspace add $WORKSPACE_NAME --resource-group $RESOURCE_GROUP_NAME --subscription $SUBSCRIPTION_NAME

# Load environment variables from .env file
set -a  # automatically export all variables
source .env
set +a  # turn off automatic export

DATA_FOLDER="data/automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10/swesmith_gen_claude__gpt-4o__t-0.00__p-1.00__c-2.00___patch_swesmith_astropy__astropy.26d14786_ps"
TRAJ_MODEL="gpt-4o"
DATA_FOLDER=data/automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10/swesmith_gen_claude__gpt-4.1__t-0.00__p-1.00__c-2.00___patch_swesmith_astropy__astropy.26d14786_ps
TRAJ_MODEL="gpt-4.1"
DATA_FOLDER=data/automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10/swesmith_gen_claude__gpt-4.1__t-0.00__p-1.00__c-2.00___patch_swesmith_pylint-dev__pylint.1f8c4d9e_ps_ms50_as1
TRAJ_MODEL="gpt-4.1"
DATA_FOLDER=data/automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10/swesmith_gen_claude__gpt-4o__t-0.00__p-1.00__c-2.00___patch_swesmith_pylint-dev__pylint.1f8c4d9e_ps_ms50_as1
TRAJ_MODEL="gpt-4o"
DATA_FOLDER=data/automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10/swesmith_gen_claude__gpt-4o_gpt-4.1__t-0.00__p-1.00__c-2.00___patch_swesmith_astropy__astropy.26d14786_ps_ms50_as1
TRAJ_MODEL="gpt-4o_gpt-4.1"

DATASETS=("astropy__astropy.26d14786_ml32700" "astropy__astropy.26d14786_full")
# DATASETS=("pylint-dev__pylint.1f8c4d9e_ml32700" "pylint-dev__pylint.1f8c4d9e_full")

DATA_FOLDER=data/automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10/swesmith_gen_claude__claude-sonnet-4_gpt4.1_gpt-4o__t-0.00__p-1.00__c-2.00___patch_swesmith_astropy__astropy.26d14786_ps
TRAJ_MODEL="claude__claude-sonnet-4_gpt4.1_gpt-4o"
DATASETS=("astropy__astropy.26d14786_submit")
MODEL="Qwen/Qwen2.5-Coder-32B-instruct"
MODEL_SLUG=$(echo "$MODEL" | sed 's|.*/||')
EPOCHS=(3 2 1)
LEARNING_RATES=(1e-4 5e-5 1e-5)
CONTEXT_LENGTH=32768
    
for EPOCH in "${EPOCHS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
            JOB_NAME="tt_${MODEL_SLUG}_cl${CONTEXT_LENGTH}_lr${LEARNING_RATE}_ep${EPOCH}_${DATASET}_${TRAJ_MODEL}"

            amlt run run_agent/amlt_config/torch_run_32B.yaml :torch_run_32B_1r1m=$JOB_NAME \
                -t $VC_NAME \
                -w $WORKSPACE_NAME \
                -x "dataset.data_files=$DATA_FOLDER/$DATASET.json epochs=$EPOCH optimizer.lr=$LEARNING_RATE tokenizer.max_seq_len=$CONTEXT_LENGTH exp_name=$JOB_NAME" \
                -y -d "Train Qwen 3 Coder 32B Model on Froggy dataset using torch tune"
        done
    done
done