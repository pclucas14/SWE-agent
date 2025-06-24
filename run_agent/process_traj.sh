
EVAL_RESULT_FILE="data/default__o3__t-0.00__p-1.00__c-2.00___patch_swesmith_astropy__astropy.26d14786_ps.1r1m_eval.json"
TRAJ_FOLDER="trajectories/zhengyanshi@microsoft.com/default__o3__t-0.00__p-1.00__c-2.00___patch_swesmith_astropy__astropy.26d14786_ps"
TASK_DATASET="automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10"
python run_script/process_trajectories_smith.py \
    --eval-file $EVAL_RESULT_FILE \
    --trajectories-folder $TRAJ_FOLDER \
    --dataset-name $TASK_DATASET