# This takes the agent trajectory on those swesmith generated instances and merges the predictions into a single file.

FOLDER_PATH="trajectories/zhengyanshi@microsoft.com/default__o3__t-0.00__p-1.00__c-2.00___patch_swesmith_astropy__astropy.26d14786_ps"
FOLDER_PATH="trajectories/zhengyanshi@microsoft.com/default__o3__t-0.00__p-1.00__c-2.00___patch_swesmith_pylint-dev__pylint.1f8c4d9e_ps"
FOLDER_PATH="trajectories/zhengyanshi@microsoft.com/default__o1__t-0.00__p-1.00__c-2.00___patch_swesmith_pylint-dev__pylint.1f8c4d9e_ps"
python sweagent/run/merge_predictions.py \
    ${FOLDER_PATH}