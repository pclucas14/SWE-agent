llamafactory-cli train run_agent/llama_factory_config/train_1r1m_openhands_05B_agent.yaml output_dir=test_1r1m

amlt workspace add $WORKSPACE_NAME --resource-group $RESOURCE_GROUP_NAME --subscription $SUBSCRIPTION_NAME
amlt run run_agent/amlt_config/run_1r1m_32B.yaml -t $VC_NAME -w $WORKSPACE_NAME -y -d "Train Qwen 2.5 Coder 32B Model on 1R1M dataset"
