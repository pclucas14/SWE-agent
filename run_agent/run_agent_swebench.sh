INSTANCE_IDS="astropy__astropy-12907|astropy__astropy-13033|astropy__astropy-13236|astropy__astropy-13398|astropy__astropy-13453|astropy__astropy-13579|astropy__astropy-13977|astropy__astropy-14096|astropy__astropy-14182|astropy__astropy-14309|astropy__astropy-14365|astropy__astropy-14369|astropy__astropy-14508|astropy__astropy-14539|astropy__astropy-14598|astropy__astropy-14995|astropy__astropy-7166|astropy__astropy-7336|astropy__astropy-7606|astropy__astropy-7671|astropy__astropy-8707|astropy__astropy-8872"

# Terminal
export OPENAI_API_BASE=http://127.0.0.1:8000/v1    # your local server
export OPENAI_API_KEY=LOCAL                    # dummy key if the server ignores it

sweagent run-batch \
  --num_workers 16 \
  --config agent/1r1m.yaml \
  --agent.type max_step \
  --agent.model.name "openai/SWE-bench/SWE-agent-LM-32B" \
  --agent.model.api_base "$OPENAI_API_BASE" \
  --agent.model.api_key  "$OPENAI_API_KEY" \
  --agent.model.per_instance_cost_limit 0 \
  --agent.model.total_cost_limit 0 \
  --agent.model.max_input_tokens 24576 \
  --agent.max_steps 50 \
  --instances.shuffle True \
  --instances.type swe_bench \
  --instances.subset verified \
  --instances.split test \
  --instances.filter "$INSTANCE_IDS" \
  --redo_existing True

predictions_path="trajectories/zhengyanshi@microsoft.com/1r1m__openai--SWE-bench--SWE-agent-LM-32B__t-0.00__p-1.00__c-0.00___swe_bench_verified_test/preds.json"
python -m swebench.harness.run_evaluation \
    --dataset_name SWE-bench/SWE-bench_Verified \
    --predictions_path $predictions_path \
    --max_workers 24 \
    --run_id swebench_verified