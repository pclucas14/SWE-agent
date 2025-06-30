MODEL_NAME="openai/SWE-smith-32B-Agent_qwen32B_bs1x8_lr5e-5_ep3"     # <-- only change here
MODEL_SLUG=$(echo "$MODEL_NAME" | sed 's|/|--|g')
python run_script/analyze_results.py \
    --results_path evaluation_results_1r1m/baselines \
    --model_name $MODEL_SLUG