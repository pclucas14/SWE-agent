#!/bin/bash

PATHS=(
    swesmith_infer__openai--tt_SWE-agent-LM-32B_cl32768_lr1e-4_ep1_astropy__astropy.26d14786_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o__t-0.00__p-1.00__c-0.00___swe_bench_verified_test__ms75_mitNone_as1_astropy__astropy
    swesmith_infer__openai--tt_SWE-agent-LM-32B_cl32768_lr1e-4_ep2_astropy__astropy.26d14786_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o__t-0.00__p-1.00__c-0.00___swe_bench_verified_test__ms75_mitNone_as1_astropy__astropy
    swesmith_infer__openai--tt_SWE-agent-LM-32B_cl32768_lr1e-4_ep3_astropy__astropy.26d14786_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o__t-0.00__p-1.00__c-0.00___swe_bench_verified_test__ms75_mitNone_as1_astropy__astropy
    swesmith_infer__openai--tt_SWE-agent-LM-32B_cl32768_lr1e-5_ep1_astropy__astropy.26d14786_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o__t-0.00__p-1.00__c-0.00___swe_bench_verified_test__ms75_mitNone_as1_astropy__astropy
    swesmith_infer__openai--tt_SWE-agent-LM-32B_cl32768_lr1e-5_ep2_astropy__astropy.26d14786_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o__t-0.00__p-1.00__c-0.00___swe_bench_verified_test__ms75_mitNone_as1_astropy__astropy
    swesmith_infer__openai--tt_SWE-agent-LM-32B_cl32768_lr1e-5_ep3_astropy__astropy.26d14786_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o__t-0.00__p-1.00__c-0.00___swe_bench_verified_test__ms75_mitNone_as1_astropy__astropy
)

for path in "${PATHS[@]}"; do
    echo "Processing: $path"
    if [ -d "trajectories/zhengyanshi@microsoft.com_astropy__astropy/$path" ]; then
        python run_script/run_error_analysis.py \
            --folder-path "trajectories/zhengyanshi@microsoft.com_astropy__astropy/$path" \
            --max-workers 5 \
            --search-path evaluation_results_1r1m/torchtune_trained_models/astropy__astropy.26d14786_only_eval \
            --model-name "claude-sonnet-4" \
            --rerun
    else
        echo "Warning: Directory not found: trajectories/zhengyanshi@microsoft.com/$path"
    fi
done

python run_script/run_error_analysis.py \
            --folder-path "trajectories/zhengyanshi@microsoft.com_astropy__astropy/swesmith_infer__openai--tt_SWE-agent-LM-32B_cl32768_lr1e-5_ep3_astropy__astropy.26d14786_submit_claude__claude-sonnet-4_gpt4.1_gpt-4o__t-0.00__p-1.00__c-0.00___swe_bench_verified_test__ms75_mitNone_as1_astropy__astropy" \
            --max-workers 5 \
            --search-path evaluation_results_1r1m/torchtune_trained_models/astropy__astropy.26d14786_only_eval \
            --model-name "claude-sonnet-4" \
            --rerun