#!/usr/bin/env bash
# Dumps what llama_cpp_benchmark.sh builds, as JSON lines, for the M3 golden tests
# (tests/test_cmd_build.py). Nothing is executed: only the *_cmd_build functions and the EXTRAS
# tables are used; run_presets_serve gets a fake binary that prints its argv.
#
# Regenerate (Git Bash, from the repo root):
#   bash tests/fixtures/bash_golden/dump_golden.sh > tests/fixtures/bash_golden/golden.jsonl

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/../../../llama_cpp_benchmark.sh"

js() {   # JSON string
    local s="$1"
    s="${s//\\/\\\\}"
    s="${s//\"/\\\"}"
    printf '"%s"' "$s"
}

jarr() {   # JSON array of the arguments
    local first=1 a
    printf '['
    for a in "$@"; do
        [[ $first -eq 1 ]] || printf ','
        first=0
        js "$a"
    done
    printf ']'
}

# 1. Every key in the EXTRAS table with its flag template.
while IFS= read -r key; do
    printf '{"kind":"extra","key":%s,"flags":%s}\n' "$(js "$key")" "$(js "${EXTRAS[$key]}")"
done < <(printf '%s\n' "${!EXTRAS[@]}" | sort)

# 2. Concrete keys (as used in the group functions) with placeholders replaced, per model.
CONCRETE=(
    extra_fa_Vq8_default extra_fa_Kq8_Vq4_default extra_fa_Kq8_Vq8_default extra_fa_Kq4_Vq8_default
    extra_fa_Kq8_Vq8_default_moe41 extra_fa_Kq8_Vq4_default_moe34 extra_fa_Kq8_Vq8_b4096_ub4096_moe41
    extra_fa_Kq8_Vq8_b2048_ub256_moe41 extra_fa_Kq8_Vq4_b8192_ub8192_moe49 extra_fa_Kq8_Vq8_b6144_ub6144_moe31
    extra_fa_Kq8_Vq8_b4096_ub4096_moe41_sd3cache extra_fa_Kq8_Vq4_b4096_ub4096_moe41_sd2cache
    extra_fa_Kq8_Vq8_b2048_ub2048_moe21_sd4mtp extra_fa_Kq8_Vq8_b2048_ub2048_moe21_sd10mtp
    extra_fa_Kq8_Vq4_default_moe34_sd2mtp extra_fa_Kq8_Vq8_b1024_ub1024 extra_fa_default extra_fa_Kq4_default
    extra_fa_Vq4_b4096_ub512 extra_fa_Kq8_b2048_ub1024_sd5cache extra_fa_Kq8_Vq8_default_sd3mtp no_extra
)
for model in "${!MODELS[@]}" DUMMY_MODEL; do
    for key in "${CONCRETE[@]}"; do
        norm="$(EXTRAS_get_flags_normalized "$key")"
        flags="$(EXTRAS_get_flags_replaced "$key" "${EXTRAS[$norm]}" "$model")"
        printf '{"kind":"replaced","model":%s,"key":%s,"flags":%s}\n' "$(js "$model")" "$(js "$key")" "$(js "$flags")"
    done
done

# 3. Serve commands: every run_SERVER_* function (incl. its commented / SD_ENABLED alternative).
serve() {   # model extra ctx [function name]
    local -a ARR=()
    run_serve_params_cmd_build "$1" cuda_flash "$2" "$3" "" ARR > /dev/null
    printf '{"kind":"serve","fn":%s,"model":%s,"extra":%s,"ctx":%s,"argv":%s}\n' \
        "$(js "$4")" "$(js "$1")" "$(js "$2")" "$3" "$(jarr "${ARR[@]}")"
}
serve omnicoder_9b                     extra_fa_Kq8_Vq8_default                    131072 run_SERVER_omnicoder_9b_CTX_cuda_flash
serve Qwen36_35B_A3B_UD_Q4_K_M_MTP     extra_fa_Kq8_Vq8_default_moe34              65536  run_SERVER_Qwen36_35B_A3B_UD_Q4_K_M_MTP_CTX_cuda_flash
serve Qwen36_35B_A3B_UD_Q4_K_M_MTP     extra_fa_Kq8_Vq4_default_moe34_sd2mtp       65536  run_SERVER_Qwen36_35B_A3B_UD_Q4_K_M_MTP_CTX_cuda_flash#commented
serve Qwen36_28B_REAP20_A3B_Q4_K_M     extra_fa_Kq8_Vq8_b4096_ub4096_moe41         65536  run_SERVER_Qwen36_28B_REAP20_A3B_Q4_K_M_CTX_cuda_flash
serve GLM_47_Flash_REAP_23B_A3B_Q4_K_M extra_fa_Kq8_Vq8_b4096_ub4096_moe49         65536  "run_SERVER_gemma_4_26B_A4b_UD_Q4_K_M_CTX_cuda_flash#first(GLM)"
serve gemma_4_26B_A4b_UD_Q4_K_M        extra_fa_Kq8_Vq8_b4096_ub4096_moe31         65536  run_SERVER_gemma_4_26B_A4b_UD_Q4_K_M_CTX_cuda_flash
serve gemma_4_26B_A4b_QAT_UD_Q4_K_XL   extra_fa_Kq8_Vq8_b2048_ub2048_moe16         65536  run_SERVER_gemma_4_26B_A4b_QAT_UD_Q4_K_XL_CTX_cuda_flash
serve gemma_4_26B_A4b_QAT_UD_Q4_K_XL   extra_fa_Kq8_Vq8_b2048_ub2048_moe21_sd4mtp  65536  "run_SERVER_gemma_4_26B_A4b_QAT_UD_Q4_K_XL_CTX_cuda_flash#SD_ENABLED=1"
serve Qwen36_35B_A3B_UD_Q4_K_M         extra_fa_Kq8_Vq8_default_sd3mtp             65536  "synthetic#mtp-without-drafter"

# 4. Bench commands: the active entries of every run_BENCH_* function plus commented ones.
bench() {   # model extra ctx fn
    local -a ARR=()
    run_bench_params_cmd_build "$1" cuda_flash "$2" "$3" "" ARR > /dev/null
    printf '{"kind":"bench","fn":%s,"model":%s,"extra":%s,"ctx":%s,"argv":%s}\n' \
        "$(js "$4")" "$(js "$1")" "$(js "$2")" "$3" "$(jarr "${ARR[@]}")"
}
for e in extra_fa_Vq8_default extra_fa_Kq8_Vq4_default extra_fa_Kq8_Vq8_default extra_fa_Kq4_Vq8_default; do
    bench omnicoder_9b "$e" 131072 run_BENCH_omnicoder_9b_CTX_cuda_flash
done
for e in extra_fa_Kq8_Vq8_default_moe41 extra_fa_Kq8_Vq4_default_moe41 extra_fa_Kq8_Vq4_default_moe34; do
    bench Qwen36_35B_A3B_UD_Q4_K_M_MTP "$e" 65536 run_BENCH_Qwen36_35B_A3B_UD_Q4_K_M_MTP_CTX_cuda_flash_CFG_MOE_on_x_CTX_x
done
for e in extra_fa_Kq8_Vq8_b4096_ub4096_moe41 extra_fa_Kq8_Vq4_b2048_ub256_moe41 extra_fa_Kq8_Vq8_default_moe23; do
    bench Qwen36_28B_REAP20_A3B_Q4_K_M "$e" 65536 run_BENCH_Qwen36_28B_REAP20_A3B_Q4_K_M_CTX_cuda_flash_CFG_MOE_on_x_CTX_x
done
for e in extra_fa_Kq8_Vq8_default_moe49 extra_fa_Kq8_Vq4_b4096_ub1024_moe49; do
    bench GLM_47_Flash_REAP_23B_A3B_Q4_K_M "$e" 65536 run_BENCH_GLM_47_Flash_REAP_23B_A3B_Q4_K_M_CTX_cuda_flash_CFG_MOE_on_x_CTX_x
done
for e in extra_fa_Kq8_Vq8_b4096_ub4096_moe31 extra_fa_Kq8_Vq8_default_moe18; do
    bench gemma_4_26B_A4b_UD_Q4_K_M "$e" 65536 run_BENCH_gemma_4_26B_A4b_UD_Q4_K_M_CTX_cuda_flash_CFG_MOE_on_x_CTX_x
done
for e in extra_fa_Kq8_Vq8_b2048_ub2048_moe16 extra_fa_Kq8_Vq4_b6144_ub6144_moe31; do
    bench gemma_4_26B_A4b_QAT_UD_Q4_K_XL "$e" 65536 run_BENCH_gemma_4_26B_A4b_QAT_UD_Q4_K_XL_CTX_cuda_flash_CFG_MOE_on_x_CTX_x
done
for c in 8192 32768; do   # run_bench_params_wrapper_PROMPTS: one extra, several prompt sizes
    bench omnicoder_9b extra_fa_Kq8_Vq8_default "$c" run_bench_params_wrapper_PROMPTS
done
bench gemma_4_26B_A4b_QAT_UD_Q4_K_XL extra_fa_Kq8_Vq8_b2048_ub2048_moe21_sd4mtp 65536 "synthetic#spec-on-llama-bench"

# 5. Benchy: run_benchy_params_with_server passes the context as PROMPT (PP sync), and the
#    background server is the normal serve command for that context.
benchy() {   # model extra ctx fn
    local -a ARR=() SRV=()
    run_benchy_params_cmd_build "$1" cuda_flash "$2" "$3" ARR > /dev/null
    run_serve_params_cmd_build "$1" cuda_flash "$2" "$3" "" SRV > /dev/null
    printf '{"kind":"benchy","fn":%s,"model":%s,"extra":%s,"ctx":%s,"argv":%s,"server":%s}\n' \
        "$(js "$4")" "$(js "$1")" "$(js "$2")" "$3" "$(jarr "${ARR[@]}")" "$(jarr "${SRV[@]}")"
}
for e in extra_fa_Kq8_Vq8_b4096_ub4096_moe41 extra_fa_Kq8_Vq4_b4096_ub4096_moe41_sd3cache; do
    benchy Qwen36_28B_REAP20_A3B_Q4_K_M "$e" 65536 run_BENCHY_WITH_SERVER_Qwen36_28B_REAP20_A3B_Q4_K_M_CTX_cuda_flash_CFG_MOE_on_x_CTX_x
done
for e in extra_fa_Kq8_Vq8_b2048_ub2048_moe21_sd4mtp extra_fa_Kq8_Vq8_b2048_ub2048_moe18; do
    benchy gemma_4_26B_A4b_QAT_UD_Q4_K_XL "$e" 65536 run_BENCHY_WITH_SERVER_gemma_4_26B_A4b_QAT_UD_Q4_K_XL_CTX_cuda_flash_CFG_MOE_on_x_CTX_x
done

# 6. Inspect: the (commented-out) builder, and the server run_INSPECT_* starts.
for t in mmlu_pro gpqa_diamond math_aime_24 swe_bench_verified; do
    ARR=()
    run_inspect_params_cmd_build "$t" ARR > /dev/null
    SRV=()
    run_serve_params_cmd_build gemma_4_26B_A4b_UD_Q4_K_M cuda_flash extra_fa_Kq8_Vq8_b4096_ub4096_moe31 65536 "" SRV > /dev/null
    printf '{"kind":"inspect","fn":%s,"model":"gemma_4_26B_A4b_UD_Q4_K_M","extra":"extra_fa_Kq8_Vq8_b4096_ub4096_moe31","ctx":65536,"task":%s,"argv":%s,"server":%s}\n' \
        "$(js "run_INSPECT_*_$t")" "$(js "$t")" "$(jarr "${ARR[@]}")" "$(jarr "${SRV[@]}")"
done

# 7. Presets router (run_PRESETS_serve) and help (run_HELP_*), with a fake binary.
dump_argv() { printf '{"kind":"%s","argv":%s}\n' "$DUMP_KIND" "$(jarr "$REAL_BIN" "$@")"; }
REAL_BIN="${SERVERS[cuda_flash]}"
SERVERS[cuda_flash]=dump_argv
DUMP_KIND=presets_serve run_presets_serve cuda_flash > >(grep '^{')
DUMP_KIND=help_server run_help_server cuda_flash | grep '^{'
REAL_BIN="${BENCHES[cuda_flash]}"
BENCHES[cuda_flash]=dump_argv
DUMP_KIND=help_bench run_help_bench cuda_flash | grep '^{'
wait
