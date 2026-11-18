#######################################################################################################################################################################################
##################################################################### CONFIG TOOLS ####################################################################################################
#######################################################################################################################################################################################

# ─────────────────────────────────────────────────────────────
# MSE_PS_DEBUG
#   Enable server on beckground debug.
# ─────────────────────────────────────────────────────────────
MSE_PS_DEBUG=false

# ─────────────────────────────────────────────────────────────
# MSE_MONITOR_ENABLED
#   Enable GPU VRAM monitoring during bench runs.
#   Tracks peak dedicated VRAM and shared memory delta.
#   Set to false to skip monitoring overhead.
# ─────────────────────────────────────────────────────────────
MSE_MONITOR_ENABLED=true

# ─────────────────────────────────────────────────────────────
# MSE_SERVER_PRESETS_DUMP  (default: false)
#   dump to help for llama.cpp model presets.
# ─────────────────────────────────────────────────────────────
MSE_SERVER_PRESETS_DUMP=false

# Presets file in this repo (fixed for llama.cpp a49db739f: [*], load-mode = none).
MSE_PRESETS_SERVE_FILE='D:\MSE\GIT\_MSE_git\mse-llm\llama_cpp_presets.ini'

###############################################################
# COMMON
###############################################################

# ─────────────────────────────────────────────────────────────
# MSE_BENCH_TG  (default: 128)
#   Token generation lengths passed to llama-bench -n flag.
#   Space-separated list, e.g. "64 128" to sweep multiple lengths.
# ─────────────────────────────────────────────────────────────
MSE_BENCH_TG="${MSE_BENCH_TG:-128}"

###############################################################
# LLAMA-CPP
###############################################################

# ─────────────────────────────────────────────────────────────
# MSE_PARALLEL  (default: 1)
#   Number of parallel slots for llama-server (concurrent requests).
#   Keep at 1 for benchmarking to avoid slot contention skewing results.
# ─────────────────────────────────────────────────────────────
MSE_PARALLEL=1

# ─────────────────────────────────────────────────────────────
# MSE_THREADS  (default: 4)
#   CPU threads passed via -t flag to llama-server and llama-bench.
#   Keep low — excess CPU threads compete with GPU and hurt throughput.
#   Recommended: 4-8; avoid matching full core count.
# ─────────────────────────────────────────────────────────────
MSE_THREADS=4

###############################################################
# BENCHY
###############################################################

# ─────────────────────────────────────────────────────────────
# MSE_BENCHY_PP_SYNCH  (default: 1)
#   When set to 1, copies the llama-server -c (context) value into
#   MSE_BENCHY_PP so the benchy PP sweep stays in sync with the
#   server context window. Set to 0 to decouple them.
# ─────────────────────────────────────────────────────────────
MSE_BENCHY_PP_SYNCH=1

# ─────────────────────────────────────────────────────────────
# MSE_BENCHY_PP  (default: "512 2048 8192 32768")
#   Space-separated list of prompt sizes swept by llama-benchy
#   via --pp flag. Each value triggers a separate measurement.
#   Overridden by MSE_BENCHY_PP_SYNCH=1 at runtime.
# ─────────────────────────────────────────────────────────────
MSE_BENCHY_PP="${MSE_BENCHY_PP:-512 2048 8192 32768}"

# ─────────────────────────────────────────────────────────────
# MSE_BENCHY_DEPTH  (default: 0)
#   KV cache pre-fill depth offset passed to llama-benchy --depth.
#   0  = cold start, empty KV cache (default bench scenario).
#   >0 = simulates mid-conversation state, e.g. 4096 or 8192,
#        useful for measuring real-world throughput degradation.
# ─────────────────────────────────────────────────────────────
MSE_BENCHY_DEPTH="${MSE_BENCHY_DEPTH:-0}"

# ─────────────────────────────────────────────────────────────
# MSE_BENCHY_RUNS  (default: 1)
#   Repetitions per PP×TG combination for llama-benchy --runs.
#   Higher values reduce noise from thermal/clock jitter.
#   Set to 1 when comparing directly with llama-bench which
#   does a single pass and does not average across runs.
# ─────────────────────────────────────────────────────────────
MSE_BENCHY_RUNS="${MSE_BENCHY_RUNS:-1}"

# ─────────────────────────────────────────────────────────────
# MSE_SERVER_READY_TIMEOUT  (default: 30)
#   Seconds to poll /health before declaring server startup failed.
#   Increase for large models that take longer to load into VRAM.
# ─────────────────────────────────────────────────────────────
MSE_SERVER_READY_TIMEOUT="${MSE_SERVER_READY_TIMEOUT:-30}"

# ─────────────────────────────────────────────────────────────
# MSE_TEST_INSPECT_PATH
#   Path to inspect-ai test repositories
#   Must be cloned 'git clone https://github.com/UKGovernmentBEIS/inspect_evals'
# ─────────────────────────────────────────────────────────────
MSE_TEST_INSPECT_PATH='/c/Users/micha/MSE/GIT/inspect_evals'

#######################################################################################################################################################################################
##################################################################### CONFIG LLM ######################################################################################################
#######################################################################################################################################################################################

declare -A MODELS=(
    # QWEN
    ["omnicoder_9b"]="d:\.lmstudio\models\bartowski\Tesslate_OmniCoder-9B-GGUF\Tesslate_OmniCoder-9B-Q4_K_M.gguf"                                   # https://huggingface.co/bartowski/Tesslate_OmniCoder-9B-GGUF              TOOL 9B_DENSE CTX_262,144 (based on qwen3.5_9B retrained for coding agents, diff native small patches)
    ["Qwen36_35B_A3B_UD_Q4_K_M"]="d:\.lmstudio\models\unsloth\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"                                  # https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF                      TOOL 23.9GB size
    ["Qwen36_35B_A3B_UD_Q4_K_M_MTP"]="d:\.lmstudio\models\unsloth\Qwen3.6-35B-A3B-MTP-GGUF\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"                          # https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF                  TOOL 24.1GB size       
    ["Qwen36_28B_REAP20_A3B_Q4_K_M"]="d:\.lmstudio\models\barozp\Qwen3.6-28B-REAP20-A3B-GGUF\Qwen3.6-28B-REAP20-A3B-Q4_K_M.gguf"                    # https://huggingface.co/barozp/Qwen3.6-28B-REAP20-A3B-GGUF                TOOL 17.26GB size      
    # GLM
    ["GLM_47_Flash_REAP_23B_A3B_Q4_K_M"]="d:\.lmstudio\models\unsloth\GLM-4.7-Flash-REAP-23B-A3B-GGUF\GLM-4.7-Flash-REAP-23B-A3B-Q4_K_M.gguf"       # https://huggingface.co/unsloth/GLM-4.7-Flash-REAP-23B-A3B-GGUF           The model has 47 MoEs (plus an extra Multi-Token Prediction (MTP) layer, bringing the absolute block count to 48 in some configuration files) 
    #GEMMA
    ["gemma_4_26B_A4b_UD_Q4_K_M"]="d:\.lmstudio\models\unsloth\gemma-4-26B-A4B-it-GGUF\gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"                           # https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF                   19.0 GB VISION_TOOL_REASONING  
    ["gemma_4_26B_A4b_QAT_UD_Q4_K_XL"]="d:\.lmstudio\models\unsloth\gemma-4-26B-A4B-it-qat-GGUF\gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"             # https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF               16.5 GB Jun 9 Update: Added MTP support
)

declare -A DRAFTER_MODELS=(
    # QWEN
    ["Qwen36_35B_A3B_UD_Q4_K_M_MTP"]="BUILD-IN"
    # GLM
    #GEMMA
    ["gemma_4_26B_A4b_QAT_UD_Q4_K_XL"]="d:\.lmstudio\models\unsloth\gemma-4-26B-A4B-it-qat-GGUF\mtp-gemma-4-26B-A4B-it.gguf"                         # https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF/tree/main               16.5 GB Jun 9 Update: Added MTP support
)

declare -A MODELS_BENCH_EXTRA=(
    #["common_2"]="-lm none --no-kv-offload 1 -v"
    #["common_1"]="-lm none -v"
    ["common"]="-lm none"
)

declare -A MODELS_SERVE_EXTRA=(
    # QWEN
    ["omnicoder_9b"]="-lm none --jinja --reasoning-format none --temp 0.3"
    ["Qwen36_35B_A3B_UD_Q4_K_M"]="-lm none --jinja --reasoning-format none --temp 0.3"
    ["Qwen36_35B_A3B_UD_Q4_K_M_MTP"]="-lm none --jinja --reasoning-format none --temp 0.3"
    ["Qwen36_28B_REAP20_A3B_Q4_K_M"]="-lm none --jinja --reasoning-format none --temp 0.3"
    # GLM
    ["GLM_47_Flash_REAP_23B_A3B_Q4_K_M"]="-lm none --jinja --reasoning-format none --temp 0.3"
    # GEMMA
    ["gemma_4_26B_A4b_UD_Q4_K_M"]="-lm none --jinja --reasoning-format auto --temp 1.0 --top-k 64 --top-p 0.95"
    ["gemma_4_26B_A4b_QAT_UD_Q4_K_XL"]="-lm none --jinja --reasoning-format auto --temp 1.0 --top-k 64 --top-p 0.95"
    # COMMON
    ["common"]="-lm none --jinja --reasoning-format auto"
)

# ─────────────────────────────────────────────────────────────
# llama.cpp binaries: the clone managed by this repo (builds\ is git-ignored), built with the
# profile config\builds\rtx4080-laptop.toml (see IMPLEMENTATION_PLAN.md section 0.4).
MSE_LLAMA_BIN_DIR='D:\MSE\GIT\_MSE_git\mse-llm\builds\llama.cpp\build\bin\Release'

declare -A BENCHES=(
    ["cuda_flash"]="${MSE_LLAMA_BIN_DIR}\\llama-bench"
)

# ─────────────────────────────────────────────────────────────

declare -A SERVERS_NET_CFG=(
    ["cuda_flash"]='0.0.0.0|8033|http://localhost'
)

declare -A SERVERS=(
    ["cuda_flash"]="${MSE_LLAMA_BIN_DIR}\\llama-server"
)

# ─────────────────────────────────────────────────────────────

EXTRAS_get_flags_replaced()
{
    local extra_user="$1"
    local lookup_data="$2"
    local model="$3"

    # If key contains _moeN, extract N and replace the placeholder value
    if [[ "$extra_user" =~ _moe([0-9]+) ]]; then
        local n="${BASH_REMATCH[1]}"
        lookup_data="${lookup_data/--n-cpu-moe N/--n-cpu-moe $n}"
    fi

    # If key contains _sdScache, extract S and replace the placeholder value
    if [[ "$extra_user" =~ _sd([0-9]+)cache ]]; then
        local s="${BASH_REMATCH[1]}"
        lookup_data="${lookup_data/--spec-draft-n-max S/--spec-draft-n-max $s}"
    fi

    # If key contains _sdSmtp, extract S and replace the placeholder value
    # Also replace drafter model if available in DRAFTER_MODELS
    if [[ "$extra_user" =~ _sd([0-9]+)mtp ]]; then
        local s="${BASH_REMATCH[1]}"
        lookup_data="${lookup_data/--spec-draft-n-max S/--spec-draft-n-max $s}"
        
        # Update drafter model if available in DRAFTER_MODELS
        if [[ -n "$model" ]] && [[ -v DRAFTER_MODELS["$model"] ]]; then
            local drafter_model="${DRAFTER_MODELS["$model"]}"
            
            # If BUILD-IN, remove the flag entirely; otherwise replace with the model path
            if [[ "$drafter_model" == "BUILD-IN" ]]; then
                lookup_data="${lookup_data/--model-draft MODEL/ }"
            else
                lookup_data="${lookup_data/--model-draft MODEL/--model-draft $drafter_model}"
            fi
        fi
    fi

    echo "$lookup_data"
}

EXTRAS_get_flags_normalized()
{
    local key="$1"
    echo "$key" \
        | sed 's/_moe[0-9]\+/_moeN/g' \
        | sed 's/_sd[0-9]\+cache/_sdScache/g' \
        | sed 's/_sd[0-9]\+mtp/_sdSmtp/g'
}


# default values are 
#-b,  --batch-size <n>                       (default: 2048)
# -ub, --ubatch-size <n>                      (default: 512)
# extra_fa_default is used as WARM-UP - as not match default params ...
declare -A EXTRAS=()
# N-Gram spec options: name_suffix -> flags
declare -A EXTRAS_NGRAM_VARIANTS=(
    ["none"]=""
    ["_moeN"]="--n-cpu-moe N"          # placeholder, 0 = default
    ["_sdScache"]="--spec-type ngram-cache --spec-draft-n-max S"
    ["_sdSmtp"]="--spec-type draft-mtp --spec-draft-n-max S --model-draft MODEL"
    ["_moeN_sdScache"]="--n-cpu-moe N --spec-type ngram-cache --spec-draft-n-max S"
    ["_moeN_sdSmtp"]="--n-cpu-moe N --spec-type draft-mtp --spec-draft-n-max S --model-draft MODEL"
)

EXTRAS_add_batch_variants()
{
    local name="$1"
    local base_flags="$2"
    local ng_suffix ng_flags
    
    local -A batch_variants=(
        ["_default"]=""
        ["_b1024_ub1024"]="-b 1024 -ub 1024"
        ["_b8192_ub8192"]="-b 8192 -ub 8192"
        ["_b6144_ub6144"]="-b 6144 -ub 6144"
        # b2048
        ["_b2048_ub256"]="-b 2048 -ub 256"
        ["_b2048_ub512"]="-b 2048 -ub 512"
        ["_b2048_ub1024"]="-b 2048 -ub 1024"
        ["_b2048_ub2048"]="-b 2048 -ub 2048"
        # b4096
        ["_b4096_ub4096"]="-b 4096 -ub 4096"
        ["_b4096_ub2048"]="-b 4096 -ub 2048"
        ["_b4096_ub1024"]="-b 4096 -ub 1024"
        ["_b4096_ub512"]="-b 4096 -ub 512"
        ["_b4096_ub256"]="-b 4096 -ub 256"
    )
    
    local b_suffix b_flags
    # FIXED: Added missing _b4096_ub2048 to for loop
    for b_suffix in _default _b1024_ub1024 _b8192_ub8192 _b6144_ub6144 _b2048_ub256 _b2048_ub512 _b2048_ub1024 _b2048_ub2048 _b4096_ub4096 _b4096_ub2048 _b4096_ub1024 _b4096_ub512 _b4096_ub256; do
        b_flags="${batch_variants[$b_suffix]}"
        
        # Create base entry without ngram variants
        EXTRAS["${name}${b_suffix}"]="$base_flags${b_flags:+ $b_flags}"
        
        # Then add ngram variants (skip "none" since it's already the base entry above)
        for ng_suffix in "_moeN" "_sdScache" "_sdSmtp" "_moeN_sdScache" "_moeN_sdSmtp"; do
            ng_flags="${EXTRAS_NGRAM_VARIANTS[$ng_suffix]}"
            local key="${name}${b_suffix}${ng_suffix}"
            EXTRAS["$key"]="$base_flags${b_flags:+ $b_flags}${ng_flags:+ $ng_flags}"
        done
    done
}

FA="-fa 1"

# Single standalone entry 
EXTRAS["no_extra"]=''

# FA only 
EXTRAS_add_batch_variants "extra_fa"            "$FA"

# V only 
EXTRAS_add_batch_variants "extra_fa_Vq4"        "$FA -ctv q4_0"
EXTRAS_add_batch_variants "extra_fa_Vq8"        "$FA -ctv q8_0"

# K only 
EXTRAS_add_batch_variants "extra_fa_Kq4"        "$FA -ctk q4_0"
EXTRAS_add_batch_variants "extra_fa_Kq8"        "$FA -ctk q8_0"

# Symmetric: K+V same type 
EXTRAS_add_batch_variants "extra_fa_Kt4_Vt4"   "$FA -ctk turbo4 -ctv turbo4"
EXTRAS_add_batch_variants "extra_fa_Kt3_Vt3"   "$FA -ctk turbo3 -ctv turbo3"
EXTRAS_add_batch_variants "extra_fa_Kt2_Vt2"   "$FA -ctk turbo2 -ctv turbo2"

# Asymmetric: q4/q8 cross 
EXTRAS_add_batch_variants "extra_fa_Kq4_Vq8"   "$FA -ctk q4_0  -ctv q8_0"
EXTRAS_add_batch_variants "extra_fa_Kq8_Vq4"   "$FA -ctk q8_0  -ctv q4_0"
EXTRAS_add_batch_variants "extra_fa_Kq8_Vq8"   "$FA -ctk q8_0  -ctv q8_0"

# Asymmetric: turbo cross 
EXTRAS_add_batch_variants "extra_fa_Kt4_Vt3"   "$FA -ctk turbo4 -ctv turbo3"
EXTRAS_add_batch_variants "extra_fa_Kt4_Vt2"   "$FA -ctk turbo4 -ctv turbo2"
EXTRAS_add_batch_variants "extra_fa_Kt3_Vt2"   "$FA -ctk turbo3 -ctv turbo2"

# Asymmetric: q8 K + turbo V 
EXTRAS_add_batch_variants "extra_fa_Kq8_Vt4"   "$FA -ctk q8_0  -ctv turbo4"
EXTRAS_add_batch_variants "extra_fa_Kq8_Vt3"   "$FA -ctk q8_0  -ctv turbo3"

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# KV-cache types are selected per-side via the standard --cache-type-k / --cache-type-v flags.
#
# Start light, then compress. Some model families — small models, certain MoE configurations, quant-sensitive instruction-tuned variants — are more delicate than others. 
# Pick a light asymmetric configuration first, verify output quality (eyeball + PPL on a hold-out set) on your specific model, then ratchet up V aggression if you have memory headroom to gain. 
# Do not start at maximum compression and work backwards !!!
#
# The core finding from the asymmetric-kv-compression paper — Asymmetric K/V Cache Compression: Why V is Free and K is Everything — drives all the configs below: V tolerates aggressive compression, K does not. 
# https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/asymmetric-kv-compression.md
# Always keep K at higher precision than V; never start symmetric. 
# That paper documents the specific failure modes you'll hit if you ignore this and compress K aggressively (PPL blow-up on certain model families, attention-rotation interaction with low-bit K, etc.) — read it before considering step 6.
#
# Higher turbo number = more bits per element = less aggressive compression. 
# The V-side compression ladder is turbo4 (lightest) → turbo3 → turbo2 (heaviest). On the K side, prefer f16 or q8_0; never lead with a turbo K.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#DUMPS
#for key in $(printf '%s\n' "${!EXTRAS[@]}" | sort); do echo "$key"; done
#for key in "${!EXTRAS[@]}"; do echo "$key"; done

#######################################################################################################################################################################################
######################################################################## EXECS ########################################################################################################
#######################################################################################################################################################################################

EXEC_user_notification_WIN_T2S()
{
    powershell -Command "(New-Object -ComObject SAPI.SpVoice).Speak(\"$*\")"
}

EXEC_cmd_with_output_mode()
{
    local output_mode=${1:-"all"}
    local output_mode_filter_ref=${2-""}
    shift 2
    local cmd=("$@")

    echo "output_mode[${output_mode}] output_mode_filter_ref=[${output_mode_filter_ref}] "

    case "$output_mode" in
        "none")
            "${cmd[@]}" > /dev/null 2>&1
            ;;
        "filter")
            if [[ -z "$output_mode_filter_ref" ]]; then
                # No filter patterns - fall back to 'all'
                "${cmd[@]}"
            else
                local -n filter_patterns="$output_mode_filter_ref"
                local pattern
                pattern=$(IFS='|'; echo "${filter_patterns[*]}")
                "${cmd[@]}" 2>&1 | grep -E --line-buffered "$pattern"
            fi
            ;;
        "all"|*)
            [[ "$output_mode" != "all" ]] && echo "WARNING: unknown output_mode '$output_mode', falling back to 'all'"
            "${cmd[@]}"
            ;;
    esac
}

EXEC_cmd_with_gpu_and_ram_monitoring()
{
    local output_mode=${1:-"all"}
    local output_mode_filter_ref=${2-""}
    local -n CMD_ARRAY=$3                  # nameref — caller passes array name
    local END_MESSAGE_2_USER="$4"
    
    echo "--------------------------------------------------------------"
    echo "TESTING: >>>>>>>>>>>>>>>>>>>>> Started <<<<<<<<<<<<<<<<<<<<<<<"
    echo "--------------------------------------------------------------"   
    
    # 1. Start Duration Timer
    local START_TIME=$(date +%s)
    
    # 2. Initialize Monitoring Variables
    local MONITOR_PID=""
    local VRAM_LOG="/tmp/vram_$$.tmp"
    local RAM_LOG="/tmp/ram_$$.tmp"
    local GPU_TEMP_LOG="/tmp/gpu_temp_$$.tmp"
    
    # 3. Get Baseline Measurements
    local BASELINE_SHARED=0
    local BASELINE_RAM=0
    local BASELINE_GPU_TEMP=0
    
    # Enable monitoring by default unless explicitly disabled
    if [ "$MSE_MONITOR_ENABLED" != "false" ]; then
        # --- GPU Baseline ---
        BASELINE_SHARED=$(powershell -Command "(Get-CimInstance -Namespace 'root\cimv2' -ClassName 'Win32_PerfRawData_GPUPerformanceCounters_GPUNonLocalAdapterMemory' | Measure-Object -Property NonLocalUsage -Sum).Sum" 2>/dev/null \
            | tr -d '\r\n' | awk '{printf "%.0f", $1/1048576}')
        BASELINE_SHARED="${BASELINE_SHARED:-0}"
        echo "  Baseline Shared Memory: ${BASELINE_SHARED} MiB"
        
        # --- System RAM Baseline (Windows) ---
        if command -v free &>/dev/null; then
            # Git Bash may have free command
            BASELINE_RAM=$(free -m 2>/dev/null | awk 'NR==2 {print $3}')
        fi
        if [ -z "$BASELINE_RAM" ] || [ "$BASELINE_RAM" = "0" ]; then
            # Fallback to PowerShell (primary for Windows)
            # Used RAM = Total - Free
            BASELINE_RAM=$(powershell.exe -Command "Get-WmiObject -Class Win32_OperatingSystem | ForEach-Object {[Math]::Round((\$_.TotalVisibleMemorySize - \$_.FreePhysicalMemory) / 1024)}" 2>/dev/null)
        fi
        BASELINE_RAM="${BASELINE_RAM:-0}"
        echo "  Baseline RAM Used: ${BASELINE_RAM} MiB"
        
        # --- GPU Temperature Baseline ---
        BASELINE_GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d '\r' | head -n 1)
        BASELINE_GPU_TEMP="${BASELINE_GPU_TEMP:-0}"
        echo "  Baseline GPU Temperature: ${BASELINE_GPU_TEMP}°C"
        
        # --- Start Monitoring Loop ---
        (
            while true; do
                # GPU Memory (VRAM)
                DEDICATED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '\r')
                DEDICATED="${DEDICATED:-0}"
                
                SHARED=$(powershell -Command "(Get-CimInstance -Namespace 'root\cimv2' -ClassName 'Win32_PerfRawData_GPUPerformanceCounters_GPUNonLocalAdapterMemory' | Measure-Object -Property NonLocalUsage -Sum).Sum" 2>/dev/null | tr -d '\r\n' | awk '{printf "%.0f", $1/1048576}')
                SHARED="${SHARED:-0}"
                
                # System RAM (Windows)
                if command -v free &>/dev/null; then
                    RAM_USED=$(free -m 2>/dev/null | awk 'NR==2 {print $3}')
                fi
                if [ -z "$RAM_USED" ] || [ "$RAM_USED" = "0" ]; then
                    # Used RAM = Total - Free
                    RAM_USED=$(powershell.exe -Command "Get-WmiObject -Class Win32_OperatingSystem | ForEach-Object {[Math]::Round((\$_.TotalVisibleMemorySize - \$_.FreePhysicalMemory) / 1024)}" 2>/dev/null)
                fi
                RAM_USED="${RAM_USED:-0}"
                
                # GPU Temperature
                GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d '\r' | head -n 1)
                GPU_TEMP="${GPU_TEMP:-0}"
                
                # Log everything
                echo "$DEDICATED,$SHARED" >> "$VRAM_LOG"
                echo "$RAM_USED" >> "$RAM_LOG"
                echo "$GPU_TEMP" >> "$GPU_TEMP_LOG"
                
                sleep 1
            done
        ) &
        MONITOR_PID=$!
    fi  # End of monitoring section
    
    # 4. Execute Command with quoted array expansion
    EXEC_cmd_with_output_mode $output_mode $output_mode_filter_ref "${CMD_ARRAY[@]}"
    
    # 5. Stop Monitoring & Calculate Duration
    if [ -n "$MONITOR_PID" ]; then
        kill $MONITOR_PID 2>/dev/null
        wait $MONITOR_PID 2>/dev/null
    fi
    local END_TIME=$(date +%s)
    local DURATION=$((END_TIME - START_TIME))
    
    # 6. Process and Display Results
    echo "--------------------------- RESULTS --------------------------"
    echo "Duration: $((DURATION / 60))m $((DURATION % 60))s"
    
    if [ "$MSE_MONITOR_ENABLED" = true ]; then
        # Debug: Show what log files exist
        echo "[DEBUG] Log files created:"
        [ -f "$VRAM_LOG" ] && echo "  ✓ VRAM_LOG exists: $(wc -l < $VRAM_LOG) lines" || echo "  ✗ VRAM_LOG missing"
        [ -f "$RAM_LOG" ] && echo "  ✓ RAM_LOG exists: $(wc -l < $RAM_LOG) lines" || echo "  ✗ RAM_LOG missing"
        [ -f "$GPU_TEMP_LOG" ] && echo "  ✓ GPU_TEMP_LOG exists: $(wc -l < $GPU_TEMP_LOG) lines" || echo "  ✗ GPU_TEMP_LOG missing"
        echo ""
        
        # --- GPU Memory Results ---
        if [ -f "$VRAM_LOG" ]; then
            local MAX_DEDICATED=$(cut -d',' -f1 "$VRAM_LOG" | sort -rn | head -n 1)
            local MAX_SHARED=$(cut -d',' -f2 "$VRAM_LOG" | sort -rn | head -n 1)
            local DELTA_SHARED=$(( ${MAX_SHARED:-0} - ${BASELINE_SHARED:-0} ))
            [ $DELTA_SHARED -lt 0 ] && DELTA_SHARED=0
            
            echo ""
            echo "GPU Memory:"
            echo "  Peak Dedicated VRAM: ${MAX_DEDICATED:-0} MiB"
            echo "  Peak Shared Memory:  ${MAX_SHARED:-0} MiB (baseline: ${BASELINE_SHARED} MiB, delta: ${DELTA_SHARED} MiB)"
            
            if [ "${DELTA_SHARED}" -gt 100 ]; then
                echo "  STATUS: [!] Performance hit detected. Offloaded to System RAM."
            else
                echo "  STATUS: [✓] Run contained within Dedicated VRAM."
            fi
            
            rm "$VRAM_LOG"
        fi
        
        # --- System RAM Results ---
        if [ -f "$RAM_LOG" ]; then
            local MAX_RAM=$(sort -rn "$RAM_LOG" | head -n 1)
            local DELTA_RAM=$(( ${MAX_RAM:-0} - ${BASELINE_RAM:-0} ))
            [ $DELTA_RAM -lt 0 ] && DELTA_RAM=0
            
            # Get total system RAM for percentage calculation
            local TOTAL_RAM=0
            if command -v free &>/dev/null; then
                TOTAL_RAM=$(free -m 2>/dev/null | awk 'NR==2 {print $2}')
            fi
            if [ -z "$TOTAL_RAM" ] || [ "$TOTAL_RAM" = "0" ]; then
                # Get total RAM from WMI
                TOTAL_RAM=$(powershell.exe -Command "Get-WmiObject -Class Win32_OperatingSystem | ForEach-Object {[Math]::Round(\$_.TotalVisibleMemorySize / 1024)}" 2>/dev/null)
            fi
            TOTAL_RAM="${TOTAL_RAM:-1}"
            
            local RAM_PERCENT=$((MAX_RAM * 100 / TOTAL_RAM))
            
            echo ""
            echo "System RAM:"
            echo "  Peak RAM Used: ${MAX_RAM:-0} MiB / ${TOTAL_RAM} MiB (${RAM_PERCENT}%)"
            echo "  Delta from Baseline: ${DELTA_RAM} MiB"
            
            if [ "${RAM_PERCENT}" -gt 80 ]; then
                echo "  STATUS: [!] High RAM usage detected (>80%)."
            elif [ "${RAM_PERCENT}" -gt 60 ]; then
                echo "  STATUS: [⚠]  Moderate RAM usage (60-80%)."
            else
                echo "  STATUS: [✓] Normal RAM usage."
            fi
            
            rm "$RAM_LOG"
        fi
        
        # --- GPU Temperature Results ---
        if [ -f "$GPU_TEMP_LOG" ]; then
            local MAX_GPU_TEMP=$(sort -rn "$GPU_TEMP_LOG" | head -n 1)
            local AVG_GPU_TEMP=$(awk '{sum+=$1; count++} END {if (count>0) printf "%.0f", sum/count}' "$GPU_TEMP_LOG")
            
            echo ""
            echo "GPU Temperature:"
            echo "  Peak: ${MAX_GPU_TEMP:-0}°C"
            echo "  Average: ${AVG_GPU_TEMP:-0}°C"
            
            if [ "${MAX_GPU_TEMP%.*}" -gt 85 ]; then
                echo "  STATUS: [!] GPU running hot (>85°C). Thermal throttling may occur."
            elif [ "${MAX_GPU_TEMP%.*}" -gt 75 ]; then
                echo "  STATUS: [⚠]  GPU warm (75-85°C). Monitor for sustained loads."
            else
                echo "  STATUS: [✓] GPU temperature normal."
            fi
            
            rm "$GPU_TEMP_LOG"
        fi
    fi
    
    echo "--------------------------------------------------------------"
    EXEC_user_notification_WIN_T2S "${END_MESSAGE_2_USER}"
}

#######################################################################################################################################################################################
################################################################# SERVER HELPERS (GitBash-safe) #######################################################################################
#######################################################################################################################################################################################
server_logfile_and_cleanup()
{
    local SERVER_LOG_FILE=$1
    local FOREGROUND_LOG_FILE=$2
    
    sed -i 's/^/>>> /' ${SERVER_LOG_FILE}
    sed -i 's/^/<<< /' ${FOREGROUND_LOG_FILE}
    echo "--------------------------------------------------------------"
    cat ${SERVER_LOG_FILE}
    rm  ${SERVER_LOG_FILE}
    echo "--------------------------------------------------------------"
    echo "--------------------------------------------------------------"
    cat ${FOREGROUND_LOG_FILE}
    rm  ${FOREGROUND_LOG_FILE}
    echo "--------------------------------------------------------------"
}

server_get_pid_by_ppid()
{
    local PARENT_PID=$1 
    
    # Look for a child of PARENT_PID, but explicitly filter for llama-server
    ps -ef | awk -v ppid="$PARENT_PID" '$3 == ppid && /llama-server/ {print $2; exit}' | tr -d '\r\n'
}

server_stop_windows()
{
    local BASH_PID=$1
    local SUBSHELL_PID=$2
    
    # 1. Get the native Windows PIDs
    local WIN_BASH_PID=$(ps -p "$BASH_PID" | awk 'NR==2 {print $4}')
    local WIN_SUB_PID=$(ps -p "$SUBSHELL_PID" | awk 'NR==2 {print $4}')
    
    if [ "$MSE_PS_DEBUG" != "false" ]; then
        ps aux # MSE: DEBUG
    fi
    echo "Shutting down server BASH_PID[linux|win]: ${BASH_PID}|${WIN_BASH_PID} SUBSHELL_PID[linux|win]: ${SUBSHELL_PID}|$WIN_SUB_PID ..."   

    # 2.a Gracefull kill the Linux process trees
    kill -9 "$BASH_PID"
    kill -9 "$SUBSHELL_PID"

    sleep 1
    
    wait "$BASH_PID" 
    wait "$SUBSHELL_PID"

    # 2.b Custom "wait" loop for Windows PIDs - loops as long as tasklist can still find the Windows PIDs running
    echo "Waiting for Windows processes to exit..."
    while (tasklist //FI "PID eq $WIN_BASH_PID" 2>/dev/null | grep -q "$WIN_BASH_PID") || \
          (tasklist //FI "PID eq $WIN_SUB_PID" 2>/dev/null | grep -q "$WIN_SUB_PID"); do
          
        echo "Shutting down server BASH_PID[linux|win]: ${BASH_PID}|${WIN_BASH_PID} SUBSHELL_PID[linux|win]: ${SUBSHELL_PID}|$WIN_SUB_PID ..."   
        
        # Force kill the Windows process trees
        if [ -n "$WIN_SUB_PID" ]; then
            taskkill //F //T //PID "$WIN_SUB_PID" 2>/dev/null
        fi
        if [ -n "$WIN_BASH_PID" ]; then
            taskkill //F //T //PID "$WIN_BASH_PID" 2>/dev/null
        fi
        
        sleep 0.5
    done
        
    echo "[✓] Server stopped"
}

# Polls /health every 1s until server responds or timeout is reached.
server_wait_for_ready()
{
    local BASE_URL="$1"
    local TIMEOUT="${2:-$MSE_SERVER_READY_TIMEOUT}"

    # Derive health URL: strip trailing /v1 if present, append /health
    local HEALTH_URL="${BASE_URL%/v1}/health"

    echo "Waiting for server at ${HEALTH_URL} (timeout ${TIMEOUT}s)..."
    local i=0
    while [ $i -lt $TIMEOUT ]; do
        curl -sf "${HEALTH_URL}" > /dev/null 2>&1
        local curl_res=$?
        echo "curl -sf \"${HEALTH_URL}\" result: ${curl_res}"
        if [ $curl_res -eq 0 ] ; then
            echo "[✓] Server ready after ${i}s"
            return 0
        fi
        sleep 1
        (( i++ ))
    done

    echo "[✗] Server did not become ready within ${TIMEOUT}s"
    return 1
}

# Extracts base URL (http://localhost:PORT/v1) from SERVERS_NET_CFG for a given server key.
_resolve_base_url()
{
    local server=$1
    local -a parts
    IFS='|' read -r -a parts <<< "${SERVERS_NET_CFG[$server]}"
    echo "${parts[2]}:${parts[1]}/v1"
}

# Derives a short display-friendly model label from the GGUF path.
# e.g. "d:\.lmstudio\models\bartowski\Tesslate_OmniCoder-9B-GGUF\Tesslate_OmniCoder-9B-Q4_K_M.gguf"
#   -> "Tesslate_OmniCoder-9B-Q4_K_M"
# llama-benchy --model accepts any string; it just needs to be non-empty and
# match what the server reports at /v1/models so benchy doesn't reject it.
# llama-server reports the bare filename (without path) at /v1/models, so we
# strip path and extension to produce the same string it advertises.
_model_label_from_path()
{
    local model_path="$1"
    # Convert backslashes to forward slashes, grab basename, strip .gguf
    local base
    base=$(echo "$model_path" | tr '\\' '/' | sed 's|.*/||; s|\.gguf$||i')
    echo "$base"
}

# Expands space-separated values into repeated flags.
# _expand_flag "--tg" "64 128"  ->  "--tg 64 --tg 128"
_expand_flag()
{
    local flag="$1"
    local values="$2"
    local out=""
    for v in $values; do
        out="${out} ${flag} ${v}"
    done
    echo "${out# }"   # trim leading space
}

#######################################################################################################################################################################################
#######################################################################################################################################################################################
################################################################ COMMANDS SINGLES #####################################################################################################
#######################################################################################################################################################################################
#######################################################################################################################################################################################


#######################################################################################################################################################################################
################################################################ COMMANDS SINGLES (HELP) ##############################################################################################
#######################################################################################################################################################################################

run_help_server()
{
    local server=$1
    local LLAMA_SERVER=${SERVERS[$server]}
    
    local CMD="${LLAMA_SERVER} --help"
    echo "--------------------------------------------------------------"
    echo "HELP: ${CMD}"
    echo "--------------------------------------------------------------"
    
    ${CMD}
}
run_HELP_server__cuda_flash() { run_help_server "cuda_flash" ; }

run_help_bench()
{
    local server=$1
    local LLAMA_BENCHE=${BENCHES[$server]}
    
    local CMD="${LLAMA_BENCHE} --help"
    echo "--------------------------------------------------------------"
    echo "HELP: ${CMD}"
    echo "--------------------------------------------------------------"

    ${CMD}
}
run_HELP_bench__cuda_flash() { run_help_bench "cuda_flash" ; }

#######################################################################################################################################################################################
################################################################ COMMANDS SINGLES (DUMP) ##############################################################################################
#######################################################################################################################################################################################
run_dump_params_wrapper_PLOTER() {
    local bench_type="${1:-auto}"  # First parameter specifies benchmark tool
    local analysis_title="${2:-"Benchmarks Performance Analysis"}"
    local input_file="$3"
    local output_png="${4:-"performance_chart.png"}"

    if [[ -z "$input_file" || ! -f "$input_file" ]]; then
        echo "Error: Input data log file path is missing or does not exist."
        echo "Usage: run_dump_params_wrapper_PLOTER [bench_type] \"Title\" \"path/to/log.txt\" [output.png]"
        echo "  bench_type: 'llama-bench', 'llama-benchy', or 'auto' (default: auto)"
        return 1
    fi

    # Execute Python via uvx with automatic inline dependency provisioning
    uvx --with matplotlib --with numpy python3 - <<EOF
import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- 0. AUTO-DETECT BENCHMARK TYPE (Format-Only, No Model Names) ---
with open("${input_file}", "r", encoding="utf-8") as f:
    log_content = f.read()

bench_type = "${bench_type}".lower()
if bench_type == "auto":
    # Detect by FORMAT MARKERS only (not model names)
    if "BENCHY:" in log_content:
        bench_type = "llama-benchy"
    elif "TESTING:" in log_content:
        bench_type = "llama-bench"
    else:
        bench_type = "llama-bench"  # default fallback

print(f"[AUTO-DETECT] Benchmark type: {bench_type}")

# --- 1. LOAD AND SEGMENT RAW TEXT FILE ---
# Split on explicit format markers ONLY - works with ANY model
block_segments = re.split(r'(?=TESTING:\s*>+\s*Started)|(?=BENCHY:\s+)', log_content)
processed_blocks = []

for block in block_segments:
    # Skip empty blocks
    if len(block.strip()) < 50:
        continue
    
    # Determine format by MARKERS ONLY (no model-specific checks)
    is_benchy = "BENCHY:" in block
    has_testing = "TESTING:" in block
    
    # Skip if neither format marker found
    if not (is_benchy or has_testing):
        continue

    # A. Generic Parsing for Model Name & Benchmark Type (Format-Agnostic)
    model_name = "Unknown_Model"
    bench_name = "default"

    if is_benchy:            
        testing_match = re.search(r'SERVING:\s+([^\s|>]+)(?:[^|]*\|\s*Bench:\s*([^\s|]+))?', block)
        if testing_match:
            model_name = testing_match.group(1).strip()
            if testing_match.group(2):
                bench_name = testing_match.group(2).strip()
    else:
        testing_match = re.search(r'TESTING:\s+([^\s|>]+)(?:[^|]*\|\s*Bench:\s*([^\s|]+))?', block)
        if testing_match:
            model_name = testing_match.group(1).strip()
            if testing_match.group(2):
                bench_name = testing_match.group(2).strip()

    config_a = f"{model_name} [{bench_name}]"

    # B. Generic Parsing for Run Parameters (Format-Agnostic)
    # All these parameters are extracted the same way regardless of format or model
    ctk_match  = re.search(r'-ctk\s+([^\s|-]+)', block)
    ctv_match  = re.search(r'-ctv\s+([^\s|-]+)', block)
    moe_match  = re.search(r'--n-cpu-moe\s+(\d+)', block)
    mmap_match = re.search(r'(?:-lm|--load-mode)\s+(\S+)|--mmap\s+(\d+)', block)
    p_match    = re.search(r'-c\s+(\d+)|--ctx.*?(\d+)|PP:\s*(\d+)', block)
    b_match    = re.search(r'-b\s+(\d+)', block)
    ub_match   = re.search(r'-ub\s+(\d+)', block)
    # Speculative decoding parameters
    spec_type_match = re.search(r'--spec-type\s+([^\s|-]+)', block)
    # Extract spec-draft-n-max - prioritize lookup_final (actual value) over template
    spec_draft_match = None
    
    # First priority: Find actual numeric value in lookup_final line
    lookup_final_draft = re.search(r'lookup_final.*?--spec-draft-n-max\s+(\d+)', block)
    if lookup_final_draft:
        spec_draft_match = lookup_final_draft
    else:
        # Second priority: Direct numeric value in command
        direct_draft = re.search(r'--spec-draft-n-max\s+(\d+)', block)
        if direct_draft:
            spec_draft_match = direct_draft

    # Extract draft model name (filename only, from full path)
    # Prioritize lookup_final (actual value) over template/placeholder
    model_draft_match = None
    
    # First priority: Find actual path in lookup_final line
    lookup_final_draft = re.search(r'lookup_final.*?--model-draft\s+([^\s|]+)', block)
    if lookup_final_draft:
        model_draft_match = lookup_final_draft
    else:
        # Second priority: Direct match, but skip placeholder "MODEL"
        direct_draft = re.search(r'--model-draft\s+([^\s|]+)', block)
        if direct_draft and direct_draft.group(1).strip() != "MODEL":
            model_draft_match = direct_draft
    
    draft_model_name = ""
    if model_draft_match:
        draft_path = model_draft_match.group(1).strip()
        # Extract filename from path (works with both / and \ separators)
        draft_model_name = os.path.basename(draft_path)
        # Remove file extension if present
        draft_model_name = draft_model_name.rsplit('.', 1)[0] if '.' in draft_model_name else draft_model_name

    # Clean extracted values
    ctk  = ctk_match.group(1).strip() if ctk_match else "default"
    ctv  = ctv_match.group(1).strip() if ctv_match else "default"
    moe  = f"MoE:{moe_match.group(1).strip()}" if moe_match else "MoE:default"
    if mmap_match and mmap_match.group(1):
        mmap = f" / lm:{mmap_match.group(1).strip()}"
    elif mmap_match:
        mmap = f" / mmap:{mmap_match.group(2).strip()}"
    else:
        mmap = ""
    
    # Handle context size (works with multiple parameter formats)
    if p_match:
        ctx_val = p_match.group(1) or p_match.group(2) or p_match.group(3)
        ctx = f"Ctx:{ctx_val.strip()}" if ctx_val else "Ctx:default"
    else:
        ctx = "Ctx:default"
    
    batch = f"Batch:{b_match.group(1).strip()}" if b_match else "Batch:default"
    ubatch = f"uBatch:{ub_match.group(1).strip()}" if ub_match else "uBatch:default"
    
    # Speculative decoding parameters (if present)
    spec_type = f"Spec:{spec_type_match.group(1).strip()}" if spec_type_match else ""
    spec_draft = f"DraftN:{spec_draft_match.group(1).strip()}" if spec_draft_match else ""
    spec_model = f"DraftModel:{draft_model_name}" if draft_model_name else ""
    spec_config = ""
    if spec_type or spec_draft or spec_model:
        spec_parts = [p for p in [spec_type, spec_draft, spec_model] if p]
        spec_config = f" / {' / '.join(spec_parts)}"

    # Build multi-line X-Axis string parameters
    config_b = f"K:{ctk} / V:{ctv}"
    config_c = f"{moe}{mmap}{spec_config}"
    config_d = f"{ctx} / {batch} / {ubatch}"

    # C. Extract Target Data Metrics (Format-Agnostic - Same Patterns for Both)
    # These patterns work for any model and any format
    pp_match = re.search(r'\|\s*pp\d+\s*\|\s*([0-9.]+)', block)
    tg_match = re.search(r'\|\s*tg\d+\s*\|\s*([0-9.]+)', block)
    
    # Duration and VRAM metrics (same format in both llama-bench and llama-benchy)
    dur_match = re.search(r'Duration:\s*(?:(\d+)m\s*)?(\d+)s', block)
    # Server startup time
    server_ready_match = re.search(r'Server ready after\s+(\d+)s', block)
    vram_match = re.search(r'Peak Dedicated VRAM:\s*([0-9.]+)\s*MiB', block)
    svram_match = re.search(r'Peak Shared Memory:\s*([0-9.]+)\s*MiB', block)

    if not (pp_match and tg_match):
        continue

    # Extract metric values
    if dur_match:
        minutes = int(dur_match.group(1)) if dur_match.group(1) else 0
        seconds = int(dur_match.group(2))
        total_duration_seconds = (minutes * 60) + seconds
    else:
        total_duration_seconds = 0
    
    # Extract server startup time
    if server_ready_match:
        server_startup_seconds = int(server_ready_match.group(1))
    else:
        server_startup_seconds = 0
    
    if vram_match:
        vram_val = float(vram_match.group(1))
    else:
        vram_val = 0
    
    if svram_match:
        svram_val = float(svram_match.group(1))
    else:
        svram_val = 0
    
    processed_blocks.append({
        'config_id': f"{config_b}\n{config_c}\n{config_d}",
        'config_a': config_a,
        'duration': float(total_duration_seconds) if total_duration_seconds > 0 else 0,
        'server_startup': float(server_startup_seconds) if server_startup_seconds > 0 else 0,
        'pp': float(pp_match.group(1)),
        'tg': float(tg_match.group(1)),
        'vram': float(vram_val) if vram_val > 0 else 0,
        'svram': float(svram_val) if svram_val > 0 else 0
    })

if not processed_blocks:
    print("Error: The parser failed to extract any configuration blocks.")
    print(f"Detected bench type: {bench_type}")
    print("Make sure log file contains TESTING: or BENCHY: markers.")
    sys.exit(1)

print(f"[SUCCESS] Processed {len(processed_blocks)} benchmark block(s)")

N = len(processed_blocks)
configs = [b['config_id'] for b in processed_blocks]
durations = [b['duration'] if b['duration'] > 0 else 100 for b in processed_blocks]
server_startups = [b['server_startup'] if b['server_startup'] > 0 else 0 for b in processed_blocks]
pps = [b['pp'] for b in processed_blocks]
tgs = [b['tg'] for b in processed_blocks]
vrams = [b['vram'] if b['vram'] > 0 else 4000 for b in processed_blocks]
svrams = [b['svram'] if b['svram'] > 0 else 12000 for b in processed_blocks]

# --- 2. COMPOSE 5-AXIS GRAPH LAYERS ---
fig, ax1 = plt.subplots(figsize=(30, 11), facecolor='white')

ax2 = ax1.twinx()
ax3 = ax1.twinx()
ax4 = ax1.twinx()
ax5 = ax1.twinx()

# Absolute multi-axis offset alignment
ax2.spines['left'].set_position(('axes', -0.06))
ax2.yaxis.set_label_position('left')
ax2.yaxis.set_ticks_position('left')

ax3.spines['left'].set_position(('axes', -0.13))
ax3.yaxis.set_label_position('left')
ax3.yaxis.set_ticks_position('left')

ax4.spines['right'].set_position(('axes', 1.06))
ax4.yaxis.set_label_position('right')
ax4.yaxis.set_ticks_position('right')

ax5.spines['right'].set_position(('axes', 1.13))
ax5.yaxis.set_label_position('right')
ax5.yaxis.set_ticks_position('right')

# Map metric coloring attributes
ax1.set_ylabel('Token Generation (t/s)', color='#1565c0', fontsize=10, fontweight='bold')
ax1.tick_params(axis='y', colors='#1565c0')
ax2.set_ylabel('Prompt Processing (t/s)', color='#5dade2', fontsize=10, fontweight='bold')
ax2.tick_params(axis='y', colors='#5dade2')
ax3.set_ylabel('Total Duration (s)', color='#e74c3c', fontsize=10, fontweight='bold')
ax3.tick_params(axis='y', colors='#e74c3c')
ax4.set_ylabel('Peak VRAM (MiB)', color='#81c784', fontsize=10, fontweight='bold')
ax4.tick_params(axis='y', colors='#81c784')
ax5.set_ylabel('Shared VRAM (MiB)', color='#2e7d32', fontsize=10, fontweight='bold')
ax5.tick_params(axis='y', colors='#2e7d32')

# --- 3. DATA PLOT LAYERING ---
ind = np.arange(N)
width = 0.16
alpha = 0.85

ax1.grid(True, axis='y', color='#e0e0e0', linewidth=0.5, alpha=0.7, zorder=1)

# Stacked duration bars: server startup (bottom) + benchmark duration (top)
rects3_startup = ax3.bar(ind - 2*width, server_startups, width, color='#c0392b', alpha=alpha, zorder=3, label='Server Startup')
rects3 = ax3.bar(ind - 2*width, durations, width, bottom=server_startups, color='#e74c3c', alpha=alpha, zorder=3, label='Benchmark Duration')
rects2 = ax2.bar(ind - width,   pps,       width, color='#5dade2', alpha=alpha, zorder=3)
rects1 = ax1.bar(ind,           tgs,       width, color='#1565c0', alpha=alpha, zorder=3)
rects4 = ax4.bar(ind + width,   vrams,     width, color='#81c784', alpha=alpha, zorder=3)
rects5 = ax5.bar(ind + 2*width, svrams,    width, color='#2e7d32', alpha=alpha, zorder=3)

# --- 4. EXPAND UPPER MARGINS FOR TOPPERS ---
for ax in [ax1, ax2, ax3, ax4, ax5]:
    ax.margins(y=0.25)

# --- 5. DYNAMIC BACKGROUND BAND & REGION LABELS ---
unique_sets = []
for b in processed_blocks:
    if b['config_a'] not in unique_sets:
        unique_sets.append(b['config_a'])

region_colors = ['#fff3e0', '#ffe0b2', '#ffcc80']
patch_colors = ['#ff9800', '#fb8c00', '#f57c00']
region_labels = ['Config Set A', 'Config Set B', 'Config Set C']

import matplotlib.transforms as transforms
trans = transforms.blended_transform_factory(ax1.transData, ax1.transAxes)

for idx, r_name in enumerate(unique_sets):
    indices = [i for i, b in enumerate(processed_blocks) if b['config_a'] == r_name]
    start = indices[0] - 0.5
    end = indices[-1] + 0.5
    
    c_idx = idx % len(region_colors)
    lbl_text = region_labels[idx] if idx < len(region_labels) else f"Config Set {idx + 1}"
    
    ax1.axvspan(start, end, facecolor=region_colors[c_idx], alpha=0.6, zorder=0)
    
    if start > -0.5:
        ax1.axvline(start, color='#e0e0e0', linewidth=1.2, linestyle='--', zorder=0.5)
        
    mid = (start + end) / 2
    ax1.text(mid, 1.02, f"{lbl_text}: {r_name}", transform=trans,
             ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#555555', zorder=5)

# --- 6. FORMAT INDIVIDUAL HISTOGRAM DATA LABELS ---
def add_labels(rects, ax, fmt_type, color):
    ylim = ax.get_ylim()
    offset = (ylim[1] - ylim[0]) * 0.008
    for rect in rects:
        val = rect.get_height()
        if fmt_type == 'time':
            lbl = f"{int(val//60):02d}:{int(val%60):02d}"
        elif fmt_type == 'dec':
            lbl = f"{val:.1f}"
        elif fmt_type == 'intM':
            lbl = f"{int(val)}M"
        ax.text(rect.get_x() + rect.get_width()/2., rect.get_y() + val + offset, lbl,
                ha='center', va='bottom', rotation=90, fontsize=7.0, color=color, fontweight='semibold', zorder=4)

add_labels(rects3_startup, ax3, 'time', '#c0392b')
add_labels(rects3, ax3, 'time', '#e74c3c')
add_labels(rects2, ax2, 'dec', '#5dade2')
add_labels(rects1, ax1, 'dec', '#1565c0')
add_labels(rects4, ax4, 'intM', '#81c784')
add_labels(rects5, ax5, 'intM', '#2e7d32')

# --- 7. X-AXIS AND TICK LABELS FORMATTING ---
ax1.set_xticks(ind)
ax1.set_xticklabels(configs, rotation=35, ha='right', fontsize=8.0)

# --- 8. LEGEND & TITLE COMPOSITION ---
metrics_labels = ['Duration (Bench)', 'Duration (Startup)', 'PP', 'TG', 'VRAM', 'SharedRAM']
metrics_colors = ['#e74c3c', '#c0392b', '#5dade2', '#1565c0', '#81c784', '#2e7d32']
legend_handles = [mpatches.Patch(color=c, label=l) for c, l in zip(metrics_colors, metrics_labels)]

for idx in range(min(len(unique_sets), 3)):
    legend_handles.append(mpatches.Patch(color=patch_colors[idx], label=region_labels[idx]))

ax1.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.14),
           ncol=8, frameon=True, facecolor='white', edgecolor='#cccccc')

plt.title("${analysis_title} [{0}]".format(bench_type.upper()), fontsize=14, fontweight='bold', pad=88, color='#222222')

# Rigid layout settings
plt.tight_layout(rect=[0.06, 0, 0.95, 0.95])
plt.savefig("${output_png}", dpi=180, bbox_inches='tight', facecolor='white')
print(f"[COMPLETE] Chart generated: ${output_png}")

EOF
}



run_dump_params_wrapper_AI_PROMPT()
{
    local analysis_title=$1

    echo "==================== DATA ANALYST PROMPT: ${analysis_title} ===================="
    echo "ROLE: Act as a Senior Data Analyst and Python Data Scientist."
    echo ""
    echo "INSTRUCTION:"
    echo "  1. Parse ALL data result blocks from the input data below."
    echo "  2. For each result block extract:"
    echo "     - Primary Y-Axis: Total Duration (s)"
    echo "     - Secondary Y-Axis: Prompt Processing (t/s)"
    echo "     - Tertiary Y-Axis: Token Generation (t/s)"
    echo "     - Quaternary Y-Axis: Peak VRAM (MiB)"
    echo "     - Quinary Y-Axis: Shared VRAM (MiB)"
    echo "     - Configuration parameters (config_a, config_b, config_c)"
    echo "  3. Use Python + matplotlib to generate a publication-quality visualization"
    echo "     saved as PNG, white background, dpi=180."
    echo ""
    echo "ANALYSIS TITLE: ${analysis_title}"
    echo ""
    echo "----------------------------------------------------------------"
    echo "VISUALIZATION SPECIFICATIONS:"
    echo ""
    echo "Chart type: Grouped/Clustered Bar Chart, 5 vertical bars per configuration."
    echo "Figure size: (30, 10) inches. White background throughout."
    echo ""
    echo "Y-AXES (5 total, color-coded):"
    echo "  Left primary         : Token Generation (t/s)    color=#1565c0  max=auto"
    echo "  Left offset (ax2)    : Prompt Processing (t/s)   color=#5dade2  max=auto"
    echo "  Left offset-2 (ax3)  : Total Duration (s)        color=#e74c3c  max=auto"
    echo "  Right primary        : Peak VRAM (MiB)           color=#81c784  max=auto"
    echo "  Right offset (ax5)   : Shared VRAM (MiB)         color=#2e7d32  max=auto"
    echo "  ax2 spine: ('axes', -0.06)  ax3 spine: ('axes', -0.13)  ax4 spine: ('axes', 1.06) ticks/label on right. ax5 spine: ('axes', 1.13) ticks/label on right."
    echo ""
    echo "BAR LAYOUT:"
    echo "  5 bars per group, bar_width=0.16, offsets=[-2, -1, 0, +1, +2] * bar_width."
    echo "  Bar order (left to right): Duration | PP | TG | Peak VRAM | Shared VRAM"
    echo "  Colors: Duration=#e74c3c  PP=#5dade2  TG=#1565c0  VRAM=#81c784  SharedRAM=#2e7d32. Alpha=0.85."
    echo ""
    echo "DATA LABELS:"
    echo "  Rotated 90deg, fontsize=7.0, above each bar (y = value + 0.8% of axis max)."
    echo "  Format: Duration->mm:ss  PP->1 decimal  TG->1 decimal  VRAM->integer+'M'  SharedRAM->integer+'M'"
    echo "  Label color matches bar color."
    echo ""
    echo "X-AXIS LABELS (3 lines each, inferred from logs):"
    echo "  Line 1: Configuration ID"
    echo "  Rotation=35, ha='right', fontsize=8.0."
    echo ""
    echo "BACKGROUND REGION BANDS (derive boundaries from inferred config parameters):"
    echo "  Region A             -> background #b3d9e8 (darker blue tint)    label 'Config Set A'"
    echo "  Region B             -> background #a8d5c0 (darker green tint)   label 'Config Set B'"
    echo "  Region C             -> background #ffd966 (darker amber tint)   label 'Config Set C'"
    echo "  Bold region label at top of each band, fontsize=9."
    echo "  Dashed vertical dividers (#666666, linewidth=1.2) between regions."
    echo ""
    echo "LEGEND:"
    echo "  Position: upper center, bbox_to_anchor=(0.5, 1.048), ncol=8, frameon=True."
    echo "  8 entries: Duration, PP, TG, VRAM, SharedRAM + 3 region patches."
    echo "  Metric patches: Duration=#e74c3c  PP=#5dade2  TG=#1565c0  VRAM=#81c784  SharedRAM=#2e7d32"
    echo "  Region patches: RegionA=#7aafd4  RegionB=#7ec3a8  RegionC=#ffc933"
    echo "  Legend placed BELOW title, ABOVE chart."
    echo ""
    echo "TITLE: '${analysis_title}'"
    echo "  fontsize=13, fontweight='bold', pad=52, color='#222222'."
    echo ""
    echo "GRID: Horizontal only, color='#cccccc', linewidth=0.5, alpha=0.6, zorder=2."
    echo "LAYOUT: plt.tight_layout(rect=[0.06, 0, 0.95, 1])"
    echo "OUTPUT: PNG, dpi=180, bbox_inches='tight', facecolor='white'."
    echo ""
    echo "================================================================"
    echo "OUTPUT FORMAT REQUIREMENT:"
    echo "1. Briefly summarize key performance/metric deltas across data points."
    echo "2. Note effect of configuration parameters where present."
    echo "3. Flag any anomalies or outliers in the data."
    echo "4. Produce the chart exactly per specifications above."
    echo "==================== END OF PROMPT ============================"
}


run_dump_params_wrapper_COMMON()
{
    local model_name=$1
    local bench=$2
    local context=$3
    local config_format=$4  # How to format test IDs
    shift 4
    local -a config_list=("$@")
    
    echo "==================== LLM DATA ANALYST PROMPT: ${model_name} [${bench}] ===================="
    echo "ROLE: Act as a Senior Performance Analyst and Python Data Scientist."
    echo "INSTRUCTION: Provide a concise analysis and Immediately use the Python tool to parse the benchmark results from the logs above. Generate a Clustered Bar Chart."
    echo "MODEL: ${model_name}"
    echo "BENCHMARK: ${bench}"
    echo "CONTEXT: ${context}"
    echo "----------------------------------------------------------------"
    
    # Common visualization specs
    echo "VISUALIZATION SPECIFICATIONS:"
    echo "- Primary Y-Axis: Total Duration (s)"
    echo "- Secondary Y-Axis: Prompt Processing (t/s)"
    echo "- Tertiary Y-Axis: Token Generation (t/s)"
    echo "- Quaternary Y-Axis: Peak VRAM (MiB)"
    echo "- Quinary Y-Axis: Shared VRAM (MiB)"
    echo "- Data Labels: Annotate bars with 'Duration', 'PP', 'TG', 'Peak VRAM', 'Shared VRAM'."
    echo "- X-Axis Labels: Use the TEST CONFIGURATION IDs provided below. IMPORTANT use full names"
    echo ""
    echo "TEST CONFIGURATIONS:"
    
    # Build config list with provided format
    local count=1
    for item in "${config_list[@]}"; do
        if [[ "$config_format" == "SIMPLE" ]]; then
            echo "TEST ID ${count}: ${item}"
        elif [[ "$config_format" == "PREFIXED" ]]; then
            echo "TEST ID ${count}: ${context}_${item}"
        fi
        ((count++))
    done
    
    echo ""
    echo "[PASTE RAW RESULTS BELOW THIS LINE]"
    echo "----------------------------------------------------------------"
    echo "OUTPUT FORMAT REQUIREMENT:"
    echo "1. Briefly summarize the performance delta, VRAM efficiency, and wall-clock time."
    echo "2. Ensure the graph is high-contrast and includes a combined legend for all 5 metrics."
    echo "3. Graph Title: ${model_name} - ${bench} @ ${context} ctx"
    echo "4. Plot data with Grouped Bar Chart (or Clustered Bar Chart) with Multiple Y-Axes. IMPORTANT include all TEST CONFIGURATIONS"
    echo "==================== END OF PROMPT ============================"
}

run_dump_params_wrapper_EXTRAS()
{
    local model_name=$1
    local bench=$2
    local runner_ctx=$3
    shift 3
    local -a extras_list=("$@")
    
    run_dump_params_wrapper_COMMON "$model_name" "$bench" "$runner_ctx" "SIMPLE" "${extras_list[@]}"
}

run_dump_params_wrapper_PROMPTS()
{
    local model_name=$1
    local bench=$2
    local extras=$3
    shift 3
    local -a prompt_list=("$@")
    
    run_dump_params_wrapper_COMMON "$model_name" "$bench" "$extras" "PREFIXED" "${prompt_list[@]}"
}

#######################################################################################################################################################################################
################################################################ COMMANDS SINGLES (SERVE) #############################################################################################
#######################################################################################################################################################################################

run_serve_params_FILTER=(
    # Memory & fit - the most useful diagnostics
    "common_memory_breakdown_print"
    "common_params_fit_impl"
    "common_fit_params"

    # Model & context sizing
    "llama_context: n_ctx"
    "llama_kv_cache: size"
    "load_tensors: offload"
    "print_info: model type"
    "print_info: model params"
    "print_info: file size"

    # Server ready signals
    "server is listening"
    "model loaded"
    "all slots are idle"

    # Errors / warnings (always want these)
    "[Ee]rror"
    "[Ww]arning"
    "WARN"
)

run_serve_params_cmd_build()
{
    local model=$1
    local server=$2
    local extra=$3
    local PROMPT=$4
    local serve_extra=$5
    #local PROMPT="512,2048,8192,16384,32768,65536,131072,262144"
    local -n SERVE_CMD_ARRAY=$6  # Reference to caller's array
    
    # Resolve model and server
    local MODEL_MSE=${MODELS[$model]}
    local LLAMA_SERVER=${SERVERS[$server]}
    
    # Process EXTRA flags
    local EXTRA
    local lookup_key=$(EXTRAS_get_flags_normalized "$extra")
    echo "EXTRA: lookup_orig = ${extra}"
    echo "EXTRA: lookup_key = ${lookup_key}"
    
    if [[ -n "${EXTRAS[$lookup_key]+isset}" ]]; then
        local lookup_data=${EXTRAS[$lookup_key]}
        echo "EXTRA: lookup_data = ${lookup_data}"
        EXTRA=$(EXTRAS_get_flags_replaced "${extra}" "${lookup_data}" "${model}")
        echo "EXTRA: lookup_final = ${EXTRA}"
    else
        echo "WARNING unhandled - extra $extra"
        EXTRA="${extra}"
    fi
    
    # Process SERVE_EXTRA
    local SERVE_EXTRA
    if [[ -n "${MODELS_SERVE_EXTRA[$model]+isset}" ]]; then
        SERVE_EXTRA="${MODELS_SERVE_EXTRA[$model]}"
    else
        SERVE_EXTRA="${MODELS_SERVE_EXTRA['common']}"
        echo "WARNING unhandled (used 'common') with user's serve_extra='$serve_extra'"
        SERVE_EXTRA="${SERVE_EXTRA} ${serve_extra}"
    fi
    
    # Parse server network config
    local -a parts_NET_CFG
    IFS='|' read -r -a parts_NET_CFG <<< "${SERVERS_NET_CFG[$server]}"
    local HOST="${parts_NET_CFG[0]}"
    local PORT="${parts_NET_CFG[1]}"
    #local URL="${parts_NET_CFG[2]}"

    # -t 16 may hurt: CPU threads don't scale past GPU saturation. Try -t 8 or -t 4. MSE TRY AVOID CPU in GENERAL !!!  
    
    # Build command array
    SERVE_CMD_ARRAY=(
        "$LLAMA_SERVER"
        -t "$MSE_THREADS"
        -c "$PROMPT"
        --parallel "$MSE_PARALLEL"
        -ngl 999
        -m "$MODEL_MSE"
        $EXTRA
        $SERVE_EXTRA
        --host "$HOST"
        --port "$PORT"
    )
    
    # Debug output
    echo "--------------------------------------------------------------"
    echo "SERVING: $model | Server: $server | P: ${PROMPT} | E: ${EXTRA} | ${MODEL_MSE}"
    echo "SERVING: ${SERVE_CMD_ARRAY[*]}"
    echo "--------------------------------------------------------------"
}

# Executor function - builds and runs the command
run_serve_params()
{
    local model=$1
    local server=$2
    local extra=$3
    local PROMPT=$4
    local serve_extra=$5
    local output_mode="${6:-"filter"}"
    
    # Build the command
    local -a SERVE_CMD
    run_serve_params_cmd_build "$model" "$server" "$extra" "$PROMPT" "$serve_extra" SERVE_CMD
    
    if [[ "$MSE_SERVER_PRESETS_DUMP" == "true" ]]; then
        echo "${SERVE_CMD[@]}"
        return
    fi
    
    # Execute with output mode handler
    EXEC_cmd_with_output_mode "${output_mode}" run_serve_params_FILTER "${SERVE_CMD[@]}"
}

# Spawns llama-server, waits for /health, runs command function, then cleanly kills the server.
run_serve_params_on_background()
{
    local model="$1"
    local server="$2"
    local extra="$3"
    local ctx="$4"
    local is_inspect="$5"
    
    # MSE: NESTED CALLS TRAP  local -n cmd=$5  # Reference to the command array - The actual command/function to run (e.g., run_benchy_params)
    shift 5
    # Safely capture the remaining arguments into a local array
    local cmd=("$@")

    local BASE_URL
    BASE_URL=$(_resolve_base_url "$server")

    echo "--------------------------------------------------------------"
    echo "SERVER WRAPPER: Starting $server for task..."
    echo "--------------------------------------------------------------"

    local SERVER_LOG_FILE="server.log"
    local FOREGROUND_LOG_FILE="foreground.log"

    # 1. Spawn server in background
    run_serve_params "$model" "$server" "$extra" "${ctx}" > "${SERVER_LOG_FILE}" 2>&1 &
    local SUBSHELL_PID=$!
    
    if [ "$MSE_PS_DEBUG" != "false" ]; then
        ps aux # MSE: DEBUG
    fi
    
    # 2. Wait for /health
    if ! server_wait_for_ready "${BASE_URL}"; then
        local FAIL_PID=$(server_get_pid_by_ppid "$SUBSHELL_PID")
        server_stop_windows "${FAIL_PID}" "${SUBSHELL_PID}"
        server_logfile_and_cleanup "${SERVER_LOG_FILE}" "${FOREGROUND_LOG_FILE}"
        EXEC_user_notification_WIN_T2S "Server failed to start for $model"
        return 1
    fi
    
    local SERVER_PID=$(server_get_pid_by_ppid "$SUBSHELL_PID")
    echo "Server ready: SERVER_PID=${SERVER_PID} SUBSHELL_PID=${SUBSHELL_PID}"
    
    # 3. Execute the payload command
    if [ "$is_inspect" = "is_inspect" ]; then
        "${cmd[@]}" | tee -a "${FOREGROUND_LOG_FILE}"
    else
        "${cmd[@]}" > "${FOREGROUND_LOG_FILE}" 2>&1
    fi
    local TASK_EXIT=$?
    server_logfile_and_cleanup "${SERVER_LOG_FILE}" "${FOREGROUND_LOG_FILE}"
    
    # 4. Cleanup
    server_stop_windows "${SERVER_PID}" "${SUBSHELL_PID}"

    if [ "$MSE_PS_DEBUG" != "false" ]; then
        ps aux # MSE: DEBUG
    fi
    
    return $TASK_EXIT
}

#######################################################################################################################################################################################
################################################################ COMMANDS SINGLES (PRESETS_SERVE) #####################################################################################
#######################################################################################################################################################################################

run_presets_dump() {
        
    # Find all functions starting with run_SERVER_
    local server_functions=$(declare -F | awk '{print $3}' | grep '^run_SERVER_')
    
    if [[ -z "$server_functions" ]]; then
        echo "No functions matching 'run_SERVER_*' found."
        return
    fi
    
    MSE_SERVER_PRESETS_DUMP=true
    
    # Execute each function
    local count=0
    while IFS= read -r func_name; do
        echo "Executing: $func_name"
        $func_name
        echo "✓ Completed: $func_name"
        echo "---"
        ((count++))
    done <<< "$server_functions"
    
    echo "Total functions executed: $count"
    
    MSE_SERVER_PRESETS_DUMP=false
}

run_presets_serve()
{
    local server=$1
    local presets_file="${6:-${MSE_PRESETS_SERVE_FILE}}"

    local LLAMA_SERVER=${SERVERS[$server]}
    
    # Parse server network config
    local -a parts_NET_CFG
    IFS='|' read -r -a parts_NET_CFG <<< "${SERVERS_NET_CFG[$server]}"
    local HOST="${parts_NET_CFG[0]}"
    local PORT="${parts_NET_CFG[1]}"
    #local URL="${parts_NET_CFG[2]}"
    
    # Build the command
    local -a SERVE_PRESETS_CMD
    
    SERVE_PRESETS_CMD=(
        "$LLAMA_SERVER"
        --models-preset ${presets_file}
        --host "$HOST"
        --port "$PORT"
    )
    
    echo "--------------------------------------------------------------"
    echo "PRESETS SERVING: File: $presets_file"
    echo "PRESETS : ${SERVE_PRESETS_CMD[*]}"
    echo "--------------------------------------------------------------"
    
    ${SERVE_PRESETS_CMD[@]}
}

run_PRESETS_serve() 
{
    run_presets_serve "cuda_flash"
}

run_PRESETS_dump() {
    run_presets_dump | grep 'SERVING: c'
}

#######################################################################################################################################################################################
################################################################ COMMANDS SINGLES (BENCH) #############################################################################################
#######################################################################################################################################################################################

run_bench_params_cmd_build()
{
    local model=$1
    local bench=$2
    local extra=$3
    local PROMPT=$4
    local bench_extra=$5
    local -n BENCH_CMD_ARRAY=$6  # Reference to caller's array
      
    #local PROMPT="512,2048,8192,16384,32768,65536,131072,262144"
    local MODEL_MSE=${MODELS[$model]}
    local LLAMA_BENCH=${BENCHES[$bench]}

    local EXTRA
    local lookup_key=$(EXTRAS_get_flags_normalized "$extra")
    echo "EXTRA: lookup_orig = ${extra}"
    echo "EXTRA: lookup_key = ${lookup_key}"
    if [[ -n "${EXTRAS[$lookup_key]+isset}" ]]; then
        local lookup_data=${EXTRAS[$lookup_key]}
        echo "EXTRA: lookup_data = ${lookup_data}"
        EXTRA=$(EXTRAS_get_flags_replaced "${extra}" "${lookup_data}" "DUMMY_MODEL")
        echo "EXTRA: lookup_final = ${EXTRA}"
    else
        echo "WARNING unhandled - extra $extra"
        EXTRA="${extra}"
    fi
    
    local BENCH_EXTRA
    if [[ -n "${MODELS_BENCH_EXTRA[$model]+isset}" ]]; then
        BENCH_EXTRA="${MODELS_BENCH_EXTRA[$model]}"
    else
        BENCH_EXTRA="${MODELS_BENCH_EXTRA['common']}"
        echo "WARNING unhandled (used 'common') with user's bench_extra='$bench_extra'"
        BENCH_EXTRA="${BENCH_EXTRA} ${BENCH_extra}"
    fi

    # -t 16 may hurt: CPU threads don't scale past GPU saturation. Try -t 8 or -t 4. MSE TRY AVOID CPU in GENERAL !!!  
    #local CMD="${LLAMA_BENCH} -t ${MSE_THREADS} -p ${PROMPT} -n ${MSE_BENCH_TG} -ngl 999 -m ${MODEL_MSE} ${EXTRA}"
    BENCH_CMD_ARRAY=(
        "${LLAMA_BENCH}"
        -t "${MSE_THREADS}"
        -p "${PROMPT}"
        -n "${MSE_BENCH_TG}"
        -ngl 999
        -m "${MODEL_MSE}"
    )
    
    if [ -n "${EXTRA}" ]; then
        local -a EXTRA_ARGS
        read -r -a EXTRA_ARGS <<< "${EXTRA}"
        BENCH_CMD_ARRAY+=("${EXTRA_ARGS[@]}")
    fi
    
    if [ -n "${BENCH_EXTRA}" ]; then
        local -a BENCH_EXTRA_ARGS
        read -r -a BENCH_EXTRA_ARGS <<< "${BENCH_EXTRA}"
        BENCH_CMD_ARRAY+=("${BENCH_EXTRA_ARGS[@]}")
    fi
    
    echo "--------------------------------------------------------------"
    echo "TESTING: $model | Bench: $bench | P: ${PROMPT} | E: ${EXTRA} | ${MODEL_MSE}"
    echo "TESTING: ${BENCH_CMD_ARRAY[*]}"
    echo "--------------------------------------------------------------"
}

run_bench_params_bundler()
{
    local model=$1
    local bench=$2
    local extra=$3
    local PROMPT=$4
    local bench_extra=$5

    # Build the command
    local -a BENCH_CMD
    run_bench_params_cmd_build "$model" "$bench" "$extra" "$PROMPT" "$bench_extra" BENCH_CMD
    
    "${BENCH_CMD[@]}"
}

run_bench_params()
{
    local model=$1
    local bench=$2
    local extra=$3
    local PROMPT=$4
    local bench_extra=$5
    local output_mode="${6:-"all"}"
    
    local -a BUNDLER_CMD=(
        run_bench_params_bundler
        "$model"
        "$bench"
        "$extra"
        "$PROMPT"
        "$bench_extra"
    )
    
    # 2. Execute Bench cmd
    EXEC_cmd_with_gpu_and_ram_monitoring "${output_mode}" "filter" BUNDLER_CMD \
        "Run bench finish - TESTING: $model with bench $bench PromptSIZE: ${PROMPT} Extra: $extra"
}

run_bench_params_wrapper_EXTRAS()
{
    local model=$1
    local bench=$2
    local runner_ctx=$3
    shift 3
    local -a extras_list=("$@")

    echo ""
    echo "Starting tests:"
    local count=1
    for item in "${extras_list[@]}"; do
        echo "TEST ${count}: ${item}"
        ((count++))
    done
    
    echo ""
    
    for extras_item in "${extras_list[@]}" ; do 
        run_bench_params ${model} ${bench} $extras_item ${runner_ctx}
    done
    
    run_dump_params_wrapper_EXTRAS ${model} ${bench} ${runner_ctx} "${extras_list[@]}"
}

run_bench_params_wrapper_PROMPTS()
{
    local model=$1
    local bench=$2
    local extras=$3
    shift 3
    local -a prompt_list=("$@")

    echo ""
    echo "Starting tests:"
    local count=1
    for item in "${prompt_list[@]}"; do
        echo "TEST ${count}: ${extras}__CTX_${item}"
        ((count++))
    done
    
    echo ""
    
    for promtp_item in "${prompt_list[@]}" ; do 
        run_bench_params ${model} ${bench} $extras ${promtp_item}
    done
    
    run_dump_params_wrapper_PROMPTS ${model} ${bench} $extras "${prompt_list[@]}"
}

#######################################################################################################################################################################################
################################################################ COMMANDS SINGLES (BENCH-Y) ###########################################################################################
#######################################################################################################################################################################################

run_benchy_params_cmd_build() 
{
    local model=$1
    local server=$2
    local extra=$3
    local PROMPT=$4
    local -n BENCHY_CMD_ARRAY=$5  # Reference to caller's array

    local BASE_URL
    BASE_URL=$(_resolve_base_url "$server")
    local PP_RAW="${PROMPT:-$MSE_BENCHY_PP}"
    local PP_LIST="${PP_RAW//,/ }"
    local MODEL_LABEL
    MODEL_LABEL=$(_model_label_from_path "${MODELS[$model]}")

    # Build directly into the referenced array
    BENCHY_CMD_ARRAY=(uvx llama-benchy
        --base-url "$BASE_URL"
        --model   "$MODEL_LABEL"
        --runs    "$MSE_BENCHY_RUNS"
        --skip-coherence
    )

    local pp
    for pp in $PP_LIST; do
        BENCHY_CMD_ARRAY+=(--pp "$pp")
    done

    local tg
    for tg in $MSE_BENCH_TG; do
        BENCHY_CMD_ARRAY+=(--tg "$tg")
    done

    local depth
    for depth in $MSE_BENCHY_DEPTH; do
        BENCHY_CMD_ARRAY+=(--depth "$depth")
    done

    echo "--------------------------------------------------------------"
    echo "BENCHY:  $model | Server: $server | PP: ${PP_LIST} | TG: ${MSE_BENCH_TG} | E: ${extra}"
    echo "BENCHY:  URL=${BASE_URL} | MODEL_LABEL=${MODEL_LABEL}"
    echo "BENCHY:  ${BENCHY_CMD_ARRAY[*]}"
    echo "--------------------------------------------------------------"
}

run_benchy_params_with_server_bundler()
{
    local model=$1
    local server=$2
    local extra=$3
    local PROMPT=$4
    local ctx=$5

    # Logic for context synchronization
    if [ "${MSE_BENCHY_PP_SYNCH}" -eq 1 ]; then
        ctx="${PROMPT}"
    fi
    
    local -a BENCHY_CMD=()
    run_benchy_params_cmd_build "$model" "$server" "$extra" "$PROMPT" BENCHY_CMD

    # MSE: NESTED CALLS TRAP run_serve_params_on_background "$model" "$server" "$extra" "$ctx" "is_benchy" BENCHY_CMD
    run_serve_params_on_background "$model" "$server" "$extra" "$ctx" "is_benchy" "${BENCHY_CMD[@]}"
}

run_benchy_params_with_server()
{
    local model=$1
    local server=$2
    local extra=$3
    local PROMPT=$4
    local ctx=$5

    local -a BACKGROUND_SERVER_CMD=(
        run_benchy_params_with_server_bundler
        "$model"
        "$server"
        "$extra"
        "$ctx"
    )
    
    if [ 0 -eq 1 ] ; then 
        "${BACKGROUND_SERVER_CMD[@]}"
    else
        EXEC_cmd_with_gpu_and_ram_monitoring "all" "dummy" \
            BACKGROUND_SERVER_CMD \
            "Complete: Server + Benchmark with full monitoring"
    fi
}

run_benchy_params_with_server_wrapper_EXTRAS()
{
    local model=$1
    local server=$2
    local runner_ctx=$3
    shift 3
    local -a extras_list=("$@")

    echo ""
    echo "Starting tests:"
    local count=1
    for item in "${extras_list[@]}"; do
        echo "TEST ${count}: ${item}"
        ((count++))
    done
    
    echo ""
    
    for extras_item in "${extras_list[@]}" ; do 
        run_benchy_params_with_server ${model} ${server} $extras_item ${runner_ctx} ${runner_ctx}
    done
    
    run_dump_params_wrapper_EXTRAS ${model} ${server} ${runner_ctx} "${extras_list[@]}"
}

#######################################################################################################################################################################################
################################################################ COMMANDS SINGLES (TESTS) ###########################################################################################
#######################################################################################################################################################################################

run_inspect_params_cmd_build() 
{
    local TEST_TYPE=$1 
    local -n INSPECT_CMD_ARRAY=$2  # Reference to caller's array
    
    # Set your environment to point to your local llama.cpp server
    export OPENAI_API_KEY=dummy
    
    # Build directly into the referenced array
    INSPECT_CMD_ARRAY=(uvx --from inspect-ai --with openai --with inspect-evals inspect eval 
        inspect_evals/${TEST_TYPE} 
        --model openai/local 
        --model-base-url http://localhost:8033/v1
    )
    echo "--------------------------------------------------------------"
    echo "INSPECT:  ${TEST_TYPE}"
    echo "INSPECT:  ${INSPECT_CMD_ARRAY[*]}"
    echo "--------------------------------------------------------------"
}

run_inspect_params_with_server_bundler()
{
    local model=$1
    local server=$2
    local extra=$3
    local ctx=$4
    local test_type=$5
    
    local -a INSPECT_CMD=( 
        eval "echo 'before' ; sleep 5 ; echo 'after'"
    )
    # !!! MSE_TODO: ESSET ISSUE !!!
    #run_inspect_params_cmd_build "$test_type" INSPECT_CMD
    
    # MSE: NESTED CALLS TRAP run_serve_params_on_background "$model" "$server" "$extra" "$ctx" "is_inspect" INSPECT_CMD
    run_serve_params_on_background "$model" "$server" "$extra" "$ctx" "is_inspect" "${INSPECT_CMD[@]}"
}

run_inspect_params_with_server()
{
    local model=$1
    local server=$2
    local extra=$3
    local ctx=$4
    local test_type=$5
    
    local -a BACKGROUND_SERVER_CMD=(
        run_inspect_params_with_server_bundler
        "$model"
        "$server"
        "$extra"
        "$ctx"
        "$test_type"
    )
    
    if [ 0 -eq 1 ] ; then 
        "${BACKGROUND_SERVER_CMD[@]}"
    else
        EXEC_cmd_with_gpu_and_ram_monitoring "all" "dummy" \
            BACKGROUND_SERVER_CMD \
            "Complete: Server + Benchmark with full monitoring"
    fi
}

run_inspect_params_with_server_wrapper_EXTRAS()
{
    local model=$1
    local server=$2
    local runner_ctx=$3
    local test_type=$4
    shift 4
    local -a extras_list=("$@")
    
    echo ""
    echo "Starting tests:"
    local count=1
    for item in "${extras_list[@]}"; do
        echo "TEST ${count}: ${test_type} ${item}"
        ((count++))
    done
    
    echo ""
    for extras_item in "${extras_list[@]}" ; do 
        run_inspect_params_with_server "$model" "$server" "$extras_item" "$runner_ctx" "$test_type"
    done
    
    run_dump_params_wrapper_EXTRAS "$model" "$server" "$runner_ctx" "${extras_list[@]}"
}

# For verified SWE-bench coding task (DOCKER)
run_inspect_params_with_server_wrapper_EXTRAS_swe_bench_verified()
{    
    local model=$1
    local server=$2
    local runner_ctx=$3
    
    shift 3
    local -a extras_list=("$@")
    
    local test_type="swe_bench_verified"
    run_inspect_params_with_server_wrapper_EXTRAS "$model" "$server" "$runner_ctx" "$test_type" "${extras_list[@]}"
}

# For General Knowledge (MMLU-Pro) - non-DOCKER
run_inspect_params_with_server_wrapper_EXTRAS_mmlu_pro()
{    
    local model=$1
    local server=$2
    local runner_ctx=$3
    
    shift 3
    local -a extras_list=("$@")
    
    local test_type="mmlu_pro"
    run_inspect_params_with_server_wrapper_EXTRAS "$model" "$server" "$runner_ctx" "$test_type" "${extras_list[@]}"
}

# For Hard Science/Reasoning (GPQA) - non-DOCKER
run_inspect_params_with_server_wrapper_EXTRAS_gpqa_diamond()
{    
    local model=$1
    local server=$2
    local runner_ctx=$3
    
    shift 3
    local -a extras_list=("$@")
    
    local test_type="gpqa_diamond"
    run_inspect_params_with_server_wrapper_EXTRAS "$model" "$server" "$runner_ctx" "$test_type" "${extras_list[@]}"
}

# For Math (AIME / AMC) - non-DOCKER
run_inspect_params_with_server_wrapper_EXTRAS_math_aime_24()
{    
    local model=$1
    local server=$2
    local runner_ctx=$3
    
    shift 3
    local -a extras_list=("$@")
    
    local test_type="math_aime_24"
    run_inspect_params_with_server_wrapper_EXTRAS "$model" "$server" "$runner_ctx" "$test_type" "${extras_list[@]}"
}
#######################################################################################################################################################################################
#######################################################################################################################################################################################
################################################################# COMMANDS GROUPS #####################################################################################################
#######################################################################################################################################################################################
#######################################################################################################################################################################################

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
################################################################# omnicoder_9b ########################################################################################################
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
run_BENCH_omnicoder_9b_CTX_cuda_flash()
{
    local runner_ctx=${1:-131072}
    local model="omnicoder_9b"
    local bench="cuda_flash"
    
    local -a extras_list=(
        "extra_fa_Vq8_default"  
        "extra_fa_Kq8_Vq4_default" 
        "extra_fa_Kq8_Vq8_default"
        "extra_fa_Kq4_Vq8_default"
    )
    
    run_bench_params_wrapper_EXTRAS ${model} ${bench} ${runner_ctx} "${extras_list[@]}"
}

run_SERVER_omnicoder_9b_CTX_cuda_flash()
{
    local runner_ctx=${1:-131072}
    run_serve_params "omnicoder_9b" "cuda_flash" "extra_fa_Kq8_Vq8_default" ${runner_ctx}
}

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
################################################################# Qwen36_35B_A3B_UD_Q4_K_M_MTP ########################################################################################
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

run_BENCH_Qwen36_35B_A3B_UD_Q4_K_M_MTP_CTX_cuda_flash_CFG_MOE_on_x_CTX_x()
{
    local runner_ctx=${1:-65536} 
    local model="Qwen36_35B_A3B_UD_Q4_K_M_MTP"
    local bench="cuda_flash"
    
    local -a extras_list=(
       "extra_fa_Kq8_Vq8_default_moe41"
       "extra_fa_Kq8_Vq4_default_moe41"
#        "extra_fa_Kq8_Vq8_default_moe36"
#        "extra_fa_Kq8_Vq4_default_moe36"
#        "extra_fa_Kq8_Vq8_default_moe34"
#        "extra_fa_Kq8_Vq4_default_moe34" # SWEETSPOT (CTX==65536)
#        "extra_fa_Kq8_Vq8_default_moe33" # PC freezing sporadicaly (CTX==65536)
#        "extra_fa_Kq8_Vq4_default_moe33" # PC freezing sporadicaly (CTX==65536)
#        "extra_fa_Kq8_Vq8_default_moe30" # PC freeze (CTX==65536)
#        "extra_fa_Kq8_Vq4_default_moe30" # PC freeze (CTX==65536)   
    )
    
    run_bench_params_wrapper_EXTRAS ${model} ${bench} ${runner_ctx} "${extras_list[@]}"
}

run_SERVER_Qwen36_35B_A3B_UD_Q4_K_M_MTP_CTX_cuda_flash()
{
    local runner_ctx=${1:-65536}
    run_serve_params "Qwen36_35B_A3B_UD_Q4_K_M_MTP" "cuda_flash" "extra_fa_Kq8_Vq8_default_moe34" ${runner_ctx} "" "all"
    #run_serve_params "Qwen36_35B_A3B_UD_Q4_K_M_MTP" "cuda_flash" "extra_fa_Kq8_Vq4_default_moe34_sd2mtp" ${runner_ctx} "" "all"
}

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
################################################################# Qwen36_28B_REAP20_A3B_Q4_K_M ########################################################################################
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

run_BENCH_Qwen36_28B_REAP20_A3B_Q4_K_M_CTX_cuda_flash_CFG_MOE_on_x_CTX_x()
{
    local runner_ctx=${1:-65536} 
    local model="Qwen36_28B_REAP20_A3B_Q4_K_M"
    local bench="cuda_flash"
    
    local -a extras_list=(
#   "extra_fa_Kq8_Vq8_default_moe41"
#   "extra_fa_Kq8_Vq4_default_moe41"
#   "extra_fa_Kq8_Vq8_b2048_ub256_moe41"
#   "extra_fa_Kq8_Vq4_b2048_ub256_moe41"
#   "extra_fa_Kq8_Vq8_b2048_ub512_moe41"  # SKIPPED like default
#   "extra_fa_Kq8_Vq4_b2048_ub512_moe41"  # SKIPPED like default
#   "extra_fa_Kq8_Vq8_b2048_ub1024_moe41"
#   "extra_fa_Kq8_Vq4_b2048_ub1024_moe41"
#   "extra_fa_Kq8_Vq8_b2048_ub2048_moe41"
#   "extra_fa_Kq8_Vq4_b2048_ub2048_moe41"
#   "extra_fa_Kq8_Vq8_b8192_ub8192_moe41" # PC freezing sporadicaly (CTX==65536)
#   "extra_fa_Kq8_Vq4_b8192_ub8192_moe41" # PC freezing sporadicaly (CTX==65536)
#   "extra_fa_Kq8_Vq8_b6144_ub6144_moe41"
#   "extra_fa_Kq8_Vq4_b6144_ub6144_moe41"
#   "extra_fa_Kq8_Vq8_b4096_ub4096_moe41" # SWEETSPOT 1. step
#   "extra_fa_Kq8_Vq4_b4096_ub4096_moe41" # SWEETSPOT 1. step
#   "extra_fa_Kq8_Vq8_b4096_ub2048_moe41"
#   "extra_fa_Kq8_Vq4_b4096_ub2048_moe41"
#   "extra_fa_Kq8_Vq8_b4096_ub1024_moe41"
#   "extra_fa_Kq8_Vq4_b4096_ub1024_moe41"
#   "extra_fa_Kq8_Vq8_b4096_ub512_moe41" 
#   "extra_fa_Kq8_Vq4_b4096_ub512_moe41"    
#   "extra_fa_Kq8_Vq8_b4096_ub256_moe41"    
#   "extra_fa_Kq8_Vq4_b4096_ub256_moe41" 



#      "extra_fa_Kq8_Vq8_default_moe34"
#      "extra_fa_Kq8_Vq4_default_moe34"
#      "extra_fa_Kq8_Vq8_default_moe28"
#      "extra_fa_Kq8_Vq4_default_moe28"
#      "extra_fa_Kq8_Vq8_default_moe23" # PC freezing sporadicaly (CTX==65536)
#      "extra_fa_Kq8_Vq4_default_moe23" # PC seems to be OK
#      "extra_fa_Kq8_Vq8_default_moe20" # PC freeze (CTX==65536)
#      "extra_fa_Kq8_Vq4_default_moe20" # PC freeze (CTX==65536)

)

    run_bench_params_wrapper_EXTRAS ${model} ${bench} ${runner_ctx} "${extras_list[@]}"
}

run_BENCHY_WITH_SERVER_Qwen36_28B_REAP20_A3B_Q4_K_M_CTX_cuda_flash_CFG_MOE_on_x_CTX_x()
{
    local runner_ctx=${1:-65536} 
    local model="Qwen36_28B_REAP20_A3B_Q4_K_M"
    local bench="cuda_flash"
    
    local -a extras_list=(
#      "extra_fa_Kq8_Vq8_default_moe41"
#      "extra_fa_Kq8_Vq4_default_moe41"
#      "extra_fa_Kq8_Vq8_b6144_ub6144_moe41"
#      "extra_fa_Kq8_Vq4_b6144_ub6144_moe41"
#      "extra_fa_Kq8_Vq8_b4096_ub4096_moe41" # SWEETSPOT 1. step
#      "extra_fa_Kq8_Vq4_b4096_ub4096_moe41" # SWEETSPOT 1. step
#      "extra_fa_Kq8_Vq8_b4096_ub4096_moe41_sd3cache"
#      "extra_fa_Kq8_Vq4_b4096_ub4096_moe41_sd3cache"
#      "extra_fa_Kq8_Vq8_b4096_ub4096_moe41_sd2cache"
#      "extra_fa_Kq8_Vq4_b4096_ub4096_moe41_sd2cache"
    )
    
    run_benchy_params_with_server_wrapper_EXTRAS ${model} ${bench} ${runner_ctx} "${extras_list[@]}"
}

run_SERVER_Qwen36_28B_REAP20_A3B_Q4_K_M_CTX_cuda_flash()
{
    local runner_ctx=${1:-65536}
    run_serve_params "Qwen36_28B_REAP20_A3B_Q4_K_M" "cuda_flash" "extra_fa_Kq8_Vq8_b4096_ub4096_moe41" ${runner_ctx} "" "all"
}

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
################################################################# GLM_47_Flash_REAP_23B_A3B_Q4_K_M ####################################################################################
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

run_BENCH_GLM_47_Flash_REAP_23B_A3B_Q4_K_M_CTX_cuda_flash_CFG_MOE_on_x_CTX_x()
{
    local runner_ctx=${1:-65536} 
    local model="GLM_47_Flash_REAP_23B_A3B_Q4_K_M"
    local bench="cuda_flash"
    
    local -a extras_list=(
#   "extra_fa_Kq8_Vq8_default_moe49"
#   "extra_fa_Kq8_Vq4_default_moe49"
#   "extra_fa_Kq8_Vq8_b2048_ub256_moe49"  # SKIPPED 
#   "extra_fa_Kq8_Vq4_b2048_ub256_moe49"  # SKIPPED 
#   "extra_fa_Kq8_Vq8_b2048_ub512_moe49"  # SKIPPED like default 
#   "extra_fa_Kq8_Vq4_b2048_ub512_moe49"  # SKIPPED like default
#   "extra_fa_Kq8_Vq8_b2048_ub1024_moe49"
#   "extra_fa_Kq8_Vq4_b2048_ub1024_moe49"
#   "extra_fa_Kq8_Vq8_b2048_ub2048_moe49"
#   "extra_fa_Kq8_Vq4_b2048_ub2048_moe49"
#   "extra_fa_Kq8_Vq8_b8192_ub8192_moe49" # PC freezing sporadicaly (CTX==65536)
#   "extra_fa_Kq8_Vq4_b8192_ub8192_moe49" # PC freezing sporadicaly (CTX==65536)
#   "extra_fa_Kq8_Vq8_b6144_ub6144_moe49" # PC freezing sporadicaly (CTX==65536)
#   "extra_fa_Kq8_Vq4_b6144_ub6144_moe49" # PC freezing sporadicaly (CTX==65536)
#   "extra_fa_Kq8_Vq8_b4096_ub4096_moe49"
#   "extra_fa_Kq8_Vq4_b4096_ub4096_moe49"
#   "extra_fa_Kq8_Vq8_b4096_ub2048_moe49"
#   "extra_fa_Kq8_Vq4_b4096_ub2048_moe49"
#   "extra_fa_Kq8_Vq8_b4096_ub1024_moe49"
#   "extra_fa_Kq8_Vq4_b4096_ub1024_moe49"
#   "extra_fa_Kq8_Vq8_b4096_ub512_moe49"   # SKIPPED 
#   "extra_fa_Kq8_Vq4_b4096_ub512_moe49"   # SKIPPED  
#   "extra_fa_Kq8_Vq8_b4096_ub256_moe49"   # SKIPPED  
#   "extra_fa_Kq8_Vq4_b4096_ub256_moe49"   # SKIPPED 


#     "extra_fa_Kq8_Vq8_default_moe49"
#     "extra_fa_Kq8_Vq4_default_moe49"
#     "extra_fa_Kq8_Vq8_default_moe40"
#     "extra_fa_Kq8_Vq4_default_moe40"
#     "extra_fa_Kq8_Vq8_default_moe28"
#     "extra_fa_Kq8_Vq4_default_moe28"
#     "extra_fa_Kq8_Vq8_default_moe23" # PC freeze (CTX==65536)
#     "extra_fa_Kq8_Vq4_default_moe23" # PC freeze (CTX==65536)
    )
    
    run_bench_params_wrapper_EXTRAS ${model} ${bench} ${runner_ctx} "${extras_list[@]}"
}

run_SERVER_gemma_4_26B_A4b_UD_Q4_K_M_CTX_cuda_flash()
{
    local runner_ctx=${1:-65536}
    run_serve_params "GLM_47_Flash_REAP_23B_A3B_Q4_K_M" "cuda_flash" "extra_fa_Kq8_Vq8_b4096_ub4096_moe49" ${runner_ctx} "" "all"
}

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
################################################################# gemma_4_26B_A4b_UD_Q4_K_M ###########################################################################################
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

run_BENCH_gemma_4_26B_A4b_UD_Q4_K_M_CTX_cuda_flash_CFG_MOE_on_x_CTX_x()
{
    local runner_ctx=${1:-65536} 
    local model="gemma_4_26B_A4b_UD_Q4_K_M"
    local bench="cuda_flash"
    
    ##################################################################
    # NOTES:
    # Batching Works - Generic quantization that benefits from batched computation to amortize overhead
    ##################################################################
    
    local -a extras_list=(
#   "extra_fa_Kq8_Vq8_default_moe31"
#   "extra_fa_Kq8_Vq4_default_moe31"
#   "extra_fa_Kq8_Vq8_b2048_ub256_moe31"  # SKIPPED 
#   "extra_fa_Kq8_Vq4_b2048_ub256_moe31"  # SKIPPED 
#   "extra_fa_Kq8_Vq8_b2048_ub512_moe31"  # SKIPPED like default 
#   "extra_fa_Kq8_Vq4_b2048_ub512_moe31"  # SKIPPED like default
#   "extra_fa_Kq8_Vq8_b2048_ub1024_moe31"
#   "extra_fa_Kq8_Vq4_b2048_ub1024_moe31"
#   "extra_fa_Kq8_Vq8_b2048_ub2048_moe31"
#   "extra_fa_Kq8_Vq4_b2048_ub2048_moe31"
#   "extra_fa_Kq8_Vq8_b8192_ub8192_moe31" # PC freeze TERRIBLE  (CTX==65536)
#   "extra_fa_Kq8_Vq4_b8192_ub8192_moe31" # PC freeze TERRIBLE  (CTX==65536)
#   "extra_fa_Kq8_Vq8_b6144_ub6144_moe31" 
#   "extra_fa_Kq8_Vq4_b6144_ub6144_moe31"
#   "extra_fa_Kq8_Vq8_b4096_ub4096_moe26" # SWEETSPOT 1. step - !!! BAD SURPRISE !!!
#   "extra_fa_Kq8_Vq8_b4096_ub4096_moe31" # SWEETSPOT 1. step
#   "extra_fa_Kq8_Vq4_b4096_ub4096_moe31" # SWEETSPOT 1. step
#   "extra_fa_Kq8_Vq8_b4096_ub2048_moe31"
#   "extra_fa_Kq8_Vq4_b4096_ub2048_moe31"
#   "extra_fa_Kq8_Vq8_b4096_ub1024_moe31"
#   "extra_fa_Kq8_Vq4_b4096_ub1024_moe31"
#   "extra_fa_Kq8_Vq8_b4096_ub512_moe31"   # SKIPPED 
#   "extra_fa_Kq8_Vq4_b4096_ub512_moe31"   # SKIPPED  
#   "extra_fa_Kq8_Vq8_b4096_ub256_moe31"   # SKIPPED  
#   "extra_fa_Kq8_Vq4_b4096_ub256_moe31"   # SKIPPED 
    
#      "extra_fa_Kq8_Vq8_default_moe31"
#      "extra_fa_Kq8_Vq4_default_moe31"
#      "extra_fa_Kq8_Vq8_default_moe26"
#      "extra_fa_Kq8_Vq4_default_moe26"       
#      "extra_fa_Kq8_Vq8_default_moe21"
#      "extra_fa_Kq8_Vq4_default_moe21"
#      "extra_fa_Kq8_Vq8_default_moe18" # SWEETSPOT (CTX==65536)
#      "extra_fa_Kq8_Vq4_default_moe18" #
#      "extra_fa_Kq8_Vq8_default_moe16" # PC freezing sporadicaly (CTX==65536)
#      "extra_fa_Kq8_Vq4_default_moe16" # PC freezing sporadicaly (CTX==65536)       
    )
    
    run_bench_params_wrapper_EXTRAS ${model} ${bench} ${runner_ctx} "${extras_list[@]}"
}

run_INSPECT_gemma_4_26B_A4b_UD_Q4_K_M_CTX_cuda_flash_mmlu_pro()
{
    local runner_ctx=${1:-65536}
    local -a extras_list=(
        "extra_fa_Kq8_Vq8_b4096_ub4096_moe31"
    )
    run_inspect_params_with_server_wrapper_EXTRAS_mmlu_pro "gemma_4_26B_A4b_UD_Q4_K_M" "cuda_flash" ${runner_ctx} "${extras_list[@]}"
}

run_SERVER_gemma_4_26B_A4b_UD_Q4_K_M_CTX_cuda_flash()
{
    local runner_ctx=${1:-65536}
    run_serve_params "gemma_4_26B_A4b_UD_Q4_K_M" "cuda_flash" "extra_fa_Kq8_Vq8_b4096_ub4096_moe31" ${runner_ctx} "" "all"
}

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
################################################################# gemma_4_26B_A4b_QAT_UD_Q4_K_XL ###########################################################################################
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

run_BENCH_gemma_4_26B_A4b_QAT_UD_Q4_K_XL_CTX_cuda_flash_CFG_MOE_on_x_CTX_x()
{
    local runner_ctx=${1:-65536} 
    local model="gemma_4_26B_A4b_QAT_UD_Q4_K_XL"
    local bench="cuda_flash"
    
    ##################################################################
    # NOTES:
    # Batching NOT Helps - Flat or minimal improvement with batching 
    ##################################################################
    
    local -a extras_list=(
#   "extra_fa_Kq8_Vq8_default_moe31"
#   "extra_fa_Kq8_Vq4_default_moe31"
#   "extra_fa_Kq8_Vq8_b2048_ub256_moe31"   # SKIPPED 
#   "extra_fa_Kq8_Vq4_b2048_ub256_moe31"   # SKIPPED 
#   "extra_fa_Kq8_Vq8_b2048_ub512_moe31"   # SKIPPED like default 
#   "extra_fa_Kq8_Vq4_b2048_ub512_moe31"   # SKIPPED like default
#   "extra_fa_Kq8_Vq8_b2048_ub1024_moe31"  # SKIPPED based on gemma_4_26B_A4b_UD_Q4_K_M 
#   "extra_fa_Kq8_Vq4_b2048_ub1024_moe31"  # SKIPPED based on gemma_4_26B_A4b_UD_Q4_K_M 
#   "extra_fa_Kq8_Vq8_b2048_ub2048_moe26"  # SWEETSPOT 1. finding
#   "extra_fa_Kq8_Vq8_b2048_ub2048_moe21"  # SWEETSPOT 1. finding
#   "extra_fa_Kq8_Vq8_b2048_ub2048_moe18"  # SWEETSPOT 1. finding
#   "extra_fa_Kq8_Vq8_b2048_ub2048_moe16"  # SWEETSPOT 1. finding FOUND (SD off)
#   "extra_fa_Kq8_Vq8_b2048_ub2048_moe31"  # SWEETSPOT 1. finding
#   "extra_fa_Kq8_Vq4_b2048_ub2048_moe31"  # SWEETSPOT 1. step
#   "extra_fa_Kq8_Vq8_b8192_ub8192_moe31"  # SKIPPED based on gemma_4_26B_A4b_UD_Q4_K_M PC - freeze TERRIBLE  (CTX==65536)
#   "extra_fa_Kq8_Vq4_b8192_ub8192_moe31"  # SKIPPED based on gemma_4_26B_A4b_UD_Q4_K_M PC - freeze TERRIBLE  (CTX==65536)
#   "extra_fa_Kq8_Vq8_b6144_ub6144_moe31" 
#   "extra_fa_Kq8_Vq4_b6144_ub6144_moe31"
#   "extra_fa_Kq8_Vq8_b4096_ub4096_moe31" 
#   "extra_fa_Kq8_Vq4_b4096_ub4096_moe31" 
#   "extra_fa_Kq8_Vq8_b4096_ub2048_moe31"
#   "extra_fa_Kq8_Vq4_b4096_ub2048_moe31"
#   "extra_fa_Kq8_Vq8_b4096_ub1024_moe31"  # SKIPPED based on gemma_4_26B_A4b_UD_Q4_K_M
#   "extra_fa_Kq8_Vq4_b4096_ub1024_moe31"  # SKIPPED based on gemma_4_26B_A4b_UD_Q4_K_M
#   "extra_fa_Kq8_Vq8_b4096_ub512_moe31"   # SKIPPED 
#   "extra_fa_Kq8_Vq4_b4096_ub512_moe31"   # SKIPPED  
#   "extra_fa_Kq8_Vq8_b4096_ub256_moe31"   # SKIPPED  
#   "extra_fa_Kq8_Vq4_b4096_ub256_moe31"   # SKIPPED 
        
    )
    
    run_bench_params_wrapper_EXTRAS ${model} ${bench} ${runner_ctx} "${extras_list[@]}"
}

run_BENCHY_WITH_SERVER_gemma_4_26B_A4b_QAT_UD_Q4_K_XL_CTX_cuda_flash_CFG_MOE_on_x_CTX_x()
{
    local runner_ctx=${1:-65536} 
    local model="gemma_4_26B_A4b_QAT_UD_Q4_K_XL"
    local bench="cuda_flash"
    
    local -a extras_list=(
#   "extra_fa_Kq8_Vq8_b2048_ub2048_moe21_sd10mtp" # SWEETSPOT 1. finding
#   "extra_fa_Kq8_Vq8_b2048_ub2048_moe21_sd6mtp"  # SWEETSPOT 1. finding
#   "extra_fa_Kq8_Vq8_b2048_ub2048_moe21_sd5mtp"  # SWEETSPOT 1. finding
#   "extra_fa_Kq8_Vq8_b2048_ub2048_moe21_sd4mtp"  # SWEETSPOT 1. finding FOUND (SD on)
#   "extra_fa_Kq8_Vq8_b2048_ub2048_moe21_sd3mtp"  # SWEETSPOT 1. finding
#   "extra_fa_Kq8_Vq8_b2048_ub2048_moe21"         # SWEETSPOT 1. finding
#   "extra_fa_Kq8_Vq8_b2048_ub2048_moe18_sd4mtp"  # SWEETSPOT 1. finding
#   "extra_fa_Kq8_Vq8_b2048_ub2048_moe18_sd3mtp"  # SWEETSPOT 1. finding
#   "extra_fa_Kq8_Vq8_b2048_ub2048_moe18"         # SWEETSPOT 1. finding
    )
    
    run_benchy_params_with_server_wrapper_EXTRAS ${model} ${bench} ${runner_ctx} "${extras_list[@]}"
}

run_SERVER_gemma_4_26B_A4b_QAT_UD_Q4_K_XL_CTX_cuda_flash()
{
    local runner_ctx=${1:-65536}
    local SD_ENABLED=${2:-0}
    
    local EXTRA="extra_fa_Kq8_Vq8_b2048_ub2048_moe16" 
    if [ $SD_ENABLED -eq 1 ]; then
        EXTRA="extra_fa_Kq8_Vq8_b2048_ub2048_moe21_sd4mtp"
    fi
        
    run_serve_params "gemma_4_26B_A4b_QAT_UD_Q4_K_XL" "cuda_flash" "$EXTRA" ${runner_ctx} "" "all"
}
