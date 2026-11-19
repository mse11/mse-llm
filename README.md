# mse-llm

Python CLI (uv + click) that replaces `llama_cpp_benchmark.sh`: build llama.cpp, detect the PC,
benchmark, tune and serve local models. See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) and
[docs/PARITY.md](docs/PARITY.md).

```
uv sync
uv run mse-llm doctor          # environment, hardware and driver checks
uv run mse-llm doctor flags    # diff llama.cpp --help against stored snapshots, audit the bash script
uv run mse-llm hw              # PC capability detection
uv run mse-llm hw identity     # hard/soft hardware identity + hash
uv run mse-llm hw processes    # other engines / GPU processes that would skew a bench
uv run mse-llm help server     # llama-server --help of the configured engine
```

Build llama.cpp (clones llama.cpp + vcpkg into the git-ignored `builds/` on first run):

```
uv run mse-llm build list                    # targets from config/builds + what is built
uv run mse-llm build plan                    # every step `build run` would do; nothing runs
uv run mse-llm build run                     # shows the plan, asks, then builds (target picked from this PC)
uv run mse-llm build run rtx4080-laptop --without zen4-avx512x   # comparison build
uv run mse-llm build run --from configure    # resume after a failed step
uv run mse-llm build show                    # every CMake flag: value, where it comes from, what it means
uv run mse-llm build info                    # build-manifest.json: commit, flags, toolchain, hashes
uv run mse-llm build versions [--remote]     # buildable releases / b<N> tags
uv run mse-llm build explain GGML_AVX512_VNNI
uv run mse-llm build clean --logs|--builds|--all [--dry-run]   # shows sizes, asks, then deletes
```

Binaries land in `builds\llama.cpp\build\bin\Release`, the folder the bash script's
`MSE_LLAMA_BIN_DIR` uses. Full logs: `builds\logs\`.

Variants and commands (M3: `--dry-run` shows the exact argv, checked against the llama.cpp help
snapshot; running them arrives in M5/M7/M12):

```
uv run mse-llm models list
uv run mse-llm variants decode extra_fa_Kq8_Vq4_b4096_ub4096_moe41_sd3cache   # fields + flags
uv run mse-llm variants expand "fa_Kq8_Vq8_b2048_ub2048_moe21_sd{3..6}mtp"
uv run mse-llm variants encode -- -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 --n-cpu-moe 16
uv run mse-llm serve gemma_4_26B_A4b_QAT_UD_Q4_K_XL --spec --dry-run      # run_SERVER_* (SD_ENABLED=1)
uv run mse-llm serve --all --dry-run                                       # run_PRESETS_dump
uv run mse-llm bench omnicoder_9b -v "extra_fa_{Vq8,Kq8_Vq4,Kq8_Vq8,Kq4_Vq8}_default" --dry-run   # run_BENCH_*
uv run mse-llm benchy Qwen36_28B_REAP20_A3B_Q4_K_M -v fa_Kq8_Vq8_b4096_ub4096_moe41 --dry-run     # run_BENCHY_*
uv run mse-llm eval gemma_4_26B_A4b_UD_Q4_K_M mmlu_pro --dry-run           # run_INSPECT_*
uv run mse-llm presets serve --dry-run                                     # run_PRESETS_serve
```

`-v` takes brace patterns; `--from FILE|-` reads a pasted bash `extras_list` (comments OK).

Memory fit before running (M4, §4.6: one `llama-fit-params --fit-print` call per variant, ~1 s, no model load):

```
uv run mse-llm fit Qwen36_28B_REAP20_A3B_Q4_K_M --ctx 65536 --from - --suggest   # paste a bash extras_list
uv run mse-llm fit gemma_4_26B_A4b_QAT_UD_Q4_K_XL                                # its default_serve(_spec)
uv run mse-llm fit Qwen36_35B_A3B_UD_Q4_K_M -v "fa_Kq8_Vq8_default_moe{34,30,28}" --ctx 65536 --ctx 131072
uv run mse-llm fit d:\models\X.gguf -v fa_Kq8_Vq8_b2048_ub2048_moe20 --free 11000 --mmproj d:\models\mmproj.gguf
```

Verdicts: `fits` (every part measured), `fits~` (drafter KV/compute or mmproj estimated), `TIGHT`
(under `--margin`, default 768 MiB), `NO-FIT`, `UNSURE` (a part NOT COUNTED, e.g. an `sd*mtp` variant with no
drafter; `--strict` exits 1). `--suggest` gives the lowest `-ncmoe` that fits, checked with extra
llama-fit-params calls, and the largest `-ub`.

Configuration: `config/config.toml` if present, else `config/config.example.toml`
(override with `--config` or `MSE_LLM_CONFIG`).
