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

Configuration: `config/config.toml` if present, else `config/config.example.toml`
(override with `--config` or `MSE_LLM_CONFIG`).
