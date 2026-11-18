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

Configuration: `config/config.toml` if present, else `config/config.example.toml`
(override with `--config` or `MSE_LLM_CONFIG`).
