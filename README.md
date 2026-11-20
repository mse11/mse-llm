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
snapshot; `serve` / `presets` run since M5, bench/benchy/sweep since M7, eval in M12):

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

Serving (M5). `serve` runs the fit check first, warns about `NO-FIT` / `TIGHT` / `UNSURE` with what
llama.cpp's `--fit` would change, and asks before starting (`--yes` skips the question). Ctrl+C stops the
server and every child process; the port is checked free afterwards. Logs: `results\servers\`.

```
uv run mse-llm serve --list                                              # what can be served (also: bare `serve`)
uv run mse-llm serve Qwen36_28B_REAP20_A3B_Q4_K_M                        # run_SERVER_* (default_serve, --fit on)
uv run mse-llm serve omnicoder_9b --output all                           # all | filter (default) | none
uv run mse-llm serve --preset gemma-4-26B-A4B-it-qat-UD-Q4_K_XL          # one INI section, outside the router
uv run mse-llm serve Qwen36_35B_A3B_UD_Q4_K_M_MTP -v fa_Kq8_Vq8_default_moe10 --let-fit   # let --fit place layers

uv run mse-llm presets validate                 # every key vs llama-server --help, model files, [*]
uv run mse-llm presets fix --write              # rewrite only the broken lines (keeps FILE.bak + comments)
uv run mse-llm presets list | dump [NAME]       # sections, variant ≈, the equivalent single-model command
uv run mse-llm presets generate --out config/presets.ini   # INI from [models.*] + default_serve
uv run mse-llm presets serve                    # run_PRESETS_serve, foreground (Ctrl+C stops all models)
uv run mse-llm presets serve --background --load Tesslate_OmniCoder-9B-Q4_K_M
uv run mse-llm presets status | load NAME | unload NAME | watch | stop
```

Monitoring + freeze prevention (M6). `serve` and `presets serve` sample VRAM, shared GPU memory, RAM, commit
and temperature every 0.5 s and stop the server on a GPU memory spill (dedicated VRAM full and the server's
shared memory still growing, or more shared memory than the host buffers `fit` expects) or RAM / commit > 95 %
(exit code 2). `-ncmoe` experts in pinned host memory show up as shared GPU memory but are not a spill.

```
uv run mse-llm monitor --duration 30                          # sample the system
uv run mse-llm monitor --guard --expect-host 15807 -- llama-bench.exe -m X.gguf ...   # bash EXEC_cmd_with_gpu_and_ram_monitoring
uv run mse-llm serve omnicoder_9b --no-guard | --no-monitor | --mem-cap 40000
uv run mse-llm --no-notify serve ...                          # no voice message at the end
```

Benchmarks, sweeps, results (M7). Every job runs the fit check first (a variant that doesn't fit is recorded
as `skipped` and not started; `--force` starts it anyway) and runs under the memory guard, which for benchmarks
also stops a job that hangs (GPU idle, no disk reads, VRAM unchanged for `[monitor].stall_stop_s`, 90 s). Each
job writes one JSON to `results/runs/YYYY-MM/` and a line to `results/index.jsonl`. Logs, benchy's own JSON
and the monitor CSV go to `results/raw/<run_id>/`. `results/` is not stored in git for now (`.gitignore`).

```
uv run mse-llm bench omnicoder_9b -v "fa_Kq8_Vq{8,4}_default" --ctx 8192 --ctx 32768   # run_bench_params*, _wrapper_PROMPTS
uv run mse-llm benchy omnicoder_9b --ctx 65536 --pp 512 --pp 8192 --depth 0 --depth 4096   # background server + llama-benchy
uv run mse-llm sweep list                       # one sweep per bash run_BENCH_* / run_BENCHY_* / run_INSPECT_* function
uv run mse-llm sweep show Qwen36_35B_A3B_UD_Q4_K_M_MTP_bench   # entries, status, bash notes, last result per variant
uv run mse-llm sweep run Qwen36_35B_A3B_UD_Q4_K_M_MTP_bench    # the entries with status on (= not commented out in bash)
uv run mse-llm sweep run Qwen36_28B_REAP20_A3B_Q4_K_M_bench --include-off --match "*b4096_ub4096*" --resume
uv run mse-llm sweep run NAME --only-failed     # re-run what failed / was aborted on this build
uv run mse-llm results list [--model M] [--sweep S]
uv run mse-llm results show RUN_ID              # tests, fit, monitor peaks, the command to reproduce it
```

Reports (M8). The bash 5-axis chart (duration, PP, TG, peak VRAM, shared memory per configuration), tables and
the AI analyst prompt, from the stored runs or from an old bash log. Output: `results/reports/<name>/`.

```
uv run mse-llm report chart --sweep Qwen36_35B_A3B_UD_Q4_K_M_MTP_bench   # bash run_dump_params_wrapper_PLOTER
uv run mse-llm report chart --legacy-log old_log.txt                       # old .txt log or _RESULTS/*.html page
uv run mse-llm report table --model gemma --sort tg [--format csv|md] [--tests]
uv run mse-llm report prompt --sweep NAME --chart -o prompt.md            # bash run_dump_params_wrapper_AI_PROMPT
uv run mse-llm results compare RUN_A RUN_B
uv run mse-llm results export --csv runs.csv [--tests]
```

One bar per configuration: the newest run of the same tool / model / variant / ctx wins (`--all-runs` for all).

Predict and search (M9, §4.6, §9 item 6). `predict` measures an `-ncmoe × -ub` grid with llama-fit-params
(endpoints measured, the rest by the "one `-ncmoe` step = one expert layer" rule; `--exact` measures every
point) and estimates TG / PP with a roofline from the GGUF (dense bytes, expert bytes, `n_used / n_expert`,
attention layers) and this PC's bandwidths (`hw` spec values). The bandwidths are calibrated on the stored ok
runs of this PC, globally and per model (a model's own runs teach it, for example, how fast the CPU runs its
experts); speculative-decoding runs are not used. `sweep search` uses the prediction to pick the order of the
stages, starts on the safe side (all experts on the CPU, smallest `-ub`), never starts a config predicted
no-fit or tight, stops a stage at the first spill / failed / skipped run, and takes the winner one step back
when the best run is right next to such a failed run (not next to an edge that is only predicted).
Earlier ok runs with the same settings and build are reused. The search is stored in
`results/search/<model>/<build>__<hw>.json`.

```
uv run mse-llm predict gemma_4_26B_A4b_QAT_UD_Q4_K_XL [--ctx 65536] [--ncmoe 30:10:5] [--ub 512,2048] [--json]
uv run mse-llm results calibrate                 # fitted speeds; measured vs predicted per run, leave-one-out
uv run mse-llm sweep search gemma_4_26B_A4b_QAT_UD_Q4_K_XL --dry-run   # the stages with predicted results
uv run mse-llm sweep search gemma_4_26B_A4b_QAT_UD_Q4_K_XL [--goal tg|pp|both] [--axes ub,ncmoe] [--apply]
```

`--apply` writes the winner as `[models.*].default_serve` into the config file (`presets generate` uses it).

Quality and speculative decoding (M10, §4.5). `quality kld` measures how much a K/V-cache choice changes the
model's token probabilities: llama-perplexity against an f16-KV reference of the same GGUF (mean / 99 % KLD,
"same top p", PPL ratio), next to a noise floor (the reference again with another placement, i.e. GPU vs CPU math).
Verdict: within 2x the floor = no measurable loss, else negligible (< 0.001) / small (< 0.01) / noticeable /
large. The reference is ~2 GiB per chunk at ctx 8192 (15 GiB for 8 chunks of a 248k-vocab model); its size is
shown and confirmed before it is created, under `results/raw/kld-ref/`. When f16 KV doesn't fit: more experts on
the CPU, then `-nkvo`, then a q8_0 reference (labelled "vs q8 ref"). Corpus: `code` (2 MB of llama.cpp sources,
default), `wikitext2` (download once), or a file. Quality runs are not stopped by the power-plan check (the
numbers don't depend on it).

```
uv run mse-llm quality kld Qwen36_35B_A3B_UD_Q4_K_M_MTP -v "fa_Kq8_Vq{8,4}_default_moe34" [--ctx 8192] [--chunks 8]
uv run mse-llm quality kld-ref MODEL [-v PLACEMENT] [--dry-run]   # only the reference (+ floor)
uv run mse-llm quality floor MODEL | list [MODEL] | clean MODEL|--all | corpus [wikitext2 --download]
uv run mse-llm sweep run NAME --with-kld        # one KLD run per distinct K/V pair, before the speed runs
```

Every server a benchmark starts gets `--metrics` (the router too); benchy runs of `sd*` variants store
`spec`: drafted / accepted tokens, acceptance rate, mean accepted length and the acceptance per draft position
(the `/metrics` delta over the run; the slot log line as fallback), plus the sampling (temperature lowers
acceptance). `--prompts code` (or `prompt_sets = ["prose", "code"]` in a sweep) runs benchy on the code corpus,
served on a local URL. Draft-side tokens: `sdmin<N>`, `dK<t>` / `dV<t>` (draft KV), `dmoe<N>` / `dcmoe`.

```
uv run mse-llm benchy Qwen36_35B_A3B_UD_Q4_K_M_MTP -v "fa_Kq8_Vq8_default_moe34_sd{2,3,4}mtp" --prompts prose --prompts code
uv run mse-llm report spec --model Qwen36_35B_A3B_UD_Q4_K_M_MTP --chart   # vs the no-spec run; slower = flagged
```

Sweep entries keep the bash comments as data: `status = "off"` (commented out: `--include-off`),
`status = "skip"` ("PC freeze": only with `--force`), `note` and `tag` (sweetspot, sporadic-freeze, ...).
`bench`, `benchy`, `sweep run` and `sweep search` stop before the first job on battery or under a quiet /
power-saving plan (e.g. Legion Quiet Mode: TG of one config varied 11-32 t/s); `--allow-low-power` or
`MSE_LLM_ALLOW_LOW_POWER=1` runs anyway. `--dry-run` never stops. `--resume` skips variants that already have an `ok` result for the same build. Exit code 1 if a job failed,
2 if the guard stopped one.

`--fit` only adjusts what you did not set: with `-ngl all` (always passed by `serve`) or `-ncmoe`, llama.cpp
aborts the fit and starts with exactly the requested settings, so a config that doesn't fit is not shrunk.
The fit check says so and shows what `--fit` would pick with both left unset (`serve --let-fit`).

Configuration: `config/config.toml` if present, else `config/config.example.toml`
(override with `--config` or `MSE_LLM_CONFIG`).
