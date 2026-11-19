# Parity checklist: `llama_cpp_benchmark.sh` → `mse-llm`

Every row must be ✓ (or "dropped + reason", approved by you) before the bash script and
`llama_cpp_presets.ini` are deleted (M13, [IMPLEMENTATION_PLAN.md §10.1](../IMPLEMENTATION_PLAN.md)).
Updated as milestones land.

Legend: ✓ done and checked · ◐ partly done · ☐ not started · ✗ dropped (reason given)

| # | Script feature (bash) | Replacement | Check | Status |
|---|---|---|---|---|
| F1 | `run_HELP_server/bench__*` | `mse-llm help server\|bench` (+ `benchy`, `eval`), `doctor flags` | output identical to `--help` | ✓ M2: real snapshots `a49db739f` + `ffc87c3a7` stored; `doctor flags` diff + audit checked against them |
| F2 | `MODELS`, `DRAFTER_MODELS`, `MODELS_BENCH_EXTRA`, `MODELS_SERVE_EXTRA` | `[models.*]` in config | every model + extra present | ✓ M3: all 7 models under their bash keys (drafter, serve_extra, ctx, default_serve/_spec); the golden test compares paths + extras |
| F3 | `EXTRAS` matrix + `EXTRAS_get_flags_normalized/_replaced` | `variants.py` codec | golden test: every generated key decodes to the same flags | ✓ M3: all 1249 `EXTRAS` keys + placeholder replacement for every model (`tests/test_cmd_build.py`); turbo keys raise "dropped" |
| F4 | `run_bench_params*`, `_wrapper_EXTRAS`, `_wrapper_PROMPTS` | `mse-llm bench` | same argv in `--dry-run` (§0.1 bugs fixed) | ✓ M3: `bench --dry-run` argv = bash (differences below); `-v` list / repeated `--ctx` = the two wrappers. M7: runs under the monitor + guard, fit-checked first, `-o jsonl` parsed, one RunResult per run in `results/` |
| F5 | `run_benchy_params_with_server*` | `mse-llm benchy` | same argv, `--pp` list passed once | ✓ M3: `benchy --dry-run` server + client argv = bash, lists after one flag; M5: `ServerHandle` for its server. M7: runs (server, then llama-benchy with `--format json --save-result`), parsed into the results store |
| F6 | `run_serve_params_on_background`, `server_wait_for_ready`, `server_stop_windows` | `ServerHandle` | 50× start/stop, no orphans | ✓ M5: `ServerHandle` (port check, spawn in its own process group + kill-on-close Job Object, `/health` poll with fail-fast, graceful CTRL_BREAK, tree kill, port verified free). 50× with the fake server (2 children each) and 50× with the real router + 10× real model load/chat, no `llama-server` left |
| F7 | `run_serve_params` + `_FILTER`, output modes | `mse-llm serve --output` | same filtered lines | ✓ M5: `serve` runs (foreground, Ctrl+C stops the tree), `--output filter` = the `run_serve_params_FILTER` regexes; + fit pre-check |
| F8 | `run_presets_serve`, `run_PRESETS_serve`, `run_PRESETS_dump`, `run_presets_dump` | `mse-llm presets …` | existing INI serves all 5 models | ✓ M5: `presets serve` (foreground / `--background`, `--load`), `validate`, `fix`, `list`, `dump`, `generate`, `status`, `load`/`unload`, `watch`, `stop`; all 5 models listed, two models switched through one endpoint; the generated INI validates and maps back to every `default_serve` |
| F9 | `EXEC_cmd_with_gpu_and_ram_monitoring` | `monitor/` | peaks within tolerance | ✓ M6: run side by side with the real bash function on one Qwen3.6-28B load: shared 16063 vs 16064 MiB, temp 53 vs 53 °C, dedicated 4728 vs 4442 MiB (bash samples every ~2 s); RAM differs because of a bash bug (below). Live memory guard added |
| F10 | `run_dump_params_wrapper_PLOTER` | `mse-llm report chart` (+ `--legacy-log`) | old logs give the same chart | ✓ M8: the bash plotter ported line for line (`report/plot.py`, `parsers/legacy_log.py`); on the same old log the PNG is pixel-identical to the one the real bash function draws (0 of 8.5 M pixels differ; opt-in test `MSE_LLM_BASH_PARITY=1`). Reads the old `.txt` logs and the `_RESULTS/*.html` pages they were pasted into; new results chart through the same code |
| F11 | `run_dump_params_wrapper_AI_PROMPT/_COMMON/_EXTRAS/_PROMPTS` | `mse-llm report prompt` | same sections + data | ✓ M8: Jinja2 template with the bash sections (role, instruction, model / benchmark / context, chart specs, TEST ID list, output requirements); the data is embedded as CSV (per run + per test) instead of "[PASTE RAW RESULTS BELOW THIS LINE]" |
| F12 | `run_inspect_params*` | `mse-llm eval` | real eval runs | ◐ M3: `eval MODEL TASK --dry-run` = the bash builder (URL from the engine); running in M12 |
| F13 | `EXEC_user_notification_WIN_T2S` | `notify.py` | voice message at end of run | ✓ M6: SAPI through COM (no powershell); `serve` / `presets serve` / `monitor -- CMD` speak at the end; `--no-notify`, `mse-llm notify TEXT` |
| F14 | `EXEC_cmd_with_output_mode` (all/filter/none) | `--output` | — | ✓ M5: `--output all\|filter\|none`, `[defaults].output`, `[defaults].output_filter` (default: the bash list) |
| F15 | `MSE_*` knobs | config fields / `MSE_LLM_*` env / `--dry-run` | each knob mapped or dropped | ◐ M0: see the knob table below |
| — | `run_BENCH_*`, `run_BENCHY_WITH_SERVER_*`, `run_SERVER_*`, `run_INSPECT_*` group functions (+ commented entries, "SWEETSPOT", "PC freeze" notes) | `[sweeps.*]` + `annotations` | one entry per function, notes kept | ◐ M7: all 9 `run_BENCH_*` / `run_BENCHY_WITH_SERVER_*` / `run_INSPECT_*` functions are `[sweeps.*]` (`bash = "…"` names the function), all 161 `extras_list` entries in order, comments as `note`, commented-out = `status = "off"`, "PC freeze" = `status = "skip"`, tags sweetspot / freeze / sporadic-freeze / skipped; `sweep run` runs bench/benchy sweeps (the inspect one in M12). `run_SERVER_*`: `default_serve(_spec)` since M3 |
| — | Model notes in comments (HF URLs, sizes, capabilities) | `[models.*].url`, `notes` | nothing lost | ✓ M3: URLs, sizes, capabilities and the per-model bench notes (SWEETSPOT / PC freeze) in `config.example.toml`; per-entry annotations go to `[sweeps.*]` in M7 |
| — | turbo2/3/4 cache variants (`*_Kt*`, `*_Vt*`) | — | — | ✗ dropped: not supported by upstream llama.cpp (approved 2026-11-18) |

## F15: `MSE_*` knobs

| Bash knob | mse-llm | Status |
|---|---|---|
| `MSE_THREADS` | `[defaults].threads`, `MSE_LLM_THREADS` | ✓ M0 |
| `MSE_PARALLEL` | `[defaults].parallel`, `MSE_LLM_PARALLEL` | ✓ M0 |
| `MSE_BENCH_TG` | `[defaults].bench_tg`, `MSE_LLM_BENCH_TG` | ✓ M0 |
| `MSE_BENCHY_PP` / `_PP_SYNCH` / `_DEPTH` / `_RUNS` | `[defaults.benchy]`, `MSE_LLM_BENCHY_*` | ✓ M0 |
| `MSE_SERVER_READY_TIMEOUT` | `[defaults].server_ready_timeout` | ✓ M0 |
| `MSE_MONITOR_ENABLED` | `[defaults].monitor`, `--no-monitor` | ✓ M0, M6 |
| `MSE_LLAMA_BIN_DIR` | `[engines.*].bin_dir`, from the build profile | ✓ M0; M2: `builds\llama.cpp\build\bin\Release` built by `mse-llm build run`, same folder in the script |
| `MSE_PS_DEBUG` | — | ✗ dropped (approved 2026-11-19): it printed `ps aux` around the bash PID juggling; `ServerHandle` has no shell processes, and `StopReport` lists every PID of the tree |
| `MSE_SERVER_PRESETS_DUMP` | `--dry-run`, `serve --all --dry-run`, `presets dump` (the INI) | ✓ M3, M5 |
| `MSE_PRESETS_SERVE_FILE` | `[presets].file`, `presets serve --file` | ✓ M3 (argv), M5 (serving) |
| `MSE_TEST_INSPECT_PATH` | `eval` config | ☐ M12 |

## M3: argv differences (bash → mse-llm)

`tests/test_cmd_build.py` compares every command against `tests/fixtures/bash_golden/golden.jsonl`,
dumped from the real script by `tests/fixtures/bash_golden/dump_golden.sh` (nothing is run).
These are the only differences. Everything else is identical, in the same order:

| Where | bash | mse-llm | Why |
|---|---|---|---|
| all | `-fa 1` | `-fa on` | §0.2: `-fa` takes on/off/auto |
| llama-server | `-ngl 999` | `-ngl all` (`[defaults].ngl`) | §0.2; llama-bench keeps `999` (it takes numbers only) |
| llama-server | (none) | `--fit on` for `serve`; `--fit off` (`[defaults].fit`) for the benchy/eval server; before `--host` | §0.3 / §11 Q3 |
| llama-bench | (none) | `-r 1 -o jsonl` at the end | bash used the implicit `-r 5`; JSONL output for the results store (M7) |
| llama-bench | `--spec-* --model-draft MODEL` passed for `sd*` | error before running: "llama-bench can't run sd …; use benchy" | §0.1: llama-bench has no `--spec-*` flags |
| sd*mtp without a drafter | literal `--model-draft MODEL` | flag left out + warning | bash bug (placeholder never replaced) |
| benchy | `--pp a --pp b` | `--pp a b` | §0.1: nargs='+', bash kept only the last value. Same argv for one value |
| benchy | (none) | warning when pp + tg + depth > ctx | PP sync makes `--pp` = ctx, so the longest request is ctx + 128 tokens |
| turbo2/3/4 | `-ctk/-ctv turbo*` | `DroppedVariantError` | dropped (approved 2026-11-18) |
| inspect | port hard-coded 8033 | URL from the engine | §0.1 |
| first `run_SERVER_gemma_4_26B_A4b_UD_Q4_K_M_*` | served GLM, shadowed | `serve GLM_47_Flash_REAP_23B_A3B_Q4_K_M` (its `default_serve`) | §0.1: names come from config now |
| `${BENCH_extra}` typo | user bench extra never appended | `[models.*].bench_extra` appended | §0.1 |

Every built llama-server / llama-bench argv is also checked against the engine's help snapshot
(the one for the commit in its `build-manifest.json`): unknown flags, missing values and enum values
(`-ctk`, `-fa`, `-lm`, `--spec-type`, ...) fail the command before anything starts.

## Milestone log

* **M0 + M1 (2026-11-18):** project skeleton, config models, `doctor` (incl. the Sysmem Fallback
  check), `doctor flags` (help snapshots, diff, script/INI audit), `help`, `hw` / `hw identity` /
  `hw processes`, `MSE_LLM_TEST_HARDWARE` / `--hw-fixture`. Baseline bash script + INI were
  already committed in the initial commit.
* **M2 (2026-11-18/19):** `mse-llm build list / plan / run / clone / show / info / versions / explain / clean`.
  First build from the repo-local clone: llama.cpp `ffc87c3a7` (master) into `builds\llama.cpp\build`,
  the folder the bash script uses, so **the bash script works again**. Also built: the comparison build
  `--without zen4-avx512x` (engine `cuda_flash-without-zen4-avx512x`) and the reference `a49db739f`
  (`--ref a49db739f --keep-previous`, engine `cuda_flash-a49db739f`). All three verified: CMakeCache flags,
  `/arch:AVX512`, VNNI/VBMI defines (present or absent as requested), sm_89, FA all quants, CUDA0 = RTX 4080 Laptop.
  * **Flag diff `a49db739f` → `ffc87c3a7`: no changes** in `llama-server` or `llama-bench` (42 commits apart).
    The audit reproduces the §0 findings in the script: `-fa 1` (deprecated) and the dropped `turbo2/3/4`
    cache types (15 findings). The fixed `llama_cpp_presets.ini` has no findings.
  * **`GGML_AVX512_BF16` removed from `zen4-avx512x`:** MSVC can't compile it
    (`sgemm.cpp(380): error C2440: cannot convert from '__m512' to '__m512bh'`). The variant is now VNNI + VBMI.
  * Under MSVC, llama.cpp enables VBMI with `__AVX512VBMI__` only (no `GGML_AVX512_VBMI` define); the
    verify step accepts either macro.
  * Build times on this laptop: vcpkg curl + openssl once ~5.5 min, configure ~45 s, full compile ~8-9.5 min.
  * Not yet: `presets validate` (M5); the INI is checked by the `doctor flags` audit for now. Done in M5.
* **M3 (2026-11-19):** `variants.py` (codec with the legacy `extra_` prefix; new tokens `cmoe mc cs lm* nohost fitt
  Kf8 sd<N><type>`; turbo → "dropped" error; brace patterns; pasted bash arrays; matrix + `exclude`), command builders
  in `runners/` (llama-server, llama-bench, llama-benchy, inspect, router), flag validation against the help snapshot
  (`flagcheck.py`), CLI `variants decode|encode|expand`, `models list|show`, `bench|benchy|serve|eval|presets serve
  --dry-run`. All 7 bash models ported to `config.example.toml`. Golden tests: the 1249 `EXTRAS` keys, placeholder
  replacement for every model, every `run_SERVER_*`, the active and some commented `run_BENCH_*` / `run_BENCHY_*`
  entries, `run_INSPECT_*`, `run_PRESETS_serve`, `run_HELP_*`. The help parser now also reads `--spec-type` (value list
  as metavar), `-lm` / `--reasoning-format` (bullet lists) and the `-fitt` / `-ts` list metavars; the `doctor flags`
  audit is unchanged (15 findings).
* **M4 (2026-11-19):** `mse-llm fit GGUF_OR_MODEL` (§4.6), new (no bash equivalent). One
  `llama-fit-params --fit off --fit-print on` call per variant × `--ctx` gives the per-device model / KV /
  compute split (`fit.py`); a small GGUF header reader (`gguf.py`, no new dependency) gives expert-layer sizes,
  vocab and the drafter / mmproj sizes. Parts with a source each: main = measured, n-gram spec = none, drafter
  weights = measured (GGUF tensor bytes), drafter KV + compute = estimated, mmproj = estimated (weights; image
  compute buffer not included), missing drafter = NOT COUNTED → `UNSURE`. Free VRAM from NVML (`--free auto`)
  or `--free MiB`; `--kv`, `--draft`, `--mmproj`, `--margin`, `--strict`, `--json`, `--include-commented`.
  Checked: the pasted Qwen28-REAP list gives the plan's table to the MiB (GPU, split, host, left at 11500 MiB
  free); `sdNmtp` with no drafter → `UNSURE`; `--suggest` proposes the lower `-ncmoe` (28 for the sweet spot).
  Tests replay recorded llama-fit-params output (`tests/fixtures/fit/record_fit.py`).
  Differences from the plan:
  * The drafter's weights come from its GGUF tensor sizes, not a second llama-fit-params call: llama-fit-params
    fails on the gemma drafter alone ("Gemma4Assistant requires ctx_other"). The gemma4-assistant drafter shares
    the main model's KV (`shared_kv_layers` = all 4 layers), so only its compute buffer is estimated.
  * `--suggest` checks its `-ncmoe` answer with up to 6 more llama-fit-params calls (~1 s each): expert layers
    differ in size, so the linear step is off by a layer near the margin (Qwen28: 381 vs 372 MiB per step).
  * The drafter comes from `--draft` or `[models.*].draft`; variant names have no `md` token, and the presets
    INI `model-draft` key is read when presets run (M5).
  * Calibrating the estimated parts from server logs needs real runs: M5 (ServerHandle) / M7 (results store).
* **M5 (2026-11-19):** `proc.py` (async spawn, log tee + output filter, process-tree stop, port checks),
  `health.py` (`/health` poll, router API, SSE), `ServerHandle` in `runners/base.py`, `serve.py` (fit pre-check,
  `--fit` preview, startup-log check, router loads, `results/servers/` state), `presets.py` (INI parse / validate /
  fix / dump / generate), commands in `cli_serve.py`. New dependency: `httpx`.
  * **`--fit on` can't shrink what `serve` passes.** llama.cpp's `--fit` adjusts only arguments the user did not
    set; with `-ngl all` it aborts ("n_gpu_layers already set by user"), with `-ncmoe` too ("tensor_buft_overrides
    already set by user"), and the server starts with exactly the requested settings. So a too-large config is
    not shrunk, contrary to the plan's assumption (§0.3). The fit check says so and shows what `--fit` would
    pick with both left unset (e.g. Qwen3.6-35B `moe10`: `-ngl all → 42`, experts of layers 15–40 on the CPU);
    `serve --let-fit` runs exactly that. The plain `-ncmoe` answer from `fit --suggest` is shown too.
  * **Graceful stop on Windows** is CTRL_BREAK to the server's own process group. llama-server only handles
    Ctrl+C itself, so CTRL_BREAK ends it through the default handler (no cleanup of its own, but it exits at
    once: 0.7 s per stop, nothing needed a kill). Ctrl+C can't be sent to another process group on Windows.
  * **Attached servers sit in a kill-on-close Job Object**, so if mse-llm itself is killed, Windows ends the
    server and every router child too. `--background` routers leave that job (they must outlive mse-llm),
    use a hidden console that their children share, and are stopped by PID from `results/servers/router-<port>.json`
    (`presets stop`), with the process create time checked so a reused PID is never killed.
    A killed router with the 26B gemma loaded (`-lm none`, ~20 GB in RAM) takes Windows ~8 s to tear down, so the
    stop waits up to 20 s after the kill before it reports a process as still running.
  * Checked with the real router (port 18033): `presets serve --background --load Tesslate_OmniCoder-9B-Q4_K_M`,
    a chat request to it, `presets load gemma-4-26B-A4B-it-qat-UD-Q4_K_XL` (the router unloaded OmniCoder:
    `models_max = 1`), a chat request to gemma through the same endpoint, `presets status`, `presets stop`:
    every process gone, state file removed.
  * `presets serve --load` waits by polling `GET /models` instead of following `/models/sse` (simpler, same
    result); `presets watch` follows `/models/sse`.
  * `[presets]` in the config gained `models_max = 1` and `memory_margin_mib = 1024` (12 GB VRAM), `autoload`,
    `load_on_start`; the golden test still compares the builder without them against `run_PRESETS_serve`.
  * `presets validate` on the original INI (reconstructed in `tests/fixtures/presets/original_broken.ini`) gives
    the §4.3 findings; `presets fix --write` turns it into the committed `llama_cpp_presets.ini` except the comments.
  * The ServerHandle cycle test runs 10× by default (~25 s); `MSE_LLM_TEST_CYCLES=50` runs the acceptance count.
* **M6 (2026-11-19):** `monitor/` (`nvidia.py` NVML, `windows.py` PDH, `system.py` psutil; `Monitor`, `Guard`,
  `summarize` in `__init__.py`), `notify.py`, Job Object memory cap in `proc.py`, commands `monitor [-- CMD]` and
  `notify` (`cli_monitor.py`); `serve` and foreground `presets serve` run under the monitor (`--no-monitor`,
  `--no-guard`, `--mem-cap`), print the bash-style RESULTS block, write `results/servers/*.monitor.csv`, and exit 2
  after a guard stop. New `[monitor]` config table. Samples every 0.5 s (all samplers in parallel threads).
  * **The bash spill rule is wrong for MoE offload.** "Shared GPU memory delta > 100 MiB = offloaded to system RAM"
    fires on every `-ncmoe` run: llama.cpp's pinned host buffers for the CPU experts count as shared GPU memory.
    Measured: a fitting Qwen3.6-28B `moe41` shows +15892 MiB shared (the bash function printed "Performance hit
    detected" for it), equal to the 15807 MiB host part `fit` predicts (0.5 % off). A real spill (Qwen3.6-35B
    `moe10`, 17.2 GB on 11.4 GB free) looks different: dedicated VRAM fills up first (12140 / 12282 MiB), then the
    server's own shared memory keeps growing. The guard stops a run when (A) dedicated VRAM is full and the
    server's shared memory grows > 100 MiB after that, (B) the server's shared memory exceeds the host buffers
    `fit` expects by > max(100 MiB, 2 %), or (C) RAM > 95 % / commit > 95 %. The summary says "no spill" with the
    numbers when fit's host size is known, and explains the bash rule otherwise.
  * **Shared memory per process:** `\GPU Process Memory(pid_*)\Shared Usage` for the server tree, matched to the
    NVIDIA adapter's LUID; the adapter-wide `\GPU Adapter Memory(*)\Shared Usage` sum the plan names would mix in
    the AMD iGPU's normal shared memory (1156 MiB here). The bash number (`Non Local Usage` sum) is kept as
    `nonlocal_mib` for comparison.
  * **Forced over-commit, real:** `serve Qwen36_35B_A3B_UD_Q4_K_M_MTP -v fa_Kq8_Vq8_default_moe10 --yes` was
    stopped by rule A 8 s after start with 586 MiB spilled (192-320 MiB over the full point, one 0.5 s sample),
    exit code 2, no process left. Without the guard the same load reached 9 GB of shared memory in 60 s.
  * **Bash RAM bug:** Git Bash has no `free`, and the monitor loop never clears `RAM_USED`, so after the first
    PowerShell sample RAM is never measured again; the bash "Peak RAM" is the value at the start (44046 MiB vs
    60015 MiB measured while 15.9 GB of expert buffers were pinned). Our RAM = total - available = the same
    formula as bash's PowerShell line (checked: 59583 vs 59574 MiB at the same moment).
  * **Job Object cap:** on Windows WDDM, VRAM allocations count against the job's committed memory too: with
    `--mem-cap 2000`, OmniCoder 9B failed with "cudaMalloc failed: out of memory" for its 5 GB CUDA buffer. A cap
    must cover VRAM + host buffers + RAM; it is off by default.
  * Not in M6: Parquet output (CSV only, no pandas dependency); thermal gating between runs (§9 item 2, M15).
* **M7 (2026-11-19):** `results.py` (RunResult, store: `results/index.jsonl`, `runs/YYYY-MM/<run_id>.json`,
  `builds/`, `hw/`, git-ignored `raw/<run_id>/`), `parsers/` (llama-bench jsonl, benchy json), `bench.py` (jobs,
  sweeps, resume), `cli_bench.py`: `bench`, `benchy`, `sweep list|show|run`, `results list|show`. The old
  dry-run-only `bench` / `benchy` commands in `cli.py` are replaced.
  * **Sweeps from the script, as data:** all 9 group functions are `[sweeps.*]` in `config.example.toml`,
    generated from the `extras_list` arrays (161 entries, a test checks the count against the script):
    commented-out entries `status = "off"`, "PC freeze" ones `status = "skip"`, every comment kept as `note`.
  * **Acceptance runs, real (build ffc87c3a7):** `sweep run omnicoder_9b_bench` (4 variants, pp 131072:
    2259-2660 t/s PP, 53-56 t/s TG), `sweep run Qwen36_35B_A3B_UD_Q4_K_M_MTP_bench` (the 2 active `moe41`
    entries, pp 65536: 727 / 745 t/s PP, 29.1 / 33.5 t/s TG), `sweep run gemma_4_26B_A4b_QAT_UD_Q4_K_XL_benchy
    --include-off --match "*moe21_sd4mtp"` (benchy, pp 65536: 1535 t/s PP, 25.4 t/s TG); then `--resume` skipped
    both Qwen entries ("ok in <run_id>") and `--only-failed` re-ran exactly the aborted gemma run. 8 KB per run JSON.
  * **A no-fit config can hang instead of spilling** (new guard rule D, §5): llama-bench, Qwen3.6-35B `moe10`,
    ctx 4096, forced past the fit check, allocated ~10 GB VRAM and then sat at 0 % GPU without disk reads for
    140 s (then spilled, rule A stopped it) and in another run for 10+ minutes, where it also blocked NVML for
    other processes and ignored the kill for ~3 minutes. Rule D: no disk reads, no VRAM change and mean GPU
    use < 5 % for 90 s (benchmarks only). Replayed on the recorded samples it stops both no-fit runs at ~97 s and
    never the fitting `moe41` run (which reads the GGUF the whole load, then runs the GPU at 99 %).
  * Monitor samplers now run in daemon threads with a time limit, so a blocked NVML call no longer stops the
    monitor (before: the guard went blind) or mse-llm's exit. After a guard stop a job waits at most 30 s for
    the process and moves on.
  * **Rule B false alarm fixed:** gemma QAT `moe21_sd4mtp` (your SD-on sweet spot) was stopped by rule B: its
    240 MiB MTP drafter added +360 MiB of shared memory that fit's host part (main model only) doesn't include,
    with 4 GB of VRAM still free. Rule B now allows the drafter / mmproj sizes on top (`serve` too).
  * Every job is fit-checked first; without `--force` a no-fit variant is a `skipped` result in ~1 s with the
    `-ncmoe` suggestion (the moe10 case above: "needs -ncmoe ≥ 24").
  * The tests now always write to a temp `MSE_LLM_RESULTS_DIR`; an M3 test that expected `bench` without
    `--dry-run` to refuse had started a real 131072-token llama-bench.
  * Not in M7: `results compare` / `export --csv` (M8 with the reports); eval sweeps (M12).
  * `results/` is git-ignored for now (your decision, 2026-11-19); the store itself is unchanged.
* **M8 (2026-11-19):** `report/` (`plot.py` the 5-axis chart, `data.py` run selection + rows, `prompts/analyst.md.j2`),
  `parsers/legacy_log.py`, `cli_report.py`: `report chart | table | prompt`, `results compare | export`.
  * **Chart parity:** the bash plotter's parser and matplotlib code are ported as they are. On the converted
    gemma-4-26B UD log (17 blocks) the bash function and `report chart --legacy-log` give PNGs with 0 differing
    pixels (same matplotlib 3.11.2). Kept bash quirks for old logs: llama-bench `-p N` shows as "Ctx:default",
    values are cut at `-` (`--spec-type draft-mtp` -> "Spec:draft"), missing duration / VRAM / shared are drawn
    as 100 s / 4000 / 12000 (labelled with those numbers).
  * **New results** chart through the same code: labels from the run's argv with the same regexes, but with the
    real ctx; duration = time under the monitor minus the server start (stacked below for benchy); VRAM = peak
    dedicated; shared = the bash number (Non Local Usage sum, adapter-wide). Missing values are drawn as 0.
    One bar per (tool, model, variant, ctx, params): the newest run wins (`--all-runs` shows every run).
    Labels differ from a bash chart of the same variant only where M3 changed the argv (`-lm none` instead of
    `--mmap 0` -> "lm:none").
  * **Old logs:** the bash `.txt` logs, and the `_RESULTS/RESULTS_run_BENCH_*.html` pages they were pasted into
    (`<br>` / `&nbsp;` converted back): all 6 pages parse (95 benchmark blocks); two are kept as test fixtures.
  * `report prompt`: the bash analyst prompt with the data embedded as CSV; `report table` (best PP / TG marked,
    `--format csv | md`, `--tests`); `results compare A B` (settings that differ, PP / TG / memory deltas);
    `results export --csv` (per run or `--tests`).
  * matplotlib, numpy and jinja2 are normal dependencies, not the optional `[report]` extra the plan named, so
    `uv run mse-llm report chart` works without an extra install step.
* **M9 (2026-11-19, acceptance runs pending):** `predict.py` (GGUF roofline shape, spec bandwidths from `hw`,
  `-ncmoe × -ub` memory grid via llama-fit-params, calibration from the results store), `search.py` (staged safe
  search), `cli_predict.py`: `predict`, `results calibrate`, `sweep search`.
  * **Memory grid:** per `-ub` only the highest and lowest `-ncmoe` are measured, the rest follows "one step = one
    expert layer" (`--exact` measures every point). `sweep search` never starts a point predicted no-fit or tight,
    and every job still gets the normal fit check and memory guard.
  * **Calibration:** six terms (VRAM / RAM bandwidth and per-token cost for TG; GPU FLOPS, PCIe bandwidth and
    per-ubatch cost for PP) fitted as scale factors on the spec values, pulled towards 1; then the same fit per
    model (a model's quant types decide how fast the CPU runs its experts). Speculative-decoding runs are not used.
    On the 6 M7 runs (OmniCoder 9B dense, Qwen3.6-35B `moe41`): leave-one-out median error 3.5 %, 11 of 12 within
    15 %; effective RAM came out at ~34 GB/s (spec 83 GB/s dual channel × 0.72 = 60).
  * **Real `sweep search gemma_4_26B_A4b_QAT_UD_Q4_K_XL` (ctx 65536, build ffc87c3a7):** order ncmoe then ub
    (predicted ×1.96 vs ×1.77), 7 runs, **no run spilled**; ncmoe stage 30 → 10, moe9 predicted tight (edge), winner
    one step back; ub stage at moe15: 512 → 2048, ub 4096 predicted no-fit; winner `fa_Kq8_Vq8_b1024_ub1024_moe15`.
  * **Power plan "Legion Quiet Mode" breaks TG.** That run was on AC but under the Lenovo quiet plan: TG with experts
    on the CPU was 5.9 t/s at moe30 (bash era, build ad1b88ca0: 27.4) and **11.1 vs 32.1 t/s for the same moe15
    config** in two runs; PP was ~3× below the bash numbers (moe15 714 t/s vs moe16 2247). PP stays predictable
    (leave-one-out within 15 % for all 7 runs), TG doesn't (7 of 14 values within 15 %). Now: `hw` / `doctor` warn
    on a quiet / power-saving plan, every run records `hw.power_plan`, and calibration and `sweep search` reuse only
    runs of the current plan (runs before M9 have no plan and are kept). The 7 quiet runs were tagged afterwards.
    `bench`, `benchy`, `sweep run` and `sweep search` now stop before the first job on battery or a quiet plan
    (`--allow-low-power` / `MSE_LLM_ALLOW_LOW_POWER=1` to run anyway; dry runs never stop).
  * **Open:** acceptance (TG/PP within ~15 % for 2 models; a search without a spill) re-run under a performance plan:
    `sweep search` for gemma QAT and Qwen3.6-35B MTP, then `results calibrate`. Whether build ffc87c3a7 itself is
    slower for gemma QAT than ad1b88ca0 (`cuda_flash` vs `cuda_flash-without-zen4-avx512x`) is checked then, since the
    quiet plan hides it.
  * Not in M9: the K/V and speculative-decoding stages of §9 item 6 (they need KLD and acceptance rate, M10);
    writing into the presets INI (`--apply` writes `default_serve`, which `presets generate` uses).
