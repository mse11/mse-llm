# Parity checklist: `llama_cpp_benchmark.sh` → `mse-llm`

Every row must be ✓ (or "dropped + reason", approved by you) before the bash script and
`llama_cpp_presets.ini` are deleted (M13, [IMPLEMENTATION_PLAN.md §10.1](../IMPLEMENTATION_PLAN.md)).
Updated as milestones land.

Legend: ✓ done and checked · ◐ partly done · ☐ not started · ✗ dropped (reason given)

| # | Script feature (bash) | Replacement | Check | Status |
|---|---|---|---|---|
| F1 | `run_HELP_server/bench__*` | `mse-llm help server\|bench` (+ `benchy`, `eval`), `doctor flags` | output identical to `--help` | ✓ M2: real snapshots `a49db739f` + `ffc87c3a7` stored; `doctor flags` diff + audit checked against them |
| F2 | `MODELS`, `DRAFTER_MODELS`, `MODELS_BENCH_EXTRA`, `MODELS_SERVE_EXTRA` | `[models.*]` in config | every model + extra present | ◐ M0: schema + the 5 INI models; full set from the script in M3 |
| F3 | `EXTRAS` matrix + `EXTRAS_get_flags_normalized/_replaced` | `variants.py` codec | golden test: every generated key decodes to the same flags | ☐ M3 |
| F4 | `run_bench_params*`, `_wrapper_EXTRAS`, `_wrapper_PROMPTS` | `mse-llm bench` | same argv in `--dry-run` (§0.1 bugs fixed) | ☐ M3/M7 |
| F5 | `run_benchy_params_with_server*` | `mse-llm benchy` | same argv, `--pp` list passed once | ☐ M7 |
| F6 | `run_serve_params_on_background`, `server_wait_for_ready`, `server_stop_windows` | `ServerHandle` | 50× start/stop, no orphans | ☐ M5 |
| F7 | `run_serve_params` + `_FILTER`, output modes | `mse-llm serve --output` | same filtered lines | ☐ M5 |
| F8 | `run_presets_serve`, `run_PRESETS_serve`, `run_PRESETS_dump`, `run_presets_dump` | `mse-llm presets …` | existing INI serves all 5 models | ☐ M5 |
| F9 | `EXEC_cmd_with_gpu_and_ram_monitoring` | `monitor/` | peaks within tolerance | ☐ M6 (M1: NVML/PDH availability checked by `doctor`) |
| F10 | `run_dump_params_wrapper_PLOTER` | `mse-llm report chart` (+ `--legacy-log`) | old logs give the same chart | ☐ M8 |
| F11 | `run_dump_params_wrapper_AI_PROMPT/_COMMON/_EXTRAS/_PROMPTS` | `mse-llm report prompt` | same sections + data | ☐ M8 |
| F12 | `run_inspect_params*` | `mse-llm eval` | real eval runs | ☐ M12 |
| F13 | `EXEC_user_notification_WIN_T2S` | `notify.py` | voice message at end of run | ☐ M6 |
| F14 | `EXEC_cmd_with_output_mode` (all/filter/none) | `--output` | — | ◐ M0: `[defaults].output`; behaviour in M5 |
| F15 | `MSE_*` knobs | config fields / `MSE_LLM_*` env / `--dry-run` | each knob mapped or dropped | ◐ M0: see the knob table below |
| — | `run_BENCH_*`, `run_BENCHY_WITH_SERVER_*`, `run_SERVER_*`, `run_INSPECT_*` group functions (+ commented entries, "SWEETSPOT", "PC freeze" notes) | `[sweeps.*]` + `annotations` | one entry per function, notes kept | ☐ M7 |
| — | Model notes in comments (HF URLs, sizes, capabilities) | `[models.*].url`, `notes` | nothing lost | ☐ M3 |
| — | turbo2/3/4 cache variants (`*_Kt*`, `*_Vt*`) | — | — | ✗ dropped: not supported by upstream llama.cpp (approved 2026-11-18) |

## F15: `MSE_*` knobs

| Bash knob | mse-llm | Status |
|---|---|---|
| `MSE_THREADS` | `[defaults].threads`, `MSE_LLM_THREADS` | ✓ M0 |
| `MSE_PARALLEL` | `[defaults].parallel`, `MSE_LLM_PARALLEL` | ✓ M0 |
| `MSE_BENCH_TG` | `[defaults].bench_tg`, `MSE_LLM_BENCH_TG` | ✓ M0 |
| `MSE_BENCHY_PP` / `_PP_SYNCH` / `_DEPTH` / `_RUNS` | `[defaults.benchy]`, `MSE_LLM_BENCHY_*` | ✓ M0 |
| `MSE_SERVER_READY_TIMEOUT` | `[defaults].server_ready_timeout` | ✓ M0 |
| `MSE_MONITOR_ENABLED` | `[defaults].monitor` | ✓ M0 (used from M6) |
| `MSE_LLAMA_BIN_DIR` | `[engines.*].bin_dir`, from the build profile | ✓ M0; M2: `builds\llama.cpp\build\bin\Release` built by `mse-llm build run`, same folder in the script |
| `MSE_PS_DEBUG` | `-v/--verbose` | ☐ (no PowerShell left to debug after M6) |
| `MSE_SERVER_PRESETS_DUMP` | `--dry-run` / `presets dump` | ☐ M5 |
| `MSE_PRESETS_SERVE_FILE` | `presets serve --file` | ☐ M5 |
| `MSE_TEST_INSPECT_PATH` | `eval` config | ☐ M12 |

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
  * Not yet: `presets validate` (M5); the INI is checked by the `doctor flags` audit for now.
