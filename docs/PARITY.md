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
| F4 | `run_bench_params*`, `_wrapper_EXTRAS`, `_wrapper_PROMPTS` | `mse-llm bench` | same argv in `--dry-run` (§0.1 bugs fixed) | ◐ M3: `bench --dry-run` argv = bash (differences below); `-v` list / repeated `--ctx` = the two wrappers; running in M7 |
| F5 | `run_benchy_params_with_server*` | `mse-llm benchy` | same argv, `--pp` list passed once | ◐ M3: `benchy --dry-run` server + client argv = bash, lists after one flag; running in M5/M7 |
| F6 | `run_serve_params_on_background`, `server_wait_for_ready`, `server_stop_windows` | `ServerHandle` | 50× start/stop, no orphans | ☐ M5 |
| F7 | `run_serve_params` + `_FILTER`, output modes | `mse-llm serve --output` | same filtered lines | ◐ M3: `serve --dry-run` argv = bash (+ `--fit on`); running + output filter in M5 |
| F8 | `run_presets_serve`, `run_PRESETS_serve`, `run_PRESETS_dump`, `run_presets_dump` | `mse-llm presets …` | existing INI serves all 5 models | ◐ M3: `presets serve --dry-run` argv = bash; `run_PRESETS_dump` = `serve --all --dry-run`; the rest in M5 |
| F9 | `EXEC_cmd_with_gpu_and_ram_monitoring` | `monitor/` | peaks within tolerance | ☐ M6 (M1: NVML/PDH availability checked by `doctor`) |
| F10 | `run_dump_params_wrapper_PLOTER` | `mse-llm report chart` (+ `--legacy-log`) | old logs give the same chart | ☐ M8 |
| F11 | `run_dump_params_wrapper_AI_PROMPT/_COMMON/_EXTRAS/_PROMPTS` | `mse-llm report prompt` | same sections + data | ☐ M8 |
| F12 | `run_inspect_params*` | `mse-llm eval` | real eval runs | ◐ M3: `eval MODEL TASK --dry-run` = the bash builder (URL from the engine); running in M12 |
| F13 | `EXEC_user_notification_WIN_T2S` | `notify.py` | voice message at end of run | ☐ M6 |
| F14 | `EXEC_cmd_with_output_mode` (all/filter/none) | `--output` | — | ◐ M0: `[defaults].output`; behaviour in M5 |
| F15 | `MSE_*` knobs | config fields / `MSE_LLM_*` env / `--dry-run` | each knob mapped or dropped | ◐ M0: see the knob table below |
| — | `run_BENCH_*`, `run_BENCHY_WITH_SERVER_*`, `run_SERVER_*`, `run_INSPECT_*` group functions (+ commented entries, "SWEETSPOT", "PC freeze" notes) | `[sweeps.*]` + `annotations` | one entry per function, notes kept | ☐ M7 |
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
| `MSE_MONITOR_ENABLED` | `[defaults].monitor` | ✓ M0 (used from M6) |
| `MSE_LLAMA_BIN_DIR` | `[engines.*].bin_dir`, from the build profile | ✓ M0; M2: `builds\llama.cpp\build\bin\Release` built by `mse-llm build run`, same folder in the script |
| `MSE_PS_DEBUG` | `-v/--verbose` | ☐ (no PowerShell left to debug after M6) |
| `MSE_SERVER_PRESETS_DUMP` | `--dry-run`, `serve --all --dry-run` (`presets dump` of the INI in M5) | ✓ M3 |
| `MSE_PRESETS_SERVE_FILE` | `[presets].file`, `presets serve --file` | ✓ M3 (argv); serving in M5 |
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
  * Not yet: `presets validate` (M5); the INI is checked by the `doctor flags` audit for now.
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
