# mse-llm — Implementation Plan

Port `llama_cpp_benchmark.sh` (≈2300 lines of Git Bash) to a cross-platform, async Python CLI
packaged with **uv**, using **click**, with a pluggable *runner* layer so engines other than
llama.cpp (e.g. [Strata](https://github.com/Niko1221/Strata)) can be added later.

Reference build for the findings in §0: llama.cpp `a49db739f` (`b12468-5`, 2026-11-17), built with
the same flags as `config/builds/rtx4080-laptop.toml`. From now on builds go to `builds\llama.cpp\build\bin\Release`.
Reference machine: RTX 4080 Laptop GPU (12 GB, sm_89, 192-bit GDDR6 ~432 GB/s, PCIe 4.0 x16) +
AMD Ryzen 9 7945HX (16C/32T, Zen 4, not hybrid, AVX-512 + VNNI/VBMI/BF16) + 80 GB DDR5-5200
(16 + 64 GB, asymmetric), Windows 11.
Build profiles: [`config/builds/`](config/builds/) (§0.4).

## Decisions and conclusions (2026-11-18)

What has been decided, checked or changed so far. The sections below have the details.

**Project and config**
* Python CLI with **uv + click**, flat layout (`mse_llm/`, `tests/`), async with `asyncio`. No shell
  needed: commands are argv lists, so it runs the same from PowerShell, cmd or Git Bash (§1, §3).
* **All configuration is TOML** (`tomllib`, no YAML). Build profiles are one TOML per hardware
  target in `config/builds/`, plus `_cmake-flags.toml` with a plain-language explanation of every
  CMake flag (§0.4).
* llama.cpp and vcpkg are **cloned into the git-ignored `builds/`** in this repo when missing.
  **Both mse-llm and the bash script use only `builds\llama.cpp\build\bin\Release`**. No checkout
  outside this repo is used (§0.4).
* **Nothing outside this repo is used.** `C:\Users\micha\MSE\GIT\_MSE_doc\AI\MODELS\app_llamacpp\`
  is being removed; the script's references to it were deleted.
* **`results/` lives in this repo; for now it is not stored in git** (`/results/` in `.gitignore`, decided
  2026-11-19 after M7; the plan below was "tracked in git"). The tool only writes files. What gets
  committed, and how, is decided later by you with plain git (no `results commit/status/prune`
  commands). Large raw data and runtime state are git-ignored (§8).
* **Preview before anything heavy runs:** `build plan`, and `--dry-run` on `build run`, `bench`,
  `benchy`, `sweep`, `serve`, `tune`, `quality`. `build run` shows its plan and asks before step 1 (§0.4).

**Build (RTX 4080 Laptop)**
* Your manual cmake build is reproduced by `config/builds/rtx4080-laptop.toml`; the check
  produced the same flags (§0.4).
* `GGML_CUDA_FA_ALL_QUANTS=ON` is required for the asymmetric K/V variants.
* `GGML_AVX512:BOOL=OFF` in CMakeCache is **misleading**: with MSVC + `GGML_NATIVE=ON` the CPU
  backend is built with `/arch:AVX512` anyway. MSVC never turns on the extensions VNNI/VBMI/BF16
  itself, so they're the variant `zen4-avx512x`, **applied by default** (`apply_variants`) but kept
  in its own config table. `build run rtx4080-laptop --without zen4-avx512x` builds the comparison
  (§0.4). Not yet verified that MSVC builds cleanly with them.
* `build show` prints every flag with its source and meaning, `build versions` lists buildable
  versions (`v1.0.0`, `b<N>` tags), `build plan` previews every step (§0.4).

**Fixed in the existing files** (checked with the real llama-server)
* `llama_cpp_presets.ini` failed at router startup (`option 'no-mmap' not recognized in preset
  'global defaults'`). Fixed: `[*]`, `load-mode = none`, `fa = on`, `version = 1`. The router now
  starts and lists all 5 models. `alias` **is** used by the router as an extra model name (§4.3).
* `llama_cpp_benchmark.sh` `MSE_PRESETS_SERVE_FILE` points at the repo's fixed INI;
  `run_PRESETS_serve` was checked on a test port (§4.3).
* Other script bugs found (to be fixed by the port, not in bash): `${BENCH_extra}` typo,
  `run_presets_serve` reads `$6`, a duplicate `run_SERVER_gemma…` that actually serves GLM,
  benchy `--pp` repeated flags (only the last value used), turbo* cache types unsupported
  (§0.1). **turbo2/3/4 are dropped** (no fork profile).

**Memory, freezes and what to tune** (measured on this laptop)
* **Most** of your laptop freezes match memory spills: when a config needs more than ~11.5 GiB of
  VRAM, the NVIDIA driver silently spills into system RAM (one exception, Qwen35 `moe30`, see §4.6). **Recommended driver setting:** NVIDIA Control
  Panel → Manage 3D settings → *CUDA – Sysmem Fallback Policy* = **Prefer No Sysmem Fallback**
  (§9 item 7). You set it yourself; mse-llm only checks it.
* `llama-fit-params --fit-print` gives llama.cpp's own VRAM/RAM estimate in ~0.5 s without loading
  the model. Two exact rules follow: **one `-ncmoe` step = one expert layer** (Qwen35 464 MiB), and
  **compute buffer ≈ `ub × vocab × 4 B` ≈ 1 MiB per ubatch token** on your 248k/262k-vocab models,
  so `ub 8192` alone costs ~8 GiB (§4.6).
* **Which setting helps** follows from `n_used / n_expert` and how many experts sit on the CPU:
  Gemma/GLM gain about twice as much TG per GiB from lowering `-ncmoe` as the sparse Qwen MoEs; a
  bigger `ub` helps PP when many experts are on the CPU (PCIe uploads per ubatch). This matches
  your notes, except Qwen35 `moe30` "freeze", which memory doesn't explain (§4.6).
* New commands: **`fit`** (paste a GGUF + your variant list + ctx → expected fit per variant,
  every memory part labelled measured / estimated / NOT COUNTED, incl. drafter and mmproj),
  **`predict`** (TG/PP estimate + recommendation), **`sweep search`** (safe automatic search from
  the safe side, stops at the first spill) (§4.6, §9).
* Hybrid models (Qwen3.6, Gemma 4) have small KV caches, so KV quantization saves < 1 GiB at 64 k;
  `-ncmoe` / `--moe-cache-mib` move far more memory (§4.5).

**Quality and speculation**
* **KL divergence** against an f16-KV reference of the same GGUF (`quality kld`), with a noise
  floor, a code corpus, a disk-size preview, and a fallback order when the reference doesn't fit
  (test context, more CPU offload, `-nkvo`, q8 reference) (§4.5).
* **Speculative-decoding acceptance** (overall, mean accepted length, per draft position) from
  llama-server's `/metrics`, the slot log and response timings (§4.5).
* New MoE-cache axes from llama.cpp 1.0: `--moe-cache-mib`, `--moe-cache-cpu-share`, load mode,
  `f8_e4m3` KV, plus a wrapper for `moe-cache-tune.py` (§4.4).

**Answers from the review of the open questions**
* **turbo2/3/4 KV cache types: dropped.** Not in upstream llama.cpp; no fork profile.
* **`serve` uses `--fit on`**, but (found in M5) llama.cpp's `--fit` only adjusts what the user did not
  set: with `-ngl all` or `-ncmoe` it aborts and the server starts with exactly the requested settings.
  So the `fit` pre-check is what protects you: it warns, says so, and shows what `--fit` would pick with
  both left unset (`serve --let-fit`). The live memory guard (§5) stops a spill. Benchmarks keep
  `--fit off` (§0.3).
* **SWE-bench:** Docker doesn't run on this Windows 11 laptop, so the model server runs on Windows and
  SWE-bench + Docker run on Linux (VirtualBox VM: host at `10.0.2.2` with NAT or `192.168.56.1`
  host-only; or another machine), with an API key and a firewall rule for the port (§7.1).
* **`zen4-avx512x` is part of the default build**, kept in its own config table for comparison.
* **llama.cpp checkout: a new clone in `builds/llama.cpp`** (vcpkg fresh in `builds/vcpkg`), used
  by mse-llm **and** the bash script (`MSE_LLAMA_BIN_DIR`). No other checkout is referenced (§11 Q5).
* No open questions remain in §11.

**Found while implementing M5 + M6 (2026-11-19)** (details in `docs/PARITY.md`, milestone log)
* `--fit on` can't shrink a `serve` config (`-ngl all` / `-ncmoe` set): see §0.3; `serve --let-fit` leaves both
  out so `--fit` decides the placement.
* **The bash spill rule is wrong for MoE offload.** llama.cpp's pinned host buffers for CPU experts count as
  shared GPU memory (+15.9 GB for a fitting Qwen3.6-28B `moe41`, equal to `fit`'s host part), so "shared delta >
  100 MiB = spill" fires on every `-ncmoe` run. A real spill fills dedicated VRAM first and then keeps growing the
  server's shared memory; the guard uses that (§5).
* Shared memory is read per server process on the NVIDIA adapter; the adapter-wide sum would include the AMD
  iGPU (§5). NVML has no per-process VRAM on Windows (WDDM); PDH gives it.
* The bash monitor samples RAM only once (bug), so its "Peak RAM" is the start value.
* A Windows Job Object memory cap also counts VRAM allocations (WDDM), so it must cover VRAM + host + RAM.
* Graceful stop on Windows is CTRL_BREAK to the server's own process group; attached servers sit in a
  kill-on-close Job Object (no orphans even if mse-llm is killed); background routers are stopped by PID.

**Found while implementing M7 (2026-11-19)** (details in `docs/PARITY.md`, milestone log)
* **A no-fit config can hang instead of spilling.** llama-bench with Qwen3.6-35B at `-ncmoe 10` (ctx 4096,
  forced past the fit check) sat at 0 % GPU with 10 GB VRAM, no disk reads and no shared-memory growth for
  140 s in one run (then spilled) and 10+ minutes in another, so guard rules A-C never fired. New rule D
  (benchmarks only): no progress for 90 s (§5). While it hangs it can block NVML for every process (the
  monitor's samplers now have time limits) and ignore the kill for minutes.
* Guard rule B (shared memory over fit's host part) needs an allowance for a drafter / mmproj: their buffers can
  land in pinned host memory, which fit doesn't place (gemma QAT + 240 MiB MTP drafter: +360 MiB, a false stop).
* The fit pre-check is what keeps sweeps safe: without `--force` such a variant is recorded as `skipped` in
  ~1 s, with the `-ncmoe` that would fit.
* The bash sweeps are data now: `[sweeps.*]` in the config, generated from the script's `extras_list` arrays,
  commented-out entries as `status = "off"`, "PC freeze" ones as `status = "skip"` (only with `--force`).

**Later**
* Strata as a second engine (managed mode, `/v1/models` probe, API key, `/metrics`), compared with
  a fair protocol (§7). PC capability detection modelled on Strata's (§5.1).
* `llama_cpp_benchmark.sh` and `llama_cpp_presets.ini` are **deleted once every row of the parity
  checklist is done** (M13, §10.1). Both are committed first, so history keeps them.

---

## 0. Findings from the current build (llama.cpp a49db739f)

Checked against `llama-server --help`, `llama-bench --help`, `uvx llama-benchy --help`, and by
running the flags against the binaries.

### 0.1 Broken today

| Where | Issue | Fix |
|---|---|---|
| `llama_cpp_presets.ini` `[global defaults]` | `no-mmap = 1` → **`error: invalid argument: --no-mmap`** (flag removed) | `load-mode = none` (`-lm none`) |
| `EXTRAS` `*_Kt*`, `*_Vt*` (turbo2/3/4) | **`Unsupported cache type: turbo4`**. The turbo types came from the turboquant fork, not upstream. Allowed now: `f32 f16 bf16 f8_e4m3 q8_0 q4_0 q4_1 iq4_nl q5_0 q5_1` | **Decided: dropped.** The `Kt*`/`Vt*` tokens aren't generated anymore; old names decode with a clear "turbo types dropped" error. Add `f8_e4m3`, `q5_0/q5_1`, `iq4_nl` |
| `llama_cpp_presets.ini` section name | Upstream docs use `[*]` for globals. `[global defaults]` becomes a *model* section | Rename to `[*]`, add `version = 1` |
| `run_bench_params_cmd_build` (l.1549) | `${BENCH_extra}` typo (wrong case), so the user's bench extra is never appended | Fixed by the port |
| `run_presets_serve` (l.1474) | Reads `$6` where `$2` was meant, so the presets file argument is ignored | Fixed by the port |
| `run_SERVER_gemma_4_26B_A4b_UD_Q4_K_M_CTX_cuda_flash` | Defined twice. The first one (l.2134) actually serves **GLM** and is shadowed by the second | Generate run definitions from config (§4), so names can't collide |
| benchy `--pp a --pp b` / `--tg` / `--depth` | benchy uses `nargs='+'`. Repeating the flag **keeps only the last value**, so PP sweeps collapse | Pass `--pp 512 2048 8192` once |
| `_sdSmtp`/`_sdScache` extras used with `llama-bench` | llama-bench has no `--spec-*` flags | Validate the flags against the target tool's capabilities (§3.3) |
| Inspect runner | Stubbed (`echo before; sleep 5`, `MSE_TODO`), port hard-coded to `8033` | Real implementation with the base URL from the server profile |
| Monitoring | Spawns `powershell` + `nvidia-smi` **3–4× per second-tick**. That costs CPU and can skew results | In-process NVML + PDH sampling (§5) |

### 0.2 Changed or deprecated (works, but should be updated)

| Flag | Now | Recommendation |
|---|---|---|
| `-fa 1` | `-fa on\|off\|auto` (1 still accepted) | emit `-fa on` |
| `-ngl 999` | `-ngl N\|auto\|all` (default `auto`) | emit `-ngl all` |
| `--jinja` | default **enabled** | drop it, or keep it explicit for clarity |
| `--reasoning-format` | default `auto`; new `-rea on\|off\|auto`, `--reasoning-effort`, `--reasoning-budget N` | expose them as typed fields |
| `--draft`, `--draft-min`, `--spec-ngram-size-*` | **removed** | already using `--spec-draft-n-max` (default 3) |
| `--parallel` | default `-1` (auto). `-kvu` is on when auto | keep `1` for benches and set it explicitly |
| `--port` | default `9931` | always set it explicitly |

### 0.3 New options worth exposing in sweeps

* **`--fit on|off` (server default `on`)**: auto-adjusts unset args to fit VRAM, so the
  measured config can silently differ from the requested one. `-fitt/--fit-target MiB`, `-fitc`.
  `llama-bench` has `-fitt/-fitc` (off by default). **Decided:**
  * **bench / benchy / sweep / quality:** `--fit off`, so the measured config is exactly the requested one.
  * **serve / presets serve:** `--fit on`. Before starting, `serve` runs the `fit` check (§4.6) and
    **warns** instead of silently adjusting:
    * `NO-FIT` / `TIGHT` / `UNSURE` for the requested settings, with the numbers, plus the
      `fit --suggest` answer (lowest `-ncmoe` that fits);
    * **what llama.cpp's `--fit` would change**: `llama-fit-params` without `--fit off` prints
      the fitted command line, which is diffed against the requested one (for example
      `-ngl all → 42` or "experts of layers 15–40 on the CPU");
    * then it asks before starting (`--yes` skips the question). After start, `serve` compares the
      server log's real `n_ctx` / offload with the request and repeats the warning if `--fit`
      changed anything.
  * **Checked in M5: `--fit` only adjusts arguments the user did not set.** `serve` always passes
    `-ngl all` and `-c`, and most variants `-ncmoe`. With `-ngl` set llama.cpp aborts the fit
    ("n_gpu_layers already set by user"), with `-ncmoe` too ("tensor_buft_overrides already set by
    user"), and the server starts with exactly the requested, too-large settings. So `--fit on`
    does **not** shrink a `serve` config. The pre-check says so and shows what `--fit` would pick
    with `-ngl` and `-ncmoe` left unset; **`serve --let-fit`** runs exactly that. A spill that
    happens anyway is stopped by the memory guard (§5).
* **MoE**: `-cmoe`, `-ncmoe N`, **`--moe-cache-mib N`**, **`--moe-cache-cpu-share F`**,
  `-ncffn N` (dense FFN to CPU). The MoE cache is the main new sweep axis next to `n-cpu-moe`; see §4.4.
* **Speculative**: `--spec-type` now lists `draft-simple, draft-eagle3, draft-mtp, draft-dflash,
  draft-dspark, draft-dflash2, draft-lilicorr, ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod,
  ngram-cache` (comma-list allowed), plus `--spec-default`, `--spec-draft-n-min`, `-ctkd/-ctvd`,
  `-ngld`, `--spec-draft-n-cpu-moe`, `--spec-ngram-mod-*`.
* **Memory/cache**: `-cram/--cache-ram`, `-ctxcp`, `--kv-unified-per-slot`, `--swa-full`, `-nkvo`,
  `-lzm/--lazy-mode`, `-lm dio`.
* **Router (presets)**: `--models-max`, **`--models-memory-margin`** (LRU unload),
  `--models-autoload`, per-preset `load-on-startup`, `stop-timeout`. Host and port are owned by the router.
* **Observability**: `--metrics` (Prometheus), `--slots` endpoint, `--log-file`.
* **llama-bench**: `-o json|jsonl|csv` (**structured output, replaces regex parsing**), `-r`
  (default **5** repetitions; the script never sets it), `-d` depth, `-pg pp,tg`, `--delay`,
  `--no-warmup`, `--progress`. It also accepts **comma lists / ranges** (`-ncmoe 30-40+2`, `-ctv q8_0,q4_0`).
* **llama-benchy**: `--format json --save-result FILE`, `--concurrency`, `--no-cache`,
  `--exact-tg`, `--latency-mode`, `--emit-progress PATH`, `--post-run-cmd`. Default `--runs 3`.
* Extra tools in the build: `llama-fit-params` (memory planning without a full run),
  `llama-batched-bench`, `llama-perplexity` (quality check for KV-quant choices).

> The plan includes `mse-llm doctor flags` (§6), which diffs each `--help` against a stored snapshot.
> Rerun it after every llama.cpp rebuild instead of auditing by hand.

### 0.4 Build under test: llama.cpp for RTX 4080 Laptop (sm_89)

Target GPU: **NVIDIA GeForce RTX 4080 Laptop, 12 GB (12281 MiB), compute capability 8.9 (Ada)**, Windows 11,
Visual Studio 17 2022 generator, vcpkg toolchain.

Current procedure (manual, from `cmd`):

```bat
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
rem d:\MSE\GIT\llama.cpp>

rem CONFIGURATION
cmake -B build ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_CUDA=ON ^
  -DCMAKE_CUDA_ARCHITECTURES=89-real ^
  -DCMAKE_TOOLCHAIN_FILE=vcpkg\scripts\buildsystems\vcpkg.cmake ^
  -DGGML_CUDA_FA_ALL_QUANTS=ON ^
  -DLLAMA_BUILD_TESTS=OFF

rem BUILD
cmake --build build --config Release -j 16
```

Binaries land in `build\bin\Release\` (`llama-server.exe`, `llama-bench.exe`, `llama-fit-params.exe`,
`llama-perplexity.exe`, ...). `scripts\moe-cache-tune.py` comes from the same checkout.

What the options mean for benchmarking:

| Option | Why it matters here |
|---|---|
| `GGML_CUDA=ON` | CUDA backend |
| `CMAKE_CUDA_ARCHITECTURES=89-real` | SASS for sm_89 only, no PTX. Faster build and smaller binaries, but **the binary only runs on Ada GPUs**. Fine for this laptop; a different GPU needs a separate build profile |
| `GGML_CUDA_FA_ALL_QUANTS=ON` | **Required by the matrix.** Without it the CUDA Flash Attention kernels exist only for a few K/V type pairs (f16/q4_0/q8_0, same type on K and V). Asymmetric configs like `Kq8_Vq4`, `Kq4_Vq8`, and `q5_*`/`iq4_nl`/`f8_e4m3` would fall back or run slowly, so the results would measure a fallback path instead of the config. Costs build time |
| `CMAKE_TOOLCHAIN_FILE=vcpkg\...` | vcpkg dependencies (libcurl/openssl for `-hf` downloads). The path is **relative**, so it assumes `vcpkg\` is cloned inside the llama.cpp checkout. Use an absolute path in automation |
| `LLAMA_BUILD_TESTS=OFF` | Faster build |
| `CMAKE_BUILD_TYPE=Release` | **Ignored by the Visual Studio generator** (it is multi-config; CMakeCache shows it `UNINITIALIZED`). `--config Release` at build time is what selects Release |
| `-j 16` | Parallel build (MSBuild `/m`). Tune to the core count |

Defaults that this build picks up without being asked (seen in the reference build's
CMakeCache): `GGML_NATIVE=ON` (CPU code tuned for this laptop's CPU, so not portable) and
`GGML_CUDA_GRAPHS=ON`. They should be recorded with the results because they affect CPU-side
speed (`-ncmoe`, `--moe-cache-cpu-share`).

**CPU: AMD Ryzen 9 7945HX** (Zen 4, 16 cores / 32 threads, not hybrid). It supports AVX2 and
AVX-512 (F/BW/DQ/VL + VNNI, VBMI, BF16). The CMake cache entry **`GGML_AVX512:BOOL=OFF` is
misleading**:
* With MSVC and `GGML_NATIVE=ON`, `ggml/src/ggml-cpu/cmake/FindSIMD.cmake` compiles and runs test
  programs at configure time (`AVX512_FOUND:BOOL=TRUE` in the cache) and sets `GGML_AVX512` as a
  normal variable, which overrides the cache option. The CPU backend **is** built with
  `/arch:AVX512` (checked in the reference build's `ggml\src\ggml-cpu.vcxproj`:
  `EnableEnhancedInstructionSet=AdvancedVectorExtensions512` plus the `GGML_AVX512` define).
  Passing `-DGGML_AVX512=OFF` would therefore be ignored, and the build profile doesn't set it.
* FindSIMD only detects base AVX-512 under MSVC. **`GGML_AVX512_VNNI`, `_VBMI` and `_BF16` stay
  OFF** even though Zen 4 has them. Variant `zen4-avx512x` turns them on. It's **applied by default** (`apply_variants`), and
  `--without zen4-avx512x` builds without it, to
  A/B the CPU-side expert compute (`-ncmoe`, `-cmoe`, `--moe-cache-cpu-share`). GPU-only speed
  shouldn't change.
* `doctor` checks the built ISA, not the cache: it reads llama.cpp's `system_info` line
  (`AVX512 = 1 | AVX512_VNNI = ...`) from the server log, plus the generated vcxproj, and
  compares them with the CPU features `mse-llm hw` detects right now.

**Checkouts:** the findings in §0 come from the reference build (`a49db739f`, tag `b12468-5`,
same flags, vcpkg classic mode `x64-windows` with `curl[ssl,sspi]`, `openssl`, `zlib`). **From now
on only the clone in this repo is used:** `builds\llama.cpp`, built into `builds\llama.cpp\build\bin\Release`.
The bash script's `MSE_LLAMA_BIN_DIR` points there too. Older checkouts elsewhere aren't referenced.

#### Project-local checkout: `builds/` (default)

`mse-llm build` doesn't depend on a hand-made checkout. By default it **clones llama.cpp and vcpkg
into the mse-llm repo** if they're missing, and builds there:

```
mse-llm/
├── config/builds/*.toml          # profiles (tracked)
└── builds/                       # git-ignored (/builds/ in .gitignore), created on first `build run`
    ├── vcpkg/                    # shared by every llama.cpp checkout: curl + openssl, x64-windows
    ├── llama.cpp/                # clone of ggml-org/llama.cpp (partial clone, all tags)
    │   ├── build/                # target rtx4080-laptop (default, incl. zen4-avx512x)
    │   └── build-without-zen4-avx512x/   # comparison: rtx4080-laptop --without zen4-avx512x
    └── llama.cpp-<fork>/         # (only if a fork is ever needed) its own checkout
```

Configured in `_base-cuda-windows.toml`:
* `[paths] builds_root = "{project}/builds"`, overridable per machine with the env var
  **`MSE_LLM_BUILDS_ROOT`**.
* `[source] src = "{builds_root}/llama.cpp"`, `on_missing = "clone"`,
  `clone_filter = "blob:none"`. A partial clone keeps the full history and every `b<N>`/`vX.Y.Z`
  tag for `build versions` and `--ref`, but fetches file contents only for checked-out commits.
* `[vcpkg] root = "{builds_root}/vcpkg"`, `on_missing = "clone"`, `triplet = "x64-windows"`,
  `packages = ["curl[ssl,sspi]", "openssl"]`. The toolchain path becomes
  `{vcpkg_root}/scripts/buildsystems/vcpkg.cmake`.

The first `build run` therefore also builds curl/openssl in vcpkg once (~10–20 min, ~3 GB). Later
builds reuse it. A profile *can* set `src` / `[vcpkg].root` to another folder (then `on_missing`
isn't used), but this laptop's profile doesn't.

**Path length (Windows).** `LongPathsEnabled` is **0** on this laptop, so paths are limited to
260 characters. The deepest file in the reference build was 194 characters (CUDA
object and tlog files under `ggml\src\ggml-cuda.dir\Release\…`). Under
`D:\MSE\GIT\_MSE_git\mse-llm\builds\llama.cpp\build-without-zen4-avx512x` the same files are about 20
characters longer (~206), which still fits but with less margin. `doctor` and `build show`
estimate the deepest path (build-dir length + the longest relative path seen in the last build,
default 165) and **warn above 240**. The fix is a shorter `MSE_LLM_BUILDS_ROOT` (for example
`D:\b`) or enabling long paths in Windows. That's a system setting, so `mse-llm` only reports it
and never changes it.

**Decided:** `cuda_flash` is built from the new clone in `builds/llama.cpp` (§11 Q5). Other
llama.cpp versions for A/B tests come from the same clone (`--ref`, `--keep-previous`), not
from separate hand-made checkouts.

#### Build support in mse-llm (`mse-llm build`)

Builds are part of what's being measured, so the tool should build them reproducibly and record
which build produced each result.

**One TOML file per hardware target** in `config/builds/`. Adding a new machine or GPU later
means adding one file, without touching the code or the main config:

```
config/builds/
├── _cmake-flags.toml           # plain-language meaning of every CMake flag (used by build show/explain)
├── _base-cuda-windows.toml     # shared CUDA + VS2022 + vcpkg defaults (bases start with "_")
├── rtx4080-laptop.toml         # this laptop: sm_89, 12 GB, extends the base   ← exists now
├── (future) rtx3090-desktop.toml    # e.g. CMAKE_CUDA_ARCHITECTURES = "86-real"
├── (future) _base-cuda-linux.toml   # Ninja generator, no vcpkg
├── (future) rocm-*.toml / vulkan-*.toml / metal-*.toml   # GGML_HIP / GGML_VULKAN / GGML_METAL
└── (future) cpu-only.toml
```

Format (see `_base-cuda-windows.toml` and `rtx4080-laptop.toml`):
* Top-level keys `schema = 1`, `kind = "build-base" | "build"`, `extends = "<file>.toml"`, `build_dir`.
  Merge is a deep merge where the child wins; `[cmake]` tables merge key by key.
* `[target]`: only `gpu_match` / `cpu_match`, substrings of the GPU and CPU names this profile is
  meant for. `doctor` and `build run` compare them with what `mse-llm hw` detects, so a profile isn't
  built or benchmarked on the wrong machine without a warning, and `build run` without a target
  picks the matching profile. **No hardware facts are stored in profiles**: VRAM, bandwidth,
  PCIe, compute capability, CPU instruction sets, RAM channels and driver settings are detected at
  runtime (§5.1) and saved with each build manifest and run result, so they can't go stale.
  `build show` checks that `CMAKE_CUDA_ARCHITECTURES` covers the detected compute capability.
* `[paths]` (`builds_root`), `[source]` (repo, `src`, `ref`, `on_missing`, `clone_filter`),
  `[vcpkg]` (root, repo, ref, triplet, packages, bootstrap), `[toolchain]` + `[toolchain.require]` (generator, config, jobs,
  vcpkg, required versions), `[cmake]` (each key becomes `-D<KEY>=<VALUE>`; values are quoted
  strings like `"ON"`), `[smoke_test]`, `[outputs]`.
* `[engine]`: the engine profile (§3.3) that gets the new `bin_dir` after a successful build.
  `RunResult` links to the build profile name, so results are grouped per hardware.
* `[variants.<name>.*]`: A/B builds on the same hardware (today only `zen4-avx512x`). Each
  optional one becomes target `rtx4080-laptop+<variant>` with its own
  `build-<variant>` dir. **`apply_variants = [...]`** lists variants that are part of the default
  build; their flags still live in their own table (see "Default variants" below).
* Placeholders: `{project}`, `{builds_root}`, `{src}`, `{vcpkg_root}`, `{build_dir}`, `{bin}`, expanded to absolute paths. Windows paths use
  TOML literal strings (`'D:\some\path'`), so backslashes need no escaping.
* Loaded with the standard-library `tomllib` (no new dependency) and validated with a pydantic
  model (`mse_llm/build.py`). All config in the project is TOML.

Commands:

```
mse-llm build list                              # targets from config/builds (+variants)
mse-llm build clone [TARGET] [--vcpkg-only|--no-vcpkg]   # only prepare sources: clone llama.cpp + vcpkg if missing
mse-llm build plan TARGET [--json]              # preview: every command that `build run` would execute, nothing runs
mse-llm build run TARGET [--step] [--yes] [--only STEP|--from STEP]   # shows the plan, asks, then runs
mse-llm build show rtx4080-laptop               # merged profile + resulting cmake argv
mse-llm build run rtx4080-laptop [--pull] [--ref REF] [--clean] [--dry-run]
mse-llm build run rtx4080-laptop --without zen4-avx512x     # comparison build without a default variant
mse-llm build info rtx4080-laptop               # manifest: commit, flags, CUDA/MSVC versions, binary hashes
mse-llm build new NAME --from rtx4080-laptop    # scaffold a profile for new hardware
mse-llm build versions [--remote] [--limit N]   # llama.cpp versions you can build + what is already built
mse-llm build explain [FLAG ...]                # plain-language meaning of CMake flags (all if none given)
```

#### Build parameter printout (`build show`, and the header of every `build run`)

`build run` always prints this block **before** it starts, and stores it as `build-plan.txt` next to
`build-manifest.json`, so you can see exactly what is compiled and why. `build show` prints the same
block without building. Explanations come from `config/builds/_cmake-flags.toml` (editable). Every
value is labelled with where it came from: `base` (`_base-cuda-windows.toml`), `profile`
(`rtx4080-laptop.toml`), `variant` (`zen4-avx512x`, a default variant), or `auto` (detected by CMake, not passed).

```
$ mse-llm build show rtx4080-laptop

══ BUILD PLAN: rtx4080-laptop (+ default variant zen4-avx512x) ═════════════════════════════════
Hardware   NVIDIA GeForce RTX 4080 Laptop GPU (cc 8.9, 12281 MiB)   ✓ matches detected GPU
           AMD Ryzen 9 7945HX (16C/32T, AVX-512 + VNNI/VBMI/BF16)  ✓ matches detected CPU
Source     D:\MSE\GIT\_MSE_git\mse-llm\builds\llama.cpp   ref master → a49db739f (b12468-5, 114 commits after v1.0.0)
Build dir  D:\MSE\GIT\_MSE_git\mse-llm\builds\llama.cpp\build   (new)
Toolchain  CMake 4.2.0 · CUDA 12.6.77 · Visual Studio Community 2022 17.11.5 · driver 617.42
           vcpkg 0df93a05b9 (2026-11-16)

── CMake flags ──────────────────────────────────────────────────────────────────────────
 FLAG                        VALUE     FROM     WHAT IT MEANS
 build
  CMAKE_BUILD_TYPE           Release   base     ignored by VS generator; --config Release is used
  CMAKE_TOOLCHAIN_FILE       …vcpkg.cmake base  find vcpkg libraries (curl/openssl)
  LLAMA_BUILD_TESTS          OFF       base     skip llama.cpp unit tests → faster build
 gpu
  GGML_CUDA                  ON        base     build the NVIDIA CUDA backend
  CMAKE_CUDA_ARCHITECTURES   89-real   profile  kernels for Ada (RTX 40xx) only
  GGML_CUDA_FA_ALL_QUANTS    ON        base     Flash Attention for every K/V cache type pair (needed for Kq8_Vq4 …)
  GGML_CUDA_GRAPHS           ON        profile  replay recorded GPU work per token → less launch overhead (TG)
 cpu
  GGML_NATIVE                ON        profile  optimize CPU code for this processor (not portable)
  GGML_AVX512                (ON)      auto     detected by FindSIMD → /arch:AVX512 (cache still shows OFF)
  GGML_AVX512_VNNI           ON        variant  8-bit int dot product in 1 instruction → faster CPU expert compute
  GGML_AVX512_VBMI           ON        variant  byte shuffles; only used for Q2_0 unpack → ~no effect on Q4 models
  GGML_AVX512_BF16           ON        variant  bfloat16 math; matters for BF16 tensors only

── Differences vs the comparison build (--without zen4-avx512x) ────────────────────────
  + GGML_AVX512_VNNI=ON  + GGML_AVX512_VBMI=ON  + GGML_AVX512_BF16=ON
  Affects: CPU-side work only (-ncmoe / -cmoe / --moe-cache-cpu-share). GPU-only runs: no change expected.

── Commands ────────────────────────────────────────────────────────────────────────────
  cmake -S D:\MSE\GIT\_MSE_git\mse-llm\builds\llama.cpp -B D:\MSE\GIT\_MSE_git\mse-llm\builds\llama.cpp\build -G "Visual Studio 17 2022" ^
        -DCMAKE_TOOLCHAIN_FILE=D:\MSE\GIT\_MSE_git\mse-llm\builds\vcpkg\scripts\buildsystems\vcpkg.cmake ^
        -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89-real ^
        -DGGML_CUDA_FA_ALL_QUANTS=ON -DLLAMA_BUILD_TESTS=OFF -DGGML_NATIVE=ON -DGGML_CUDA_GRAPHS=ON ^
        -DGGML_AVX512_VNNI=ON -DGGML_AVX512_VBMI=ON -DGGML_AVX512_BF16=ON
  cmake --build D:\MSE\GIT\_MSE_git\mse-llm\builds\llama.cpp\build --config Release -j 16
═════════════════════════════════════════════════════════════════════════════════════════
```

Options: `--explain/--no-explain` (drop the meaning column), `--json` (machine-readable, the same
data that goes into the manifest), `--diff OTHER` (compare two targets, e.g.
`build show rtx4080-laptop --diff "rtx4080-laptop --without zen4-avx512x"`). A flag missing from
`_cmake-flags.toml` is shown with `?` and a hint to add it, never left unexplained.

**After the build**, `build run` checks the flags really got in (they can be silently overridden,
as `GGML_AVX512` shows): it reads `CMakeCache.txt`, the generated `ggml-cpu.vcxproj`
(`EnableEnhancedInstructionSet`, `PreprocessorDefinitions`), and the `system_info` line printed by
`llama-bench`/`llama-server` (`AVX512 = 1 | AVX512_VNNI = 1 | …`). Then it prints:

```
── Verified in binaries ─────────────────────────────────────────────────────────────────
  CPU: /arch:AVX512 ✓  AVX512_VNNI ✓  AVX512_VBMI ✓  AVX512_BF16 ✓   (system_info)
  GPU: sm_89 ✓  FA all quants ✓   (llama-bench --list-devices, CMakeCache)
  Built  b12468-5-ga49db739f in <duration> → D:\MSE\GIT\_MSE_git\mse-llm\builds\llama.cpp\build\bin\Release
```

#### Preview before running: `build plan` / `build run --dry-run`

You should know exactly what will happen before anything is cloned, downloaded or compiled.
`build plan TARGET` (same as `build run TARGET --dry-run`) **executes nothing**. It only reads the
profiles, checks which folders exist, and prints every step in order: the exact command, what it
does in plain words, whether it will run or be skipped and why, and roughly how long and how much
disk it costs. Every action is numbered, and the commands can be copied and run by hand.

```
$ mse-llm build plan rtx4080-laptop

══ WHAT `build run rtx4080-laptop` WILL DO (nothing has been executed) ══════════════════
Profile   config/builds/rtx4080-laptop.toml  (extends _base-cuda-windows.toml)
Root      D:\MSE\GIT\_MSE_git\mse-llm\builds   (from [paths].builds_root; MSE_LLM_BUILDS_ROOT not set)

 #  STEP                 STATUS   COMMAND / ACTION
 1  Clone llama.cpp      RUN      git clone --filter=blob:none https://github.com/ggml-org/llama.cpp
                                    D:\MSE\GIT\_MSE_git\mse-llm\builds\llama.cpp
                                  why: builds\llama.cpp does not exist  (on_missing = "clone")
                                  cost: network download, ~0.5 GB disk
 2  Checkout version     RUN      git -C …\builds\llama.cpp checkout master
                                  → builds whatever master is at clone time (shown after step 1)
 3  Clone vcpkg          RUN      git clone https://github.com/microsoft/vcpkg D:\…\builds\vcpkg
                                  why: builds\vcpkg does not exist
 4  Bootstrap vcpkg      RUN      D:\…\builds\vcpkg\bootstrap-vcpkg.bat -disableMetrics
                                  what: downloads the vcpkg.exe tool from GitHub (Microsoft)
 5  Install libraries    RUN      D:\…\builds\vcpkg\vcpkg.exe install curl[ssl,sspi] openssl --triplet x64-windows
                                  what: compiles curl + openssl from source (for -hf downloads / https)
                                  cost: ~10–20 min, ~3 GB disk
 6  Configure (CMake)    RUN      cmake -S …\builds\llama.cpp -B …\builds\llama.cpp\build -G "Visual Studio 17 2022"
                                    -DCMAKE_TOOLCHAIN_FILE=…\builds\vcpkg\scripts\buildsystems\vcpkg.cmake
                                    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON
                                    -DLLAMA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=89-real
                                    -DGGML_NATIVE=ON -DGGML_CUDA_GRAPHS=ON
                                    -DGGML_AVX512_VNNI=ON -DGGML_AVX512_VBMI=ON -DGGML_AVX512_BF16=ON  (default variant zen4-avx512x)
                                  what: checks the compilers, picks CPU/GPU features, writes the VS solution
                                  (meaning of every flag: `mse-llm build show rtx4080-laptop`)
 7  Compile              RUN      cmake --build …\builds\llama.cpp\build --config Release -j 16
                                  cost: long (FA_ALL_QUANTS compiles many CUDA kernels), several GB disk
 8  Smoke test           RUN      …\build\bin\Release\llama-server.exe --version
                                  …\build\bin\Release\llama-bench.exe --list-devices   (expects RTX 4080 Laptop)
 9  Verify flags         RUN      read CMakeCache.txt, ggml-cpu.vcxproj, system_info (AVX512 …)
10  Write manifest       RUN      …\build\bin\Release\build-manifest.json + build-plan.txt
11  Help snapshot        RUN      llama-server --help / llama-bench --help → diff vs previous build
12  Update engine        RUN      engine "cuda_flash".bin_dir → …\builds\llama.cpp\build\bin\Release
                                  (same folder the bash script's MSE_LLAMA_BIN_DIR uses)

Checks    ✓ RTX 4080 Laptop detected  ✓ Ryzen 9 7945HX detected  ✓ git 2.51.0  ✓ CMake 4.2.0
          ✓ CUDA 12.6.77  ✓ VS 2022 17.11.5  ✓ free disk on D: …
          ⚠ Windows long paths disabled: deepest path ≈ 215 of 260 chars (OK, little margin)
Will NOT  touch anything outside D:\MSE\GIT\_MSE_git\mse-llm\builds
          change any Windows setting, or delete anything

Run it:   mse-llm build run rtx4080-laptop          (asks once before step 1)
          mse-llm build run rtx4080-laptop --step   (asks before every step)
═════════════════════════════════════════════════════════════════════════════════════════
```

When the folders already exist, steps 1, 3, 4 and 5 show **SKIP** with the reason (for example
`builds\vcpkg exists, curl + openssl already installed`), and step 6 shows **SKIP** when
CMakeCache already matches the requested flags. You can see in advance that a rebuild only
recompiles.

How it behaves:
* **`build run` always shows this plan first and asks `Proceed? [y/N]`.** `--yes` skips the
  question, for scripts. Nothing runs without either the prompt or `--yes`.
* **`--step`** asks before each numbered step (`[y]es / [s]kip / [q]uit`), so you can watch one
  step finish before allowing the next.
* **`--only STEP` / `--from STEP`**: run just one step or resume from one (`--only clone`,
  `--from configure`), for example to clone first, look around, and build later.
* **`--dry-run --json`**: the same plan as data, used by tests (golden plans per profile) and stored
  as `build-plan.json`.
* While running, each step prints `▶ [6/12] Configure (CMake)` with the exact command, streams
  its output (or summarizes it with `--quiet`, keeping the full log in
  `builds\logs\<target>-<timestamp>.log`), and ends with `✓ done in 42 s` or `✗ failed (exit 1)`
  plus the last 30 log lines. A failure stops the run and says how to resume (`--from <step>`).
* `build clone TARGET --dry-run` previews only steps 1–5.

The same preview rule applies to the other commands that start long or heavy work:
`bench`/`benchy`/`sweep run`/`serve`/`tune moe-cache` accept **`--dry-run`**, which prints the
exact `llama-server`/`llama-bench`/`llama-benchy` command lines, the variant decoded into flags,
ports, the expected number of runs and the estimated duration, without starting anything
(the bash `MSE_SERVER_PRESETS_DUMP` idea, generalized).

#### Buildable versions (`build versions`)

llama.cpp has two kinds of version tags: **daily build tags** `b<N>` (for example `b12468`, several
per day) and, since 1.0, **semantic releases** `vX.Y.Z` (`v1.0.0` = 2026-11-13). `build versions`
lists both from the local checkout (`git tag`); `--remote` runs `git fetch --tags` first. Each
version also shows whether it is already built and with which profile, from the
`build-manifest.json` files:

```
$ mse-llm build versions --limit 8

llama.cpp at D:\MSE\GIT\_MSE_git\mse-llm\builds\llama.cpp   checked out: a49db739f (b12468-5), branch master
Releases
  v1.0.0     2026-11-13   ← your checkout is 114 commits newer
  v0.9.0     …
Build tags (newest first)
  b12468     2026-11-17   ← your checkout is 5 commits after this
  b12466     2026-11-17
  b12465     2026-11-17
  b12464     2026-11-16
Built here
  rtx4080-laptop          (not built yet)        build\bin\Release
Toolchain  CMake 4.2.0 · CUDA 12.6.77 · VS 2022 17.11.5 · git 2.51.0

Build one:  mse-llm build run rtx4080-laptop --ref v1.0.0
```

`build run --ref <tag|commit>` checks out that version (detached, the checkout must be clean) and
builds it into `build-<ref>` when `--keep-previous` is set, so two llama.cpp versions can be
benchmarked side by side as separate engine profiles. `build versions` also warns when the
installed CUDA is older than the version requires (from llama.cpp's CMake minimum).

`build run` does the following (async subprocess, streamed log, no shell):
1. **Sources.** If `src` is missing and `on_missing = "clone"`:
   `git clone --filter=blob:none <repo> <src>`. With `on_missing = "error"`, stop with a hint.
   Otherwise an optional `git fetch --tags` (`--pull`), then `git checkout <ref>`. Refuses to run
   on a dirty tree unless `--allow-dirty`. Prints the commit and `git describe` it will build.
1b. **vcpkg.** If `[vcpkg].root` is missing: `git clone <vcpkg repo> <root>`, checkout `ref`, run
   `bootstrap-vcpkg.bat -disableMetrics` (downloads the vcpkg tool), then
   `vcpkg install curl[ssl,sspi] openssl --triplet x64-windows`. If the root exists, check that
   the packages are installed (`vcpkg list`) and install only the missing ones. The first run takes
   a while and uses about 3 GB, so it's announced before it starts, with `--yes` to skip the prompt.
2. `cmake -B <build_dir> -G <generator> -DCMAKE_TOOLCHAIN_FILE=... -D...` (configure; skipped when
   CMakeCache matches the requested defs, unless `--clean`).
3. `cmake --build <build_dir> --config Release -j <jobs>`.
4. Smoke test: `llama-server --version`, `llama-bench --list-devices` (expects the sm_89 device).
5. Writes `build-manifest.json` next to the binaries: git commit + `git describe`, all CMake defs
   (from CMakeCache), nvcc / MSVC / CMake versions, driver version (NVML), SHA-256 of the main
   binaries, timestamp.
6. Captures `--help` of server/bench into a new help snapshot and runs `doctor flags` against the
   previous one, so new, removed and changed flags show up immediately after each `git pull`.
7. Points the engine profile at the new `bin_dir` (keeping the old build dir if `--keep-previous`,
   for A/B runs).

Every `RunResult.build` (§8) stores the manifest's commit, CMake defs and binary hash. `results
compare` then refuses (or warns) when two runs come from different builds and the difference
isn't the point of the comparison.

`doctor` checks for this build: `nvcc` on PATH and CUDA version, CMake ≥ 3.21, VS 2022 C++ workload,
vcpkg toolchain file exists, `GGML_CUDA_FA_ALL_QUANTS=ON` in CMakeCache (warns if a K/V sweep
targets a build without it), and `CMAKE_CUDA_ARCHITECTURES` covers the detected GPU.

Variants defined in `rtx4080-laptop.toml`: only `zen4-avx512x` (AVX-512 VNNI/VBMI/BF16,
**default**). Not planned:
* `GGML_CUDA_FORCE_MMQ`: no effect on Ada, because llama.cpp already uses MMQ there.
* `GGML_CUDA_FORCE_CUBLAS`: only changes prompt processing, usually for the worse, since
  generation at batch ≤ 8 uses the MMVQ kernels either way. Checked in
  `ggml-cuda/mmq.cu` / `mmvq.cu`; both variants were removed on 2026-11-18.
* The turboquant fork (turbo* cache types dropped).

---

## 1. Goals and constraints

1. **Feature parity** with the bash script (inventory in §2).
2. **Async first**: `asyncio` subprocesses, concurrent monitor sampling, health polling, and
   log streaming in one event loop. No busy shells.
3. **Platform independent**: no Git Bash, `ps`, `tasklist`, `sed`, or `powershell` shell-outs.
   Windows is primary; Linux and macOS must work wherever the backend exists.
4. **Pluggable runners**: llama.cpp now, Strata / vLLM / Ollama / LM Studio later.
5. **Config, not code**: models, variants, and sweeps live in TOML. No more editing functions and
   commenting lines out.
6. **Structured results**: every run writes JSON and is appended to a results store, so charts
   come from data instead of regex over console logs.

---

## 2. Feature inventory (bash → Python)

| # | Bash feature | Python home |
|---|---|---|
| F1 | `run_HELP_server/bench` dump help | `mse-llm help {server,bench,benchy}` + `doctor flags` |
| F2 | `MODELS`, `DRAFTER_MODELS`, `*_EXTRA` maps | `config.toml` `[models.*]` |
| F3 | `EXTRAS` matrix (FA × K/V × batch × moe/spec variants), `_moeN/_sdNcache/_sdNmtp` name parsing | `variants.py`: typed axes + name codec (§4.2) |
| F4 | `llama-bench` runs with monitoring, list wrappers (`_wrapper_EXTRAS`, `_wrapper_PROMPTS`) | `mse-llm bench` |
| F5 | `llama-benchy` against a background server | `mse-llm benchy` |
| F6 | Background server lifecycle: spawn, `/health` poll, PID tree kill | `runners/base.py` `ServerHandle` async context manager |
| F7 | `llama-server` foreground serve with filtered output | `mse-llm serve` |
| F8 | Router serve with `--models-preset` + preset dump | `mse-llm presets {serve,dump,generate,validate}` |
| F9 | GPU VRAM / shared mem / RAM / temp monitoring + status thresholds | `monitor/` (§5) |
| F10 | Log parsing → 5-axis matplotlib chart | `report/plot.py` reading JSON results (legacy log parser kept for old logs) |
| F11 | AI analyst prompt dumps | `mse-llm report prompt` (templated, Jinja2) |
| F12 | inspect-ai evals (mmlu_pro, gpqa_diamond, math_aime_24, swe_bench_verified) | `mse-llm eval` |
| F13 | Windows TTS notification on completion | `notify.py` (pyttsx3 / SAPI on Windows, `notify-send`/`say` elsewhere, optional toast) |
| F14 | Output modes `all/filter/none` + filter regex list | `--output all\|filter\|none`, filters in config |
| F15 | `MSE_*` env knobs (threads, parallel, TG, PP sync, depth, runs, timeout, debug) | Config fields; env override `MSE_LLM_*`; CLI flags win |

---

## 3. Architecture

### 3.1 Layout (flat)

```
mse-llm/
├── pyproject.toml            # uv project, [project.scripts] mse-llm = "mse_llm.cli:main"
├── uv.lock
├── config/
│   ├── builds/               # one TOML per hardware target (§0.4): _cmake-flags.toml, _base-cuda-windows.toml, rtx4080-laptop.toml
│   ├── config.example.toml   # committed
│   └── presets.ini           # generated or hand-written router presets
├── mse_llm/
│   ├── __init__.py
│   ├── cli.py                # click group + subcommands (thin)
│   ├── config.py             # pydantic-settings models, TOML load, env/CLI merge
│   ├── variants.py           # variant axes, name codec, matrix expansion
│   ├── runners/
│   │   ├── __init__.py       # registry (entry points: "mse_llm.runners")
│   │   ├── base.py           # Runner / ServerRunner / BenchRunner protocols, ServerHandle
│   │   ├── llamacpp.py       # llama-server, llama-bench, router presets
│   │   ├── benchy.py         # llama-benchy client (any OpenAI-compatible server)
│   │   ├── inspect_eval.py   # inspect-ai client (any OpenAI-compatible server)
│   │   └── strata.py         # stub → later (§7)
│   ├── proc.py               # async subprocess, process-tree kill (psutil), log tee/filter
│   ├── health.py             # async readiness probes (/health, /v1/models, /metrics)
│   ├── hw/                   # PC capability detection, identity, fit (§5.1)
│   ├── monitor/
│   │   ├── __init__.py       # Monitor async context manager, Sample, Summary
│   │   ├── nvidia.py         # pynvml: VRAM, temp, power, clocks, util, throttle reasons
│   │   ├── windows.py        # PDH counters: GPU shared/non-local memory (no powershell)
│   │   ├── system.py         # psutil: RAM, swap, CPU, per-process RSS of server
│   │   └── amd.py / apple.py # later
│   ├── results.py            # RunResult schema, JSONL/SQLite store, run IDs
│   ├── parsers/
│   │   ├── llama_bench.py    # -o jsonl parsing (primary)
│   │   ├── benchy.py         # --format json parsing
│   │   ├── server_log.py     # memory breakdown, n_ctx, offload, KV size, fit decisions
│   │   └── legacy_log.py     # port of the current regex parser (old .txt logs)
│   ├── report/
│   │   ├── plot.py           # 5-axis chart (port), plus new views
│   │   ├── table.py          # rich tables, best-config ranking
│   │   └── prompts/          # Jinja2 templates for AI analyst prompts
│   ├── notify.py
│   ├── build.py              # llama.cpp git + cmake build, build-manifest.json
│   └── doctor.py             # env checks, help snapshot diff, flag validation
└── tests/
    ├── conftest.py
    ├── fixtures/             # captured --help outputs, bench jsonl, server logs, old .txt logs
    ├── fake_server.py        # aiohttp/uvicorn fake OpenAI server for lifecycle tests
    ├── test_variants.py
    ├── test_cmd_build.py     # golden command lines (parity with bash)
    ├── test_parsers.py
    ├── test_proc.py
    ├── test_monitor.py       # with a fake sampler
    └── test_cli.py           # click CliRunner
```

### 3.2 Dependencies

* Core: `click`, `pydantic>=2`, `pydantic-settings`, `httpx` (async), `psutil`, `rich`,
  `anyio` (structured concurrency / task groups), `jinja2`. All config is TOML, read with stdlib `tomllib`.
* Monitoring / hw detection: `nvidia-ml-py` (pynvml), `pywin32` (PDH) on Windows only (environment
  marker), `gguf` (GGUF metadata for fit estimates). CPU topology and power use stdlib `ctypes`/`winreg`.
* Report: `matplotlib`, `numpy`, optional `pandas`. Optional extra `[report]` keeps the core light.
* Tools run through `uvx` as today: `llama-benchy`, `inspect-ai` (+ `inspect-evals`, `openai`).
  Pin their versions in config so results stay reproducible.
* Dev: `pytest`, `pytest-asyncio` (or anyio plugin), `ruff`, `mypy`/`pyright`.

Python ≥ 3.12 (`asyncio.TaskGroup`, `tomllib`).

### 3.3 Runner abstraction

```python
class ServerRunner(Protocol):
    name: str                                  # "llamacpp", "strata", ...
    capabilities: set[str]                     # {"serve", "router", "spec", "kv-quant", "metrics", ...}
    def build_command(self, model: ModelCfg, variant: Variant, net: NetCfg) -> list[str]: ...
    async def start(self, ...) -> ServerHandle: ...      # spawn + readiness
    def readiness_probe(self) -> Probe: ...             # /health for llama.cpp, /v1/models or /metrics for others
    def parse_startup_log(self, lines) -> StartupInfo: ...  # n_ctx, KV size, offload, fit adjustments

class BenchRunner(Protocol):                   # llama-bench (in-process, no server)
    async def run(self, model, variant, sweep) -> list[RunResult]: ...

class ClientBench(Protocol):                   # benchy, inspect: work against ANY OpenAI endpoint
    async def run(self, endpoint: Endpoint, ...) -> list[RunResult]: ...
```

Main split: **server engines** (llama.cpp, Strata, vLLM...) vs **clients** (benchy, inspect,
future: own load generator). Any client works against any server engine through the
OpenAI-compatible endpoint. Adding Strata means writing one `ServerRunner`.

`ServerHandle` is an `async with` context manager:
spawn → stream stdout/stderr to log file + filtered console → wait for readiness (timeout, also
fails fast if the process exits) → yield endpoint → on exit: graceful terminate (CTRL_BREAK on
Windows / SIGTERM) → wait → `psutil` kill of the whole process tree → **verify the port is free**
before the next run. This replaces `server_stop_windows`, `server_get_pid_by_ppid`, `tasklist`
loops, and `/tmp` logs.

As built (M5): the server runs in its own process group (CREATE_NEW_PROCESS_GROUP), so Ctrl+C reaches
only mse-llm, which stops the tree itself. CTRL_BREAK is the only console signal that can be sent to
another group; llama-server handles just Ctrl+C, so CTRL_BREAK ends it through the default handler at
once (~0.7 s, no kill needed). An attached server is also in a **kill-on-close Job Object**: if mse-llm
dies, Windows ends the server and every router child with it. A `--background` router (must outlive
mse-llm) gets a hidden console instead and is stopped by PID from `results/servers/router-<port>.json`
(create time checked, so a reused PID is never killed). A killed process holding ~20 GB (a model loaded
with `-lm none`) needs ~8 s to end, so the stop waits up to 20 s after the kill.

Engine profiles (what `BENCHES`/`SERVERS`/`SERVERS_NET_CFG` are today). Each build profile's
`[engine]` table (§0.4) declares one, and `build run` updates its `bin_dir`. It points at the
repo's own build, the same folder as the bash script's `MSE_LLAMA_BIN_DIR`:

```toml
[engines.cuda_flash]
runner = "llamacpp"
bin_dir = '{project}\builds\llama.cpp\build\bin\Release'
host = "0.0.0.0"
port = 8033
help_snapshot = "fixtures/help/a49db739f"   # used by doctor + flag validation
```

A capability check validates every built command against the engine's parsed `--help` (flag
exists, enum value allowed, e.g. `ctk ∈ {...}`). Bad combos (an unsupported cache type, `--spec-type`
on llama-bench) fail **before** a 10-minute run starts.

---

## 4. Configuration model

### 4.1 `config.toml`

```toml
[defaults]
threads = 4
parallel = 1
ngl = "all"
fit = "off"                 # reproducible benches
load_mode = "none"
bench_tg = [128]
bench_repetitions = 1       # llama-bench -r (bash used implicit 5)
server_ready_timeout = 120
notify = true

[defaults.benchy]
pp = [512, 2048, 8192, 32768]
pp_sync_ctx = true
depth = [0]
runs = 1
version = "llama-benchy==X.Y.Z"

[models.gemma_4_26B_A4b_QAT_UD_Q4_K_XL]
path = 'd:\.lmstudio\models\unsloth\gemma-4-26B-A4B-it-qat-GGUF\gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf'
alias = "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL"
draft = 'd:\.lmstudio\models\unsloth\gemma-4-26B-A4B-it-qat-GGUF\mtp-gemma-4-26B-A4B-it.gguf'   # or "builtin"
url = "https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF"
tags = ["moe", "vision", "reasoning"]
serve_extra = { reasoning-format = "auto", temp = 1.0, top-k = 64, top-p = 0.95 }
default_serve = "fa_Kq8_Vq8_b2048_ub2048_moe16"
default_serve_spec = "fa_Kq8_Vq8_b2048_ub2048_moe21_sd4mtp"
notes = "Batching does NOT help (flat)."

[sweeps.gemma_qat_moe]              # replaces run_BENCH_* / run_BENCHY_* functions
model = "gemma_4_26B_A4b_QAT_UD_Q4_K_XL"
engine = "cuda_flash"
tool = "benchy"                     # bench | benchy | eval
ctx = 65536
variants = [
  "fa_Kq8_Vq8_b2048_ub2048_moe21_sd{3,4,5,6,10}mtp",   # brace expansion
  "fa_Kq8_Vq8_b2048_ub2048_moe{18,21}",
]
# or a structured matrix:
# matrix = { ctk=["q8_0"], ctv=["q8_0","q4_0"], b_ub=[[2048,2048],[4096,4096]], n_cpu_moe=[18,21,26] }

[sweeps.gemma_qat_moe.annotations]  # bash comments like "PC freeze" become data
"fa_Kq8_Vq8_b8192_ub8192_moe31" = { status = "skip", reason = "PC freeze (ctx 65536)" }
"fa_Kq8_Vq8_b2048_ub2048_moe16" = { status = "sweetspot" }
```

### 4.2 Variants

Variant = typed object (`fa`, `ctk`, `ctv`, `b`, `ub`, `n_cpu_moe`, `spec_type`, `spec_n_max`,
`draft`, `moe_cache_mib`, ... + free `extra: list[str]`). Two-way codec:

* `decode("extra_fa_Kq8_Vq4_b4096_ub4096_moe41_sd3cache")` → Variant. The legacy `extra_` prefix
  is accepted, so old names and logs keep working.
* `encode(variant)` → canonical short name used in run IDs, charts, and file names.
* Matrix expansion (cartesian product with filters) replaces `EXTRAS_add_batch_variants`, without
  materialising ~1000 dict entries up front.
* Tokens for new flags: `mc{N}` = `--moe-cache-mib N`, `fitt{N}`, `lm{mode}`, `Kf8` = f8_e4m3, `sdN{type}`
  for any spec type (`sd4eagle3`, `sd3ngrammod`, ...).

### 4.3 Presets: multi-model server from an INI file (`mse-llm presets …`)

This replaces `run_PRESETS_serve` / `run_presets_serve`, which ran
`llama-server --models-preset <ini> --host 0.0.0.0 --port 8033`. llama-server then runs as a
**router**: one HTTP endpoint, one child `llama-server` per model, started on demand and picked by
the `"model"` field of each request. Switching models in a client needs no restart.

**Which INI file.** Resolved in this order: `--file PATH` → `[presets].file` in `config.toml` →
`config/presets.ini`. Until M13 the config points at the **existing `llama_cpp_presets.ini`** in
the repo root. This is the fixed file, which the bash script's `MSE_PRESETS_SERVE_FILE` now also
points at. It's the only presets file; nothing outside this repo is used. Today's file works unchanged:

```toml
[presets]
file = "llama_cpp_presets.ini"        # later: "config/presets.ini" (generated)
engine = "cuda_flash"                  # which llama-server binary
host = "0.0.0.0"
port = 8033
models_max = 1                         # 12 GB VRAM: one large model at a time
memory_margin_mib = 1024               # router unloads LRU models to keep this much VRAM free
autoload = true                        # load a model on its first request
load_on_start = []                     # e.g. ["Qwen3.6-35B-A3B-UD-Q4_K_M-MTP"]
```

**Commands:**

```
mse-llm presets list     [--file F]           # sections, model file found?, key args, router-owned keys
mse-llm presets validate [--file F]           # every key vs the engine's --help; fails on invalid keys
mse-llm presets fix      [--file F] [--write] # show the fixes as a diff; --write applies them (keeps .bak)
mse-llm presets dump     [--file F] [NAME]    # merged [*] + section → equivalent single-model llama-server argv
mse-llm presets generate [--out F]            # write an INI from [models.*] + default_serve variants
mse-llm presets serve    [--file F] [--port N] [--models-max N] [--memory-margin MiB]
                         [--no-autoload] [--load NAME ...] [--foreground|--background] [--dry-run]
mse-llm presets status                         # GET /models: loaded/unloaded, args, memory min/max/effective
mse-llm presets load NAME | unload NAME        # POST /models/load | /models/unload
mse-llm presets watch                          # follow /models/sse: loads, unloads, memory measurements
mse-llm presets stop                           # stop a --background router (whole process tree)
mse-llm serve  --preset NAME [--file F]        # ONE model, outside the router, with that section's args
mse-llm benchy --preset NAME [--file F]        # benchmark exactly the config the router serves
```

**`presets validate` / `presets fix` on today's file.** These are the findings the check would
report (verified against this build, §0.1 and the server README):

```
$ mse-llm presets validate

llama_cpp_presets.ini → engine cuda_flash (a49db739f)
 ✗ [global defaults]  section name: globals must be [*]; this becomes a model preset without a model
 ✗ [global defaults]  no-mmap = 1   flag removed in this build ("invalid argument: --no-mmap") → load-mode = none
 ⚠ [global defaults]  fa = 1        accepted, but the flag now takes on|off|auto → fa = on
 ⚠ [global defaults]  jinja = 1     already the default (enabled)
 ⚠ [global defaults]  ngl = 999     ok; 'all' is the explicit spelling
 ⚠ (file)             no "version = 1" line
 ℹ every section       alias = …     used by the router as an extra model name (log: "aliases: …");
                                     the section name is the main id
 ✓ 5 model files found (d:\.lmstudio\models\…)
 ✓ all other keys exist in llama-server --help (ctk, ctv, b, ub, ctx-size, temp, top-k, top-p,
   reasoning-format, n-cpu-moe, threads, parallel)
Result: 2 errors, 4 warnings.  `mse-llm presets fix` shows the corrections.
```

> Checked on 2026-11-18 with the real router (`--no-models-autoload`, port 18033). The original
> file fails at startup: `failed to initialize router models: option 'no-mmap' not recognized in
> preset 'global defaults'`. With `[*]`, `load-mode = none`, `fa = on` and `version = 1` (now
> applied to `llama_cpp_presets.ini`), the router starts and lists all 5 models. The bash script's
> `MSE_PRESETS_SERVE_FILE` was switched to this repo file, and `run_PRESETS_serve` was checked on
> a test port: healthy in 1 s, 5 models listed.

`presets fix --write` rewrites only those lines. It keeps comments, the commented-out
`#[Qwen3.6-35B-A3B-UD-Q4_K_M]` block and the original command-line comments, and saves
`llama_cpp_presets.ini.bak` first.

**`presets serve` behaviour:**
* It always runs `validate` first and refuses to start on errors (`--force` to try anyway).
* It prints the exact command and settings before starting, and `--dry-run` stops there:

  ```
  ▶ llama-server --models-preset D:\…\llama_cpp_presets.ini --host 0.0.0.0 --port 8033
                 --models-max 1 --models-memory-margin 1024
    engine     cuda_flash → …\bin\Release\llama-server.exe (a49db739f)
    models     Qwen3.6-28B-REAP20-A3B-Q4_K_M, Qwen3.6-35B-A3B-UD-Q4_K_M-MTP,
               gemma-4-26B-A4B-it-qat-UD-Q4_K_XL, gemma-4-26B-A4B-it-UD-Q4_K_M, Tesslate_OmniCoder-9B-Q4_K_M
    endpoint   http://localhost:8033/v1   (model = section name)
  ```

* **Foreground** (default, like the script): streams router and child logs with the same filter
  as `serve --output filter`. Ctrl+C stops the router **and all children**.
* **Background**: uses the same `ServerHandle` as benches (§3.3). It waits for `/health`, then
  for every `--load NAME` to reach `loaded` (by polling `GET /models`; `presets watch` follows
  `/models/sse`), prints the endpoint and returns.
  `presets stop` ends it. The PID file and logs go under `results/servers/`.
* Before starting, it checks the port is free and no other engine is running (§5.1), and checks
  free VRAM against `memory_margin_mib`.
* Monitoring (§5) follows the **router's whole process tree** (foreground mode), so each child
  model's VRAM and shared memory is attributed via the PDH per-process counters (NVML has no
  per-process VRAM on Windows WDDM). `presets status` also shows the router's own
  `memory.min/max/effective` per model.

**Things the router does differently from single-model `serve`** (shown by `presets dump` and
`list`, so nothing is surprising):
* `host`, `port`, `api-key` in the INI are router-owned and ignored. That's why the INI comments
  say "host/port NOT SUPPORTED". `alias` **is** used, as extra model names. This was checked:
  the router log lists `aliases: …` per preset.
* The router instances run with `--fit on` unless the preset sets `fit = off`. A preset that
  pins `ctx-size` (all of yours do) keeps that context. Without it, `--fit` would pick the largest
  context that fits at load time. As with `serve` (§0.3), a preset that sets `ngl` or `n-cpu-moe`
  (all of yours do) leaves `--fit` nothing to adjust.
* `mmproj` (vision) memory is **not** counted by the router's memory check, so for Gemma 4 keep
  extra margin or set `memory_margin_mib` higher.
* Precedence: command line > model section > `[*]`.

**`presets dump`** (replaces `run_PRESETS_dump`): for each section it prints the equivalent
standalone command, the merge of `[*]` and the section, mapped to long flag names:

```
[gemma-4-26B-A4B-it-qat-UD-Q4_K_XL]
  llama-server -m d:\.lmstudio\…\gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf -c 65536 -t 4 -np 1 -fa on
               -ngl all -lm none -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 --n-cpu-moe 16
               --reasoning-format auto --temp 1.0 --top-k 64 --top-p 0.95
  variant ≈ fa_Kq8_Vq8_b2048_ub2048_moe16   (matches models.gemma_4_26B_A4b_QAT_UD_Q4_K_XL.default_serve)
```

That command is what `serve --preset NAME` and `benchy --preset NAME` run, so a benchmark
measures exactly what the router serves. The `variant ≈` line maps the preset back to a variant
name, so preset configs and sweep results can be compared directly.

**`presets generate`**: builds the INI from `[models.*]` + each model's `default_serve` variant
(one source of truth instead of a hand-kept INI plus bash maps). It writes `version = 1` and `[*]`,
and keeps the original command line as a comment per section, as today. In M13 the config switches
to the generated `config/presets.ini` and `llama_cpp_presets.ini` is deleted (§10.1).

### 4.4 New sweep axes: MoE expert cache, load mode, FP8 KV

Source: video *"llama.cpp 1.0 Is Here — Can It Finally Catch Strata?"* (Code Unpacked, 2026-11-14),
checked against your checkout (`docs/moe-cache.md`, `scripts/moe-cache-tune.py`, `src/llama-moe-cache.cpp`;
PRs #30966, #31233 and #31888 are all in `a49db739f`).

**How it works.** With experts kept in host RAM (`-cmoe` or `-ncmoe N`), `--moe-cache-mib N` keeps
the most-used experts in VRAM (ClassLRU score, #31233). On a miss the expert is uploaded over PCIe.
With `--moe-cache-cpu-share F`, a share F of the misses is computed **on the CPU instead**, in
parallel with the GPU. The share applies only once the cache is full; until then every miss is
uploaded. The cache is used only for **small batches** (decode, spec drafting), so it mainly affects
**TG**, and PP should stay roughly the same. This is a check worth running.

**Published numbers (other people's machines, not yours):**

| Setup | No cache | share 0 | 0.5 | 0.95 |
|---|---|---|---|---|
| RTX 2080 Ti, Qwen3.6-35B, 2.7 GB cache | 42.1 | 23.5 (slower than no cache) | — | 51.9 |
| RTX 3090, Qwen3.8-Flash-Next IQ4_XS, 5.5 GB, **mmap** | — | 19.5 | 26.6 | 31.8 |
| same, **`-lm dio`** (pinned) | — | 32.1 | 35.9 | 33.4 |
| dev card, 6 GB / 19 GB cache | — | 24.2 / 65.5 | — | 47.6 / 54.1 |
| R9700 on PCIe 5 | — | best | ~15 % slower | ~15 % slower |

Takeaways for the matrix:
1. **Load mode matters as much as the share.** Experts are pinned with `-lm none|mlock|dio` and
   not with mmap (`auto`, the default). With pinned experts the share often stops helping. Your
   configs already use `-lm none`, so **`-lm` has to be a sweep axis** (`none`, `auto`, `dio`,
   maybe `mlock`) and not a constant. `--no-host` turns pinning off.
2. **A small cache at share 0 can be slower than no cache.** Always include a **no-cache baseline**
   (your current best `-ncmoe N`) in the same sweep.
3. The best share depends on the machine. Your RTX 4080 **Laptop** (12 GB, mobile PCIe link) is
   likely in the "slow link" class where the share helps, but only measuring will tell.
4. Leave VRAM headroom: the tuner OOM'd when a vision model (mmproj) also loaded. Gemma 4 is
   multimodal, so include the mmproj in the budget or don't load it during benches.
5. **FP8 KV cache** (`-ctk/-ctv f8_e4m3`) is new in 1.0. It's an 8-bit alternative to `q8_0`
   that also sits between `q8_0` and `q4_0` on the V side.

**New variant tokens** (codec §4.2):

| Token | Flags | Values to sweep |
|---|---|---|
| `cmoe` | `-cmoe` (all experts in RAM) | vs `moe{N}` |
| `mc{MiB}` | `--moe-cache-mib MiB` | 0, 2048, 4096, 6144, 8192 (cap by free VRAM) |
| `cs{pct}` | `--moe-cache-cpu-share pct/100` | 0, 50, 80, 90, 95 |
| `lmN` / `lmA` / `lmD` / `lmL` | `-lm none` / `auto` / `dio` / `mlock` | none, auto, dio |
| `nohost` | `--no-host` | rarely |
| `Kf8`, `Vf8` | `-ctk f8_e4m3`, `-ctv f8_e4m3` | next to `Kq8`/`Vq8`/`Vq4` |

Example: `fa_Kq8_Vq8_b2048_ub2048_cmoe_mc6144_cs95_lmD`.

**Tool support.** `llama-bench` (this build) has `-lm` (incl. `dio`) and `-ncmoe`, but **no
`--moe-cache-*`**, so cache sweeps must run as **server + benchy**. The capability check (§3.3)
rejects `mc`/`cs` tokens for `tool = "bench"`.

**Measurement rules for cache runs:**
* Warm the cache: TG long enough for the cache to fill and the share to apply (the tuner
  generates 2048 tokens on its first run and 512 later). Use benchy `--tg 512 2048` and run
  `--runs ≥ 2`, keeping cold and warm runs separate, or add an explicit warm-up request.
* Hold the cache size constant when sweeping the share, and vice versa (one axis at a time, the
  same way the video's tests were run).
* Report PP and TG at short **and** long prompts (`--pp 512 8192 32768`) and with `--depth`.
* Parse the server log into `RunResult.moe_cache` (`parsers/server_log.py`):
  * `MoE cache size = X MiB for Y MiB of host experts, Z% of the misses on the CPU`
  * `llama_moe_cache: ...: hits = H, misses = M (C on the CPU), hit rate = R%, uploaded = U MiB`
  * warnings `MoE cache is disabled` / `budget is too small` → mark the run `invalid_config`
* Log PCIe TX/RX from NVML (§5), which explains whether the link is the bottleneck.

**Integrated tuner.** `mse-llm tune moe-cache MODEL -v VARIANT [--max-mib N]` builds the normal
server argv for the variant and runs
`python scripts/moe-cache-tune.py --out <results>/tune/<run_id> -- <argv>` from the configured
llama.cpp checkout. It parses `recommended: --moe-cache-mib N --moe-cache-cpu-share F` and
stores the recommendation as a new variant, so the next sweep can include it. Its 6 calibration
runs are reused by the script while the command stays the same.

**Proposed sweeps** (config-driven, staged so each stage fixes the winner of the previous one):

```toml
[sweeps.qwen36_mtp_moecache]
model = "Qwen36_35B_A3B_UD_Q4_K_M_MTP"
tool = "benchy"
ctx = 65536
benchy = { pp = [512, 8192], tg = [512, 2048], runs = 2 }
stages = [
  # 0. baselines: current sweet spot, with each load mode
  ["fa_Kq8_Vq8_default_moe34_lm{N,A,D}"],
  # 1. all experts in RAM + cache size at share 0 vs 0.95
  ["fa_Kq8_Vq8_default_cmoe_mc{2048,4096,6144}_cs{0,95}_lm{N,D}"],
  # 2. share sweep at the best size and load mode from stage 1
  ["fa_Kq8_Vq8_default_cmoe_mc{best}_cs{0,50,80,90,95}_lm{best}"],
  # 3. cache on top of partial offload, then MTP drafting on top
  ["fa_Kq8_Vq8_default_moe{34,41}_mc{best}_cs{best}_lm{best}",
   "fa_Kq8_Vq8_default_moe34_mc{best}_cs{best}_lm{best}_sd{2,3}mtp"],
  # 4. KV type at the winner
  ["fa_K{q8,f8}_V{q8,f8,q4}_default_..._winner"],
]
```

The same template applies to `gemma_4_26B_A4b_QAT_UD_Q4_K_XL` (baseline `b2048_ub2048_moe16`,
spec `moe21_sd4mtp`), `Qwen36_28B_REAP20` (`b4096_ub4096_moe41`) and GLM (`moe49`). Configs that
froze the PC before (low `moe` + large `ub`) go through the memory guard (§5).
Each sweep produces a chart with TG, PP, duration, peak VRAM, shared-mem delta and RAM
(the 5-axis chart) **plus** hit rate and uploaded MiB.

### 4.5 Quality and speculative-decoding metrics

Speed alone can't tell a good config from a fast but broken one. Two metrics go next to PP/TG in
every result and chart. Both were checked against this build (`a49db739f`).

#### KL divergence for KV-cache and quant choices (`mse-llm quality kld`)

**Why KLD and not perplexity.** Perplexity averages over all tokens, so small distortions cancel
out. `Kq8_Vq4` and `Kq8_Vq8` can show nearly the same PPL even though the model picks different
tokens. KL divergence compares the **full token probability distribution** of the tested config
with a reference at every position, so it shows exactly the drift that K/V quantization causes.
This follows the asymmetric-KV rule in the script ("K is everything, V is free"): KLD is how to
check it on your own models.

**How it works with llama.cpp** (`tools/perplexity`):
1. **Reference run, once per model/context/corpus:**
   `llama-perplexity -m MODEL -f CORPUS -c CTX --chunks N -fa on -ctk f16 -ctv f16 <offload flags>
   --kl-divergence-base <ref>.kld`. This saves the reference logits.
2. **Test run per variant:** the same command with the variant's K/V types, plus
   `--kl-divergence-base <ref>.kld --kl-divergence`.
3. Output parsed into `RunResult.quality`: `Mean KLD ± err`, `Median`, `95/99/99.9 % KLD`,
   `Maximum KLD`, **`Same top p`** (how often the top token matches the reference), `Mean Δp`,
   `RMS Δp`, `Mean PPL(Q)/PPL(base)`.

**Decisions that make the numbers meaningful:**
* **Reference = the same GGUF with an f16 KV cache**, not f16 weights. An f16 Gemma-26B or Qwen-35B
  doesn't fit this laptop, and comparing against the same weights isolates the **KV-cache effect**,
  which is what the `K*_V*` variants change. Weight-quant comparisons (Q4_K_M vs Q4_K_XL vs QAT)
  are a separate, optional mode (`--base-model`), only for models where a bigger quant fits.
* **Context length matters.** KV quantization errors build up with context, so chunks of 512
  tokens (the perplexity default) understate them. The default is `-c 8192` with fewer chunks,
  and an optional second point at `-c 32768`.
* **Noise floor first.** GPU vs CPU math (`-ncmoe`, `-cmoe`) gives tiny logit differences even
  with identical KV types. `quality kld` runs **reference vs reference with different offload**
  once and reports that as the floor. Variant KLDs within ~2× the floor count as "no measurable
  loss".
* **Corpus:** `wikitext-2` (via llama.cpp's `scripts/get-wikitext-2.sh`, the community standard),
  **plus a code corpus**, because your models are mainly used for coding. By default it's a fixed
  set of source files from the llama.cpp checkout, recorded by hash. `--corpus FILE` takes your own text.
* **Disk size.** The reference file stores `tokens × vocab × 2 bytes`. Example: 8 chunks × 8192 =
  65,536 tokens × 262,144 vocab (Gemma 4) × 2 B ≈ **34 GB**; with Qwen's ~152k vocab ≈ 20 GB.
  `quality kld` prints the size **before** creating it (same preview rule as builds, §0.4) and
  lets `--chunks` trade accuracy for size. Reference files go to `results/raw/kld-ref/`, which is
  local and not tracked, keyed by (model identity, ctx, corpus hash, build).
* Same `-fa on`, `-b/-ub`, threads and offload as the variant, so only the K/V types differ,
  **unless the reference doesn't fit** (next point).

**When the f16 reference doesn't fit in 12 GB.** The reference only needs to produce correct
logits once; speed doesn't matter. So it may use **different memory placement** than the
variant, and a fallback order is applied automatically (shown in the `--dry-run` preview):
1. **Use the test context, not the serving context.** The KLD context (default 8192) is chosen
   for the test; it isn't the 65536 the server uses. The f16 KV cache at 8192 is small (table below).
2. **Move more weights to the CPU for the reference only** (higher `-ncmoe`, or `-cmoe`). That
   frees GB of VRAM, while KV f16 vs q8 only differs by MB at 8k.
3. **Keep the KV cache in system RAM for the reference** (`-nkvo`). It's slow but a one-time run.
4. **Last resort: q8_0/q8_0 KV as reference.** q8_0 KV is close to lossless, so variants are
   still ranked correctly against each other. Results are then labelled **"KLD vs q8 ref"**,
   never mixed with f16-ref numbers.

The **noise floor** run (reference vs the same reference with different placement) measures
exactly what steps 2–3 change. A placement difference never shows up as KV-quant loss.

**Estimated f16 KV-cache size for your models** (from GGUF metadata. These models are
hybrid: only some layers keep a KV cache that grows with context. The exact size comes from the
`llama_kv_cache: size` log line):

| Model | Attention layout (GGUF) | f16 KV @ 8 k | f16 KV @ 64 k | q8_0 saves @ 64 k |
|---|---|---|---|---|
| Qwen3.6-35B-A3B / 28B-REAP20 (`qwen35moe`) | 40–41 layers, full attention every 4th (~10), 2 KV heads × 256 | ~0.16 GiB | ~1.25 GiB | ~0.6 GiB |
| gemma-4-26B-A4B (`gemma4`) | 30 layers, sliding window 1024 on most, few global (2 KV heads × 512) | ~0.7 GiB | ~1.9 GiB | ~0.9 GiB |
| OmniCoder-9B (`qwen35`) | 32 layers, full attention every 4th (8), 4 KV heads × 256 | ~0.25 GiB | ~2 GiB (131 k: ~4 GiB) | ~1–2 GiB |

Two consequences:
* **The f16 reference fits at 8 k** wherever the variant fits; the difference is a few hundred MB.
  Steps 2–4 matter only for long-context KLD (32 k–64 k) or a model that's already at the VRAM limit.
* On these hybrid models **KV quantization saves less memory than on classic models** (well under
  1 GiB at 64 k for the Qwen MoEs). `-ncmoe` / `--moe-cache-mib` move far more memory. So
  KLD's main job here is to **confirm a K/V choice is free** before using it, not to justify a big
  saving. If `Kq8_Vq4` saves 0.3 GiB but shows measurable KLD, the saving isn't worth it.

**Commands:**

```
mse-llm quality kld-ref MODEL [--ctx 8192] [--chunks 8] [--corpus wikitext2|code|FILE] [--dry-run]
mse-llm quality kld     MODEL -v VARIANT [-v VARIANT ...] [--ctx 8192] [--corpus ...] [--dry-run]
mse-llm quality floor   MODEL            # reference vs reference with different offload = noise floor
mse-llm sweep run NAME --with-kld        # add a KLD run per distinct K/V combination in the sweep
```

```
$ mse-llm quality kld gemma_4_26B_A4b_QAT_UD_Q4_K_XL -v fa_Kq8_Vq8_b2048_ub2048_moe16 -v fa_Kq8_Vq4_b2048_ub2048_moe16 -v fa_Kf8_Vf8_b2048_ub2048_moe16
reference  f16/f16 KV, ctx 8192, 8 chunks, corpus code@3f2a…   (noise floor: mean KLD 0.0004)
VARIANT                 K/V        Mean KLD   99% KLD   Same top p   PPL ratio   verdict
Kq8_Vq8                 q8_0/q8_0  …          …         …            …           (example layout; values come from the run)
Kq8_Vq4                 q8_0/q4_0  …
Kf8_Vf8                 f8_e4m3/…  …
```

Results charts get a **quality axis** (mean KLD, or "Same top p") next to TG/PP/VRAM. The Pareto
chart (§8) becomes **TG vs KLD at a given VRAM**, which is the real KV trade-off.

#### Speculative decoding acceptance (`RunResult.spec`)

**Why.** For `sd{N}mtp`, `sd{N}cache` and the other spec types, TG alone doesn't say *why* `sd4mtp`
beats `sd6mtp`. The answer is the **acceptance rate**: how many drafted tokens the main model keeps,
and how far into the draft they stay accepted. Longer drafts only pay off while the later positions
are still accepted.

**Where llama-server reports it** (verified in `tools/server`):
| Source | Fields | Used for |
|---|---|---|
| Response `timings` of each request | `draft_n`, `draft_n_accepted` | per request (when mse-llm sends its own requests) |
| Slot log line after each request | `draft acceptance = R (A accepted / G generated), mean len = L` | always available, parsed by `parsers/server_log.py` |
| Prometheus `/metrics` (needs `--metrics`) | `spec_decode_num_draft_tokens_total`, `..._accepted_tokens_total`, `..._drafts_total`, `..._accepted_tokens_per_pos_total{position="N"}` | **primary**: counter delta before/after a benchy run gives exact totals, **including acceptance per draft position** |

mse-llm adds `--metrics` to every server it starts for a bench, and scrapes `/metrics` before and
after each benchy run. The router (`presets serve`) gets it too.

**Stored per run:** `spec = {type, n_max, draft_model, drafted, accepted, acceptance_rate,
verif_steps, mean_accept_len, accepted_per_pos[]}`. `mean_accept_len` is tokens gained per
verification step, so the expected speedup is visible next to the measured TG.

**Charts / reports:**
* TG and acceptance rate vs `spec-draft-n-max` (for example `sd{2,3,4,5,6,10}mtp`), one line per model.
* **Acceptance per draft position** (bar per position). The point where it falls under ~0.3
  usually marks where a longer draft stops paying off.
* A spec run with TG **lower** than the same variant without spec is flagged automatically.

**Measurement rules:**
* Acceptance depends heavily on **content** and **sampling**. Benchy's default prompt is prose
  (Sherlock Holmes); coding use needs a code prompt set. Spec sweeps run both and report them
  separately. The sampling (temp, top-k/p) is taken from the model's serve settings and recorded,
  because a high temperature (Gemma 4 uses 1.0) lowers acceptance.
* `--spec-draft-n-min`, `-ctkd/-ctvd` (draft KV types) and `--spec-draft-n-cpu-moe` become
  variant fields, so the draft side of the config is recorded too.
* This build has `--spec-synth-len` / `--spec-synth-rates` (**synthetic acceptance** for
  benchmarking). Optional mode `spec-ceiling`: run the same variant at fixed acceptance rates
  to measure the speed you *would* get at 50/70/90 % acceptance. That separates "the draft is too
  slow" from "the draft is too often wrong".

### 4.6 Predicting which setting helps, before running (`mse-llm predict`)

Your notes show that `n-cpu-moe` matters a lot for some models while `-b/-ub` matters for others
("Batching works" for gemma UD, "Batching NOT helps" for gemma QAT). That follows from a few
numbers that can be read **before** any benchmark: from the GGUF file, from llama.cpp's own memory
estimate, and from the laptop's memory speeds. All figures below were measured or read on this
laptop on 2026-11-18.

#### Inputs

**Per model, from GGUF tensors** (`gguf` package; exact sizes, not estimates):

| Model | experts in RAM if all on CPU | VRAM per expert layer on GPU | experts read per token per layer | `n_used / n_expert` |
|---|---|---|---|---|
| Qwen3.6-35B-A3B MTP (41 layers) | 18.7 GiB | 466 MiB | 14.6 MiB | 8/256 = 3.1 % |
| Qwen3.6-28B-REAP20 (40 layers) | 14.5 GiB | 372 MiB | 14.5 MiB | 8/205 = 3.9 % |
| gemma-4-26B QAT Q4_K_XL (30 layers) | 12.0 GiB | 408 MiB | 25.5 MiB | 8/128 = 6.3 % |
| gemma-4-26B UD-Q4_K_M (30 layers) | 13.4 GiB | 456 MiB | 28.5 MiB | 8/128 = 6.3 % |
| GLM-4.7-Flash-REAP-23B (46 MoE layers) | 11.7 GiB | 261 MiB | 21.7 MiB | 4/48 = 8.3 % (+ 0.3 GiB shared experts) |

**Per config, from llama.cpp itself**: `llama-fit-params … --fit off --fit-print on` returns the
GPU and host memory split in **~0.5 s, without loading the model**:

| Config (ctx 65536, q8 KV) | GPU model | GPU KV | GPU compute | **GPU total** | host |
|---|---|---|---|---|---|
| Qwen35 `-ncmoe 34` | 4807 | 745 | 505 | **6057 MiB** | 16371 |
| Qwen35 `-ncmoe 30` | 6663 | 745 | 505 | **7913 MiB** | 14515 |
| Qwen35 `-ncmoe 20` | 11303 | 745 | 505 | **12553 MiB** ✗ | 9875 |
| Qwen35 `-ncmoe 34 -ub 4096` | 4807 | 745 | 4040 | **9592 MiB** | 16931 |
| Qwen35 `-ncmoe 34 -ub 8192` | 4807 | 745 | 8080 | **13632 MiB** ✗ | 17571 |
| gemma QAT `-ncmoe 16 -ub 2048` (your SD-off sweet spot) | 7039 | 1010 | 2170 | **10219 MiB** | 7286 |
| gemma QAT `-ncmoe 21 -ub 2048` | 4998 | 1010 | 2170 | **8178 MiB** | 9327 |
| gemma QAT `-ncmoe 16 -ub 8192` ("freeze TERRIBLE") | 7039 | 1647 | 8688 | **17374 MiB** ✗ | 8450 |

Usable VRAM is 12282 MiB minus what Windows and other apps already use (742 MiB at the time of
measurement), so about **11.5 GiB**. ✗ means the config can't fit and would spill into shared
memory (the freezes).

**Two rules for VRAM that follow from it:**
* **Each `-ncmoe` step moves exactly one expert layer** (464 MiB for Qwen35, matching the GGUF).
  VRAM for any `n-cpu-moe` value is a straight line from one estimate.
* **The compute buffer is about `ub × vocab × 4 bytes`**. That's the logits buffer for a full
  ubatch. With vocabularies of 248k (Qwen) and 262k (Gemma) tokens, that's **~1 MiB per ubatch
  token**: `ub 512` ≈ 0.5 GiB, `ub 4096` ≈ 4 GiB, `ub 8192` ≈ 8 GiB. That's why
  `b8192_ub8192` froze every model. Raising `ub` costs as much VRAM as moving 8–17 expert layers
  back to the CPU.

**Per laptop, from `hw`:**
* VRAM bandwidth: 192-bit GDDR6 at 18 Gbps ≈ **432 GB/s**.
* PCIe: **Gen4 x16** ≈ 31.5 GB/s theoretical. The link idles at Gen2, so it's read under load.
* RAM: DDR5-5200, **2 channels but asymmetric (16 GB + 64 GB)**. Only the first 32 GB run
  dual-channel (83 GB/s theoretical). The remaining 48 GB run **single-channel (~42 GB/s)**. Experts
  that end up in that region are read about half as fast. That's a possible source of run-to-run
  variance; `hw` reports it, and `predict` uses the slower value as a pessimistic bound.
* CPU threads: `-t` limits how fast the CPU part runs. With `-t 4`, the CPU-side expert compute may
  not reach the RAM bandwidth, so `-t 8` is worth testing when many experts are on the CPU.

#### What each setting changes (simple roofline model)

**TG (token generation, one token at a time) is limited by memory reads.** Each token reads the
dense weights plus `n_used` experts per layer. For each layer on the CPU, those expert bytes come
from RAM (~60 GB/s effective) instead of VRAM (~300 GB/s effective):

> extra time per token for each expert layer on the CPU ≈ `active_expert_bytes × (1/BW_ram − 1/BW_vram)`

| Model | extra TG time per CPU layer | **TG gained per GiB of VRAM given to experts** |
|---|---|---|
| Qwen35-MTP | ~0.20 ms | ~0.45 ms/token |
| Qwen28-REAP20 | ~0.20 ms | ~0.55 ms/token |
| gemma QAT / UD | ~0.36–0.40 ms | **~0.9 ms/token** |
| GLM-4.7-REAP | ~0.30 ms | **~1.2 ms/token** |

The last column is proportional to **`n_used / n_expert`**: the share of an expert layer that each
token actually reads. **Gemma and GLM gain about twice as much TG per GiB from lowering
`n-cpu-moe` as the sparse Qwen models (256 experts, 8 used).** So `n-cpu-moe` is the main setting for
Gemma and GLM, and much less so for Qwen.

**PP (prompt processing, large batches) is limited by PCIe uploads when experts sit on the CPU.**
For each ubatch, llama.cpp uploads the CPU-resident expert weights it needs to the GPU (with
hundreds of tokens per ubatch nearly all experts are hit). So:

> upload time per ubatch ≈ `N_cpu_layers × expert_layer_bytes / BW_pcie` &nbsp;&nbsp;→&nbsp;&nbsp; PP ≤ `ub / upload_time`

| Config | upload per ubatch (~25 GB/s) | PP ceiling at `ub 512` | at `ub 2048` | at `ub 4096` |
|---|---|---|---|---|
| Qwen35 `ncmoe 34` | ~0.67 s | ~770 t/s | ~3 000 t/s | ~6 100 t/s |
| Qwen28-REAP `ncmoe 41` (all 40 layers) | ~0.62 s | ~820 t/s | ~3 300 t/s | ~6 600 t/s |
| gemma UD `ncmoe 31` (all 30 layers) | ~0.57 s | ~890 t/s | ~3 600 t/s | ~7 100 t/s |
| gemma QAT `ncmoe 16` | ~0.27 s | ~1 900 t/s | ~7 500 t/s | (doesn't fit) |

When the PP ceiling at the default `ub` is far below what the GPU could compute, a bigger `ub`
gives an almost proportional PP gain ("batching works"). When few expert layers are on the CPU,
the ceiling is already high and the GPU's own compute speed is the limit, so a bigger `ub` adds
little and costs ~1 MiB VRAM per token ("batching NOT helps").

**This matches your notes:**
* **gemma UD `moe31`** (all experts on CPU): "Batching works", sweet spot `b4096_ub4096`.
* **Qwen28-REAP `moe41`** (all experts on CPU): sweet spot `b4096_ub4096`.
* **gemma QAT `moe16–26`** (only part of the experts on CPU): "Batching NOT helps"; the sweet spot
  is `moe16` + `ub 2048` at **10.2 of ~11.5 GiB**, right at the edge.
* **`b8192_ub8192` froze everything**: +8 GiB of compute buffer.

**Not explained by memory: Qwen35 `moe30` "PC freeze" at 65 k.** The estimate is 7.9 GiB, which
fits. Possible causes are other VRAM users at that time (desktop, browser, LM Studio), an older
llama.cpp build, or the GPU being so busy that the Windows desktop (if it runs on the 4080) stops
responding. With the "Prefer No Sysmem Fallback" driver setting and the live monitor, a re-run
will show which it is. `predict` marks such configs "fits, but unsafe in your notes" instead of
hiding the conflict.

#### `mse-llm fit`: paste a GGUF + your variant list + context, get the expected fit

The quick helper for everyday tuning. You give it a model file, the variants you're thinking of
running (in any of the forms you already use), and one or more context sizes. It tells you, before
anything starts, how much VRAM and RAM each one needs and whether it fits. It runs nothing heavy:
one `llama-fit-params --fit-print` call per variant (~0.5 s each, no model load).

```
mse-llm fit GGUF_OR_MODEL --ctx 65536 [--ctx 131072 ...]
            [-v VARIANT ...] [--from FILE | -]        # list from a file, or paste on stdin
            [--kv q8_0/q8_0] [--free auto|MiB] [--margin 768] [--suggest] [--json]
```

* **`GGUF_OR_MODEL`**: a path (`d:\.lmstudio\models\…\X.gguf`) or a name from `[models.*]`.
* **Variants, in any form you already have:**
  * `-v fa_Kq8_Vq8_b4096_ub4096_moe41` (repeatable, with or without the old `extra_` prefix);
  * `--from list.txt` or `-` (stdin): **paste the bash array as it is**, including
    `local -a extras_list=(`, `#` commented lines, quotes and trailing `# SWEETSPOT …` comments.
    Those comments are kept and shown as notes;
  * brace patterns: `-v "fa_Kq8_V{q8,q4}_b4096_ub4096_moe{41,34,28}"`.
* **`--free auto`** reads free VRAM right now (NVML), because the desktop, a browser or LM Studio
  change it. `--margin` is the headroom below which a fit counts as **TIGHT** (default 768 MiB).
* **Speculative variants and extras (drafter, vision):** `llama-fit-params` rejects `--spec-*`
  and doesn't count a vision projector, so `fit` adds those parts itself and **always says what it
  counted**. It never leaves a part out silently. Each variant shows a breakdown with a
  **source per part**:

  | Part | When present | Source |
  |---|---|---|
  | main model weights, KV, compute | always | `measured` (llama-fit-params) |
  | n-gram speculation (`sdNcache`, `ngram-*`) | spec type is n-gram | `none`: runs on the CPU, 0 VRAM |
  | draft model weights | `sdNmtp` with a separate draft GGUF (e.g. gemma `mtp-gemma-4-26B-A4B-it.gguf`, 240 MiB) | `measured`: second llama-fit-params call on the draft GGUF |
  | draft KV + compute | any `draft-*` spec type, incl. built-in MTP | `estimated` from GGUF metadata (layers × KV heads × ctx × type; ub × vocab × 4 B), until a real run measured it |
  | vision projector (`mmproj`) | the variant/preset passes `--mmproj` (gemma: `mmproj-F32.gguf`, 2185 MiB) | `measured` from file size + projector buffers |
  | anything else unknown | e.g. `sdNmtp` but no drafter configured for the model | `NOT COUNTED` |

  **Where the drafter comes from**, in this order: the variant (`--model-draft` / `md`), the model's
  `draft` field in `[models.*]` (the old `DRAFTER_MODELS`: a path or `builtin`), the preset's
  `model-draft` key. If a `draft-*` variant has none of these, the part is **NOT COUNTED** and
  the verdict can't be trusted.

  **How you see it in the output:**
  * Verdict **`fits`** = every part measured. **`fits~`** = some parts estimated (normally accurate to
    ~100 MiB). **`UNSURE`** = some part NOT COUNTED. TIGHT and NO-FIT work the same way.
  * A footer per variant lists the parts that aren't plain `measured`, for example
    `draft KV+compute estimated (+410 MiB)` or `drafter for draft-mtp NOT COUNTED: set models.X.draft`.
  * `--strict` makes `fit` exit with an error when any part is NOT COUNTED (useful in `sweep search`,
    which then refuses to start that variant).
  * **Calibration:** after the first real run of a variant with a drafter or mmproj, the server log's
    buffer lines (draft context `CUDA0 … buffer size`, mmproj buffers) are stored per
    (model, drafter, build). From then on those parts show as `measured (run 2026-11-…)`
    instead of `estimated`.
* **`--suggest`** uses the two linear rules (one expert layer per `-ncmoe` step, ~`vocab × 4 B`
  per ubatch token) to show how to use the remaining VRAM: the lowest `-ncmoe` that still fits
  with margin, and the largest `-ub`, per fitting variant.

**Real output for your pasted list** (Qwen3.6-28B-REAP20-A3B-Q4_K_M, `--ctx 65536`; computed on this
laptop on 2026-11-18 with a prototype of this command):

```
$ mse-llm fit Qwen36_28B_REAP20_A3B_Q4_K_M --ctx 65536 --from - --suggest   (← pasted extras_list)

model Qwen3.6-28B-REAP20-A3B-Q4_K_M.gguf  ctx 65536  free VRAM now 11500 MiB
VARIANT                                       GPU MiB   (model+KV+compute)  host MiB    left  VERDICT   NOTE
fa_Kq8_Vq8_default_moe41                         2568       (1286+745+537)     15247    8932  fits
fa_Kq8_Vq4_default_moe41                         2408       (1286+585+537)     15247    9092  fits
fa_Kq8_Vq8_b6144_ub6144_moe41                    8091      (1286+745+6060)     16127    3409  fits
fa_Kq8_Vq4_b6144_ub6144_moe41                    7931      (1286+585+6060)     16127    3569  fits
fa_Kq8_Vq8_b4096_ub4096_moe41                    6071      (1286+745+4040)     15807    5429  fits      SWEETSPOT 1. step
fa_Kq8_Vq4_b4096_ub4096_moe41                    5911      (1286+585+4040)     15807    5589  fits      SWEETSPOT 1. step
fa_Kq8_Vq8_b4096_ub4096_moe41_sd3cache           6071      (1286+745+4040)     15807    5429  fits      +spec ngram: no extra VRAM
fa_Kq8_Vq4_b4096_ub4096_moe41_sd3cache           5911      (1286+585+4040)     15807    5589  fits      +spec ngram: no extra VRAM
fa_Kq8_Vq8_b4096_ub4096_moe41_sd2cache           6071      (1286+745+4040)     15807    5429  fits      +spec ngram: no extra VRAM
fa_Kq8_Vq4_b4096_ub4096_moe41_sd2cache           5911      (1286+585+4040)     15807    5589  fits      +spec ngram: no extra VRAM

suggest (worked out by hand from the two rules; the prototype printed only the table above)
        (model has 40 MoE layers; moe41 = ALL experts on CPU; 1 layer = 372 MiB, 1 ub token ≈ 0.99 MiB):
  fa_Kq8_Vq8_b4096_ub4096_moe41 → 5429 MiB unused: lowest safe -ncmoe ≈ 28 (12 expert layers back on the GPU,
                                  ~0.55 ms/token TG gain per GiB → worth testing: fa_Kq8_Vq8_b4096_ub4096_moe{34,30,28})
  K/V: Vq4 instead of Vq8 saves only 160 MiB here (hybrid model, §4.5) → check KLD before preferring it
```

What this table shows that the bash runs didn't:
* Every variant in the list fits with **3.4–9 GiB left**, because `moe41` is more than the
  model's 40 MoE layers, so **all experts stay on the CPU**. The sweet spot leaves 5.4 GiB unused.
  That VRAM could hold ~12 expert layers.
* The `ub` cost is plain: `ub 6144` costs 6060 MiB of compute buffer vs 537 MiB at the default.
* `Vq4` vs `Vq8` differs by only 160 MiB at 64 k on this hybrid model.
* Context: pass several `--ctx` values to see where a variant stops fitting, e.g.
  `--ctx 65536 --ctx 131072 --ctx 262144`.

**How the same command reports a drafter** (gemma QAT, your SD-on sweet spot; layout example:
the draft-side numbers are filled in by the implementation, only the main-model figure was measured here):

```
$ mse-llm fit gemma_4_26B_A4b_QAT_UD_Q4_K_XL --ctx 65536 -v fa_Kq8_Vq8_b2048_ub2048_moe21_sd4mtp

VARIANT                                   GPU MiB   left   VERDICT  PARTS
fa_Kq8_Vq8_b2048_ub2048_moe21_sd4mtp      8178+…    …      fits~    main 8178 measured (4998+1010+2170)
                                                                    draft weights  240 measured (mtp-gemma-4-26B-A4B-it.gguf, from models.….draft)
                                                                    draft KV+compute … estimated
                                                                    mmproj         not used (no --mmproj)
  ~ estimated parts: draft KV+compute. Run it once to replace the estimate with measured buffers.
```

And when the drafter is missing from the config:

```
fa_Kq8_Vq8_default_moe34_sd2mtp           …         …      UNSURE   draft-mtp requested, but models.Qwen36_….draft is not set
  ! drafter NOT COUNTED: set draft = "builtin" (MTP inside the GGUF) or a draft GGUF path
```

`fit` is the memory half of `predict`. `predict` adds the TG/PP estimates and recommendations,
`sweep search` uses both, and `fit` is the part you'll use most by hand.

#### `mse-llm predict` and how it's used

```
mse-llm predict MODEL [--ctx 65536] [--kv q8_0/q8_0] [--axes ncmoe,ub] [--engine cuda_flash]
```

1. Reads the GGUF table above, the `hw` bandwidths and the free VRAM right now.
2. Runs `llama-fit-params --fit-print` for a small grid of `n-cpu-moe × ub` (each call ~0.5 s,
   no model load), then interpolates with the two linear rules.
3. Prints for each config: GPU/host memory, **fits / tight (< 0.75 GiB left) / no-fit**,
   predicted TG and PP ceiling, and a recommendation like *"Qwen35: TG gains little below
   ncmoe 34 (0.45 ms per GiB); spend VRAM on ub 2048–4096 for PP"* or *"gemma QAT: lower ncmoe
   first; keep ub ≤ 2048"*.
4. **Calibration:** after the first two real runs of a model, `predict` fits the effective
   bandwidths (RAM, PCIe, VRAM) to the measured TG/PP, the same idea as `moe-cache-tune.py`. From
   then on, predictions use this laptop's real speeds instead of the spec values above.

`sweep search` (§9 item 6) uses `predict` to decide **which setting to explore first** and which
configs to skip without running them. That replaces "try everything and see what freezes".

---

## 5. Monitoring (async, in-process)

`async with Monitor(interval=0.5, pid=server_pid) as mon:` runs samplers concurrently in the same
loop. As built (M7): each sampler runs in its own daemon thread with a time limit (2 × interval). A
hung CUDA process can block every NVML call on the machine; the monitor then keeps sampling PDH and psutil,
reports `stuck_nvml_s`, and the blocked thread never keeps mse-llm from exiting.

| Metric | Source (no shell-outs) |
|---|---|
| Dedicated VRAM used, total | NVML `nvmlDeviceGetMemoryInfo` |
| **Per-process VRAM** of the server | PDH `\GPU Process Memory(pid_*)\Dedicated Usage` on Windows (NVML `nvmlDeviceGetComputeRunningProcesses` reports no per-process memory under WDDM; used elsewhere) |
| Shared GPU memory (Windows spill) | PDH `\GPU Process Memory(pid_<server tree>_<NVIDIA luid>)\Shared Usage` (the server's own), `\GPU Adapter Memory(<NVIDIA luid>)\Shared Usage` (the GPU), via `win32pdh` with open queries. Not the sum over all adapters: that adds the AMD iGPU's normal shared memory (1.1 GB here) |
| Bash "shared" number (for comparison) | PDH sum of `\GPU Non Local Adapter Memory(*)\Non Local Usage` (= the script's `GPUNonLocalAdapterMemory.NonLocalUsage`) |
| GPU temp, power, SM/mem clocks, util, **throttle reasons** | NVML |
| System RAM, swap/pagefile, commit | psutil |
| Server RSS / private bytes | psutil on the server PID tree |
| PCIe TX/RX | NVML `nvmlDeviceGetPcieThroughput` (useful for `n-cpu-moe` / `moe-cache` analysis) |

Output: baseline, peak, mean, p95, delta; the bash status thresholds for RAM (> 60/80 %) and temperature
(> 75/85 °C); and the raw timeseries saved as CSV next to the result (no Parquet: it would need pandas).

**Spill status (changed from bash, M6).** The bash rule "shared delta > 100 MiB = offloaded to system RAM"
is wrong for MoE offload: llama.cpp's **pinned host buffers** for the experts `-ncmoe` keeps on the CPU
count as shared GPU memory. Measured: a fitting Qwen3.6-28B `moe41` shows +15892 MiB shared, and `fit`
predicts 15807 MiB of host buffers (0.5 % off); the bash function printed "Performance hit detected"
for it. A real spill (Qwen3.6-35B `moe10`, 17.2 GB on 11.4 GB free) looks different: dedicated VRAM
fills up first (12140 / 12282 MiB), then the server's shared memory keeps growing. So the status
compares the server's shared memory with the host buffers `fit` expects when that is known, and
otherwise explains the bash rule instead of calling it a spill.

**Safety guard.** Your notes show several configs freezing the PC. The monitor **stops the run live**
(it kills the server tree, the result is `aborted: memory_guard`, a sweep moves to the next variant,
`serve` exits with code 2) when:
* **A.** dedicated VRAM is full (free <= 256 MiB) and the server's shared memory grows by more than
  100 MiB after that, or
* **B.** the server's shared memory exceeds the host buffers `fit` expects by more than max(100 MiB, 2 %), or
* **C.** system RAM > 95 % or the commit charge > 95 % of the limit.
* **D.** (benchmarks only, M7) **no progress** for `stall_stop_s` (90 s): the process tree reads nothing from
  disk, its VRAM doesn't change, and the GPU's mean use over the window is < 5 % (single blips from other
  programs average out); or NVML blocked that long. Found in M7: a config that doesn't fit can sit **quiet**
  instead of spilling right away: llama-bench, Qwen3.6-35B `-ncmoe 10`, ctx 4096 allocated ~10 GB VRAM, read
  only 23 MiB of the GGUF, GPU at 0 %, for 140 s (then it spilled, rule A) and in another run for 10+ minutes.
  A fitting run never looks like that: `-ncmoe 41` read the GGUF the whole load (21 GB in 25 s), then the GPU
  ran at 99 %. Replayed on the recorded samples, D stops both no-fit runs at ~97 s and never the fitting one.
  `serve` doesn't use D: a server waiting for requests looks the same. A process stuck in the driver can also
  ignore the kill for minutes (seen once: ~3 min), so after a guard stop a job waits at most 30 s and moves on
  (the process ends when the driver lets go; it is in the kill-on-close Job Object).
Checked in M6: the `moe10` over-commit was stopped 8 s after start with 586 MiB spilled; without the
guard it reached 9 GB of shared memory in 60 s. A sweep can also use `llama-fit-params` to predict OOM
and skip known-bad configs (§4.6). Thresholds live in `[monitor]`.

### 5.1 PC capability detection (`mse-llm hw`), modelled on Strata

Strata (MIT) has hardware detection in two places: `strata_control/tune_machine.py` (Python,
stdlib only) and `apps/host/server/hardware.mjs`, `device-probes.mjs`, `windows-gpu.mjs`,
`system-telemetry.mjs` (Node). Fit and verdict logic lives in `strata_control/gpu_fit.py`,
`vram_fit.py` and `fit_label.py`. What it does, and what we take from it:

| Strata does | How | mse-llm |
|---|---|---|
| **Never-fatal probes**: a missing tool gives `None`, never an exception. A guard that can't read its input says so and doesn't assume the memory is free | `_run()` with timeout, try/except per probe | **Adopt** as a rule for `mse_llm/hw/` (fields are `Optional`; each has a `source` and `error`) |
| RAM total + **available** (reclaimable, not the free list) | Win `GlobalMemoryStatusEx` (ctypes), Linux `/proc/meminfo MemAvailable` + cgroup limit, macOS `vm_stat` | **Adopt** via `psutil.virtual_memory().available`, plus commit/pagefile (Win) for the freeze guard |
| Memory pressure label | free/total > 35 % low, > 18 % medium, else high | **Adopt**, shown in `hw` and stored per run |
| CPU model, physical/logical cores, **hybrid (P+E cores)** | Win registry `ProcessorNameString` + `GetLogicalProcessorInformationEx` (`EfficiencyClass` > 1 class = hybrid); Linux `/proc/cpuinfo`, `/sys/devices/cpu_core` + `cpu_atom` | **Adopt**, and also record **which logical CPUs are P-cores** (see below) |
| NVIDIA GPUs: index, uuid, PCI bus, name, VRAM, driver | `nvidia-smi --query-gpu=...` | **Adopt with NVML** (`nvidia-ml-py`), which also gives compute capability, PCIe gen current/max + width, power limit, clocks |
| Non-NVIDIA GPUs on Windows, **correct VRAM size** | DirectX registry `HKLM\SOFTWARE\Microsoft\DirectX` `DedicatedVideoMemory` (64-bit) + adapter LUID. `Win32_VideoController.AdapterRAM` is 32-bit and reads 0 or 4 GB for big cards | **Adopt** (`winreg`, no PowerShell). LUID also matches PDH `GPU Adapter Memory` counters, which gives per-adapter shared memory |
| Drop virtual adapters | regex `virtual\|remote\|indirect\|mirror\|basic render\|parsec\|displaylink\|idd` and no VRAM | **Adopt** |
| Linux AMD/Intel GPUs | `/sys/class/drm/card*/device` (vendor id, `mem_info_vram_*`, hwmon temp/power) | Adopt later (M14) |
| Live telemetry: temp, util, mem used, power, PCIe gen/width, **PCIe RX MB/s**, fan, clock | `nvidia-smi` + `nvidia-smi dmon -s t`; long-lived PowerShell for Windows perf counters | **Already in §5** via NVML + PDH, in-process, so no polling PowerShell |
| Disk free at the models dir; disk read/write rate | `statfs` / PDH PhysicalDisk | **Adopt**. Disk read rate during load explains `-lm` differences (§4.4) |
| **Other engines running** ("a second engine changes every number") | Process scan for `strata.exe ... --serve` | **Adopt**, widened: any `llama-server`, `strata`, `ollama`, `lms`, plus every GPU compute process from NVML. Warn or refuse before a bench |
| **Hardware identity, hard vs soft** | `hard` = GPU names + VRAM, CPU model/cores/hybrid, RAM (rounded down to 2 GiB), PCIe, OS. A change means stored settings may not start. `soft` = driver, OS release, so a re-check is due | **Adopt.** `stable_hash(hard)` goes on every `RunResult` and build profile. Sweet spots and presets carry the hash they were found on; `hw` warns on mismatch |
| Engine identity: version, build id, exe SHA-256, `tuning_epoch` | engine manifest | Covered by `build-manifest.json` (§0.4) |
| **Model identity without hashing 20 GB** | file size + digest of head, middle and tail of the first shard | **Adopt** (`hw.model_identity(path)`), stored per run so a re-downloaded GGUF with the same name is detected |
| Checks → verdict ("can run comfortably / limits / below minimum") | pass/warn/fail on GPU, RAM ≥ 16/32 GB, cores ≥ 4/8, disk ≥ 25/60 GB | **Adopt** as `doctor` section with thresholds in config |
| **Fit label** fits / tight / spills, plus a recommended variant | `files ≤ RAM + VRAM`; per-backend measured constants for VRAM slack (`vram_fit.py`) | **Adapt**: use llama.cpp's own numbers instead of constants (below) |
| Test override | `STRATA_TEST_HARDWARE='{...}'` JSON stands in for probes | **Adopt** as `MSE_LLM_TEST_HARDWARE` and `--hw-fixture FILE` for tests and CI |

**Additions for this laptop that Strata doesn't cover:**
* **Power source and mode**: AC vs battery (`GetSystemPowerStatus`), active power plan
  (`powercfg /getactivescheme`, read-only), and NVML power limit, which changes with the vendor's
  performance mode. On a laptop these swing TG more than many flags. They are stored per run,
  and a bench **refuses to start on battery** unless `--allow-battery`.
* **Hybrid CPU → thread placement**: if the CPU is hybrid, record the P-core logical IDs and add
  variant tokens `tP` (threads = number of P-cores) and `pinP` (`-C <mask of P-cores>` /
  `--cpu-strict 1`). The script uses a fixed `-t 4`; on a hybrid CPU, threads landing on E-cores
  slow down `-ncmoe` and `--moe-cache-cpu-share` runs. *This laptop's Ryzen 9 7945HX is not
  hybrid* (16 identical Zen 4 cores on two CCDs), so `tP`/`pinP` don't apply here. The useful
  axes are `-t` itself (4 / 8 / 16) and pinning to **one CCD** (`-C` mask of logical CPUs 0–15),
  which keeps expert compute on one L3 cache for CPU-share runs. Variant tokens: `t{N}` (`-t N`)
  and `ccd0` (`-C 0xFFFF --cpu-strict 1`). The CCD-to-logical-CPU mapping comes from `hw` and
  is verified there, not hard-coded.
* **CPU ISA** (AVX2 / AVX-VNNI / AVX-512 + extensions): read from llama.cpp's own `system_info`
  line in the server log, which reflects the flags the binary was built with, and compared with
  the CPU's real features (`IsProcessorFeaturePresent` / CPUID). A mismatch, such as AVX-512
  VNNI present but not compiled in, is reported as a build suggestion (§0.4).
* **Optimus / iGPU**: detect the integrated GPU alongside the 4080 and check that CUDA device 0
  is the 4080 (`llama-bench --list-devices`). A monitor or browser on the dGPU raises baseline VRAM,
  so the baseline is recorded before every run (§5).
* **Idle-state gate**: before a run, wait until GPU temperature, utilization and VRAM are back to
  baseline (shared with the thermal gating in §9.2).

**Fit estimate (adapted from Strata's fits / tight / spills):**
Strata uses measured constants for its own engine. For llama.cpp, the engine gives better answers:
1. GGUF metadata (`gguf` Python package): layers, experts, expert tensor bytes, attention layers,
   head dims. This gives KV bytes per token per K/V type and the split between experts and dense weights.
2. `llama-fit-params` with the variant's args, or a short `--fit off` load that parses
   `common_memory_breakdown_print` / `llama_kv_cache: size` / `load_tensors: offload` from the log.
   That gives exact device and host buffer sizes for this build.
3. Verdict: **fits** (VRAM headroom ≥ reserve, RAM spare ≥ 4 GB), **tight** (little RAM left,
   which is the freeze zone in your notes), **spills** (needs Windows shared GPU memory), **no-fit**.
   Values are cached per (hardware hash, build hash, model identity, variant).
4. Use: `sweep run` skips **no-fit** variants and asks before **tight** ones; `fit --suggest` (§4.6) suggests
   the smallest `-ncmoe N` or the largest `--moe-cache-mib` that still fits.

**Module and CLI:**

```
mse_llm/hw/
├── __init__.py      # detect(force=False) -> HardwareInfo (cached 30 s, like Strata), identity(), checks()
├── memory.py        # psutil + Win commit/pagefile
├── cpu.py           # model, cores, hybrid + P-core ids (ctypes GetLogicalProcessorInformationEx / sysfs)
├── gpu_nvidia.py    # NVML inventory
├── gpu_windows.py   # DirectX registry adapters, LUID, virtual-adapter filter
├── power.py         # AC/battery, power plan, NVML power limit
├── processes.py     # other engines + GPU compute processes
├── model_id.py      # size + head/mid/tail digest, GGUF metadata
└── fit.py           # fit estimate + verdict
```

```
mse-llm hw                       # table: host, CPU (hybrid, P-cores), RAM/pressure, GPUs, PCIe, power, disk, checks, verdict
mse-llm hw --json                # HardwareInfo as JSON (stored into each RunResult.host)
mse-llm hw identity              # hard/soft identity + hash; compare with the build profile's [target]
mse-llm hw processes             # other engines / GPU processes that would skew a bench
```

(The memory-fit check is the top-level `mse-llm fit`, §4.6. It uses `hw` for free VRAM.)

What `hw` detects that the earlier drafts of the build profile stored by hand, measured on this
laptop on 2026-11-18:
* **GPU:** RTX 4080 Laptop, 12281 MiB, cc 8.9, 192-bit GDDR6 at 18 Gbps (~432 GB/s), **PCIe 4.0 x16**
  (idles at Gen2, so the link is read under load), driver 617.42.
* **CPU:** Ryzen 9 7945HX, 16C/32T, not hybrid, AVX-512 incl. VNNI/VBMI/BF16.
* **RAM:** 79.2 GiB DDR5-5200 (rated 5600), **16 GB + 64 GB, asymmetric**: 32 GB dual-channel
  (~83 GB/s), 48 GB single-channel (~42 GB/s). Read from `Win32_PhysicalMemory`.
* **iGPU:** AMD Radeon (integrated). `hw` checks that CUDA device 0 is the 4080.
* **Driver setting** *CUDA – Sysmem Fallback Policy* (§9 item 7).

`doctor` runs `hw` checks first. `build run` checks `[target].gpu_match` / `cpu_match` against the detection.
`bench`/`benchy`/`sweep` take a `HardwareInfo` snapshot at start and refuse to run on a hardware
identity mismatch with the engine's build profile, on battery, or with another engine running
(each overridable). The code is a reimplementation in Python; where logic follows Strata closely
(DirectX registry LUID, hybrid detection, model digest), the module docstring credits it (MIT).

---

## 6. CLI surface (click)

```
mse-llm
├── build list|clone|plan|show|run|info|versions|explain|new   # llama.cpp build profiles, clone into builds/, preview, explained flags, versions (§0.4)
├── hw [--json]|identity|fit|processes   # PC capability detection (§5.1)
├── doctor                    # hw checks, binaries, CUDA device, NVML/PDH, uv/uvx, ports free, build toolchain
│   └── flags [--engine E]    # diff --help vs snapshot; list added/removed/changed flags
├── help {server|bench|benchy|eval} [--engine E]
├── models list|show NAME
├── variants decode NAME | encode ... | expand PATTERN
├── fit GGUF|MODEL --ctx N [-v VARIANT ... | --from FILE|-] [--suggest]   # expected VRAM/RAM per variant (§4.6)
├── predict MODEL [--ctx N] [--kv K/V] [--axes ncmoe,ub]   # VRAM fit + TG/PP prediction before running (§4.6)
├── bench   MODEL [-v VARIANT ...] [--ctx N] [--tg ...] [--sweep NAME] [--dry-run]
├── benchy  MODEL [-v VARIANT ...] [--ctx N] [--pp ...] [--depth ...] [--concurrency ...]
├── serve   MODEL [-v VARIANT] [--ctx N] [--spec] [--output all|filter|none] | --preset NAME [--file F]
├── presets list|validate|fix|dump|generate|serve|status|load|unload|watch|stop   # router from INI (§4.3)
├── eval    MODEL TASK [-v VARIANT] [--limit N] [--remote-client]   # mmlu_pro, gpqa_diamond, math_aime_24 local; swe_bench via Linux client (§7.1)
│   └── import FILE                                  # join a Linux-side SWE-bench/inspect result with the server-side run
├── quality kld-ref|kld|floor MODEL [-v VARIANT ...]  # KL divergence vs f16-KV reference (§4.5)
├── sweep   run NAME [--resume] [--only-failed] | list | show NAME
├── results list|show RUN_ID|compare A B|export --csv
└── report  chart [--sweep NAME|--runs ...] [--legacy-log FILE] | table | prompt
```

Global options: `--config PATH`, `--engine`, `--dry-run` (print the commands, like
`MSE_SERVER_PRESETS_DUMP`), `--json` (machine-readable output), `-v/--verbose`, `--no-notify`.

click runs synchronously, so each command calls `asyncio.run(main_coro(...))` through a small
`@async_command` decorator.

**Shell independence:** the tool never runs a shell, and all commands are argv lists. A `tasks`
runner (`poethepoet` in pyproject or a `justfile`) covers common chains, such as
`uv run poe bench-gemma`, which run the same from PowerShell, cmd, Git Bash, and Linux shells.

---

## 7. Strata (and other engines)

Facts from the Strata repo (v0.2.15): GGUF models (Qwen3.8-Flash-Next, Qwen3.6-35B-A3B), its own
CUDA/HIP/SYCL/Metal engine, CLI `strata launch --family qwen --model q2_0 --yes --no-tui`,
OpenAI- and Anthropic-compatible server on `127.0.0.1:8080` (`/v1/chat/completions`,
`/v1/responses`, `/v1/messages`), `/metrics` endpoint, flags `--context`, `--kv int8|q4_0|k8v4|fp16`,
`--parallel`, `--mtp on|off`, `--drafting auto|mtp|dflash2|off`, `--tune`, `--set KEY=VALUE`,
`--port`, API key on by default (`--no-api-key` / `--show-key`), `launch --stop/--status`.
No health route is documented. Not yet checked: `docs/DETAILS.md`, which documents the full settings.

Design implications, already covered above:

* `StrataRunner.build_command` maps the common Variant fields (ctx, kv, parallel, spec) to Strata
  flags. Unmappable fields fail validation. Engine-specific knobs go through `extra`/`--set`.
* Readiness probe: `GET /v1/models` with retry (or `/metrics`), not `/health`.
* Lifecycle: Strata manages its own daemon (`launch` returns), so `ServerHandle` needs a
  **"managed" mode**: start = `strata launch ... --no-tui`, stop = `strata launch --stop`, PID from
  `--status`. The same mode fits Ollama and LM Studio (`lms server start`).
* Auth: `Endpoint` carries an optional API key, passed to benchy `--api-key` and inspect `OPENAI_API_KEY`.
* Metrics: optional `MetricsScraper` that reads Prometheus `/metrics` during a run. It works for both
  llama.cpp (`--metrics`) and Strata, and adds server-side tokens/s, KV usage, and queue depth.
* Fair comparison: same client (benchy) with the same pp/tg/depth/concurrency against both engines,
  so the `RunResult` rows stay comparable. The protocol from the video, built in as
  `sweep compare-engines`: **the same GGUF** on both engines, the same machine, llama.cpp with the
  MoE cache sized by the tuner and experts pinned (`-lm dio|none`), **drafting off on both, then on
  on both**, short and long prompts, PP and TG reported separately.
  Published reference points (different machines, so not comparable to each other): on an RTX 3090,
  tuned llama.cpp before the cache did 19.1 t/s, Strata did 77.6 with no drafting and 118.9 as
  shipped, and llama.cpp 1.0 with the cache did about 32–36.

Other candidates for the same interface: **vLLM** / **SGLang** (Linux/WSL), **Ollama**,
**LM Studio** (`lms`), **ik_llama.cpp** (just another `llamacpp` engine profile with its own
help snapshot).

### 7.1 SWE-bench with the client on Linux (`mse-llm eval --remote-client`)

SWE-bench (`swe_bench_verified`, and smaller sets like *SWE-bench Lite* or a *Verified mini*
subset) runs every task in a Docker container. Docker isn't usable on this Windows 11 laptop
(the setup needs VirtualBox, which conflicts with the Hyper-V/WSL2 that Docker Desktop needs). So the
work is split:
* **Windows 11 laptop:** only the **model server** (`llama-server` on the RTX 4080), plus mse-llm's
  monitoring of it.
* **Linux:** the **SWE-bench harness + Docker**, using the Windows server as its OpenAI-compatible
  endpoint. This can be a **Linux VM in VirtualBox on the same laptop**, or a separate Linux machine.

**How mse-llm supports it:**

```
mse-llm eval MODEL swe_bench_verified -v VARIANT --remote-client [--bind lan|vbox-hostonly|IP] [--api-key auto]
```

1. Starts `llama-server` like `serve` (fit pre-check, `--fit on`, monitoring + guard), but bound so the
   Linux side can reach it, and **with an API key**, because it's no longer localhost-only:
   * VirtualBox VM with **NAT**: the VM reaches the host at `10.0.2.2:8033`. The server must listen
     on `0.0.0.0` (or the host's VirtualBox adapter);
   * VirtualBox **host-only** adapter: the host is usually `192.168.56.1`;
   * separate Linux machine: the laptop's LAN IP.
2. Prints **the exact commands to run on Linux**: the endpoint, model name, API key, and a
   ready-to-paste command for the chosen harness (`inspect eval inspect_evals/swe_bench_verified
   --model openai/local --model-base-url http://10.0.2.2:8033/v1 …`, or the equivalent for
   SWE-bench's own harness or `mini-swe-agent`), plus a `curl …/v1/models` connectivity test.
3. Keeps the server running and monitored (VRAM, RAM, temperature, acceptance rate) until you stop
   it, or until `--max-hours` passes. The monitor timeline is stored with the run.
4. **`mse-llm eval import FILE`** later reads the result file produced on Linux (inspect log or
   SWE-bench report JSON) and joins it with the server-side run by time window, so pass rate and
   server metrics end up in one `RunResult`.

**Checks and notes:**
* `doctor --remote-client` checks that the port is reachable from outside localhost. A **Windows
  Firewall rule** for port 8033 is usually needed. That's a system setting: mse-llm prints the
  rule but doesn't add it.
* **RAM:** a VirtualBox VM takes its memory from the same 80 GB that holds the CPU-side experts
  (`-ncmoe`). The fit check uses *available* RAM, so start the VM before `eval` to get a correct
  number. SWE-bench Docker images also need a lot of disk inside the VM.
* SWE-bench runs are long and use many requests, so the server keeps `--parallel 1` unless you set
  more. Agent harnesses expect long context, so pick a variant that fits at the harness's context.
* Non-Docker evals (`mmlu_pro`, `gpqa_diamond`, `math_aime_24`) still run locally on Windows as before.

---

## 8. Results and reporting

`RunResult` (pydantic, JSON):
`run_id, sweep, timestamp, git (mse-llm + llama.cpp commit/build), host (HardwareInfo snapshot
§5.1: GPU, driver, CUDA, RAM, CPU/hybrid, power source/limit, hw identity hash), model identity, engine, tool, model, variant (canonical + full argv), ctx, pp, tg, depth, concurrency,
metrics {pp_tps, tg_tps, ttft, latency, stdev}, startup {ready_s, n_ctx, kv_size, offload,
fit_adjustments}, monitor {peaks, deltas, status}, quality {mean/median/p99/max KLD, same_top_p,
ppl_ratio, reference id} (§4.5), spec {type, n_max, drafted, accepted, acceptance_rate,
mean_accept_len, accepted_per_pos[]} (§4.5), status (ok|failed|aborted|skipped), log paths`.

* Store: `results/` **in this repo**. **For now not stored in git** (`/results/` git-ignored, 2026-11-19); the
  original idea below is to track the small JSON files later, so benchmark history travels with the code
  and every number can be traced to a commit. One JSON per run + `results/index.jsonl` (SQLite
  later if needed, generated from the JSON and not tracked). What is tracked and what stays local:

  ```
  results/
  ├── index.jsonl                      # tracked: one line per run (id, model, variant, build, hw hash, PP/TG, status)
  ├── runs/2026-11/<run_id>.json       # tracked: full RunResult (argv, hw snapshot, metrics, monitor summary, moe_cache)
  ├── builds/<build_hash>.json         # tracked: copy of build-manifest.json (builds/ itself is git-ignored)
  ├── hw/<hw_hash>.json                # tracked: hardware identity the runs refer to
  ├── reports/<sweep>/*.png|*.html     # tracked: charts and HTML reports (regenerable, small)
  ├── tune/<run_id>/recommendation.json  # tracked: moe-cache-tune result
  ├── tune/<run_id>/runs/              # local: tuner calibration runs + routing traces (large)
  ├── raw/<run_id>/                    # local: monitor timeseries (csv), full server/client logs
  └── servers/                         # local: PID files and state of --background servers
  ```

  The run JSON stays small (target < 50 KB; it stores summaries, not timeseries). The tool only
  writes files. It never commits, stages or deletes anything in `results/`: what gets saved to git,
  and how, is decided by you later, with plain git.
  `sweep run --resume` skips variants that already have an `ok` result for the same build.
* Parsing: `llama-bench -o jsonl` and `benchy --format json --save-result`. No more regex on
  markdown tables. The legacy regex parser stays only to import old logs.
* As built (M7): `mse_llm/results.py` (RunResult, store), `mse_llm/parsers/` (llama_bench, benchy),
  `mse_llm/bench.py` (jobs, sweeps, resume), `mse_llm/cli_bench.py` (`bench`, `benchy`, `sweep`, `results`).
  One RunResult per job: llama-bench's pp and tg lines (or benchy's pp+tg per prompt size / depth) are
  `tests[]`, and `metrics` holds the headline numbers of the bash chart (PP of the largest prompt, TG at
  depth 0). Also stored: `key` (tool + model + variant + ctx + params; `--resume` matches it with the build id),
  `fit` (verdict, GPU / host MiB, free, left), `startup` (llama-bench's effective settings, or the server's
  n_ctx / offload / ready time), `monitor` (the summary, CSV path), `build` (id = commit + hash of the
  manifest, copied to `results/builds/`), `hw` (hash, copied to `results/hw/`), `model_identity`, argv, logs.
  Measured size: 8 KB per run. `MSE_LLM_RESULTS_DIR` moves the whole tree (the tests use it, so they never
  write into the repo's `results/`).
* Every job is fit-checked first: a no-fit variant becomes a `skipped` result with the reason and the
  `-ncmoe` suggestion, without starting the tool (`--force` starts it anyway).
* `results list` / `results show` (with the reproduce command, §9 item 1) are in M7; `results compare` and
  `results export --csv` come with the reports (M8).
* Charts: port the 5-axis grouped bar chart as is (parity), then add:
  * Pareto chart: TG t/s vs peak VRAM (or vs shared-mem spill), one point per variant.
  * Heatmap: `n-cpu-moe` × `ub` → TG (or PP).
  * PP/TG vs ctx or depth line plots (benchy depth sweeps).
  * Spec decoding: acceptance rate and speedup vs `spec-draft-n-max`.
  * Optional self-contained HTML report (plotly), generated into `results/` from the stored runs.
* `report prompt`: AI analyst prompt from a Jinja template, with the result **data embedded as
  JSON/CSV** instead of "paste raw results below".

---

## 9. Other proposals

1. **Run manifest + reproducibility**: record the exact argv, env, and binary hash of each run.
   `results show` prints a copy-paste command to reproduce it.
2. **Cooldown/thermal gating** between runs: wait until GPU temp is below X °C and VRAM is back to
   baseline before the next variant. That removes the order bias in sweeps.
3. **Warm-up policy**: explicit warm-up run (today `extra_fa_default` serves as a hidden warm-up).
4. **Repetitions + statistics**: `-r`/`--runs` ≥ 3 with mean ± stdev in charts. Flag differences
   below noise.
5. **Quality guard for KV quant**: KL divergence per K/V combo (§4.5), plus optional short
   inspect evals. This follows the asymmetric-KV notes in the script: never accept a faster config
   without a quality check.
6. **`sweep search`: safe automatic search** (automates the manual "SWEETSPOT" hunt):
   * Order of settings from `predict` (§4.6): `n-cpu-moe` first for models with high
     `n_used/n_expert` (Gemma, GLM), `-b/-ub` first when many experts stay on the CPU (Qwen MoE).
     Then K/V types (checked with KLD, §4.5), then speculative decoding (`sd{N}cache`, `sd{N}mtp`,
     with acceptance rate).
   * Each stage starts from the previous winner, like "SWEETSPOT 1. step".
   * It starts from the **safe side** (high `n-cpu-moe`, small `ub`) and steps toward the edge.
     Configs that `predict` marks **no-fit** are never started. In each direction it stops at the
     first run that spills into shared GPU memory or trips the memory guard.
   * The found edge is stored per (model, build, hardware hash) as data (your "PC freeze" notes).
     The winner is taken **one step back from the edge**, because your notes show sporadic
     freezes right next to working configs.
   * Output: the fastest safe config, written as the model's `default_serve` and into the presets.
7. **Freeze prevention:**
   * `doctor` checks the NVIDIA driver setting **CUDA – Sysmem Fallback Policy** and recommends
     *Prefer No Sysmem Fallback* (Global, or per program for `llama-server.exe`, `llama-bench.exe`,
     `llama-perplexity.exe`). With it, a too-large config fails with CUDA out-of-memory instead of
     spilling into system RAM. mse-llm only reads and reports this setting; you change it in
     NVIDIA Control Panel.
     * Where: NVIDIA Control Panel → Manage 3D settings → **Global Settings** (all CUDA programs)
       or **Program Settings** → add `llama-server.exe`, `llama-bench.exe`, `llama-perplexity.exe`.
       Global is simplest; per program keeps games and other GPU apps unchanged.
     * The setting is **permanent**: it's stored in the driver's profile database and survives
       reboots and normal driver updates. A **clean** driver install resets it, so `doctor` checks
       it again after every driver version change.
     * It needs driver ≥ 536.40 (installed: 617.42). There's no environment variable or CUDA flag
       that does the same.
     * Program profiles match by file name, so one entry should cover every `builds\…\bin\Release`
       folder. `doctor` confirms this after the first build in a new folder.
     * Effect: configs that used to "run very slowly" or freeze now stop with a CUDA out-of-memory
       error. For `sweep search` that error marks the edge.
   * Optional Windows Job Object memory cap on the server process tree (`--mem-cap`, off by default),
     so it can't push the system into swap. Found in M6: under WDDM, **VRAM allocations count against
     the cap too** (a 2000 MiB cap made OmniCoder's 5 GB `cudaMalloc` fail), so the cap must cover
     VRAM + host buffers + RAM.
   * The monitor stops a run live on a GPU memory spill instead of reporting it afterwards. Not with
     the bash rule "shared memory grew by more than 100 MiB" (pinned `-ncmoe` host buffers count as
     shared memory), but with the guard rules A-C of §5.
8. **Port management**: pick a free port automatically for background servers, so a forgotten
   foreground server on 8033 can't collide with a benchy run.
9. **Notifications**: TTS plus optional desktop toast or ntfy/webhook for long sweeps.
10. **Logging**: `rich` console + structured JSON log file per run. Server logs prefixed `>>>`,
   client logs `<<<` (as today), kept on disk instead of `cat` + `rm`.
11. **CI**: GitHub Actions on windows-latest + ubuntu-latest running unit tests with the fake
    server and fixture help/bench outputs. No GPU needed.

**Backlog: ideas discussed but not chosen for now** (KL divergence and acceptance rate were chosen
and are in §4.5; freeze prevention is item 7):
* `build bisect`: binary-search `b<N>` tags with a 1-minute bench to find the commit that made a build slower.
* Canary run: build latest master, run a short fixed bench, warn when it's ±5 % off the baseline.
* Golden output check: a fixed prompt with greedy decoding, compared across builds and KV settings
  to catch correctness bugs.
* Sustained vs burst speed: a 20–30 min soak run with NVML throttle reasons (laptop heat and power limits).
* Replay your own prompts (captured with `--log-prompts-dir`): TTFT on multi-turn agent sessions,
  prefix-cache reuse (`--cache-reuse`, `-cram`).
* Energy per token from the NVML energy counter (tokens per joule, useful on battery).
* `recommend`: pick the best stored variant for a goal (e.g. max TG with ≥ 64 k ctx, no spill) and
  write it into the presets.
* `models scan`: scan `d:\.lmstudio\models`, read GGUF metadata and propose a starting config per model.

---

## 10. Milestones

**Order and why.** Each milestone only needs the ones before it, and the tools you use every day
come first:
1. **Foundation (M0–M2):** project skeleton, hardware detection, and the llama.cpp build. Until
   `builds\llama.cpp` is built, neither mse-llm nor the bash script (`MSE_LLAMA_BIN_DIR`) has
   binaries to run, so the build comes before anything that starts a model.
2. **Daily tools (M3–M5):** the variant decoder, then **`fit`** (it needs only the decoder and
   `llama-fit-params`), then **`serve` + `presets`** (your usual way of running models).
3. **Safe benchmarking (M6–M8):** monitoring and freeze prevention **before** the tool runs long
   benchmarks, then `bench`/`benchy`/`sweep` with the results store, then charts/reports.
4. **Smarter tuning (M9–M11):** `predict` and `sweep search` (they calibrate from M7 results), KLD
   and acceptance rate, then the MoE-cache axes.
5. **Evals and retirement (M12–M13):** evals incl. SWE-bench from Linux, then the bash script is
   deleted. At that point every parity row (F1–F15) is covered.
6. **After retirement (M14–M15):** Strata, and the extras and backlog. They're not needed to
   replace the script.

| M | Scope | Done when |
|---|---|---|
| **M0** | Commit the untracked `llama_cpp_benchmark.sh` + `llama_cpp_presets.ini` (already fixed) as the baseline (§10.1); uv project skeleton, click group, config models (TOML), `doctor` (incl. the NVIDIA Sysmem Fallback check), `help`, `doctor flags` with help snapshots; start `docs/PARITY.md` | `uv run mse-llm doctor` passes; snapshot diff reproduces §0 findings |
| **M1** | `mse_llm/hw/` (§5.1): detection (GPU, VRAM free, PCIe, CPU/ISA, RAM channels, power, driver setting), identity hash, checks/verdict, `hw` commands, `MSE_LLM_TEST_HARDWARE` fixtures | `mse-llm hw` shows the RTX 4080 Laptop (12281 MiB, cc 8.9, PCIe 4.0 x16), Ryzen 9 7945HX (16C/32T, not hybrid, AVX-512 + VNNI/VBMI/BF16), 16 + 64 GB asymmetric RAM, AC power; identity hash stable across runs |
| **M2** | `mse-llm build` (§0.4): `plan` / `clone` / `show` / `run` / `info` / `versions` / `explain`; clone llama.cpp + vcpkg into `builds/`, configure, build, smoke test, verify flags, `build-manifest.json`, help snapshot + flag diff; default variant `zen4-avx512x` + `--without` | `build plan` lists every step without executing anything; `build run rtx4080-laptop` asks before step 1, clones into `builds/` and builds `builds\llama.cpp\build\bin\Release` (the folder the bash script uses); `build run … --without zen4-avx512x` builds the comparison; the flag diff vs `a49db739f` is shown; `presets validate` still passes on the fixed INI |
| **M3** | `variants.py` codec + matrix (incl. legacy `extra_` names; turbo* → "dropped" error); llama.cpp command builders; flag validation against help snapshots; golden tests vs bash command lines | Every bash `run_*` function has an equivalent `--dry-run` with identical argv (minus fixed bugs) |
| **M4** | `mse-llm fit` (§4.6): GGUF/model + variants (`-v`, brace patterns, pasted bash arrays with comments) + `--ctx` → `llama-fit-params --fit-print` per variant, per-part breakdown with source (measured / estimated / NOT COUNTED) for drafter + mmproj, verdicts `fits` / `fits~` / `UNSURE`, `--strict`, `--suggest` | Pasting the Qwen28-REAP `extras_list` reproduces the §4.6 table; an `sdNmtp` variant without a configured drafter shows `UNSURE`; `--suggest` proposes the lower `-ncmoe` |
| **M5** | `proc.py` + `ServerHandle` (async spawn, readiness, tree kill, port-free check); `serve` (+ `--preset`, `--fit on` with the `fit` pre-check warning, §0.3); `presets *` (§4.3) | Server starts/stops cleanly 50× in a loop on Windows with no orphan processes; `serve` warns before a bad fit and shows what `--fit` would change; `presets serve` + `presets load` switch between two models through one endpoint |
| **M6** | `monitor/` (NVML, PDH, psutil) + memory guard + **freeze prevention** (§9 item 7: live spill / RAM / commit stop with the guard rules of §5, optional Job Object cap) + notify | Peaks match the bash monitor within tolerance; a forced over-commit run is stopped before the laptop stalls |
| **M7** | `bench` (llama-bench jsonl) + `benchy` (json) + results store in `results/` + `sweep run --resume` | Old sweeps from the script run from TOML; results JSON written. As built: all 9 bash group functions are `[sweeps.*]` (161 entries, comments as status / note / tag), `sweep list/show/run`, `results list/show`, fit pre-check per job, guard rule D (hang) |
| **M8** | `report chart` (parity 5-axis + legacy log import), `report table`, `report prompt` | Chart from new results matches the old chart for the same data; old `.txt` logs still chart |
| **M9** | §4.6 `predict` (GGUF expert sizes, `llama-fit-params` grid, roofline TG/PP, calibration from M7 runs) + §9 item 6 `sweep search` (safe side first, stops at the first spill, winner one step back) | After calibration, predicted TG/PP are within ~15 % of measured for 2 models; `sweep search` finds a sweet spot without a single run that spills |
| **M10** | §4.5: `quality kld-ref/kld/floor` (llama-perplexity, size preview, wikitext-2 + code corpus, reference fallbacks); `--metrics` on bench servers + `/metrics` delta scraping, spec log parsing, `RunResult.quality` / `.spec` | KLD table for 3 K/V variants of one model, with noise floor; a `sd{2,3,4}mtp` benchy sweep shows acceptance rate + per-position acceptance next to TG |
| **M11** | MoE-cache axes (§4.4): `cmoe/mc/cs/lm*/Kf8` tokens, staged sweeps, server-log cache stats, `tune moe-cache` wrapper | Stage 0–2 sweep for Qwen36 MTP completes; chart shows TG/PP/duration/VRAM/hit rate per variant |
| **M12** | `eval` (inspect-ai: mmlu_pro, gpqa_diamond, math_aime_24 locally); `eval --remote-client` + `eval import` for SWE-bench from a Linux VM/host (§7.1) | mmlu_pro `--limit 20` completes against background server; a VirtualBox Linux VM reaches the Windows server (`curl /v1/models`), runs a few SWE-bench (Lite/mini) tasks, and `eval import` joins the result with server metrics |
| **M13** | Retire the bash runner (§10.1): parity checklist complete, final acceptance runs, delete `llama_cpp_benchmark.sh` + `llama_cpp_presets.ini` | Every row in `docs/PARITY.md` is ✓ or "dropped + reason" approved by you; both files removed in one commit, recoverable from git history |
| **M14** | Runner registry + `StrataRunner` (managed mode, `/v1/models` probe, API key, `/metrics` scrape); Linux AMD/Intel GPU detection | Same benchy sweep runs against llama.cpp and Strata; `results compare` works |
| **M15** | Extras: Pareto/heatmap charts, HTML report, thermal gating, items from the §9 backlog | — |

**First usable checkpoints:** after **M2** the bash script works again (binaries in `builds\`).
After **M4** you can check any variant list's memory fit before running it. After **M5**
`mse-llm serve` / `presets serve` replace `run_SERVER_*` / `run_PRESETS_serve`.

### 10.1 Retiring `llama_cpp_benchmark.sh` (milestone M13)

Once **everything** the script does is implemented in `mse-llm` and checked, the script is
**deleted** from the repo. It isn't kept as a second, drifting copy.

**Before any code is written:** `llama_cpp_benchmark.sh` and `llama_cpp_presets.ini` are
currently **untracked** (`git status: ??`). Deleting an untracked file removes it for good, so
both are committed first (M0). After that the deleted script stays recoverable from git history
(`git show <commit>:llama_cpp_benchmark.sh`).

**Parity checklist.** Every item must be ✓ before deletion. It's kept as
`docs/PARITY.md` and updated as milestones land:

| Script feature (bash) | Replacement | Check |
|---|---|---|
| F1 `run_HELP_server/bench__*` | `mse-llm help server\|bench` | output identical to `--help` |
| F2 `MODELS`, `DRAFTER_MODELS`, `MODELS_BENCH_EXTRA`, `MODELS_SERVE_EXTRA` | `[models.*]` in config | every model + extra present |
| F3 `EXTRAS` matrix + `EXTRAS_get_flags_normalized/_replaced` (`_moeN`, `_sdNcache`, `_sdNmtp`, `MODEL`/`BUILD-IN` drafter) | `variants.py` codec | golden test: every key the script can generate decodes to the same flags |
| F4 `run_bench_params*`, `_wrapper_EXTRAS`, `_wrapper_PROMPTS` | `mse-llm bench` | same argv in `--dry-run` (bugs from §0.1 fixed) |
| F5 `run_benchy_params_with_server*` (PP sync, depth, runs) | `mse-llm benchy` | same server + benchy argv, `--pp` list passed correctly |
| F6 `run_serve_params_on_background`, `server_wait_for_ready`, `server_stop_windows` | `ServerHandle` | 50× start/stop, no orphan processes (M5) |
| F7 `run_serve_params` + `run_serve_params_FILTER`, output modes | `mse-llm serve --output` | same filtered lines |
| F8 `run_presets_serve`, `run_PRESETS_serve`, `run_PRESETS_dump`, `run_presets_dump` | `mse-llm presets serve\|dump\|generate` (+ `validate`, `status`, `load/unload`) | `presets serve` on the existing INI serves all 5 models; generated INI loads too |
| F9 `EXEC_cmd_with_gpu_and_ram_monitoring` (VRAM, shared, RAM, temp + statuses) | `monitor/` | peaks within tolerance of the bash monitor (M6) |
| F10 `run_dump_params_wrapper_PLOTER` (5-axis chart, auto-detect) | `mse-llm report chart` (+ `--legacy-log`) | old logs give the same chart |
| F11 `run_dump_params_wrapper_AI_PROMPT/_COMMON/_EXTRAS/_PROMPTS` | `mse-llm report prompt` | prompt contains the same sections + data |
| F12 `run_inspect_params*` (mmlu_pro, gpqa_diamond, math_aime_24, swe_bench_verified) | `mse-llm eval` | real eval runs (the script's version is a stub) |
| F13 `EXEC_user_notification_WIN_T2S` | `notify.py` | voice message at end of run |
| F14 `EXEC_cmd_with_output_mode` (all/filter/none) | `--output` | — |
| F15 `MSE_*` knobs (`MSE_THREADS`, `MSE_PARALLEL`, `MSE_BENCH_TG`, `MSE_BENCHY_*`, `MSE_SERVER_READY_TIMEOUT`, `MSE_MONITOR_ENABLED`, `MSE_PS_DEBUG`, `MSE_SERVER_PRESETS_DUMP`, `MSE_PRESETS_SERVE_FILE`, `MSE_TEST_INSPECT_PATH`) | config fields / `MSE_LLM_*` env / `--dry-run` | each knob mapped or explicitly dropped with a reason |
| Every `run_BENCH_*`, `run_BENCHY_WITH_SERVER_*`, `run_SERVER_*`, `run_INSPECT_*` group function (model-specific lists, incl. commented-out entries and their notes "SWEETSPOT", "PC freeze") | `[sweeps.*]` + `annotations` in config | one sweep or serve entry per function; notes preserved as data |
| Model notes in comments (HF URLs, sizes, capabilities) | `[models.*].url`, `notes` | nothing lost |

**Deletion step (M13), after all rows are ✓:**
1. Run `mse-llm` for one real sweep per model family, and one `serve` + `presets serve`, as
   the final acceptance check.
2. Generate `config/presets.ini` with `mse-llm presets generate`, check it serves the same models,
   then **delete `llama_cpp_presets.ini`** too (it's replaced by the generated file).
3. `git rm llama_cpp_benchmark.sh llama_cpp_presets.ini` in a commit named, for example,
   `Remove bash runner: replaced by mse-llm (see docs/PARITY.md)`, with the parity table
   linked in the commit message.
4. Remove the legacy references that are no longer needed: the `extra_` prefix stays accepted by the
   variant codec, and `report chart --legacy-log` stays for importing old result logs. Those are
   about old *data*, not the script.

If one feature is intentionally dropped, the parity table records "dropped + reason" instead of ✓.
Already approved: the **turbo2/3/4 cache variants** (`*_Kt*`, `*_Vt*`): dropped, not supported by
upstream llama.cpp (2026-11-18); the **`MSE_PS_DEBUG`** knob: dropped, it only printed `ps aux` around the
bash PID handling, which `ServerHandle` replaced (2026-11-19). Dropped items
need your explicit OK before the script is deleted.

---

## 11. Open questions

1. ~~turbo* cache types~~ **Decided: dropped** (no fork profile; parity table "dropped + reason").
2. ~~Results location~~ **Decided:** results live inside this repo under `results/`. Nothing is
   read from or written to `_MSE_doc\AI\MODELS\app_llamacpp\` (that folder is being removed).
   `results/` was planned as **tracked in git**; **update 2026-11-19: not stored in git for now** (git-ignored).
   Which files get committed later, and how, is your decision;
   the tool only writes them (§8).
3. ~~Default `--fit` for serve~~ **Decided:** `serve` uses `--fit on`, and the `fit`
   pre-check **warns** about a bad fit and shows what `--fit` would change before starting.
   Bench/benchy/sweep/quality use `--fit off` (§0.3). **Update (M5):** `--fit` can't shrink the
   settings `serve` passes (`-ngl all`, `-ncmoe`); the pre-check says so, `serve --let-fit` lets
   `--fit` place the layers, and the memory guard (§5) stops a spill.
4. ~~Docker check for SWE-bench~~ **Decided:** Docker doesn't run on this Windows 11 laptop
   (VirtualBox). The model server runs on Windows, and SWE-bench + Docker run on Linux (a VirtualBox
   VM or another machine) against that endpoint: `eval --remote-client` + `eval import` (§7.1).
5. ~~Which llama.cpp checkout~~ **Decided:** a **new clone in `builds/llama.cpp`**, managed by
   mse-llm (the default in `rtx4080-laptop.toml`), with vcpkg created fresh in `builds/vcpkg`.
   The **bash script uses it too** (`MSE_LLAMA_BIN_DIR`). No other checkout is referenced.
6. ~~zen4-avx512x default~~ **Decided:** it's part of the default build (`apply_variants`), kept in
   its own `[variants.zen4-avx512x]` table; `--without zen4-avx512x` builds the comparison.
