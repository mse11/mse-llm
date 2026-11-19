"""In-process monitoring + memory guard (§5, §9 item 7). Replaces bash EXEC_cmd_with_gpu_and_ram_monitoring.

    async with Monitor(interval=0.5, guard=GuardCfg(...), on_trip=stop_server) as mon:
        mon.track(server_pid)          # the server's whole process tree
        ...
    print(mon.summary())

Samplers (NVML, PDH, psutil) run concurrently in threads, once per tick. No powershell / nvidia-smi.

**Spill detection.** The bash rule ("shared delta > 100 MiB = offloaded to system RAM") is wrong for MoE
offload: llama.cpp's pinned host buffers for CPU experts count as shared GPU memory (15.9 GB for a fitting
Qwen3.6-28B `moe41`). Measured on this laptop, a real spill (Qwen3.6-35B `moe10`, 17.2 GB on a 12 GB GPU)
looks like this: dedicated VRAM fills up first (12140 / 12282 MiB), then the server's shared memory keeps
growing while dedicated stays full. The guard therefore stops a run when

  A. dedicated VRAM is full (free <= vram_full_mib) and the server's shared memory has grown by more than
     spill_stop_mib since it became full, or
  B. the server's shared memory exceeds the host buffers `fit` expects (expected_host_mib) by more than
     spill_stop_mib (or host_tolerance_pct of the expected size, whichever is larger), or
  C. system RAM or the commit charge passes ram_stop_pct / commit_stop_pct.

For benchmarks (stall_stop_s set) a no-fit config can also *hang* instead of spilling (found in M7: llama-bench
with the 35B at -ncmoe 10 sat 10 min at 0 % GPU, VRAM nearly full, one thread spinning in the driver), so:

  D. no progress for stall_stop_s (GPU idle, no disk reads by the tree, its VRAM unchanged), or the GPU driver
     not answering NVML calls for that long (a hung llama-bench blocks NVML for every process).

Each sampler runs in its own daemon thread with a time limit, so a blocked driver call never stops the
monitor loop (the other samplers and the guard keep running) and never keeps mse-llm from exiting.
"""

from __future__ import annotations

import asyncio
import csv
import inspect
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

from mse_llm.util import IS_WINDOWS

Sample = dict[str, Any]


# ---------------------------------------------------------------- guard

@dataclass
class GuardCfg:
    spill_stop_mib: int = 100
    vram_full_mib: int = 256
    ram_stop_pct: float = 95.0
    commit_stop_pct: float = 95.0
    expected_host_mib: float | None = None
    host_tolerance_pct: float = 2.0
    host_extra_mib: float = 0.0          # drafter / mmproj buffers that may be host or VRAM (fit can't tell)
    enabled: bool = True
    # no progress for this long (GPU idle, no disk reads, VRAM unchanged) = stalled; None = off.
    # Only for benchmarks: an idle server waiting for requests looks the same.
    stall_stop_s: float | None = None


class Guard:
    def __init__(self, cfg: GuardCfg, baseline: Sample) -> None:
        self.cfg = cfg
        self.baseline = baseline
        self._full_at: float | None = None      # server shared MiB when dedicated VRAM became full
        self._stall_ref: tuple[float, float, float] | None = None   # (t, read MiB, VRAM MiB) at the last progress
        self._util: list[float] = []                 # GPU use since then

    STALL_GPU_PCT = 5.0       # mean GPU use below this over the window = idle (blips from other apps average out)

    def _stalled(self, s: Sample) -> str | None:
        """A config that doesn't fit can also hang instead of spilling: the process allocates ~10 GB of VRAM and
        then waits in the driver, reading nothing from disk, GPU idle (llama-bench + 35B at -ncmoe 10: 140 s to
        10+ min). A fitting run never looks like that: it reads the GGUF the whole load (21 GB in 25 s), then
        the GPU runs at ~99 %. Progress = disk reads or VRAM change of the tree; GPU use is averaged over the
        window because the device-wide value has blips from other programs."""
        lim = self.cfg.stall_stop_s
        if not lim or "proc_rss_mib" not in s:           # off, or the process doesn't exist yet
            return None
        t = float(s.get("t") or 0.0)
        stuck = s.get("stuck_nvml_s")
        if stuck is not None and stuck >= lim:
            return (f"GPU driver not responding: NVML calls blocked for {stuck:.0f} s (a hung CUDA process holds the "
                    "driver); a config that doesn't fit can hang like this")
        read = float(s.get("proc_read_mib") or 0)
        vram = float(s.get("proc_vram_mib", s.get("vram_used_mib")) or 0)
        util = s.get("gpu_util_pct")
        ref = self._stall_ref
        if ref is None or abs(read - ref[1]) >= 1 or abs(vram - ref[2]) >= 16:
            self._stall_ref = (t, read, vram)
            self._util = []
            return None
        if util is not None:
            self._util.append(float(util))
        if t - ref[0] < lim:
            return None
        mean = sum(self._util) / len(self._util) if self._util else 0.0
        if mean >= self.STALL_GPU_PCT:                    # computing: progress, start a new window
            self._stall_ref = (t, read, vram)
            self._util = []
            return None
        return (f"no progress for {t - ref[0]:.0f} s: GPU idle (mean {mean:.0f} %), no disk reads, VRAM unchanged at "
                f"{vram:.0f} MiB ({s.get('vram_free_mib')} MiB free); a config that doesn't fit can hang like this")

    def shared(self, s: Sample) -> float | None:
        """The server's shared GPU memory: per process when PDH has it, else the adapter's growth."""
        if "proc_shared_mib" in s:
            return s["proc_shared_mib"]
        if "shared_mib" in s and "shared_mib" in self.baseline:
            return s["shared_mib"] - self.baseline["shared_mib"]
        if "nonlocal_mib" in s and "nonlocal_mib" in self.baseline:
            return s["nonlocal_mib"] - self.baseline["nonlocal_mib"]
        return None

    def check(self, s: Sample) -> str | None:
        c = self.cfg
        if not c.enabled:
            return None
        sh = self.shared(s)
        if sh is not None and c.expected_host_mib is not None:
            allow = max(c.spill_stop_mib, c.expected_host_mib * c.host_tolerance_pct / 100) + c.host_extra_mib
            over = sh - c.expected_host_mib
            if over > allow:
                return (f"GPU memory spill: the server uses {sh:.0f} MiB of shared GPU memory, {over:.0f} MiB more "
                        f"than the {c.expected_host_mib:.0f} MiB of host buffers `fit` expects"
                        + (f" (+{c.host_extra_mib:.0f} MiB allowed for drafter / mmproj)" if c.host_extra_mib else ""))
        free = s.get("vram_free_mib")
        if sh is not None and free is not None:
            if free <= c.vram_full_mib:
                if self._full_at is None:
                    self._full_at = sh
                elif sh - self._full_at > c.spill_stop_mib:
                    return (f"GPU memory spill: dedicated VRAM is full ({s.get('vram_used_mib')} of "
                            f"{s.get('vram_total_mib')} MiB) and the server's shared GPU memory grew by "
                            f"{sh - self._full_at:.0f} MiB since")
            else:
                self._full_at = None
        if s.get("ram_pct", 0) >= c.ram_stop_pct:
            return f"system RAM at {s['ram_pct']:.0f} % (stop at {c.ram_stop_pct:.0f} %)"
        if s.get("commit_pct", 0) >= c.commit_stop_pct:
            return (f"commit charge at {s['commit_pct']:.0f} % of the limit ({s.get('commit_mib')} of "
                    f"{s.get('commit_limit_mib')} MiB; stop at {c.commit_stop_pct:.0f} %)")
        return self._stalled(s)


# ---------------------------------------------------------------- summary

@dataclass
class Stat:
    baseline: float | None
    peak: float
    mean: float
    p95: float

    @property
    def delta(self) -> float | None:
        return None if self.baseline is None else self.peak - self.baseline


def _p95(xs: list[float]) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, math.ceil(0.95 * len(s)) - 1))]


@dataclass
class Status:
    area: str
    level: str          # pass | warn | fail
    text: str


@dataclass
class Summary:
    samples: int
    interval: float
    duration_s: float
    stats: dict[str, Stat]
    statuses: list[Status]
    tripped: str | None = None
    throttle: list[str] = field(default_factory=list)
    spill_mib: float | None = None
    expected_host_mib: float | None = None
    csv_path: str | None = None
    errors: list[str] = field(default_factory=list)

    def get(self, key: str) -> Stat | None:
        return self.stats.get(key)

    def as_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples, "interval": self.interval, "duration_s": round(self.duration_s, 1),
            "tripped": self.tripped, "throttle": self.throttle, "spill_mib": self.spill_mib,
            "expected_host_mib": self.expected_host_mib, "csv": self.csv_path, "errors": self.errors,
            "stats": {k: {"baseline": v.baseline, "peak": v.peak, "mean": round(v.mean, 1), "p95": v.p95,
                          "delta": v.delta} for k, v in self.stats.items()},
            "statuses": [s.__dict__ for s in self.statuses],
        }


@dataclass
class Thresholds:
    """The bash status thresholds."""
    spill_mib: int = 100
    ram_warn_pct: float = 60
    ram_fail_pct: float = 80
    temp_warn_c: float = 75
    temp_fail_c: float = 85


def summarize(samples: list[Sample], baseline: Sample | None, *, interval: float, guard: Guard | None = None,
              tripped: str | None = None, th: Thresholds | None = None, errors: list[str] | None = None) -> Summary:
    th = th or Thresholds()
    base = baseline or {}
    keys = sorted({k for s in samples for k, v in s.items() if isinstance(v, (int, float)) and k not in ("t",)})
    stats: dict[str, Stat] = {}
    for k in keys:
        xs = [float(s[k]) for s in samples if isinstance(s.get(k), (int, float))]
        if xs:
            b = base.get(k)
            stats[k] = Stat(float(b) if isinstance(b, (int, float)) else None, max(xs), sum(xs) / len(xs), _p95(xs))
    dur = samples[-1]["t"] if samples else 0.0
    out: list[Status] = []
    exp = guard.cfg.expected_host_mib if guard else None
    spill = None
    sh_peak = max((guard.shared(s) or 0) for s in samples) if guard and samples else None
    if sh_peak is not None:
        spill = sh_peak - (exp or 0)
    nl = stats.get("nonlocal_mib") or stats.get("shared_mib")
    bash_spill = bool(nl and (nl.delta or 0) > th.spill_mib)
    if tripped and ("spill" in tripped or "no progress" in tripped):
        out.append(Status("gpu", "fail", f"stopped: {tripped}"))
    elif spill is not None and exp is not None:
        extra = guard.cfg.host_extra_mib if guard else 0.0
        lvl = "fail" if spill > max(th.spill_mib, exp * 0.02) + extra else "pass"
        out.append(Status("gpu", lvl, f"server shared GPU memory peak {sh_peak:.0f} MiB, expected host buffers "
                          f"{exp:.0f} MiB → {'spill' if lvl == 'fail' else 'no spill'} ({spill:+.0f} MiB)"))
    elif bash_spill:
        out.append(Status("gpu", "warn", f"shared GPU memory +{nl.delta:.0f} MiB: the bash rule calls this "
                          "\"offloaded to system RAM\", but pinned host buffers (-ncmoe experts) count too; "
                          "give the expected host size (fit) to tell them apart"))
    elif nl is not None:
        out.append(Status("gpu", "pass", "run contained within dedicated VRAM"))
    ram = stats.get("ram_pct")
    if ram:
        lvl = "fail" if ram.peak > th.ram_fail_pct else "warn" if ram.peak > th.ram_warn_pct else "pass"
        txt = {"fail": f"high RAM usage (>{th.ram_fail_pct:.0f} %)", "warn": f"moderate RAM usage "
               f"({th.ram_warn_pct:.0f}-{th.ram_fail_pct:.0f} %)", "pass": "normal RAM usage"}[lvl]
        out.append(Status("ram", lvl, txt))
    if tripped and "spill" not in tripped and "no progress" not in tripped:
        out.append(Status("ram", "fail", f"stopped: {tripped}"))
    t = stats.get("gpu_temp_c")
    if t:
        lvl = "fail" if t.peak > th.temp_fail_c else "warn" if t.peak > th.temp_warn_c else "pass"
        txt = {"fail": f"GPU running hot (>{th.temp_fail_c:.0f} °C), thermal throttling may occur",
               "warn": f"GPU warm ({th.temp_warn_c:.0f}-{th.temp_fail_c:.0f} °C)", "pass": "GPU temperature normal"}
        out.append(Status("temp", lvl, txt[lvl]))
    from mse_llm.monitor.nvidia import throttle_names

    mask = 0
    for s in samples:
        mask |= int(s.get("gpu_throttle") or 0)
    return Summary(len(samples), interval, dur, stats, out, tripped, throttle_names(mask), spill, exp,
                   errors=list(errors or []))


# ---------------------------------------------------------------- monitor

def default_samplers(pcie: bool = True) -> list[Any]:
    from mse_llm.monitor.nvidia import NvmlSampler
    from mse_llm.monitor.system import SystemSampler

    out: list[Any] = [NvmlSampler(0, pcie=pcie), SystemSampler()]
    if IS_WINDOWS:
        from mse_llm.monitor.windows import PdhSampler, nvidia_luid

        out.insert(1, PdhSampler(nvidia_luid()))
    return out


class Monitor:
    def __init__(self, interval: float = 0.5, *, guard: GuardCfg | None = None, samplers: list[Any] | None = None,
                 csv_path: str | Path | None = None, on_trip: Callable[[str], Any] | None = None,
                 on_sample: Callable[[Sample], Any] | None = None, thresholds: Thresholds | None = None) -> None:
        self.interval = interval
        self.guard_cfg = guard
        self.samplers = samplers if samplers is not None else default_samplers()
        self.csv_path = Path(csv_path) if csv_path else None
        self.on_trip = on_trip
        self.on_sample = on_sample
        self.thresholds = thresholds
        self.samples: list[Sample] = []
        self.baseline: Sample | None = None
        self.guard: Guard | None = None
        self.tripped: str | None = None
        self.trip_event = asyncio.Event()
        self._pid: int | Callable[[], int | None] | None = None
        self._t0 = 0.0
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self.sample_timeout = max(1.0, 2 * interval)
        self._pending: dict[str, asyncio.Future] = {}
        self._stuck_since: dict[str, float] = {}

    @property
    def errors(self) -> list[str]:
        return [s.error for s in self.samplers if getattr(s, "error", None)]

    def track(self, pid: int | Callable[[], int | None] | None) -> None:
        """Follow this process and its children (router models included). A callable is asked every tick,
        so a monitor can start before the server (baseline first) and pick the pid up once it exists."""
        self._pid = pid

    def pids(self) -> set[int]:
        pid = self._pid() if callable(self._pid) else self._pid
        if pid is None:
            return set()
        try:
            p = psutil.Process(pid)
            return {p.pid, *(c.pid for c in p.children(recursive=True))}
        except psutil.Error:
            return set()

    async def _sample(self) -> Sample:
        """One tick: every sampler in its own daemon thread, at most sample_timeout s. A sampler still busy from
        an earlier tick is skipped and reported as stuck_<name>_s (seconds since it stopped answering)."""
        pids = await asyncio.to_thread(self.pids)
        now = time.monotonic()
        futs: dict[str, asyncio.Future] = {}
        for s in self.samplers:
            name = getattr(s, "name", type(s).__name__)
            busy = self._pending.get(name)
            if busy is not None and not busy.done():
                continue
            self._pending.pop(name, None)
            self._stuck_since.pop(name, None)
            futs[name] = _in_thread(s.sample, pids)
        if futs:
            await asyncio.wait(list(futs.values()), timeout=self.sample_timeout)
        out: Sample = {"t": round(now - self._t0, 2)}
        for name, f in futs.items():
            if not f.done():
                self._pending[name] = f
                self._stuck_since.setdefault(name, now)
            elif f.exception() is None and isinstance(f.result(), dict):
                out.update(f.result())
        for name, since in self._stuck_since.items():
            out[f"stuck_{name}_s"] = round(time.monotonic() - since, 1)
        return out

    async def __aenter__(self) -> Monitor:
        opens = [_in_thread(s.open) for s in self.samplers]
        await asyncio.wait(opens, timeout=10)          # nvmlInit can block while a CUDA process hangs the driver
        self._t0 = time.monotonic()
        self.baseline = await self._sample()
        # without a guard config the Guard still gives the summary its view of the server's shared memory
        self.guard = Guard(self.guard_cfg or GuardCfg(enabled=False), self.baseline)
        self._task = asyncio.create_task(self._loop())
        return self

    async def _loop(self) -> None:
        while not self._stop.is_set():
            t = time.monotonic()
            s = await self._sample()
            self.samples.append(s)
            if self.on_sample:
                self.on_sample(s)
            if self.guard and not self.tripped:
                reason = self.guard.check(s)
                if reason:
                    self.tripped = reason
                    self.trip_event.set()
                    if self.on_trip:
                        r = self.on_trip(reason)
                        if inspect.isawaitable(r):
                            await r
            try:
                await asyncio.wait_for(self._stop.wait(), max(0.0, self.interval - (time.monotonic() - t)))
            except TimeoutError:
                pass

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def __aexit__(self, *exc) -> None:
        await self.stop()
        closes = [_in_thread(s.close) for s in self.samplers]
        await asyncio.wait(closes, timeout=5)
        if self.csv_path and self.samples:
            self.write_csv(self.csv_path)

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cols = ["t", *sorted({k for s in self.samples for k in s} - {"t"})]
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for s in self.samples:
                w.writerow(s)

    def summary(self) -> Summary:
        s = summarize(self.samples, self.baseline, interval=self.interval, guard=self.guard, tripped=self.tripped,
                      th=self.thresholds, errors=self.errors)
        s.csv_path = str(self.csv_path) if self.csv_path else None
        return s


def _in_thread(fn: Callable[..., Any], *args: Any) -> asyncio.Future:
    """fn(*args) in a new daemon thread. Unlike asyncio.to_thread, a call that never returns (a driver call
    blocked by a hung CUDA process) doesn't hold up the event loop's executor or the interpreter's exit."""
    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()

    def done(ok: bool, val: Any) -> None:
        if not fut.done():
            fut.set_result(val) if ok else fut.set_exception(val)

    def run() -> None:
        try:
            r, ok = fn(*args), True
        except BaseException as e:      # noqa: BLE001 - handed to the future
            r, ok = e, False
        try:
            loop.call_soon_threadsafe(done, ok, r)
        except RuntimeError:            # loop already closed: nobody waits any more
            pass

    threading.Thread(target=run, daemon=True, name=f"monitor-{getattr(fn, '__qualname__', 'call')}").start()
    return fut


# ---------------------------------------------------------------- from [monitor] config

def guard_cfg_from(m: Any, *, enabled: bool = True, expected_host_mib: float | None = None,
                   stall: bool = False, host_extra_mib: float = 0.0) -> GuardCfg:
    """GuardCfg from the [monitor] table (config.MonitorCfg). stall: also stop on no progress (benchmarks)."""
    return GuardCfg(spill_stop_mib=m.spill_stop_mib, vram_full_mib=m.vram_full_mib, ram_stop_pct=m.ram_stop_pct,
                    commit_stop_pct=m.commit_stop_pct, expected_host_mib=expected_host_mib,
                    host_tolerance_pct=m.host_tolerance_pct, host_extra_mib=host_extra_mib,
                    enabled=enabled and m.guard,
                    stall_stop_s=(m.stall_stop_s or None) if stall else None)


def thresholds_from(m: Any) -> Thresholds:
    return Thresholds(m.spill_stop_mib, m.ram_warn_pct, m.ram_fail_pct, m.temp_warn_c, m.temp_fail_c)


def monitor_from(m: Any, *, guard: bool, expected_host_mib: float | None = None, csv_path: Path | None = None,
                 on_trip: Callable[[str], Any] | None = None, on_sample: Callable[[Sample], Any] | None = None,
                 samplers: list[Any] | None = None, stall: bool = False, host_extra_mib: float = 0.0) -> Monitor:
    return Monitor(m.interval, guard=guard_cfg_from(m, enabled=guard, expected_host_mib=expected_host_mib,
                                                    stall=stall, host_extra_mib=host_extra_mib),
                   samplers=samplers, csv_path=csv_path, on_trip=on_trip, on_sample=on_sample,
                   thresholds=thresholds_from(m))
