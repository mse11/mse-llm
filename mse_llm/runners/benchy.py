"""llama-benchy client command (bash run_benchy_params_cmd_build) + its background llama-server."""

from __future__ import annotations

from mse_llm.config import Config
from mse_llm.runners.base import Command
from mse_llm.runners.llamacpp import as_variant, server_command
from mse_llm.variants import Variant


def benchy_command(cfg: Config, model: str, variant: Variant | str | None, ctx: int | None = None,
                   engine: str | None = None, *, pp: list[int] | None = None, tg: list[int] | None = None,
                   depth: list[int] | None = None, runs: int | None = None,
                   concurrency: list[int] | None = None, port: int | None = None,
                   save_result: str | None = None) -> Command:
    """`uvx llama-benchy ...` against the background server for (model, variant, ctx).

    PP list: --pp, else the server context when [defaults.benchy].pp_sync_ctx (bash MSE_BENCHY_PP_SYNCH=1,
    which made the PP sweep the context itself), else [defaults.benchy].pp. Every list goes after ONE
    flag (`--pp 512 2048`): benchy uses nargs='+', so the bash `--pp a --pp b` kept only the last value.
    save_result: benchy writes its JSON there (`--format json --save-result`), which the results store reads."""
    v = as_variant(variant)
    server = server_command(cfg, model, v, ctx, engine, fit=cfg.defaults.fit, purpose="benchy-server",
                            port=port)
    mkey, m = cfg.model(model)
    _, eng = cfg.engine(engine)
    b = cfg.defaults.benchy
    ctx = server.ctx
    if pp:
        pps = list(pp)
    elif b.pp_sync_ctx and ctx:
        pps = [ctx]
    else:
        pps = list(b.pp)
    tgs = list(tg or cfg.defaults.bench_tg)
    depths = list(depth if depth is not None else b.depth)
    nruns = runs if runs is not None else b.runs
    argv = ["uvx", b.version, "--base-url", eng.base_url(port), "--model", m.label,
            "--runs", str(nruns), "--skip-coherence",
            "--pp", *map(str, pps), "--tg", *map(str, tgs), "--depth", *map(str, depths)]
    if concurrency:
        argv += ["--concurrency", *map(str, concurrency)]
    if save_result:
        argv += ["--format", "json", "--save-result", save_result]
    cmd = Command("llama-benchy", argv, label=f"benchy {mkey} {v.name}", model=mkey, variant=v.name, ctx=ctx,
                  server=server, params={"pp": pps, "tg": tgs, "depth": depths, "runs": nruns,
                                         "concurrency": list(concurrency or [1])})
    if ctx:
        need = max(pps) + max(tgs) + max(depths)
        if need > ctx:
            cmd.warnings.append(f"pp {max(pps)} + tg {max(tgs)} + depth {max(depths)} = {need} > ctx {ctx}: "
                                "the longest request doesn't fit the server context")
    return cmd
