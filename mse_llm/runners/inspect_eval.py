"""inspect-ai eval command (bash run_inspect_params_cmd_build) + its background llama-server.

The bash runner was a stub (`echo before; sleep 5`) with the base URL hard-coded to port 8033; here
the URL comes from the engine. Running it arrives in M12; M3 builds and shows the command.
"""

from __future__ import annotations

from mse_llm.config import Config
from mse_llm.runners.base import Command
from mse_llm.runners.llamacpp import as_variant, server_command
from mse_llm.variants import Variant

LOCAL_TASKS = ("mmlu_pro", "gpqa_diamond", "math_aime_24")
DOCKER_TASKS = ("swe_bench_verified",)   # needs Docker: runs from a Linux client (§7.1, M12)


def inspect_command(cfg: Config, model: str, task: str, variant: Variant | str | None, ctx: int | None = None,
                    engine: str | None = None) -> Command:
    v = as_variant(variant)
    server = server_command(cfg, model, v, ctx, engine, fit=cfg.defaults.fit, purpose="eval-server")
    mkey, _ = cfg.model(model)
    _, eng = cfg.engine(engine)
    argv = ["uvx", "--from", "inspect-ai", "--with", "openai", "--with", "inspect-evals", "inspect", "eval",
            f"inspect_evals/{task}", "--model", "openai/local", "--model-base-url", eng.base_url()]
    cmd = Command("inspect", argv, label=f"eval {task} {mkey} {v.name}", model=mkey, variant=v.name,
                  ctx=server.ctx, env={"OPENAI_API_KEY": "dummy"}, server=server)
    if task in DOCKER_TASKS:
        cmd.warnings.append(f"{task} needs Docker, which doesn't run on this laptop: run it from a Linux client "
                            "(eval --remote-client, M12)")
    elif task not in LOCAL_TASKS:
        known = ", ".join(LOCAL_TASKS + DOCKER_TASKS)
        cmd.warnings.append(f"{task}: not one of the tasks the bash script ran ({known})")
    return cmd
