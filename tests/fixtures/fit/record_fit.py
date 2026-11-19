"""Record real llama-fit-params output + GGUF header info for tests/test_fit.py.

Needs the models from config/config.example.toml and a built llama-fit-params; takes ~1 minute.
Regenerate (from the repo root) after a llama.cpp update changes the memory estimates:

    uv run python tests/fixtures/fit/record_fit.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from mse_llm import fit as ff  # noqa: E402
from mse_llm import gguf  # noqa: E402
from mse_llm import variants as vv  # noqa: E402
from mse_llm.config import load_config  # noqa: E402

OUT = Path(__file__).with_name("recorded.json")
QWEN28_LIST = (Path(__file__).with_name("qwen28_extras_list.txt")).read_text(encoding="utf-8")

RUNS = [("Qwen36_28B_REAP20_A3B_Q4_K_M", e.name, 65536) for e in vv.parse_list(QWEN28_LIST)] + [
    ("Qwen36_28B_REAP20_A3B_Q4_K_M", "fa_Kq8_Vq8_b4096_ub4096_moe41", 131072),
    *[("Qwen36_28B_REAP20_A3B_Q4_K_M", f"fa_Kq8_Vq8_b4096_ub4096_moe{n}", 65536) for n in range(26, 31)],
    ("gemma_4_26B_A4b_QAT_UD_Q4_K_XL", "fa_Kq8_Vq8_b2048_ub2048_moe16", 65536),
    ("gemma_4_26B_A4b_QAT_UD_Q4_K_XL", "fa_Kq8_Vq8_b2048_ub2048_moe21_sd4mtp", 65536),
    ("gemma_4_26B_A4b_QAT_UD_Q4_K_XL", "fa_Kq8_Vq8_b8192_ub8192_moe16", 65536),
    ("Qwen36_35B_A3B_UD_Q4_K_M_MTP", "fa_Kq8_Vq4_default_moe34_sd2mtp", 65536),
    ("Qwen36_35B_A3B_UD_Q4_K_M", "fa_Kq8_Vq8_default_moe34_sd2mtp", 65536),
    ("Qwen36_35B_A3B_UD_Q4_K_M", "fa_Kq8_Vq8_b8192_ub8192_moe20", 65536),
]


def key(argv: list[str]) -> str:
    """Runner lookup key: the argv without the binary path (which depends on the checkout)."""
    return " ".join(argv[1:])


def main() -> None:
    cfg = load_config(ROOT / "config" / "config.example.toml")
    runs, infos = {}, {}
    for model, variant, ctx in RUNS:
        m = cfg.models[model]
        argv = ff.fit_params_argv(cfg, None, m.path, vv.decode(variant), ctx)
        cp = ff.run_fit_params(argv)
        if cp.returncode:
            raise SystemExit(f"{model} {variant}: {cp.stderr[-500:]}")
        runs[key(argv)] = cp.stdout
        print(model, variant, ctx, cp.stdout.split())
        for p in (m.path, m.draft if m.draft and m.draft != "builtin" else None):
            if p and p not in infos:
                infos[p] = gguf.info(p).as_dict()
    mm = r"d:\.lmstudio\models\unsloth\gemma-4-26B-A4B-it-qat-GGUF\mmproj-F32.gguf"
    infos[mm] = gguf.info(mm).as_dict()
    OUT.write_text(json.dumps({"runs": runs, "infos": infos}, indent=1) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
