"""Minimal GGUF reader (§4.6 inputs): metadata + tensor sizes, without loading any weights.

Only the header is read (metadata and tensor infos); big arrays such as the tokenizer vocabulary are
skipped and only their length is kept. Tensor byte sizes come from the offsets in the header (the
data region is laid out in header order, each tensor padded to `general.alignment`), so they are the
exact bytes llama.cpp loads, no quantization tables needed.
"""

from __future__ import annotations

import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO

MAGIC = b"GGUF"
_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9
_KEEP_ARRAY = 64          # arrays up to this length are kept (per-layer head counts); longer: length only

EXPERT_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down|gate_up)_exps\.")
LAYER_RE = re.compile(r"^blk\.(\d+)\.")


class GgufError(ValueError):
    pass


@dataclass
class ArrayLen:
    """A skipped long array (e.g. tokenizer.ggml.tokens): its length only."""
    n: int


class _R:
    def __init__(self, f: BinaryIO):
        self.f = f

    def take(self, n: int) -> bytes:
        b = self.f.read(n)
        if len(b) != n:
            raise GgufError("truncated GGUF header")
        return b

    def scalar(self, t: int) -> Any:
        fmt = _SCALARS[t]
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def u32(self) -> int:
        return self.scalar(4)

    def u64(self) -> int:
        return self.scalar(10)

    def string(self) -> str:
        return self.take(self.u64()).decode("utf-8", "replace")

    def skip_string(self) -> None:
        self.f.seek(self.u64(), 1)

    def value(self, t: int) -> Any:
        if t in _SCALARS:
            return self.scalar(t)
        if t == _STRING:
            return self.string()
        if t == _ARRAY:
            et, n = self.u32(), self.u64()
            if n <= _KEEP_ARRAY:
                return [self.value(et) for _ in range(n)]
            if et in _SCALARS:
                self.f.seek(n * struct.calcsize(_SCALARS[et]), 1)
            elif et == _STRING:
                for _ in range(n):
                    self.skip_string()
            else:
                for _ in range(n):
                    self.value(et)
            return ArrayLen(n)
        raise GgufError(f"unknown GGUF value type {t}")


@dataclass
class Tensor:
    name: str
    shape: list[int]
    type: int
    offset: int
    nbytes: int = 0


@dataclass
class Gguf:
    path: str
    version: int
    metadata: dict[str, Any]
    tensors: list[Tensor]
    file_size: int

    @property
    def arch(self) -> str:
        return str(self.metadata.get("general.architecture", ""))

    def get(self, key: str, default: Any = None) -> Any:
        """`{arch}.key` first, then the plain key."""
        return self.metadata.get(f"{self.arch}.{key}", self.metadata.get(key, default))


def read(path: str | Path) -> Gguf:
    p = Path(path)
    size = p.stat().st_size
    with p.open("rb") as f:
        r = _R(f)
        if r.take(4) != MAGIC:
            raise GgufError(f"{p}: not a GGUF file")
        version = r.u32()
        if version < 2:
            raise GgufError(f"{p}: GGUF v{version} is not supported")
        n_tensors, n_kv = r.u64(), r.u64()
        meta: dict[str, Any] = {}
        for _ in range(n_kv):
            key = r.string()
            meta[key] = r.value(r.u32())
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            shape = [r.u64() for _ in range(r.u32())]
            tensors.append(Tensor(name, shape, r.u32(), r.u64()))
        align = int(meta.get("general.alignment", 32))
        pos = f.tell()
    data_start = (pos + align - 1) // align * align
    ordered = sorted(tensors, key=lambda t: t.offset)
    for t, nxt in zip(ordered, ordered[1:] + [None], strict=True):
        end = nxt.offset if nxt else size - data_start
        t.nbytes = end - t.offset
    return Gguf(str(p), version, meta, tensors, size)


# ---------------------------------------------------------------- what `fit` needs from a GGUF

def _first(v: Any) -> Any:
    """Per-layer arrays (e.g. head_count_kv on hybrid / SWA models): the largest value."""
    if isinstance(v, list):
        vals = [x for x in v if isinstance(x, (int, float))]
        return max(vals) if vals else None
    return v


@dataclass
class ModelInfo:
    """The numbers from one GGUF that `fit --suggest` and the draft estimates use (MiB = 2**20 bytes)."""
    path: str
    arch: str
    weights_mib: float                     # all tensors
    n_layers: int | None
    moe_layers: int                        # layers with *_exps tensors
    expert_layer_mib: float                # mean expert bytes per MoE layer (one -ncmoe step)
    n_expert: int | None
    n_expert_used: int | None
    vocab: int | None
    n_head_kv: int | None
    key_length: int | None
    value_length: int | None
    nextn_layers: int                      # built-in MTP / NextN layers (0 = none)
    attn_layers: int | None = None         # layers with a KV cache (hybrid models have fewer)
    shared_kv_layers: int = 0              # layers reusing another context's KV (gemma4-assistant drafter)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelInfo:
        return cls(**d)


def model_info(g: Gguf) -> ModelInfo:
    mib = 2 ** 20
    experts: dict[int, int] = {}
    attn: set[int] = set()
    for t in g.tensors:
        m = EXPERT_RE.match(t.name)
        if m:
            experts[int(m.group(1))] = experts.get(int(m.group(1)), 0) + t.nbytes
        m = LAYER_RE.match(t.name)
        if m and (".attn_k." in t.name or ".attn_v." in t.name or ".attn_qkv." in t.name):
            attn.add(int(m.group(1)))
    vocab = g.metadata.get("tokenizer.ggml.tokens")
    vocab_n = vocab.n if isinstance(vocab, ArrayLen) else (len(vocab) if isinstance(vocab, list) else None)
    vocab_n = g.get("vocab_size", vocab_n)
    n_head = _first(g.get("attention.head_count"))
    emb = g.get("embedding_length")
    key_len = g.get("attention.key_length") or (emb // n_head if emb and n_head else None)
    return ModelInfo(
        path=g.path,
        arch=g.arch,
        weights_mib=sum(t.nbytes for t in g.tensors) / mib,
        n_layers=g.get("block_count"),
        moe_layers=len(experts),
        expert_layer_mib=(sum(experts.values()) / len(experts) / mib) if experts else 0.0,
        n_expert=g.get("expert_count"),
        n_expert_used=g.get("expert_used_count"),
        vocab=vocab_n,
        n_head_kv=_first(g.get("attention.head_count_kv")) or n_head,
        key_length=key_len,
        value_length=g.get("attention.value_length") or key_len,
        nextn_layers=int(g.get("nextn_predict_layers", 0) or 0),
        attn_layers=len(attn) or None,
        shared_kv_layers=int(g.get("attention.shared_kv_layers", 0) or 0),
    )


def info(path: str | Path) -> ModelInfo:
    return model_info(read(path))
