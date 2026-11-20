"""Quality metrics (M10, §4.5): KL divergence of a K/V-cache choice against an f16-KV reference of the same GGUF.

    quality kld-ref  one llama-perplexity run that saves the reference logits (results/raw/kld-ref/)
    quality floor    the reference again with a different placement (-ncmoe / -nkvo): GPU vs CPU math noise
    quality kld      one run per variant against the reference: mean KLD, 99 % KLD, same top p, PPL ratio

The reference keeps the variant's weights and only changes the KV cache to f16, so the numbers isolate the
KV-cache effect (what the K*/V* tokens change). It may use a different placement than the variants (more
experts on the CPU, KV in RAM): placement changes the logits only by the noise floor, which is measured.
When even that doesn't fit, q8_0/q8_0 is the reference and every number says "vs q8 ref".

Reference size: per chunk llama-perplexity stores the second half of the tokens' log-probs as uint16
(tools/perplexity: n_ctx - 1 - n_ctx/2 tokens × (n_vocab rounded to even + 4) × 2 bytes) plus the tokens.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import hashlib
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from mse_llm import results as rs
from mse_llm.config import Config
from mse_llm.util import project_root, stable_hash
from mse_llm.variants import Variant, decode

DEFAULT_CTX = 8192          # KV errors grow with context; 512 (the perplexity default) understates them
DEFAULT_CHUNKS = 8
DEFAULT_CORPUS = "code"
FLOOR_FACTOR = 2.0          # mean KLD within 2x the floor = no measurable loss
CODE_BYTES = 2_000_000      # fixed size: ~500-650k tokens, enough for 8 chunks at 64k
CODE_GLOBS = ("src/*.cpp", "src/*.h", "common/*.cpp", "common/*.h", "tools/server/*.cpp", "tools/server/*.h")
WIKITEXT_URL = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
WIKITEXT_FILE = "wikitext-2-raw/wiki.test.raw"
WIKITEXT_ZIP_MB = 4.5       # what scripts/get-wikitext-2.sh downloads (unzipped test split: ~1.3 MB)
REF_KV = ("f16", "f16")
REF_KV_FALLBACK = ("q8_0", "q8_0")


# ---------------------------------------------------------------- corpus

@dataclass
class Corpus:
    name: str                  # code | wikitext2 | file
    path: Path
    sha: str                   # sha256 of the text (first 16 hex)
    size: int
    source: str = ""

    @property
    def label(self) -> str:
        return f"{self.name}@{self.sha[:8]}"

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "path": rs.rel(self.path), "sha": self.sha, "size": self.size,
                "source": self.source}


def corpus_dir() -> Path:
    return rs.results_root() / "raw" / "corpus"


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def llama_src(cfg: Config | None = None) -> Path:
    return project_root() / "builds" / "llama.cpp"


def code_corpus(src: Path | None = None, *, create: bool = True, size: int = CODE_BYTES) -> Corpus:
    """A fixed, recorded set of C++ sources from the llama.cpp checkout (sorted paths, cut at `size` bytes).
    The same checkout gives the same text, so the same hash; a new checkout gives a new corpus (and reference)."""
    src = src or llama_src()
    files = sorted({p for g in CODE_GLOBS for p in src.glob(g) if p.is_file()},
                   key=lambda p: p.relative_to(src).as_posix())
    if not files:
        raise ValueError(f"no llama.cpp sources under {src} (clone them with `mse-llm build run`, or use "
                         "--corpus wikitext2 / FILE)")
    parts: list[str] = []
    used: list[str] = []
    n = 0
    for p in files:
        rel = p.relative_to(src).as_posix()
        txt = f"// ---- {rel}\n" + p.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
        parts.append(txt)
        used.append(rel)
        n += len(txt.encode("utf-8"))
        if n >= size:
            break
    text = "".join(parts)
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    out = corpus_dir() / f"code-{sha}.txt"
    if create and not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8", newline="\n")
        commit = _git_head(src)
        (out.with_suffix(".json")).write_text(json.dumps({"sha": sha, "llama_cpp_commit": commit, "files": used},
                                                         indent=2) + "\n", encoding="utf-8")
    return Corpus("code", out, sha, len(text.encode("utf-8")), f"llama.cpp {_git_head(src) or '?'}: "
                  f"{len(used)} files")


def _git_head(src: Path) -> str | None:
    head = src / ".git" / "HEAD"
    try:
        ref = head.read_text(encoding="utf-8").strip()
        if ref.startswith("ref: "):
            return (src / ".git" / ref[5:]).read_text(encoding="utf-8").strip()[:9]
        return ref[:9]
    except OSError:
        return None


def wikitext_path() -> Path:
    return corpus_dir() / WIKITEXT_FILE


def download_wikitext(progress: Callable[[int], None] | None = None) -> Path:
    """scripts/get-wikitext-2.sh in Python: the zip from Hugging Face, only wiki.test.raw kept."""
    import io
    import urllib.request
    import zipfile

    dst = wikitext_path()
    if dst.is_file():
        return dst
    buf = io.BytesIO()
    with urllib.request.urlopen(WIKITEXT_URL, timeout=60) as r:   # noqa: S310 (fixed https URL)
        while True:
            b = r.read(1 << 16)
            if not b:
                break
            buf.write(b)
            if progress:
                progress(buf.tell())
    with zipfile.ZipFile(buf) as z:
        data = z.read(WIKITEXT_FILE)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
    return dst


def resolve_corpus(name: str, *, create: bool = True) -> Corpus:
    """code (default) | wikitext2 | a file path."""
    if name == "code":
        return code_corpus(create=create)
    if name in ("wikitext2", "wikitext-2", "wiki"):
        p = wikitext_path()
        if not p.is_file():
            raise ValueError(f"wikitext-2 is not downloaded ({p}); get it with `mse-llm quality corpus wikitext2 "
                             "--download`")
        return Corpus("wikitext2", p, _sha(p), p.stat().st_size, "wikitext-2-raw test split")
    p = Path(name)
    if not p.is_file():
        raise ValueError(f"corpus {name!r}: not code / wikitext2 and no such file")
    return Corpus("file", p, _sha(p), p.stat().st_size, str(p))


def prompt_text(cfg: Config, name: str, *, create: bool = True) -> Path:
    """The text behind a benchy prompt set (`prompts = "code"`): the code corpus, wikitext-2 or a file."""
    return resolve_corpus(name, create=create).path


class TextServer:
    """A file over http://127.0.0.1:<free port>/ for llama-benchy --book-url (it only fetches over HTTP)."""

    def __init__(self, path: Path) -> None:
        import http.server

        data = Path(path).read_bytes()
        name = Path(path).name

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:          # noqa: N802
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a) -> None:  # quiet
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/{name}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


# ---------------------------------------------------------------- reference store

def ref_size_bytes(n_vocab: int, ctx: int, chunks: int) -> int:
    nv = 2 * ((n_vocab + 1) // 2) + 4
    return 20 + chunks * ctx * 4 + chunks * (ctx - 1 - ctx // 2) * nv * 2


def refs_dir() -> Path:
    return rs.results_root() / "raw" / "kld-ref"


@dataclass
class RefMeta:
    id: str
    model: str
    ctx: int
    chunks: int
    corpus: dict[str, Any]
    build: str | None
    kv: tuple[str, str]
    variant: str                       # placement (and KV) the reference ran with
    extra: list[str] = field(default_factory=list)     # raw flags not in the name (-nkvo)
    model_digest: str | None = None
    steps: list[str] = field(default_factory=list)     # fallback steps taken
    size_bytes: int | None = None
    run_id: str | None = None
    ppl: float | None = None
    created: str | None = None
    floor: dict[str, Any] | None = None               # {run_id, variant, mean_kld, same_top_p, ...}

    @property
    def q8_ref(self) -> bool:
        return tuple(self.kv) != REF_KV

    @property
    def label(self) -> str:
        return f"{'/'.join(self.kv)} KV, ctx {self.ctx}, {self.chunks} chunks, corpus {self.corpus.get('name')}@" \
               f"{str(self.corpus.get('sha'))[:8]}"

    def path(self) -> Path:
        return refs_dir() / self.model / f"{self.id}.kld"

    def meta_path(self) -> Path:
        return self.path().with_suffix(".json")

    @property
    def ready(self) -> bool:
        return self.run_id is not None and self.path().is_file()

    def as_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["kv"] = list(self.kv)
        return d

    def save(self) -> None:
        p = self.meta_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, p: Path) -> RefMeta:
        d = json.loads(p.read_text(encoding="utf-8"))
        d["kv"] = tuple(d.get("kv") or REF_KV)
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__ if k in d})


def ref_id(model_digest: str | None, ctx: int, chunks: int, corpus_sha: str, build: str | None) -> str:
    return stable_hash({"model": model_digest, "ctx": ctx, "chunks": chunks, "corpus": corpus_sha,
                        "build": build}, 12)


def list_refs(model: str | None = None) -> list[RefMeta]:
    root = refs_dir()
    if not root.is_dir():
        return []
    out = []
    for p in sorted(root.glob(f"{model or '*'}/*.json")):
        try:
            out.append(RefMeta.load(p))
        except (OSError, ValueError, TypeError):
            continue
    return out


def find_ref(model: str, ctx: int, corpus_sha: str, build: str | None, model_digest: str | None,
             chunks: int | None = None) -> RefMeta | None:
    """The newest ready reference for (model file, ctx, corpus, build); chunks only when given."""
    hits = [r for r in list_refs(model) if r.ready and r.ctx == ctx and r.corpus.get("sha") == corpus_sha
            and r.build == build and (model_digest is None or r.model_digest in (None, model_digest))
            and (chunks is None or r.chunks == chunks)]
    return sorted(hits, key=lambda r: r.created or "")[-1] if hits else None


# ---------------------------------------------------------------- placement planning (fallbacks)

FitFn = Callable[[Variant], Any]       # variant -> serve.FitCheck (at the KLD ctx)


@dataclass
class Placement:
    variant: Variant
    kv: tuple[str, str]
    steps: list[str]
    verdict: str
    fits: bool

    @property
    def name(self) -> str:
        return vname(self.variant)


def vname(v: Variant) -> str:
    """Name for display / storage: the decodable variant name, raw extra flags appended after a space."""
    base = replace(v, extra=()).name
    return base + (" " + " ".join(v.extra) if v.extra else "")


def from_vname(name: str) -> Variant:
    head, *extra = name.split(" ")
    return replace(decode(head), extra=tuple(extra))


def with_kv(v: Variant, kv: tuple[str, str]) -> Variant:
    return replace(v.without_spec(), ctk=kv[0], ctv=kv[1])


def _moe_layers(cfg: Config, model: str) -> int:
    from mse_llm import gguf

    _, m = cfg.model(model)
    try:
        return gguf.info(m.path).moe_layers or 0
    except (OSError, gguf.GgufError):
        return 0


def plan_reference(base: Variant, fit: FitFn, moe_layers: int) -> Placement:
    """§4.5 fallback order for the f16 reference: (1) the test ctx (the caller passes it); (2) more experts on
    the CPU; (3) KV cache in system RAM (-nkvo); (4) q8_0/q8_0 KV as reference."""
    steps: list[str] = []
    cand = with_kv(base, REF_KV)
    chk = fit(cand)
    if not chk.bad:
        return Placement(cand, REF_KV, steps, chk.verdict, True)
    steps.append(f"{vname(cand)}: {chk.verdict}")
    tried: list[Variant] = []
    if moe_layers:
        n0 = base.n_cpu_moe or 0
        step = max(2, moe_layers // 6)
        tried = [replace(cand, n_cpu_moe=n, cpu_moe=False) for n in range(n0 + step, moe_layers, step)]
        tried.append(replace(cand, n_cpu_moe=None, cpu_moe=True))
    for v in tried:
        chk = fit(v)
        if not chk.bad:
            steps.append(f"more experts on the CPU: {vname(v)} {chk.verdict}")
            return Placement(v, REF_KV, steps, chk.verdict, True)
    last = tried[-1] if tried else cand
    for v in (replace(cand, extra=("-nkvo",)), replace(last, extra=("-nkvo",))):
        chk = fit(v)
        if not chk.bad:
            steps.append(f"KV in system RAM: {vname(v)} {chk.verdict}")
            return Placement(v, REF_KV, steps, chk.verdict, True)
    q8 = with_kv(last, REF_KV_FALLBACK)
    chk = fit(q8)
    steps.append(f"f16 KV doesn't fit anywhere: q8_0/q8_0 reference ({vname(q8)} {chk.verdict})")
    return Placement(q8, REF_KV_FALLBACK, steps, chk.verdict, not chk.bad)


def floor_variant(ref: Variant, moe_layers: int, fit: FitFn | None = None) -> tuple[Variant, str]:
    """The reference's own KV types with a different placement: more (or fewer) experts on the CPU for a MoE
    model, the KV cache moved for a dense one. The KLD between the two is pure GPU-vs-CPU math noise."""
    if moe_layers:
        if ref.cpu_moe:
            n = moe_layers - max(4, moe_layers // 4)
            alt = replace(ref, cpu_moe=False, n_cpu_moe=n)
            if fit is None or not fit(alt).bad:
                return alt, f"-ncmoe {n} instead of -cmoe"
        else:
            n0 = ref.n_cpu_moe or 0
            n = min(moe_layers, n0 + max(4, moe_layers // 4))
            if n != n0:
                return replace(ref, n_cpu_moe=n), f"-ncmoe {n} instead of {n0}"
    if "-nkvo" in ref.extra:
        return replace(ref, extra=tuple(x for x in ref.extra if x != "-nkvo")), "KV on the GPU instead of in RAM"
    return replace(ref, extra=(*ref.extra, "-nkvo")), "KV in system RAM (-nkvo)"


KLD_RESOLUTION = 1e-6       # llama-perplexity prints the mean with 6 decimals
# absolute levels for a loss that is above the noise floor (mean KLD, nats per token)
KLD_LEVELS = ((1e-3, "negligible"), (1e-2, "small"), (5e-2, "noticeable"))


def verdict(mean_kld: float | None, floor: float | None, err: float | None = None) -> str:
    """"no measurable loss" when the mean is within 2x the floor (or within 2 sigma of it); otherwise a level from
    the absolute mean KLD. A floor of 0.000000 means "below the printed resolution"."""
    if mean_kld is None:
        return "-"
    if floor is None:
        return "no floor yet"
    f = max(floor, KLD_RESOLUTION)
    if mean_kld <= FLOOR_FACTOR * f or (err and mean_kld - 2 * err <= FLOOR_FACTOR * f):
        return "no measurable loss"
    for lim, name in KLD_LEVELS:
        if mean_kld < lim:
            return f"{name} loss"
    return "large loss"


# ---------------------------------------------------------------- running llama-perplexity

PPL_FILTER = [r"[Ee]rror", r"[Ww]arn", r"[Ff]ail", r"out of memory", r"^\[\d+\]", r"Final estimate",
              r"KL divergence", r"Same top p", r"Mean\s+KLD", r"computing over", r"calculating perplexity",
              r"tokenizing", r"llama_kv_cache"]


@dataclass
class QualityJob:
    model: str
    variant: Variant             # what runs (KV types + placement)
    mode: str                    # ref | kld | floor
    ctx: int
    chunks: int
    corpus: Corpus
    logits_file: Path
    ref: RefMeta | None = None   # kld / floor: the reference compared against
    engine: str | None = None
    sweep: str | None = None
    note: str | None = None
    tested: str | None = None    # kld: the variant name the user asked for (before placement changes)

    def command(self, cfg: Config):
        from mse_llm.runners import perplexity_command

        cmd = perplexity_command(cfg, self.model, self.variant, self.ctx, self.engine, corpus=str(self.corpus.path),
                                 chunks=self.chunks, logits_file=str(self.logits_file), kld=self.mode != "ref")
        return cmd

    def params(self) -> dict[str, Any]:
        return {"mode": self.mode, "chunks": self.chunks, "corpus": self.corpus.sha,
                "ref": self.ref.id if self.ref else None, "extra": list(self.variant.extra) or None}


async def run_quality(cfg: Config, job: QualityJob, opts, env=None) -> rs.RunResult:
    """One llama-perplexity run under the monitor guard -> RunResult(tool="quality", quality={...})."""
    from mse_llm import bench as bb
    from mse_llm import proc as pr
    from mse_llm.parsers.perplexity import parse_perplexity

    env = env or bb.RunEnv()
    t = rs.now()
    run_id = rs.new_run_id("quality", t)
    cmd = job.command(cfg)
    mkey, m = cfg.model(job.model)
    ename, _ = cfg.engine(job.engine)
    params = job.params()
    name = replace(job.variant, extra=()).name
    res = rs.RunResult(run_id=run_id, key=rs.run_key("quality", mkey, name, job.ctx, params),
                       timestamp=t.isoformat(timespec="seconds"), sweep=job.sweep, tool="quality", engine=ename,
                       model=mkey, model_path=m.path, model_identity=env.model_id(m.path), variant=name,
                       variant_fields=job.variant.as_dict(), ctx=job.ctx, params=params, argv=list(cmd.argv),
                       build=env.build_for(cfg, job.engine), hw=env.hw_info(), note=job.note)
    q: dict[str, Any] = {"mode": job.mode, "corpus": job.corpus.as_dict(), "chunks": job.chunks,
                         "kv": [job.variant.ctk, job.variant.ctv]}
    if job.ref:
        q.update(ref_id=job.ref.id, ref_kv=list(job.ref.kv), ref_variant=job.ref.variant, vs_q8_ref=job.ref.q8_ref)
    if job.tested:
        q["tested"] = job.tested
    res.quality = q
    t0 = time.monotonic()

    def done(status: str, reason: str | None = None) -> rs.RunResult:
        res.status, res.reason = status, reason      # type: ignore[assignment]
        res.duration_s = round(time.monotonic() - t0, 1)
        if opts.write:
            rs.write_run(res)
        return res

    if not cmd.ok:
        return done("failed", "flag check: " + "; ".join(cmd.errors))
    raw = rs.raw_dir(run_id)
    raw.mkdir(parents=True, exist_ok=True)
    log = raw / "llama-perplexity.log"
    res.logs["tool"] = rs.rel(log) or str(log)
    mon = None
    if opts.monitor and cfg.defaults.monitor:
        from mse_llm.monitor import monitor_from

        mon = monitor_from(cfg.monitor, guard=opts.guard, expected_host_mib=None, csv_path=raw / "monitor.csv",
                           samplers=opts.samplers, stall=True)
    lines: list[str] = []
    sp = pr.Spawned(cmd.argv, log_path=log, sink=opts.sink, memory_cap_mib=opts.mem_cap_mib, keep_lines=4000)
    sp.watchers.append(lines.append)
    stopped = asyncio.Event()

    async def trip(reason: str) -> None:
        opts.say(f"[red]■ memory guard:[/] {reason} → stopping llama-perplexity")
        if sp.proc is not None and sp.returncode is None:
            await sp.stop(grace=2)
        stopped.set()

    async def body() -> None:
        try:
            await sp.start()
        except pr.ProcError as e:
            res.reason = str(e)
            return
        waiter = asyncio.create_task(sp.wait())
        try:
            rc = await bb._wait_or_give_up(waiter, stopped, res, sp.pid, opts)
        except asyncio.CancelledError:
            waiter.cancel()
            await sp.stop(grace=2)
            raise
        if rc not in (None, 0) and not (mon is not None and mon.tripped):
            res.reason = f"llama-perplexity exit code {rc}: {bb._tail_reason(lines)}"

    try:
        await bb._with_monitor(mon, lambda: sp.proc.pid if sp.proc is not None else None, body, trip)
    except asyncio.CancelledError:
        bb._finish_monitor(res, mon)
        done("aborted", "interrupted (Ctrl+C)")
        raise
    bb._finish_monitor(res, mon)
    parsed = parse_perplexity(lines)
    q.update(parsed)
    if mon is not None and mon.tripped:
        return done("aborted", f"memory guard: {mon.tripped}")
    if parsed.get("tokens_needed"):
        return done("failed", f"corpus too short: llama-perplexity needs {parsed['tokens_needed']} tokens "
                              f"for ctx {job.ctx} (use fewer --chunks / a smaller --ctx / a longer corpus)")
    if res.reason:
        return done("failed", res.reason)
    if job.mode == "ref":
        if not job.logits_file.is_file():
            return done("failed", "no reference file written: " + "; ".join(parsed.get("errors") or ["?"]))
        q["size_bytes"] = job.logits_file.stat().st_size
        return done("ok")
    if parsed.get("mean_kld") is None:
        return done("failed", "no KLD summary in the output: " + "; ".join(parsed.get("errors") or ["?"]))
    floor = (job.ref.floor or {}).get("mean_kld") if job.ref else None
    if job.mode == "kld":
        q["floor_kld"] = floor
        q["verdict"] = verdict(parsed["mean_kld"], floor, parsed.get("mean_kld_err"))
    return done("ok")


def now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------- high level: reference, floor, variants

@dataclass
class KldSetup:
    """What `quality kld-ref / kld / floor` resolve before running anything (also the --dry-run preview)."""
    model: str
    ctx: int
    chunks: int
    corpus: Corpus
    build: str | None
    digest: str | None
    n_vocab: int | None
    moe_layers: int
    ref: RefMeta | None                    # existing ready reference
    placement: Placement | None = None     # for a new reference
    size_bytes: int | None = None

    @property
    def size_gib(self) -> float | None:
        return None if self.size_bytes is None else self.size_bytes / 2**30


def setup(cfg: Config, model: str, *, ctx: int = DEFAULT_CTX, chunks: int = DEFAULT_CHUNKS,
          corpus: str = DEFAULT_CORPUS, engine: str | None = None, base: Variant | None = None,
          fit: FitFn | None = None, plan_new: bool = True, env=None) -> KldSetup:
    from mse_llm import bench as bb
    from mse_llm import gguf

    env = env or bb.RunEnv()
    mkey, m = cfg.model(model)
    cp = resolve_corpus(corpus)
    build = env.build_for(cfg, engine).get("id")
    ident = env.model_id(m.path) or {}
    digest = ident.get("digest")
    try:
        gi = gguf.info(m.path)
        n_vocab, moe_layers = gi.vocab, gi.moe_layers or 0
    except (OSError, gguf.GgufError):
        n_vocab, moe_layers = None, 0
    ref = find_ref(mkey, ctx, cp.sha, build, digest, chunks)
    st = KldSetup(mkey, ctx, chunks, cp, build, digest, n_vocab, moe_layers, ref)
    if n_vocab:
        st.size_bytes = ref_size_bytes(n_vocab, ctx, chunks)
    if ref is None and plan_new:
        fit = fit or default_fit(cfg, mkey, ctx, engine)
        b = base if base is not None else (decode(m.default_serve) if m.default_serve else Variant(fa="on"))
        st.placement = plan_reference(b, fit, moe_layers)
    return st


def default_fit(cfg: Config, model: str, ctx: int, engine: str | None) -> FitFn:
    from mse_llm import serve as sv

    cache: dict[str, Any] = {}

    def fit(v: Variant):
        k = v.name + "|" + " ".join(v.extra)
        if k not in cache:
            cache[k] = sv.fit_check(cfg, model, v, ctx, engine)
        return cache[k]
    return fit


def new_ref_meta(st: KldSetup) -> RefMeta:
    assert st.placement is not None
    rid = ref_id(st.digest, st.ctx, st.chunks, st.corpus.sha, st.build)
    v = st.placement.variant
    return RefMeta(rid, st.model, st.ctx, st.chunks, st.corpus.as_dict(), st.build, st.placement.kv,
                   replace(v, extra=()).name, list(v.extra), st.digest, list(st.placement.steps), st.size_bytes,
                   created=now_iso())


async def make_reference(cfg: Config, st: KldSetup, opts, *, engine: str | None = None, env=None,
                         sweep: str | None = None) -> tuple[RefMeta, rs.RunResult]:
    meta = new_ref_meta(st)
    meta.path().parent.mkdir(parents=True, exist_ok=True)
    job = QualityJob(st.model, st.placement.variant, "ref", st.ctx, st.chunks, st.corpus, meta.path(),
                     engine=engine, sweep=sweep, note="; ".join(meta.steps) or None)
    r = await run_quality(cfg, job, opts, env)
    if r.status == "ok":
        meta.run_id = r.run_id
        meta.ppl = (r.quality or {}).get("ppl")
        meta.size_bytes = (r.quality or {}).get("size_bytes") or meta.size_bytes
        meta.save()
        if r.quality is not None:
            r.quality["ref_id"] = meta.id
    else:
        meta.path().unlink(missing_ok=True)
    return meta, r


async def run_floor(cfg: Config, st: KldSetup, ref: RefMeta, opts, *, engine: str | None = None, env=None,
                    fit: FitFn | None = None, sweep: str | None = None) -> rs.RunResult:
    rv = ref_variant(ref)
    fv, why = floor_variant(rv, st.moe_layers, fit)
    job = QualityJob(st.model, fv, "floor", ref.ctx, ref.chunks, st.corpus, ref.path(), ref, engine, sweep,
                     note=f"noise floor: {why}")
    r = await run_quality(cfg, job, opts, env)
    if r.status == "ok" and r.quality:
        ref.floor = {"run_id": r.run_id, "variant": vname(fv), "change": why, "mean_kld": r.quality.get("mean_kld"),
                     "kld_p99": r.quality.get("kld_p99"), "same_top_p": r.quality.get("same_top_p")}
        ref.save()
    return r


def ref_variant(ref: RefMeta) -> Variant:
    return replace(decode(ref.variant), extra=tuple(ref.extra or ()))


def tested_variant(v: Variant, ref: RefMeta, fit: FitFn | None) -> tuple[Variant, str | None]:
    """The variant as tested: its own placement when it fits at the KLD ctx, else the reference's placement with
    the variant's K/V types (the KV effect is what is measured; the floor covers the placement change)."""
    t = v.without_spec()
    if fit is None or not fit(t).bad:
        return t, None
    rv = replace(ref_variant(ref), ctk=t.ctk, ctv=t.ctv)
    return rv, f"{t.name} doesn't fit at ctx {ref.ctx}: tested with the reference's placement ({vname(rv)})"


def kld_rows(runs: list[rs.RunResult]) -> list[dict[str, Any]]:
    """Table rows for KLD results (newest per tested variant)."""
    out = []
    for r in runs:
        q = r.quality or {}
        kv = q.get("kv") or [None, None]
        out.append({"run_id": r.run_id, "mode": q.get("mode"), "variant": q.get("tested") or r.variant,
                    "ran": r.variant, "kv": f"{kv[0] or 'f16'}/{kv[1] or 'f16'}", "ctx": r.ctx,
                    "mean_kld": q.get("mean_kld"), "mean_kld_err": q.get("mean_kld_err"),
                    "kld_p99": q.get("kld_p99"), "same_top_p": q.get("same_top_p"),
                    "ppl_ratio": q.get("ppl_ratio"), "floor": q.get("floor_kld"), "verdict": q.get("verdict"),
                    "vs_q8_ref": q.get("vs_q8_ref"), "status": r.status, "reason": r.reason})
    return out
