"""Common diagnostic interface: every model and every diagnostic plug into this one shape.

Three pieces
    ZooModel        wraps any model behind two calls, so a diagnostic never deals with
                    architecture details:
                        logits(tokens)       (batch, positions, 61) next move scores
                        activations(tokens)  (layers, batch, positions, width) the vector after
                                             each layer: residual stream for transformers,
                                             hidden state for LSTM and Mamba
    the contract    every diagnostic module defines
                        DIAGNOSTIC = "D1" (or D2, D3, D4)
                        def score(model: ZooModel, split: str, n_games: int, data_seed: int,
                                  **config) -> Result
                    and builds its Result with result_for(model, ...) so the fields follow
                    src/component/results.py.
    run()           scores a list of models with one diagnostic, saves every result, and turns
                    a crash into a failed result with the error as the reason, so one bad model
                    never stops a sweep.

Models available now
    TransformerLensModel    the reference Othello GPT (TransformerLens weights)
    TrainPyModel            a model built by src/zoo/train.py; the transformer works now,
                            LSTM and Mamba activations come in A7
    othello_gpt()           loads a reference Othello GPT
    from_checkpoint()       loads a zoo model from outputs/zoo/{model_id}/model.pt

Example (EC2, repo root):
    from src.diagnostics.interface import othello_gpt, from_checkpoint, run   # or load with importlib
    models = [othello_gpt("synthetic"), from_checkpoint("othello_transformer_large_synthetic_s0")]
    run(my_diagnostic_module, models, split="test", n_games=1000, data_seed=0)
"""

import importlib.util
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import torch
import torch.nn as nn

log = logging.getLogger("interface")

ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, rel: str) -> ModuleType:
    """Import a file from src/ by path (src/ is not a package). Registered in sys.modules, so
    dataclasses and pickling work inside the loaded module."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


oth = load_module("othello", "src/domains/othello.py")
results = load_module("results", "src/component/results.py")

GPT_DIR = os.environ.get("WMC_OTHELLO_GPT", os.path.expanduser("~/artifacts/othello_gpt_tl"))
ZOO_DIR = ROOT / "outputs" / "zoo"
MODEL_FIELDS = ("domain", "arch", "scale", "distribution", "seed")


# ==========================================================================
# ZooModel
# ==========================================================================

class ZooModel:
    """One model behind the common calls. Subclasses implement _logits and _activations."""

    def __init__(self, module: nn.Module, model_id: str, arch: str, n_layers: int, d_model: int,
                 device: str | None = None, meta: dict[str, Any] | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.module = module.to(self.device).eval()
        self.model_id = model_id
        self.arch = arch
        self.n_layers = n_layers
        self.d_model = d_model
        # domain, arch, scale, distribution, seed: only needed for models that are not in
        # models.csv (the reference Othello GPTs). Zoo models get them from models.csv.
        self.meta = meta or {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.model_id}, {self.arch}, {self.n_layers} layers x {self.d_model})"

    def _to(self, tokens: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(tokens):
            tokens = torch.as_tensor(tokens)
        return tokens.to(self.device, dtype=torch.long)

    @torch.no_grad()
    def logits(self, tokens: torch.Tensor) -> torch.Tensor:
        """(batch, positions) tokens -> (batch, positions, 61) next move scores."""
        return self._logits(self._to(tokens))

    @torch.no_grad()
    def activations(self, tokens: torch.Tensor) -> torch.Tensor:
        """(batch, positions) tokens -> (layers, batch, positions, width): the vector after each layer."""
        acts = self._activations(self._to(tokens))
        assert acts.shape[0] == self.n_layers and acts.shape[-1] == self.d_model, acts.shape
        return acts

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        """So a ZooModel can be passed wherever a model(tokens) -> logits is expected
        (for example othello.legal_rate)."""
        return self.logits(tokens)

    def _logits(self, tokens: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _activations(self, tokens: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerLensModel(ZooModel):
    """A TransformerLens HookedTransformer, such as the reference Othello GPT."""

    def _logits(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.module(tokens)

    def _activations(self, tokens: torch.Tensor) -> torch.Tensor:
        names = [f"blocks.{L}.hook_resid_post" for L in range(self.n_layers)]
        _, cache = self.module.run_with_cache(tokens, names_filter=lambda n: n in names)
        return torch.stack([cache[n] for n in names])


class TrainPyModel(ZooModel):
    """A model built by src/zoo/train.py (classes GPT, LSTM, Mamba)."""

    def _logits(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.module(tokens)

    def _activations(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.arch == "transformer":
            m = self.module
            T = tokens.shape[1]
            mask = nn.Transformer.generate_square_subsequent_mask(T, device=tokens.device)
            h = m.tok(tokens) + m.pos(torch.arange(T, device=tokens.device))
            out = []
            for layer in m.layers:
                h = layer(h, src_mask=mask, is_causal=True)
                out.append(h)
            return torch.stack(out)
        raise NotImplementedError(f"activations for {self.arch} come in A7 (activation hooks)")


# ==========================================================================
# Loading models
# ==========================================================================

def othello_gpt(which: str = "synthetic", device: str | None = None, gpt_dir: str = GPT_DIR) -> ZooModel:
    """A reference Othello GPT (Li et al. 2023, TransformerLens weights from
    NeelNanda/Othello-GPT-Transformer-Lens). which: synthetic or championship. Reference models
    are scored by every diagnostic as anchors but are not part of the zoo statistics."""
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = HookedTransformerConfig(n_layers=8, d_model=512, d_head=64, n_heads=8, d_mlp=2048,
                                  d_vocab=oth.VOCAB, n_ctx=oth.N_CTX, act_fn="gelu",
                                  normalization_type="LNPre", device=dev)
    module = HookedTransformer(cfg)
    module.load_state_dict(torch.load(os.path.join(gpt_dir, f"{which}_model.pth"), map_location=dev))
    meta = {"domain": "othello", "arch": "transformer", "scale": "othello_gpt",
            "distribution": which, "seed": 0}
    return TransformerLensModel(module, f"othello_gpt_{which}", "transformer", 8, 512, dev, meta)


def from_checkpoint(model_id: str, device: str | None = None, zoo_dir: Path | str = ZOO_DIR) -> ZooModel:
    """A trained zoo model from {zoo_dir}/{model_id}/model.pt. Architecture and scale come from
    models.csv. (A6 adds the registry; this is the minimal loader it builds on.)"""
    train = load_module("zoo_train", "src/zoo/train.py")
    row = results.load_models()[model_id]
    module = train.build_model(row["arch"], row["scale"])
    path = Path(zoo_dir) / model_id / "model.pt"
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    module.load_state_dict(torch.load(path, map_location=dev))
    return TrainPyModel(module, model_id, row["arch"], int(row["n_layers"]), int(row["d_model"]), dev)


# ==========================================================================
# The diagnostic contract and the runner
# ==========================================================================

def result_for(model: ZooModel, diagnostic: str, **kwargs: Any) -> "results.Result":
    """Build a Result for this model: zoo fields come from models.csv, reference model fields
    from model.meta. kwargs are the Result fields (method, split, n_games, data_seed, score, ...)."""
    extra = {k: v for k, v in model.meta.items() if k in MODEL_FIELDS}
    return results.make_result(model.model_id, diagnostic, **extra, **kwargs)


def check_diagnostic(module: ModuleType) -> None:
    """A diagnostic module must define DIAGNOSTIC (D1 to D4) and a callable score."""
    if getattr(module, "DIAGNOSTIC", None) not in results.DIAGNOSTICS:
        raise TypeError(f"{module.__name__}: DIAGNOSTIC must be one of {results.DIAGNOSTICS}")
    if not callable(getattr(module, "score", None)):
        raise TypeError(f"{module.__name__}: needs a function score(model, split, n_games, data_seed, **config)")


def run(diagnostic: ModuleType, models: Iterable[ZooModel], split: str = "test", n_games: int = 1000,
        data_seed: int = 0, results_dir: Path | str | None = None, save: bool = True,
        **config: Any) -> list["results.Result"]:
    """Score every model with one diagnostic. Each result is validated and saved (unless
    save=False). A crash on one model becomes a failed result whose reason is the error, and
    the sweep goes on."""
    check_diagnostic(diagnostic)
    rdir = results_dir if results_dir is not None else results.RESULTS_DIR
    out = []
    for model in models:
        t0 = time.time()
        log.info("%s on %s", diagnostic.DIAGNOSTIC, model.model_id)
        try:
            res = diagnostic.score(model, split=split, n_games=n_games, data_seed=data_seed, **config)
            if res.diagnostic != diagnostic.DIAGNOSTIC or res.model_id != model.model_id:
                raise ValueError(f"score() returned a result for {res.model_id} {res.diagnostic}")
        except Exception as e:  # one bad model must not stop the sweep
            log.warning("  failed: %s", e)
            tb = traceback.format_exc(limit=3)
            res = result_for(model, diagnostic.DIAGNOSTIC, method=getattr(diagnostic, "METHOD", "see module"),
                             split=split, n_games=n_games, data_seed=data_seed, score=None,
                             status="failed", reason=f"{type(e).__name__}: {e}",
                             details={"traceback": tb}, runtime_sec=round(time.time() - t0, 1))
        if save:
            results.save(res, rdir)
        log.info("  %s  score %s  (%.0f s)", res.status, res.score, time.time() - t0)
        out.append(res)
    return out
