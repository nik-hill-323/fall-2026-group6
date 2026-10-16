"""Tests for Diagnostic 4, representation probing (src/diagnostics/d4/representation_probe.py).

They use random legal games and small models, so they run anywhere (including GitHub Actions):
    planted    layer 0 is noise, layer 1 holds the true board as a one hot code: the probe must
               read layer 1 almost perfectly, pick it as the best layer, and beat the control.
               Its next move scores are read from layer 1, so it USES its board: the causal check
               must call it causal
    stored     the same layers, but its next move scores come from the move list, not from
               layer 1: decodable but not causal
    tiny       a real (untrained) train.py transformer through the full score() contract
    patching   logits_patched on all three train.py architectures (Amendment 2)

Run from the repo root:
    python -m pytest src/tests/test_d4_representation_probe.py -q
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("d4_probe", ROOT / "src/diagnostics/d4/representation_probe.py")
d4 = importlib.util.module_from_spec(_spec)
sys.modules["d4_probe"] = d4
_spec.loader.exec_module(d4)
oth, itf = d4.oth, d4.itf
train = itf.load_module("zoo_train", "src/zoo/train.py")

ZOO_ID = "othello_transformer_small_synthetic_s0"   # a real models.csv row (synthetic distribution)


def legal_logits_from_board(rel: torch.Tensor) -> torch.Tensor:
    """(B, T, 64) mine / theirs / empty boards -> (B, T, 61) scores: 1 on the legal next moves."""
    B, T, _ = rel.shape
    out = torch.zeros(B, T, oth.VOCAB)
    for b in range(B):
        for t in range(T):
            r = rel[b, t].numpy()
            bw = np.where(r == 1, oth.BLACK, np.where(r == 2, oth.WHITE, oth.EMPTY)).astype(np.int8)
            for s in oth.legal_next(bw, oth.BLACK):        # the player who just moved owns "mine"
                out[b, t, oth.SQ_TO_TOKEN[s]] = 1.0
    return out


class Planted(itf.ZooModel):
    """Layer 0: random noise. Layer 1: the true mine / theirs / empty board, one hot (64 x 3).
    Next move scores are read from layer 1, so an edit of layer 1 changes the predictions."""

    def __init__(self) -> None:
        super().__init__(nn.Identity(), ZOO_ID, "transformer", 2, oth.N_SQ * 3, "cpu")

    @staticmethod
    def readout(layer1: torch.Tensor) -> torch.Tensor:
        B, T, _ = layer1.shape
        return legal_logits_from_board(layer1.view(B, T, oth.N_SQ, 3).argmax(-1))

    def _logits_edited(self, tokens, positions, edits):
        acts = self._activations(tokens)
        for layer in sorted(edits):            # layers do not feed each other here; only layer 1 is read
            acts[layer] = itf._edit_at(acts[layer], positions, edits[layer])
        return self.readout(acts[1])

    def _activations(self, tokens: torch.Tensor) -> torch.Tensor:
        tok = tokens.cpu().numpy()
        B, T = tok.shape
        board = np.zeros((B, T, oth.N_SQ), dtype=np.int64)
        for b in range(B):
            moves = np.array([oth.TOKEN_TO_SQ[int(t)] for t in tok[b] if t > 0])
            boards, movers = oth.replay(moves)
            rel = np.where(boards == 0, 0, np.where(boards == movers[:, None], 1, 2))
            board[b, : len(moves)] = rel
        onehot = torch.nn.functional.one_hot(torch.as_tensor(board), 3).float().view(B, T, -1)
        noise = torch.randn(B, T, oth.N_SQ * 3, generator=torch.Generator().manual_seed(int(tok.sum())))
        return torch.stack([noise, onehot])

    def _logits(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.readout(self._activations(tokens)[1])


class Stored(Planted):
    """Same activations as Planted, but the next move scores come from replaying the move list,
    so whatever is written into layer 1 never reaches the predictions."""

    def _logits(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.readout(Planted._activations(self, tokens)[1])

    def _logits_edited(self, tokens, positions, edits):
        acts = self._activations(tokens)
        for layer in sorted(edits):            # the edits run, but the predictions never read them
            acts[layer] = itf._edit_at(acts[layer], positions, edits[layer])
        return self._logits(tokens)


def tiny() -> "itf.ZooModel":
    torch.manual_seed(0)
    module = train.ARCHS["transformer"](n_layers=2, d_model=16, n_heads=2)
    return itf.TrainPyModel(module, ZOO_ID, "transformer", 2, 16, "cpu")


@pytest.fixture()
def fake_data(monkeypatch):
    def fake_load(name, split, n_games, seed=0, **kw):
        return oth.random_games(n_games, seed=seed + (0 if split == "val" else 1000))
    monkeypatch.setattr(oth, "load_distribution", fake_load)


SMALL = dict(n_fit=150, n_choose=50, epochs=3, batch=50, pos_batch=32)   # small data, so small steps
FAST = dict(SMALL, mlp_hidden=0, causal=False)                         # linear probe only


# ==========================================================================
# Pieces
# ==========================================================================

def test_type_ids_are_unique_per_square_and_position() -> None:
    tok = np.array([[1, 2, 60], [60, 2, 1]])
    ids = d4.type_ids(tok)
    assert ids.max() < d4.N_TYPES and ids.min() >= 0
    assert ids[0, 1] == ids[1, 1]                  # same square at the same position: same type
    assert ids[0, 0] != ids[1, 2]                  # same square at another position: different type


def test_control_table_is_fixed_and_follows_frequencies() -> None:
    freq = np.tile([0.6, 0.3, 0.1], (oth.N_SQ, 1))
    a, b = d4.control_table(freq, 0), d4.control_table(freq, 0)
    assert np.array_equal(a, b) and a.shape == (d4.N_TYPES, oth.N_SQ)
    share = np.bincount(a.ravel(), minlength=3) / a.size
    assert np.allclose(share, [0.6, 0.3, 0.1], atol=0.02)


def test_occupancy_baseline_knows_empty_squares_only() -> None:
    labels = np.array([[[0, 1, 2, 1]]])            # one position, four squares
    mask = np.array([[True]])
    rule = np.array([1, 1, 1, 1])                  # occupied squares always called mine
    assert d4.occupancy_accuracy(labels, mask, rule) == pytest.approx(3 / 4)   # wrong only on the theirs square


# ==========================================================================
# Full runs
# ==========================================================================

def test_planted_board_is_read_at_the_right_layer(fake_data) -> None:
    r = d4.score(Planted(), split="test", n_games=50, data_seed=0, **SMALL)
    d = r.details
    assert d["best_layer"] == 1
    assert d["probe_acc_by_layer"][1] > 0.95                   # the board is linearly readable
    assert d["probe_acc_occupied_by_layer"][1] > 0.9           # including whose disc it is
    assert d["probe_acc_by_layer"][1] > d["occupancy_baseline_acc"]
    assert r.score == pytest.approx(d["selectivity_by_layer"][1], abs=1e-4)
    assert r.score > 0.2                                       # far above the control


def test_tiny_model_follows_contract_and_saves(fake_data, tmp_path: Path) -> None:
    out = itf.run(d4, [tiny()], split="test", n_games=40, data_seed=0, results_dir=tmp_path, **SMALL)
    r = out[0]
    assert r.status == "ok"
    d = r.details
    assert len(d["probe_acc_by_layer"]) == len(d["selectivity_by_layer"]) == 2
    assert d["gain_over_layer0_by_layer"][0] == pytest.approx(0.0)
    assert 0.0 <= d["occupancy_baseline_acc"] <= 1.0
    best = d["best_layer"]
    assert best == int(np.argmax(d["selectivity_by_layer_choose"]))   # chosen on val, not on test
    assert r.score == pytest.approx(d["selectivity_by_layer"][best], abs=1e-4)
    saved = itf.results.load(itf.results.result_path(ZOO_ID, "D4", tmp_path))
    assert saved.diagnostic == "D4" and saved.score == r.score


def test_same_seed_same_score(fake_data) -> None:
    a = d4.score(tiny(), n_games=40, data_seed=0, **SMALL).score
    b = d4.score(tiny(), n_games=40, data_seed=0, **SMALL).score
    assert a == pytest.approx(b, abs=1e-6)


def test_module_passes_interface_check() -> None:
    itf.check_diagnostic(d4)


# ==========================================================================
# Amendment 2: MLP probe, causal check, patching
# ==========================================================================

def test_mlp_probe_does_not_change_the_score(fake_data) -> None:
    """The score is the linear selectivity (A6): adding the MLP probe and the causal check must
    leave it exactly as it was."""
    with_all = d4.score(tiny(), n_games=40, data_seed=0, **SMALL, causal_cases_max=10)
    linear_only = d4.score(tiny(), n_games=40, data_seed=0, **FAST)
    assert with_all.score == pytest.approx(linear_only.score, abs=1e-7)
    assert "mlp" in with_all.details and "mlp" not in linear_only.details
    assert "causal" in with_all.details and "causal" not in linear_only.details


def test_planted_board_mlp_and_causal(fake_data) -> None:
    r = d4.score(Planted(), split="test", n_games=40, data_seed=0, **SMALL)
    m, c = r.details["mlp"], r.details["causal"]
    assert m["probe_acc_by_layer"][1] > 0.9 and m["best_layer"] == 1
    assert m["selectivity_at_best"] > 0.2
    assert c["layers"] == [1] and c["n_cases"] >= 30          # from the best layer (1) to the last (1)
    assert c["edit_success_rate"] > 0.9                 # the probe reads the edited board
    assert c["boundary_acc_none"] < 0.2                 # unedited, it follows the true board
    assert c["boundary_acc_edit"] > 0.8                 # edited, it follows the edited board
    assert c["causal_score"] > 0.5 and c["causal_score_ci"][0] > 0
    assert c["causal_label"] == "decodable and causal"


def test_stored_board_is_decodable_but_not_causal(fake_data) -> None:
    r = d4.score(Stored(), split="test", n_games=40, data_seed=0, **SMALL)
    c = r.details["causal"]
    assert r.score > 0.2                                # the board is there to read
    assert c["edit_success_rate"] > 0.9
    assert abs(c["causal_score"]) < 0.05                # but editing it changes nothing
    assert c["causal_label"] == "decodable but not causal"


def test_causal_cases_flip_one_square_and_change_the_legal_moves() -> None:
    data = oth.build(oth.random_games(30, seed=3))
    cases = d4.causal_cases(data, seed=0)
    assert len(cases["game"]) >= 25
    for i in range(len(cases["game"])):
        g, t, sq = cases["game"][i], cases["pos"][i], cases["square"][i]
        true = data["mine_theirs"][g, t]
        diff = np.flatnonzero(cases["target"][i] != true)
        assert list(diff) == [sq] and true[sq] != 0       # exactly one occupied square flipped
        assert not np.array_equal(cases["legal_before"][i], cases["legal_after"][i])
        assert cases["legal_after"][i].any() and not cases["legal_after"][i][0]
        assert t < data["mask"][g].sum() - 1               # a recorded next move exists
    again = d4.causal_cases(data, seed=0)
    assert np.array_equal(again["square"], cases["square"])


def test_boundary_accuracy_counts_gained_and_lost_moves() -> None:
    before = np.zeros((1, oth.VOCAB), bool)
    after = np.zeros((1, oth.VOCAB), bool)
    before[0, [1, 2]] = True                           # legal before: 1, 2
    after[0, [2, 3]] = True                            # legal after: 2, 3 (1 lost, 3 gained)
    follows = np.zeros((1, oth.VOCAB))
    follows[0, [2, 3]] = 5.0                           # the model accepts the new legal set
    ignores = np.zeros((1, oth.VOCAB))
    ignores[0, [1, 2]] = 5.0                           # the model keeps the old legal set
    assert d4.boundary_accuracy(follows, before, after)[0] == pytest.approx(1.0)
    assert d4.boundary_accuracy(ignores, before, after)[0] == pytest.approx(0.0)


def test_causal_label_rules() -> None:
    assert d4.causal_label(0.05, 0.9, 0.5) == "not decodable"
    assert d4.causal_label(0.3, 0.25, 0.01) == "decodable and causal"
    assert d4.causal_label(0.3, 0.15, 0.05) == "decodable but not causal"
    assert d4.causal_label(0.3, 0.25, -0.01) == "decodable but not causal"


@pytest.mark.parametrize("arch", ["transformer", "lstm", "mamba"])
def test_logits_patched_on_every_architecture(arch: str) -> None:
    torch.manual_seed(0)
    module = train.ARCHS[arch](n_layers=2, d_model=16, n_heads=2)
    model = itf.TrainPyModel(module, ZOO_ID, arch, 2, 16, "cpu")
    tok = torch.as_tensor(oth.build(oth.random_games(3, seed=1))["tokens"][:, :12])
    pos = torch.tensor([5, 7, 9])
    rows = torch.arange(3)
    base = model.logits(tok)
    for layer in (0, 1):
        same = model.activations(tok)[layer][rows, pos]
        assert torch.allclose(model.logits_patched(tok, layer, pos, same), base, atol=1e-5)
        # a random edit, not a constant: LayerNorm removes a constant added to every feature
        bump = 3.0 * torch.randn(same.shape, generator=torch.Generator().manual_seed(layer))
        moved = model.logits_patched(tok, layer, pos, same + bump)
        for b in range(3):
            p = int(pos[b])
            assert torch.allclose(moved[b, :p], base[b, :p], atol=1e-5)      # the past is untouched
            assert not torch.allclose(moved[b, p], base[b, p], atol=1e-4)    # the edited position changes
    assert torch.allclose(model.logits(tok), base)                            # the hook is removed


def test_edit_layers_modes() -> None:
    assert d4.edit_layers(5, 8, "from_best") == [5, 6, 7]
    assert d4.edit_layers(5, 8, "best") == [5]
    assert d4.edit_layers(6, 8, "from_middle") == [4, 5, 6, 7]
    assert d4.edit_layers(2, 8, "from_middle") == [2, 3, 4, 5, 6, 7]
    assert d4.edit_layers(5, 8, "all") == list(range(8))


@pytest.mark.parametrize("arch", ["transformer", "lstm", "mamba"])
def test_edits_run_in_layer_order_and_later_layers_see_them(arch: str) -> None:
    torch.manual_seed(0)
    module = train.ARCHS[arch](n_layers=3, d_model=16, n_heads=2)
    model = itf.TrainPyModel(module, ZOO_ID, arch, 3, 16, "cpu")
    tok = torch.as_tensor(oth.build(oth.random_games(2, seed=1))["tokens"][:, :10])
    pos = torch.tensor([4, 6])
    seen = []
    def record(layer):
        def fn(v):
            seen.append(layer)
            return v + torch.randn(v.shape, generator=torch.Generator().manual_seed(layer))
        return fn
    both = model.logits_edited(tok, pos, {2: record(2), 0: record(0)})
    assert seen == [0, 2]                                   # forward order, whatever the dict order
    only_last = model.logits_edited(tok, pos, {2: record(2)})
    assert not torch.allclose(both, only_last, atol=1e-5)   # the layer 0 edit reached the output
