"""v3 CPU tests (run after unit_test_cpu.py and unit_test_cpu_v2.py, which must also
pass on the v3 package):

  1. engine s_rel scale: ||Hp||_F == s_rel * ||H||_F for spectrum_matched, frame_matched,
     random_ext; cos(Hp, H) == 1/scale_log (v3 doc §3.1); spectrum_matched's applied update is
     diagonal in the SVD basis of W_before (moves only singular values).
  2. per-step basis refresh (default refresh_every=1) and random_ext short-circuit
     (no SVD basis is built).
  3. exact_iso keeps the singular values of the anchor exactly.
  4. Instrumenter stores G/H/W in fp32.
"""

import os
import shutil
import sys
import tempfile

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from specgeom_v3.instrument import Instrumenter  # noqa: E402
from specgeom_v3.intervene_engine import InterventionEngine  # noqa: E402

torch.manual_seed(0)


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(d, d, bias=False)
        self.self_attn.o_proj = nn.Linear(d, d, bias=False)
        self.mlp = nn.Module()
        self.mlp.up_proj = nn.Linear(d, h, bias=False)
        self.mlp.down_proj = nn.Linear(h, d, bias=False)


class Tiny(nn.Module):
    def __init__(self, d=24, h=40, L=3):
        super().__init__()
        self.layers = nn.ModuleList([Block(d, h) for _ in range(L)])
        self.config = type("C", (), {"num_hidden_layers": L})()


def fake_step(model, gen, lr=1e-3):
    """Simulate an optimizer step: W <- W + lr * N(0,1)."""
    with torch.no_grad():
        for p in model.parameters():
            p.add_(lr * torch.randn(p.shape, generator=gen))


def run_mode(mode, scale, refresh_every=1):
    model = Tiny()
    gen = torch.Generator().manual_seed(1)
    eng = InterventionEngine(model, mode, seed=0, scale=scale, refresh_every=refresh_every)
    eng.pre_step()
    w0 = {n: p.detach().clone() for n, p in eng.params.items()}
    fake_step(model, gen)
    h = {n: p.detach().clone() - w0[n] for n, p in eng.params.items()}
    eng.post_step()
    hp = {n: p.detach().clone() - w0[n] for n, p in eng.params.items()}
    return eng, w0, h, hp


def test_scale_and_cos():
    for mode in ("spectrum_matched", "frame_matched", "random_ext"):
        for scale in (1.0, 3.0):
            eng, w0, h, hp = run_mode(mode, scale)
            for n in h:
                ratio = hp[n].norm() / h[n].norm()
                assert abs(ratio - scale) < 1e-4, (mode, scale, n, ratio)
                cos = (hp[n] * h[n]).sum() / (hp[n].norm() * h[n].norm())
                assert abs(cos - eng.cos_log[n]) < 1e-4, (mode, n, cos, eng.cos_log[n])
                assert abs(eng.cos_log[n] * eng.scale_log[n] - 1) < 1e-6
            if mode == "spectrum_matched":
                for n in h:
                    # spectrum-only: the applied update is diagonal in the SVD
                    # basis of W_before (Hp = U diag(c) V^T), so it moves only the
                    # singular values of that decomposition. (Checking U itself is
                    # ill-posed on random tiny matrices: near-degenerate singular
                    # values let the SVD reorder/mix columns.)
                    U, S, Vh = torch.linalg.svd(w0[n], full_matrices=False)
                    C = U.T @ hp[n] @ Vh.T
                    off = (C - torch.diag(C.diagonal())).norm() / C.norm()
                    assert off < 1e-4, (n, off)
                    assert C.diagonal().abs().max() > 0, "singular values must move"
            if mode == "random_ext":
                assert not eng.basis, "random_ext must not build an SVD basis"
    print("scale / cos / spectrum-only ok")


def test_refresh_every_step():
    model = Tiny()
    gen = torch.Generator().manual_seed(2)
    eng = InterventionEngine(model, "spectrum_matched", seed=0)
    eng.pre_step()
    b0 = {n: eng.basis[n][0].clone() for n in eng.params}
    fake_step(model, gen, lr=1e-2)
    eng.post_step()
    eng.pre_step()
    moved = any((eng.basis[n][0] - b0[n]).abs().max() > 1e-6 for n in eng.params)
    assert moved, "basis must be refreshed at every step (refresh_every=1)"
    print("per-step basis refresh ok")


def test_exact_iso():
    eng, w0, h, hp = run_mode("exact_iso", 1.0)
    for n in h:
        S0 = torch.linalg.svdvals(w0[n])
        S1 = torch.linalg.svdvals(w0[n] + hp[n])
        assert (S1 - S0).abs().max() < 1e-4, (n, (S1 - S0).abs().max())
    print("exact_iso ok")


def test_capture_fp32():
    model = Tiny()
    out = tempfile.mkdtemp()
    try:
        ins = Instrumenter(model, out, 3, save_every=1)
        ins.begin_step(1)
        for p in ins.tracked.values():
            p.grad = torch.randn_like(p)
        ins.capture_grad()
        ins.pre_optimizer()
        fake_step(model, torch.Generator().manual_seed(3))
        ins.post_optimizer()
        path = ins.flush()
        rec = torch.load(path)
        for n, d in rec["mats"].items():
            for k in ("G", "H", "W"):
                assert d[k].dtype == torch.float32, (n, k, d[k].dtype)
        print("capture G/H/W fp32 ok")
    finally:
        shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    test_scale_and_cos()
    test_refresh_every_step()
    test_exact_iso()
    test_capture_fp32()
    print("ALL_CPU_TESTS_V3_PASS")
