"""CPU unit tests for the v2 additions (torch + pandas only, no transformers).

  interventions_v2 : matched modes have ||Hp|| == ||H||; exact_iso restores
                     the anchor spectrum; random_ext dictionary is orthonormal
  intervene_engine_v2 : every v2 mode runs on a tiny model whose parameter
                     names match the engine regex; matched modes log scales;
                     exact_iso keeps singular values fixed across steps
  compute_metrics_v2 : split-half statistics recover a planted diagonal
                     signal (R_sigma_cross >> r_null) while pooled
                     spearman_absC_snr stays high even with NO signal
                     (documents the v1 coupling); H self-check passes on a
                     synthetic spectrum_only capture and fails on a raw one

Run: ~/envs_spectral/bin/python analysis_v3/unit_test_cpu_v2.py
"""

import json
import os
import shutil
import sys
import tempfile

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from specgeom_v3 import metrics
from specgeom_v3.interventions import (MATCHED, project_update, random_basis)
from specgeom_v3.intervene_engine import InterventionEngine

torch.manual_seed(0)
m, n = 64, 48


def test_matched_modes():
    W = torch.randn(m, n)
    H = 1e-3 * torch.randn(m, n)
    U, S, V = metrics.weight_svd(W)
    for mode in ("spectrum_matched", "frame_matched"):
        Hp = project_update(W, H, mode)
        assert abs(Hp.norm() / H.norm() - 1) < 1e-4, mode
    spec = project_update(W, H, "spectrum_only")
    Hs = project_update(W, H, "spectrum_matched")
    # matched spectrum is a pure rescale of the unmatched one
    ratio = (Hs / spec)[spec.abs() > 1e-9]
    assert ratio.std() / ratio.mean().abs() < 1e-4
    assert ratio.mean() > 1.0, "spectral part of a random H must be amplified"
    # random_ext: orthonormal dictionary, output diagonal in that basis, matched
    Ur, Vr = random_basis(m, n, min(m, n), torch.Generator().manual_seed(1))
    assert (Ur.T @ Ur - torch.eye(min(m, n))).abs().max() < 1e-5
    assert (Vr.T @ Vr - torch.eye(min(m, n))).abs().max() < 1e-5
    Hr = project_update(W, H, "random_ext", rand_basis=(Ur, Vr))
    Cr = Ur.T @ Hr @ Vr
    off = Cr - torch.diag(Cr.diagonal())
    assert off.abs().max() < 1e-5 * Cr.abs().max()
    assert abs(Hr.norm() / H.norm() - 1) < 1e-4
    # exact_iso: singular values of W_before + Hp equal the anchor
    S_anchor = torch.linalg.svdvals(W)
    Hi = project_update(W, H, "exact_iso", S_anchor=S_anchor)
    S_new = torch.linalg.svdvals(W + Hi)
    assert (S_new - S_anchor).abs().max() < 1e-4
    print("matched modes ok")


class Tiny(nn.Module):
    """Parameter names match the engine regex: layers.N.mlp.X.weight."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({"mlp": nn.ModuleDict({
                "up_proj": nn.Linear(n, m, bias=False),
                "down_proj": nn.Linear(m, n, bias=False)})})
            for _ in range(2)])

    def forward(self, x):
        for l in self.layers:
            x = l["mlp"]["down_proj"](torch.tanh(l["mlp"]["up_proj"](x)))
        return x


def _run_engine(mode, steps=3, seed=0):
    torch.manual_seed(seed)
    model = Tiny()
    eng = InterventionEngine(model, mode, refresh_every=2, seed=seed)
    assert len(eng.params) == 4
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(16, n)
    svals = []
    for _ in range(steps):
        opt.zero_grad()
        (model(x) ** 2).mean().backward()
        eng.observe_grad()
        eng.pre_step()
        opt.step()
        eng.post_step()
        svals.append({k: torch.linalg.svdvals(p.detach()) for k, p in eng.params.items()})
        for p in eng.params.values():
            assert torch.isfinite(p).all()
    return eng, svals


def test_engine_modes():
    for mode in MATCHED:
        eng, svals = _run_engine(mode)
        if mode in ("spectrum_matched", "frame_matched", "random_ext"):
            assert len(eng.scale_log) == 4, mode
            assert all(v > 0 for v in eng.scale_log.values()), mode
        if mode == "exact_iso":
            for k, S0 in eng.S_anchor.items():
                for sv in svals:
                    assert (sv[k] - S0).abs().max() < 1e-4, "exact_iso drifted"
    # spectrum_matched must move the weights as much as the raw step would
    torch.manual_seed(0)
    model = Tiny()
    eng = InterventionEngine(model, "spectrum_matched")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(16, n)
    opt.zero_grad()
    (model(x) ** 2).mean().backward()
    eng.observe_grad()
    eng.pre_step()
    w0 = {k: p.detach().clone() for k, p in eng.params.items()}
    opt.step()
    raw = {k: (p.detach() - w0[k]).norm() for k, p in eng.params.items()}
    eng.post_step()
    for k, p in eng.params.items():
        assert abs((p.detach() - w0[k]).norm() / raw[k] - 1) < 1e-3
    print("engine modes ok")


def _fake_capture(run, intervention, B=8, planted="diag", seed=0):
    """Write args.json + one capture file with a planted mean gradient.

    planted: "diag" (signal only on the spectral diagonal), "dense" (signal on
    a random half of the entries of C), or None (pure noise).
    """
    from analysis_v3 import compute_metrics as cm
    g = torch.Generator().manual_seed(seed)
    os.makedirs(os.path.join(run, "capture"), exist_ok=True)
    json.dump({"intervention": intervention}, open(os.path.join(run, "args.json"), "w"))
    W = torch.randn(m, n, generator=g)
    U, S, V = metrics.weight_svd(W)
    r = min(m, n)
    if planted == "diag":
        # per-sample SNR ~1 on the diagonal (entries ~3 vs noise std 3)
        M = torch.diag(3.0 * torch.randn(r, generator=g))
    elif planted == "dense":
        M = 3.0 * torch.randn(r, r, generator=g) * (torch.rand(r, r, generator=g) < 0.5)
    else:
        M = torch.zeros(r, r)
    mu = U @ M @ V.T
    Gb = {}
    for b in range(B):
        noise = torch.randn(m, n, generator=g) * 3.0
        Gb[f"b{b}"] = (mu + noise).to(torch.bfloat16)
    G = torch.stack([v.float() for v in Gb.values()]).mean(0)
    H = -1e-3 * G
    if intervention == "spectrum_only":
        H = project_update(W, H, "spectrum_only")
    rec = {"step": 10, "mats": {"model.layers.0.mlp.up_proj.weight": {
        "W": W.to(torch.bfloat16), "G": G.to(torch.bfloat16),
        "H": H.to(torch.bfloat16), "G_b": Gb}}}
    torch.save(rec, os.path.join(run, "capture", "step_000010.pt"))
    return cm


def test_cross_metrics():
    import pandas as pd
    tmp = tempfile.mkdtemp()
    try:
        # planted diagonal signal, spectrum_only H -> self-check must pass
        run = os.path.join(tmp, "planted_spec")
        cm = _fake_capture(run, "spectrum_only", planted="diag")
        cm.process_run(run)
        df = pd.read_csv(os.path.join(run, "metrics.csv"))
        man = json.load(open(os.path.join(run, "manifest.json")))
        r = df.iloc[0]
        assert r.R_sigma_cross > 10 * r.r_null, (r.R_sigma_cross, r.r_null)
        assert r.sig_energy_total > 0 and r.noise_energy > 0
        assert man["h_check"]["passed"] is True
        # dense signal on half the entries: the split-half statistics must
        # detect it (magnitude on half A predicts reliability on half B)
        rund = os.path.join(tmp, "planted_dense")
        cm = _fake_capture(rund, "none", planted="dense", seed=3)
        cm.process_run(rund)
        dd = pd.read_csv(os.path.join(rund, "metrics.csv")).iloc[0]
        assert dd.spearman_absC_snr_cross > 0.15, dd.spearman_absC_snr_cross
        assert dd.spearman_absC_split > 0.15, dd.spearman_absC_split
        # no signal at all: cross statistic must sit near zero / undefined,
        # while the v1 pooled spearman stays high (the coupling artefact)
        run0 = os.path.join(tmp, "null_raw")
        cm = _fake_capture(run0, "none", planted=None, seed=1)
        cm.process_run(run0)
        d0 = pd.read_csv(os.path.join(run0, "metrics.csv")).iloc[0]
        assert d0.spearman_absC_snr > 0.5, "v1 statistic should be coupled"
        assert abs(d0.spearman_absC_snr_cross) < 0.2, d0.spearman_absC_snr_cross
        assert abs(d0.spearman_absC_split) < 0.2, d0.spearman_absC_split
        # raw H labelled spectrum_only -> self-check must fail
        runf = os.path.join(tmp, "bad_label")
        cm = _fake_capture(runf, "spectrum_matched", planted="diag", seed=2)
        cm.process_run(runf)
        manf = json.load(open(os.path.join(runf, "manifest.json")))
        assert manf["h_check"]["passed"] is False
    finally:
        shutil.rmtree(tmp)
    print("cross metrics ok")


if __name__ == "__main__":
    test_matched_modes()
    test_engine_modes()
    test_cross_metrics()
    print("ALL_CPU_TESTS_V2_PASS")
