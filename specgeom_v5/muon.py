"""Muon optimizer (Jordan et al.) — momentum + Newton-Schulz orthogonalization.

Applied only to 2D weight matrices of the transformer blocks; embeddings, norms
and lm_head should go to a separate AdamW group (standard Muon practice).
"""

import torch


@torch.no_grad()
def newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Quintic Newton-Schulz iteration approximating polar(G) = U V^T."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5,
                 weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(g)
                g = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                u = newton_schulz(g.float(), steps=group["ns_steps"])
                # scale as in the reference impl: sqrt(max(m,n)/min(m,n)) keeps
                # update RMS comparable across shapes
                scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
                if group["weight_decay"] > 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(u, alpha=-group["lr"] * scale)
        return loss
