# -*- coding: utf-8 -*-
"""Model zoo and training utilities for the connection-invariant surrogate.

All models map (theta, i_c) -> (psi_c, T). Feature layout:
    x = [ i_c / 60  (12),  phi(theta) (7) ]
with phi = [1, sin2t, cos2t, sin4t, cos4t, sin6t, cos6t], t = theta in rad.

Models
------
MLP          : plain black-box regressor (baseline).
GradNet      : vector-field gradient network. g(x) = A0 x + b0 + A1^T s(A1 x + c1)
               with s = softplus and A0 = blkdiag(P, 0), P = B^T B + eps I PSD.
               By construction g = grad_x W-hat, so psi = g[:12] and
    T = I_SCALE g[12:] . dphi/dtheta share one scalar potential and the
    Jacobian is symmetric PSD (reciprocity + monotonicity exact).
ScalarEnergy : scalar co-energy head W-hat(i, theta), psi/T via autograd.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import exp.wmi_fix  # noqa: F401  (WMI hang workaround, before numpy/torch)

import numpy as np
import torch
from torch import nn

I_SCALE = 60.0
HARM = (2, 4, 6)


def theta_features(theta_deg):
    t = np.deg2rad(theta_deg)
    cols = [np.ones_like(t)]
    for k in HARM:
        cols += [np.sin(k * t), np.cos(k * t)]
    return np.stack(cols, axis=1)


def theta_feature_grad(theta_deg):
    """d(phi)/d(theta_rad), theta in radians."""
    t = np.deg2rad(theta_deg)
    cols = [np.zeros_like(t)]
    for k in HARM:
        cols += [k * np.cos(k * t), -k * np.sin(k * t)]
    return np.stack(cols, axis=1)


def make_features(theta_deg, ic):
    return np.concatenate(
        [ic / I_SCALE, theta_features(theta_deg)], axis=1
    ).astype(np.float32)


class MLP(nn.Module):
    def __init__(self, d_in=19, d_out=13, width=128, depth=3):
        super().__init__()
        layers = [nn.Linear(d_in, width), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.GELU()]
        layers += [nn.Linear(width, d_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GradNet(nn.Module):
    """Vector-field gradient network: exact gradient of an implicit scalar."""

    def __init__(self, n_i=12, n_phi=7, hidden=96, eps=1e-3):
        super().__init__()
        self.n_i, self.n_phi = n_i, n_phi
        d = n_i + n_phi
        self.B = nn.Parameter(torch.randn(n_i, n_i) * 0.05)
        self.eps = eps
        self.b0 = nn.Parameter(torch.zeros(d))
        self.A1 = nn.Parameter(torch.randn(hidden, d) * (2.0 / d) ** 0.5)
        self.c1 = nn.Parameter(torch.zeros(hidden))

    def forward(self, x):
        # g(x) = A0 x + b0 + A1^T softplus(A1 x + c1), A0 = blkdiag(B^T B + eps I, 0)
        z = torch.nn.functional.softplus(x @ self.A1.T + self.c1)
        quad = x.clone()
        quad[:, : self.n_i] = x[:, : self.n_i] @ (
            self.B.T @ self.B + self.eps * torch.eye(self.n_i, device=x.device)
        )
        quad[:, self.n_i :] = 0.0
        return quad + self.b0 + z @ self.A1

    def psi_T(self, x, dphi_dtheta):
        """x: features; dphi_dtheta: (n, n_phi) d phi/d theta_rad. Returns psi(12), T."""
        g = self.forward(x)
        return g[:, : self.n_i], I_SCALE * (
            g[:, self.n_i :] * dphi_dtheta
        ).sum(dim=1)


class ScalarEnergy(nn.Module):
    """Scalar magnetic co-energy W(i, theta); psi/T by autograd."""

    def __init__(self, n_coil=12, width=128):
        super().__init__()
        self.L = nn.Parameter(torch.randn(n_coil, n_coil) * 0.02)

        def mlp(d_in, d_out, w, depth):
            layers = [nn.Linear(d_in, w), nn.GELU()]
            for _ in range(depth - 1):
                layers += [nn.Linear(w, w), nn.GELU()]
            layers += [nn.Linear(w, d_out)]
            return nn.Sequential(*layers)

        self.n_phi = 7
        self.bias = mlp(self.n_phi, n_coil, width, 2)
        self.res = mlp(n_coil + self.n_phi, 1, width, 3)
        # Start as the pure quadratic + PM-affine model; the residual learns
        # to leave it only where saturation demands it.
        last = self.res[-1]
        last.weight.data.zero_()
        last.bias.data.zero_()

    def forward(self, ic, phi):
        lic = ic @ self.L
        quad = 0.5 * torch.sum(lic * lic, dim=1)
        bias = torch.sum(self.bias(phi) * ic, dim=1)
        res = self.res(torch.cat([ic, phi], dim=1)).squeeze(1)
        return quad + bias + res

    def psi_T(self, ic, phi, dphi_dtheta):
        ic = ic.clone().requires_grad_(True)
        phi = phi.clone().requires_grad_(True)
        W = self.forward(ic, phi).sum()
        psi = torch.autograd.grad(W, ic, create_graph=True)[0]
        gphi = torch.autograd.grad(W, phi, create_graph=True)[0]
        return psi, I_SCALE * (gphi * dphi_dtheta).sum(dim=1)


class StructuredEnergy(nn.Module):
    """Structured co-energy: explicit Fourier inductance + PM flux + NN residual.

    W(i, phi) = 0.5 i' L(phi) i + pm(phi)' i + R(i, phi), with t = theta/1 rad
    encoded by the harmonic features phi = [1, sin2t, cos2t, sin4t, cos4t,
    sin6t, cos6t]. L = A0 A0' + eps I + sum_{h in (4,8)} [C_h cos(ht) + S_h
    sin(ht)] (L0 PSD, harmonics symmetrized; cos/sin(8t) from double-angle on
    the 4t column); pm = P f(phi) with f the harmonic basis. R is a
    zero-initialized MLP (saturation/cross-coupling only).

    psi = dW/di and T = dW/dt = (dW/dphi) . dphi/dt share one scalar
    potential, so reciprocity and torque-flux consistency hold exactly.
    """

    H_L = (4, 8)
    H_PM = (2, 4, 6)

    def __init__(self, n_coil=12, width=96, depth=3):
        super().__init__()
        nc = n_coil
        self.nc = nc
        self.A0 = nn.Parameter(torch.randn(nc, nc) * 0.08)
        self.eps = 1e-4
        self.C = nn.ParameterDict({
            f"c{h}": nn.Parameter(torch.randn(nc, nc) * 0.02) for h in self.H_L})
        self.S = nn.ParameterDict({
            f"s{h}": nn.Parameter(torch.randn(nc, nc) * 0.02) for h in self.H_L})
        # pm basis: 1, cos/sin(2t), cos/sin(4t), cos/sin(6t) -> columns of phi
        self.P = nn.Parameter(torch.randn(nc, 7) * 0.05)

        layers = [nn.Linear(nc + 7, width), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.GELU()]
        layers += [nn.Linear(width, 1)]
        self.res = nn.Sequential(*layers)
        self.res[-1].weight.data.zero_()
        self.res[-1].bias.data.zero_()

    def L_of_phi(self, phi):
        c4, s4 = phi[:, 4], phi[:, 3]
        c8 = c4 * c4 - s4 * s4
        s8 = 2.0 * s4 * c4
        L = self.A0 @ self.A0.T + self.eps * torch.eye(self.nc, device=phi.device)
        terms = [(self.C["c4"], c4), (self.S["s4"], s4),
                 (self.C["c8"], c8), (self.S["s8"], s8)]
        for M, u in terms:
            Ms = 0.5 * (M + M.T)
            L = L + Ms * u[:, None, None]
        return L

    def pm_of_phi(self, phi):
        return phi @ self.P.T              # (n, nc)

    def forward(self, ic, phi):
        L = self.L_of_phi(phi)
        lin = 0.5 * torch.einsum("ni,nij,nj->n", ic, L, ic)
        pm = torch.sum(self.pm_of_phi(phi) * ic, dim=1)
        r = self.res(torch.cat([ic, phi], 1)).squeeze(1)
        return lin + pm + r

    def psi_T(self, ic, phi, dphi_dtheta):
        ic = ic.clone().requires_grad_(True)
        phi = phi.clone().requires_grad_(True)
        W = self.forward(ic, phi).sum()
        psi = torch.autograd.grad(W, ic, create_graph=True)[0]
        gphi = torch.autograd.grad(W, phi, create_graph=True)[0]
        return psi, I_SCALE * (gphi * dphi_dtheta).sum(dim=1)


class Trainer:
    """Full-batch AdamW with validation-based early stopping (test never touched)."""

    def __init__(self, model, lr=1e-3, wd=1e-4, max_epochs=8000, eval_every=50,
                 patience=800, clip=1.0, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        self.max_epochs, self.eval_every = max_epochs, eval_every
        self.patience, self.clip = patience, clip
        self.y_scale = None

    def loss(self, batch):
        raise NotImplementedError

    def evaluate(self, batch):
        self.model.eval()
        with torch.no_grad():
            return self.loss(batch).item()

    def fit(self, train, val, verbose=False):
        xt, dphi_t, yt = train
        best_val, best_state, bad = float("inf"), None, 0
        for ep in range(self.max_epochs):
            self.model.train()
            self.opt.zero_grad()
            loss = self.loss((xt, dphi_t, yt))
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.opt.step()
            if ep % self.eval_every == 0 or ep == self.max_epochs - 1:
                vl = self.evaluate(val)
                if vl < best_val - 1e-9:
                    best_val, bad = vl, 0
                    best_state = {k: v.detach().clone() for k, v in
                                  self.model.state_dict().items()}
                else:
                    bad += self.eval_every
                    if bad >= self.patience:
                        break
                if verbose and ep % 500 == 0:
                    print(f"  ep {ep:5d} train {loss.item():.3e} val {vl:.3e}",
                          flush=True)
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return best_val


class DirectTrainer(Trainer):
    """For MLP: input features -> 13 outputs (psi(12), T), standardized. Batch = (x, dphi, y)."""

    def __init__(self, model, **kw):
        super().__init__(model, **kw)

    def loss(self, batch):
        x, dphi, y = batch
        return nn.functional.mse_loss(self.model(x), y)


class GradNetTrainer(Trainer):
    """For GradNet: psi from g[:12], T from g[12:] . dphi/dtheta. Batch = (x, dphi, y)."""

    def __init__(self, model, **kw):
        super().__init__(model, **kw)

    def loss(self, batch):
        x, dphi, y = batch
        psi, T = self.model.psi_T(x, dphi)
        out = torch.cat([psi, T[:, None]], dim=1)
        scale = self.y_scale if self.y_scale is not None else 1.0
        return nn.functional.mse_loss((out - y) / scale, torch.zeros_like(y))


class EnergyTrainer(Trainer):
    """For scalar/structured energy models: psi/T by autograd; needs grad on
    eval. Batch = (x, dphi, y). w_T up-weights the single torque output."""

    w_T = 3.0

    def __init__(self, model, **kw):
        super().__init__(model, **kw)

    def loss(self, batch):
        x, dphi, y = batch
        psi, T = self.model.psi_T(x[:, :12], x[:, 12:], dphi)
        out = torch.cat([psi, T[:, None]], dim=1)
        scale = self.y_scale if self.y_scale is not None else 1.0
        w = torch.ones(13, device=out.device)
        w[12] = self.w_T
        return (w * ((out - y) / scale) ** 2).mean()

    def evaluate(self, batch):
        self.model.eval()
        with torch.enable_grad():
            return self.loss(batch).item()


def jacobian_asymmetry(predict_psi, x, h=1e-2, n_i=12):
    """Central-difference Jacobian d psi / d i_scaled on probe points x (numpy).

    predict_psi: callable (n, 19) -> (n, 12) in scaled output units.
    Returns per-point asymmetry ||J - J^T||_F / ||J||_F and min eig of sym part.
    """
    n = x.shape[0]
    J = np.zeros((n, n_i, n_i), dtype=np.float64)
    for j in range(n_i):
        xp = x.copy(); xp[:, j] += h
        xm = x.copy(); xm[:, j] -= h
        J[:, :, j] = (predict_psi(xp) - predict_psi(xm)) / (2 * h)
    asym = np.linalg.norm(J - J.transpose(0, 2, 1), axis=(1, 2)) / np.maximum(
        np.linalg.norm(J, axis=(1, 2)), 1e-12)
    eigs = np.linalg.eigvalsh(0.5 * (J + J.transpose(0, 2, 1)))
    return asym, eigs.min(axis=1)
