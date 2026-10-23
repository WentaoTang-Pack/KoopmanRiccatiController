"""
kedmd.py
========

Kernel EDMD (kEDMD) surrogate of a discrete-time control-affine system and
the corresponding MPC, following

* L. Bold, F. M. Philipp, M. Schaller, K. Worthmann, "Kernel-based Koopman
  approximants for control: flexible sampling, error analysis, and
  stability", SIAM J. Control Optim. 63(6), 2025  -- surrogate (Algorithm 1);
* L. Bold, M. Schaller, I. Schimperna, K. Worthmann, "Kernel EDMD for
  data-driven nonlinear Koopman MPC with stability guarantees",
  IFAC-PapersOnLine, 2025  -- kEDMD-MPC (Algorithm 1).

System
------
    x+ = F(x, u) = g_0(x) + G(x) u,     x in Omega c R^n,  u in U c R^m,

e.g. a continuous-time control-affine system sampled with zero-order hold.

Surrogate (SICON, Algorithm 1)
------------------------------
Data: triplets (x_i, u_i, x_i^+ = F(x_i, u_i)) with *arbitrary* sampling,
and cluster centers  X = {x_1, ..., x_d}  (e.g. a Chebyshev grid with x_1 = 0).

* Step 1 (clustering + local regression).  For each center x_l, take the
  N_c >= m + 1 nearest data points and solve
      H_l = argmin_H || [x^+_{l,1} ... x^+_{l,N_c}] - H U_l ||_F = X_l^+ U_l^+,
      U_l = [1 ... 1; u_{l,1} ... u_{l,N_c}],
  an estimate of [g_0(x_l) | G(x_l)]  (n x (m+1)).
* Step 2 (kernel interpolation).  Each entry is interpolated,
      H_hat_pq(x) = ((H_l)_pq)_l' (K_X + lambda I)^{-1} k_X(x),
  (lambda = 0: interpolation as in the paper), giving
      x+ = F_hat(x, u) = g_hat_0(x) + G_hat(x) u.

The surrogate is an n-dimensional nonlinear system whose right-hand side is
a kernel expansion with d centers,  [g_hat_0 | G_hat](x) = C k_X(x);  the
number of centers d affects its accuracy and evaluation cost, not its state
dimension.

kEDMD-MPC (IFAC, Algorithm 1) -- :class:`KedmdMPC`
--------------------------------------------------
    min_{u(0..N-1) in U}  sum_{i=0}^{N-1}  x(i)' Q x(i) + u(i)' R u(i)
    s.t.  x(0) = x_hat,   x(i+1) = F_hat(x(i), u(i)),
          x(i) in X_box (-) B_{i eps}(0)   (optional, tightened state box),

without terminal cost or terminal constraint; the first input is applied.
Solved by multiple shooting with CasADi + IPOPT; the kernel must provide a
CasADi expression (e.g. :func:`kernel.wendland_kernel`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Grids and data generation
# ---------------------------------------------------------------------------

def chebyshev_grid(n_per_dim, half_width=1.0, dim=2):
    """
    Tensor grid of Chebyshev nodes  a_k cos(pi (2i + 1) / (2 n)),  i = 0..n-1,
    in  prod_k [-a_k, a_k]  (a = half_width, scalar or per dimension).
    For odd n the grid contains the origin.  Returns (dim, n^dim).

    (Bold et al. index i = 1..n, which repeats one node and misses another;
    we use the standard i = 0..n-1.)
    """
    n = int(n_per_dim)
    a = np.broadcast_to(np.asarray(half_width, dtype=float), (dim,))
    nodes = np.cos(np.pi * (2 * np.arange(n) + 1) / (2 * n))
    nodes[np.abs(nodes) < 1e-15] = 0.0
    mesh = np.meshgrid(*[a[k] * nodes for k in range(dim)], indexing="ij")
    return np.vstack([g.ravel() for g in mesh])


@dataclass
class TransitionData:
    """Triplets (X, U, Xp) with  Xp[:, i] = F(X[:, i], U[:, i]);  shapes (n, D), (m, D), (n, D)."""
    X: np.ndarray
    U: np.ndarray
    Xp: np.ndarray
    wall_time: float = 0.0

    @property
    def n_samples(self):
        return self.X.shape[1]


def _apply(F, X, U):
    return np.column_stack([np.asarray(F(X[:, i], U[:, i]), dtype=float).ravel()
                            for i in range(X.shape[1])])


def sample_transitions_iid(F, n_samples, x_half_width, u_min, u_max,
                           n, m, rng=None):
    """i.i.d. uniform (x, u) in  prod[-a, a] x [u_min, u_max];  x+ = F(x, u)."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(rng)
    a = np.broadcast_to(np.asarray(x_half_width, dtype=float), (n,))
    X = rng.uniform(-a[:, None], a[:, None], (n, n_samples))
    U = rng.uniform(np.broadcast_to(u_min, (m,))[:, None], np.broadcast_to(u_max, (m,))[:, None],
                    (m, n_samples))
    return TransitionData(X, U, _apply(F, X, U), time.perf_counter() - t0)


def sample_transitions_clusters(F, centers, n_per_cluster,
                                radius, u_min, u_max, m, rng=None):
    """
    Cluster sampling (Bold et al.): for each center x_l, n_per_cluster states
    drawn uniformly in the ball of the given radius around x_l (radius = 0:
    at x_l itself) with uniformly random inputs in [u_min, u_max].
    Samples are ordered cluster by cluster.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(rng)
    n, d = centers.shape
    Np = int(n_per_cluster)
    dirs = rng.normal(size=(n, d * Np))
    dirs /= np.linalg.norm(dirs, axis=0, keepdims=True)
    rad = radius * rng.uniform(size=d * Np) ** (1.0 / n)            # uniform in the ball
    X = np.repeat(centers, Np, axis=1) + dirs * rad
    U = rng.uniform(np.broadcast_to(u_min, (m,))[:, None], np.broadcast_to(u_max, (m,))[:, None],
                    (m, d * Np))
    return TransitionData(X, U, _apply(F, X, U), time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# kEDMD surrogate
# ---------------------------------------------------------------------------

class KedmdSurrogate:
    """
    kEDMD surrogate  x+ = g_hat_0(x) + G_hat(x) u  (SICON, Algorithm 1).

    Parameters
    ----------
    kernel : MercerKernel
        Interpolation kernel (Wendland in the paper).
    centers : (n, d) array
        Cluster centers X (should contain the origin, the controlled steady state).
    n_neighbors : int
        Cluster size N_c >= m + 1 (nearest data points of each center).
    regularization : float
        lambda >= 0 in (K_X + lambda I)^{-1}  (0: interpolation).
    drift_kernel : MercerKernel, optional
        Separate kernel for g_hat_0 (default: ``kernel``).  With a linear-type
        kernel (e.g. :func:`kernel.linear_wendland_kernel`), g_hat_0(0) = 0 holds
        by construction for any data and regularization; centers at the origin
        are then left out of the drift interpolation (where g_0 = 0 anyway).

    After :meth:`fit`:
    H : (d, n, m+1) local estimates of [g_0(x_l) | G(x_l)];
    coef_drift : (n, d_0) and coef_input : (n m, d) interpolation coefficients, so that
           g_hat_0(x) = coef_drift k_0(x),  vec G_hat(x) = coef_input k_X(x);
    info : diagnostics (condition number of K_X, cluster radii, ranks, time).
    """

    def __init__(self, kernel, centers, n_neighbors, regularization=0.0, drift_kernel=None):
        self.kernel = kernel
        self.drift_kernel = kernel if drift_kernel is None else drift_kernel
        self.centers = np.asarray(centers, dtype=float)
        self.n, self.d = self.centers.shape
        self.n_neighbors = int(n_neighbors)
        self.regularization = float(regularization)
        self.m = self.coef_drift = self.coef_input = None
        # centers used for g_hat_0: all, except the origin for a separate (linear-type) drift kernel
        self._drift_mask = (np.ones(self.d, dtype=bool) if drift_kernel is None
                            else np.linalg.norm(self.centers, axis=0) > 1e-14)
        self.drift_centers = self.centers[:, self._drift_mask]

    def fit(self, data, report=True):
        t0 = time.perf_counter()
        X, U, Xp = data.X, data.U, data.Xp
        self.m = m = U.shape[0]
        Nc = self.n_neighbors

        # Step 1: N_c nearest data points of each center, local least squares
        dist, idx = cKDTree(X.T).query(self.centers.T, k=Nc)
        H = np.empty((self.d, self.n, m + 1))
        for l in range(self.d):
            Ul = np.vstack([np.ones((1, Nc)), U[:, idx[l]]])
            H[l] = Xp[:, idx[l]] @ np.linalg.pinv(Ul)

        # Step 2: kernel interpolation of every entry of H (g_0 and G with their own kernels)
        lam = self.regularization
        K = self.kernel.matrix(self.centers)
        K0 = self.drift_kernel.matrix(self.drift_centers)
        Y0 = H[self._drift_mask, :, 0]                        # (d_0, n), rows = centers
        Y1 = H[:, :, 1:].reshape(self.d, -1)                  # (d, n m)
        self.coef_drift = np.linalg.solve(K0 + lam * np.eye(K0.shape[0]), Y0).T
        self.coef_input = np.linalg.solve(K + lam * np.eye(self.d), Y1).T
        self.H = H
        ev, ev0 = np.linalg.eigvalsh(K), np.linalg.eigvalsh(K0)
        self.info = {"cond_K": float(ev[-1] / ev[0]), "cond_K_drift": float(ev0[-1] / ev0[0]),
                     "cluster_radius_max": float(dist.max()), "wall_time": time.perf_counter() - t0}
        if report:
            print(f"[kEDMD] d = {self.d} centers, N_c = {Nc}, lambda = {self.regularization:g}, "
                  f"{X.shape[1]} data triplets; cond(K_X) = {self.info['cond_K']:.2e}"
                  + (f", cond(K_drift) = {self.info['cond_K_drift']:.2e}" if self.drift_kernel is not self.kernel else "")
                  + f", max cluster radius = {self.info['cluster_radius_max']:.2e}: "
                  f"total computational time = {self.info['wall_time']:.4f} s")
        return self

    # -- evaluation -----------------------------------------------------------

    def _HX(self, X):
        g0 = self.coef_drift @ self.drift_kernel.matrix(self.drift_centers, X)          # (n, M)
        G = (self.coef_input @ self.kernel.matrix(self.centers, X)).reshape(self.n, self.m, -1)
        return np.concatenate([g0[:, None, :], G], axis=1)

    def g0(self, X):
        """g_hat_0 at states X (n,) or (n, M)."""
        X = np.asarray(X, dtype=float)
        out = self._HX(X.reshape(self.n, -1))[:, 0, :]
        return out[:, 0] if X.ndim == 1 else out

    def G(self, x):
        """G_hat at a single state, (n, m)."""
        return self._HX(np.asarray(x, dtype=float).reshape(self.n, 1))[:, 1:, 0]

    def step(self, x, u):
        """x+ = g_hat_0(x) + G_hat(x) u  for a single state and input."""
        Hx = self._HX(np.asarray(x, dtype=float).reshape(self.n, 1))[:, :, 0]
        return Hx[:, 0] + Hx[:, 1:] @ np.atleast_1d(np.asarray(u, dtype=float))

    def step_batch(self, X, U):
        """Vectorized one-step prediction for X (n, M), U (m, M)."""
        HX = self._HX(X)
        return HX[:, 0, :] + np.einsum("nmk,mk->nk", HX[:, 1:, :], U)

    def casadi_function(self):
        """CasADi Function  F(x, u) -> x+  of the surrogate."""
        import casadi as ca
        x = ca.SX.sym("x", self.n)
        u = ca.SX.sym("u", self.m)
        g0 = ca.mtimes(ca.DM(self.coef_drift), self.drift_kernel.casadi_row(x, self.drift_centers).T)
        G = ca.reshape(ca.mtimes(ca.DM(self.coef_input), self.kernel.casadi_row(x, self.centers).T),
                       self.m, self.n).T                      # (n, m)
        xp = g0 + ca.mtimes(G, u)
        return ca.Function("F_hat", [x, u], [xp], ["x", "u"], ["xp"])

    def __repr__(self):
        drift = "" if self.drift_kernel is self.kernel else f", drift_kernel={self.drift_kernel.name!r}"
        return (f"KedmdSurrogate(kernel={self.kernel.name!r}{drift}, d={self.d}, "
                f"N_c={self.n_neighbors}, lambda={self.regularization:g}, fitted={self.coef_input is not None})")


# ---------------------------------------------------------------------------
# kEDMD-MPC
# ---------------------------------------------------------------------------

@dataclass
class KedmdMPCResult:
    """Closed loop of the (discrete-time) plant under KedmdMPC.

    k : (K+1,) time steps;  x : (n, K+1) states;  u : (m, K) applied inputs;
    costs : (K,) optimal OCP values;  solve_times : (K,);  success : (K,);
    wall_time : total computational time [s].
    """
    k: np.ndarray
    x: np.ndarray
    u: np.ndarray
    costs: np.ndarray
    solve_times: np.ndarray
    success: np.ndarray
    wall_time: float = 0.0

    def __repr__(self):
        return (f"KedmdMPCResult(steps={self.u.shape[1]}, all_success={bool(np.all(self.success))}, "
                f"mean_solve_time={np.mean(self.solve_times):.4f}s, wall_time={self.wall_time:.3f}s)")


class KedmdMPC:
    """
    kEDMD-MPC (Bold et al.): MPC without terminal ingredients, with the kEDMD
    surrogate as the (discrete-time) prediction model.

    Parameters
    ----------
    surrogate : KedmdSurrogate (fitted; its kernel must have a CasADi expression)
    horizon : int, N
    Q : (n, n), R : (m, m) stage-cost weights
    u_min, u_max : input bounds (scalars or (m,))
    x_min, x_max : optional state box X; with ``tightening = eps`` the
        constraint on x(i) is shrunk to  [x_min + i eps, x_max - i eps]
        (Pontryagin difference with the ball B_{i eps}).
    tightening : eps >= 0 (default 0)
    tol, max_iter : IPOPT settings.  The objective is scaled
        by 1 / (x_hat' Q x_hat) at each solve, so that IPOPT's tolerance acts
        relative to the current state (the closed loop converges to tiny states).
    """

    name = "kEDMD-MPC (Bold et al.)"

    def __init__(self, surrogate, horizon, Q=None, R=None, u_min=None, u_max=None,
                 x_min=None, x_max=None, tightening=0.0, tol=1e-10, max_iter=500):
        import casadi as ca
        self.sur = surrogate
        n, m, N = surrogate.n, surrogate.m, int(horizon)
        self.n, self.m, self.N = n, m, N
        self.Q = np.eye(n) if Q is None else np.atleast_2d(np.asarray(Q, dtype=float))
        self.R = np.eye(m) if R is None else np.atleast_2d(np.asarray(R, dtype=float))
        bc = lambda b, dim, fill: np.full(dim, fill) if b is None else np.broadcast_to(
            np.asarray(b, dtype=float), (dim,)).copy()
        self.u_min, self.u_max = bc(u_min, m, -np.inf), bc(u_max, m, np.inf)
        self.x_min, self.x_max = bc(x_min, n, -np.inf), bc(x_max, n, np.inf)
        self.tightening = float(tightening)

        F = surrogate.casadi_function()
        Xv = ca.MX.sym("X", n, N + 1)
        Uv = ca.MX.sym("U", m, N)
        p = ca.MX.sym("p", n + 1)                         # [x_hat; objective scale]
        Qd, Rd = ca.DM(self.Q), ca.DM(self.R)
        J = 0
        g = [Xv[:, 0] - p[:n]]
        for i in range(N):
            J += ca.bilin(Qd, Xv[:, i], Xv[:, i]) + ca.bilin(Rd, Uv[:, i], Uv[:, i])
            g.append(Xv[:, i + 1] - F(Xv[:, i], Uv[:, i]))
        nlp = {"x": ca.vertcat(ca.vec(Xv), ca.vec(Uv)), "f": p[n] * J,
               "g": ca.vertcat(*g), "p": p}
        opts = {"print_time": False, "show_eval_warnings": False,
                "ipopt.print_level": 0, "ipopt.sb": "yes",
                "ipopt.tol": tol, "ipopt.max_iter": max_iter}
        self._solver = ca.nlpsol("kedmd_mpc", "ipopt", nlp, opts)
        self._F = F
        self._cost = ca.Function("J", [Xv, Uv], [J])

        lbX = np.empty((n, N + 1)); ubX = np.empty((n, N + 1))
        for i in range(N + 1):
            shrink = 0.0 if i == 0 else i * self.tightening
            lbX[:, i] = self.x_min + shrink if i > 0 else -np.inf
            ubX[:, i] = self.x_max - shrink if i > 0 else np.inf
        lbU = np.tile(self.u_min[:, None], (1, N)); ubU = np.tile(self.u_max[:, None], (1, N))
        self._lbw = np.concatenate([lbX.ravel(order="F"), lbU.ravel(order="F")])
        self._ubw = np.concatenate([ubX.ravel(order="F"), ubU.ravel(order="F")])
        self._ng = n * (N + 1)
        self._prev = None

    def reset(self):
        self._prev = None

    def _rollout(self, x0, U):
        X = np.empty((self.n, self.N + 1)); X[:, 0] = x0
        for i in range(self.N):
            X[:, i + 1] = np.asarray(self._F(X[:, i], U[:, i])).ravel()
        return X

    def solve(self, x_hat):
        """Solve the OCP at x_hat; returns an :class:`mpc.OCPSolution`."""
        from mpc import OCPSolution
        n, m, N = self.n, self.m, self.N
        x_hat = np.asarray(x_hat, dtype=float).ravel()
        if self._prev is None:
            U0 = np.clip(np.zeros((m, N)), self.u_min[:, None], self.u_max[:, None])
        else:
            U0 = np.hstack([self._prev[:, 1:], self._prev[:, -1:]])
        X0 = self._rollout(x_hat, U0)
        if not np.all(np.isfinite(X0)):
            X0 = np.tile(x_hat[:, None], (1, N + 1))
        w0 = np.concatenate([X0.ravel(order="F"), U0.ravel(order="F")])
        scale = 1.0 / max(float(x_hat @ self.Q @ x_hat), 1e-300)
        t0 = time.perf_counter()
        sol = self._solver(x0=w0, p=np.r_[x_hat, scale], lbx=self._lbw, ubx=self._ubw,
                           lbg=np.zeros(self._ng), ubg=np.zeros(self._ng))
        st = time.perf_counter() - t0
        stats = self._solver.stats()
        w = np.asarray(sol["x"]).ravel()
        X = w[:n * (N + 1)].reshape((n, N + 1), order="F")
        U = w[n * (N + 1):].reshape((m, N), order="F")
        self._prev = U
        ok = bool(stats.get("success", False))
        if not ok:
            print(f"warning: IPOPT did not converge ({stats.get('return_status')})")
        return OCPSolution(u=U, x=X, t=np.arange(N + 1, dtype=float), cost=float(self._cost(X, U)),
                           success=ok, message=str(stats.get("return_status")),
                           nit=int(stats.get("iter_count", -1)), solve_time=st)

    def __call__(self, x):
        return self.solve(x).u[:, 0]

    def simulate_closed_loop(self, plant_step, x0, n_steps, report=True, label=""):
        """
        Closed loop with the (true) discrete-time plant  x(k+1) = plant_step(x(k), u(k)),
        u(k) = first optimal input of the OCP at x(k).  Reports the total time.
        """
        t0 = time.perf_counter()
        self.reset()
        x = np.asarray(x0, dtype=float).ravel()
        xs, us, costs, times, succ = [x.copy()], [], [], [], []
        for _ in range(int(n_steps)):
            sol = self.solve(x)
            u = sol.u[:, 0]
            x = np.asarray(plant_step(x, u), dtype=float).ravel()
            xs.append(x.copy()); us.append(u); costs.append(sol.cost)
            times.append(sol.solve_time); succ.append(sol.success)
        res = KedmdMPCResult(np.arange(int(n_steps) + 1), np.column_stack(xs), np.column_stack(us),
                               np.array(costs), np.array(times), np.array(succ),
                             time.perf_counter() - t0)
        if report:
            print(f"[{self.name}{', ' + label if label else ''}] closed loop, {int(n_steps)} steps: "
                  f"total computational time = {res.wall_time:.3f} s (OCP solves "
                  f"{res.solve_times.sum():.3f} s, mean {1e3 * res.solve_times.mean():.1f} ms)")
        return res

    def __repr__(self):
        return (f"KedmdMPC(d={self.sur.d}, N={self.N}, kernel={self.sur.kernel.name!r}, "
                f"tightening={self.tightening:g})")
