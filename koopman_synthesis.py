"""
koopman_synthesis.py
====================

Data-driven Koopman controller synthesis on the linear-Sobolev RKHS
(W. Tang, "Koopman-Lyapunov and Koopman-Riccati Controller Synthesis").

Data: samples x_1..x_n (0 excluded) and the flows of f_0, f_1, ..., f_m from
each of them (:meth:`dynamics.InputAffineSystem.collect_data`).

Matrices (eqs. 19-20), with the finite-time Yosida approximation on [0, theta]:
    G_ii'   = k(x_i, x_i')
    A^j_ii' = lam^2 int_0^theta e^{-lam t} k(x_i, S^t_{f_j}(x_i')) dt - lam k(x_i, x_i'),  j = 0..m
    Q_ii'   = <phi(x_i)|Q|phi(x_i')>   (e.g. x_i . x_i' for q(x) = |x|^2)

Koopman-Riccati (Section 4.4, Kleinman iteration), with Y = G P G,
A = G^{-1} A^0, B_j = G^{-1} A^j and A_u = A + sum_j B_j Diag(u_j):
    policy evaluation:   Y A_u + A_u' Y + Q + sum_j u_j u_j' = 0
    policy improvement:  u_j = -diag(G P A^j) = -diag(Y B_j)
The value function is v(x) = g(x)' P g(x), g(x) = [k(x, x_i)]_i; the policy is
known at the samples and interpolated elsewhere (Remark 3.18).

Koopman-Lyapunov (Section 3.4): an SDP in X = (G P G)^{-1}, with a box and a
reference (LQR) bound on X, followed by the policy recovery u_j = -diag(Y B_j)
safeguarded by the diagonal descent condition (21); see :func:`koopman_lyapunov`.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.integrate import trapezoid
from scipy.linalg import solve_continuous_lyapunov


def yosida_matrices(data, kernel, lam):
    """G and [A^0, A^1, ..., A^m] from the flows in ``data`` (trapezoidal rule in time)."""
    X0, t = data.X0, data.t
    G = kernel.matrix(X0)
    w = lam ** 2 * np.exp(-lam * t)
    As = []
    for j in range(data.n_fields):
        K = np.stack([kernel.matrix(X0, data.X[j, :, l, :]) for l in range(len(t))])   # (N_t, n, n)
        As.append(trapezoid(w[:, None, None] * K, t, axis=0) - lam * G)
    return G, As


def koopman_riccati(G, As, Qm, u0, tol=1e-9, max_iter=100, mu=0.0):
    """
    Kleinman iteration for the sample-based operator Riccati equation (35a)-(35b).

    G, As = [A^0, A^1, ..., A^m] : data matrices;  Qm : (n, n) cost matrix;
    u0 : (m, n) initial policy values at the samples (A_u0 must be Hurwitz);
    mu : regularization, G + mu I is used wherever G is inverted.

    The iteration stops when the policy changes by less than ``tol``, or when
    the closed-loop data matrix A_u is no longer Hurwitz; then the last
    Hurwitz iterate is returned (with finite data, A_u can acquire spurious
    unstable modes even though the policy values themselves are accurate).
    Returns P (n, n), u (m, n) and a dict with the iteration history.
    """
    t0 = time.perf_counter()
    G = G + mu * np.eye(len(G))
    A = np.linalg.solve(G, As[0])
    Bs = [np.linalg.solve(G, Aj) for Aj in As[1:]]
    u = np.atleast_2d(np.asarray(u0, dtype=float))
    history, stop = [], "max_iter"
    for it in range(max_iter):
        Au = A + sum(B * uj[None, :] for B, uj in zip(Bs, u))          # A + sum_j B_j Diag(u_j)
        if np.max(np.linalg.eigvals(Au).real) >= 0:                      # u no longer stabilizing
            stop = "A_u not Hurwitz"
            break
        Y = solve_continuous_lyapunov(Au.T, -(Qm + u.T @ u))           # policy evaluation of u
        P, u_eval = np.linalg.solve(G, np.linalg.solve(G, Y).T).T, u    # P = G^{-1} Y G^{-1}
        u = np.stack([-np.diag(Y @ B) for B in Bs])                     # policy improvement
        change = np.max(np.abs(u - u_eval))
        history.append((change, np.max(np.linalg.eigvals(Au).real)))
        if change < tol:
            stop = "converged"
            break
    if stop == "A_u not Hurwitz":
        if not history:
            raise ValueError("A_u of the initial policy u0 is not Hurwitz; Kleinman cannot start.")
        u = u_eval                                                       # last stabilizing policy and its P
    info = {"iterations": len(history), "stop": stop, "history": np.array(history),
            "wall_time": time.perf_counter() - t0}
    return P, u, info


def koopman_lyapunov(G, As, L, L0, mu=0.0, eps=1e-4, alpha_box=300.0, alpha_ref=1.0, beta=0.05):
    """
    Koopman-Lyapunov synthesis (Section 3.4) with a reference bound on the Lyapunov function.

    G, As = [A^0, A^1, ..., A^m] : data matrices;  L : (n, p) with Q = L L';
    L0 : (n, r) with G P0 G = L0 L0' for a reference P0 (e.g. the LQR value of the linearization);
    mu : regularization, G + mu I is used wherever G is inverted.

    SDP (A = G^{-1} A^0, B_j = G^{-1} A^j, X = (G P G)^{-1}, U^j = F^j X):
        min -log det X   s.t.   [ -(A X + X A' + sum_j (B_j U^j + U^j' B_j'))   X L ]
                                [ L' X                                          I   ]  >= 0,
                                eps I <= X <= alpha_box I,     L0' X L0 <= alpha_ref I,
    the last constraint being equivalent to  G P G >= G P0 G / alpha_ref  (v >= v_0 / alpha_ref at the samples).

    Policy recovery: u_j = -diag(Y B_j) with Y = G P G (i.e. u_j(x_i) = -(1/2) L_{f_j} v(x_i) on the data).
    At the samples where the diagonal descent condition (21),
        c_i + 2 sum_j a_ji u_ji <= 0,   a_ji = (Y B_j)_ii,   c_i = 2 (Y A)_ii + Q_ii,
    fails and the input acts on v (||a_i|| >= beta max_k ||a_k||), u_i is projected onto it.
    Returns P (n, n), u (m, n) and a dict with the solver status and the recovery statistics.
    """
    import cvxpy as cp
    t0 = time.perf_counter()
    n = len(G)
    Gm = G + mu * np.eye(n)
    A = np.linalg.solve(Gm, As[0])
    Bs = [np.linalg.solve(Gm, Aj) for Aj in As[1:]]
    X = cp.Variable((n, n), symmetric=True)
    Us = [cp.Variable((n, n)) for _ in Bs]
    M = A @ X + X @ A.T + sum(B @ U + U.T @ B.T for B, U in zip(Bs, Us))
    LMI = cp.bmat([[-M, X @ L], [L.T @ X, np.eye(L.shape[1])]])
    cons = [0.5 * (LMI + LMI.T) >> 0, X >> eps * np.eye(n), X << alpha_box * np.eye(n),
            L0.T @ X @ L0 << alpha_ref * np.eye(L0.shape[1])]
    prob = cp.Problem(cp.Minimize(-cp.log_det(X)), cons)
    prob.solve(solver="MOSEK")
    sdp_time = time.perf_counter() - t0
    Y = np.linalg.inv(0.5 * (X.value + X.value.T))
    P = np.linalg.solve(Gm, np.linalg.solve(Gm, Y).T).T                    # P = G^{-1} Y G^{-1}
    a = np.stack([np.diag(Y @ B) for B in Bs])                              # (m, n)
    c = 2 * np.diag(Y @ A) + np.sum(L ** 2, axis=1)
    u = -a.copy()                                                           # u_j = -diag(Y B_j)
    viol = c + 2 * np.sum(a * u, axis=0)
    norm_a = np.sum(a ** 2, axis=0)
    k = (viol > 0) & (np.sqrt(norm_a) >= beta * np.sqrt(norm_a).max())
    u[:, k] -= viol[k] / (2 * norm_a[k]) * a[:, k]                          # projection onto (21)
    info = {"status": prob.status, "sdp_time": sdp_time, "wall_time": time.perf_counter() - t0,
            "n_safeguarded": int(k.sum()), "n_violated": int(np.sum(c + 2 * np.sum(a * u, axis=0) > 1e-12)),
            "a": a, "c": c}
    return P, u, info


class KoopmanController:
    """
    v(x) = g(x)' P g(x) and the policy pi_j(x), interpolated from its sample
    values u_j by kernel ridge regression with ``interp_kernel`` (Remark 3.18).
    """

    name = "Koopman-Riccati"

    def __init__(self, kernel, X0, P, u, interp_kernel, reg=1e-8):
        self.kernel, self.X0, self.P = kernel, X0, P
        self.policy_model = interp_kernel.fit(X0, u, reg)

    def value(self, X):
        """v at the columns of X (d, M)."""
        g = self.kernel.matrix(self.X0, X)                               # (n, M)
        return np.einsum("im,ij,jm->m", g, self.P, g)

    def __call__(self, x):
        return np.atleast_1d(self.policy_model.predict(np.asarray(x, dtype=float)[:, None])[..., 0])
