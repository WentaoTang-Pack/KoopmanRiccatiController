"""
edmd.py
=======

Sampled-data EDMD of an input-affine system  dx/dt = f_0(x) + sum_j f_j(x) u_j
into a bilinear lifted model, and control methods for it.

Model (Strasser et al., IEEE TAC 2025; Peitz et al., SIADS 2020):

    dz/dt = A z + B_0 u + sum_{j=1}^m u_j B_j z,      z = Phi_hat(x),  Phi_hat(0) = 0,

where Phi_hat(x) = [x; psi(x) - psi(0)] (the state is always included).

Estimation: for each vector field f_j, snapshot pairs (x_i, phi^dt_{f_j}(x_i))
from :meth:`dynamics.InputAffineSystem.collect_data` give a least-squares
Koopman matrix on [1; z], with Strasser's structure (constant observable; no
constant column for the drift, since f_0(0) = 0).  The generator is
logm(K)/dt.  The data contain the flow of each f_j alone, so B_j is
estimated directly.

Controllers (all simulated in closed loop with the true system):
* EdmdRobustController -- Strasser et al., Theorem 7: LMIs (54)-(55), rational
  feedback  mu(x) = (I - L_w (Lambda^{-1} kron Phi_hat(x)))^{-1} L P^{-1} Phi_hat(x),
  region of attraction {x : Phi_hat(x)' P^{-1} Phi_hat(x) <= 1}.
* EdmdLQR -- LQR on the linear part (A, B_0):  u = -K Phi_hat(x).
* EdmdMPC -- MPC with the bilinear model as the predictor (lifted state, IPOPT).
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import logm, solve_continuous_are


# ---------------------------------------------------------------------------
# Dictionaries
# ---------------------------------------------------------------------------

class Dictionary:
    """Lifting Phi_hat: R^n -> R^N with Phi_hat(x)[:n] = x and Phi_hat(0) = 0.
    Subclasses define _extra(X) (extra observables of the columns of X) and
    _extra_jacobian(x)."""

    def __init__(self, n, names):
        self.n = int(n)
        self.names = [f"x{k + 1}" for k in range(self.n)] + list(names)
        self.N = len(self.names)

    def __call__(self, X):
        """(n,) -> (N,);  (n, M) -> (N, M)."""
        X = np.asarray(X, dtype=float)
        X2 = X.reshape(self.n, -1)
        Z = np.vstack([X2, self._extra(X2)])
        return Z[:, 0] if X.ndim == 1 else Z

    def jacobian(self, x):
        """d Phi_hat / dx at one state, (N, n)."""
        return np.vstack([np.eye(self.n), self._extra_jacobian(np.asarray(x, dtype=float))])

    def __repr__(self):
        return f"{type(self).__name__}(N={self.N}: {', '.join(self.names)})"


class MonomialDictionary(Dictionary):
    """All monomials of total degree 1..degree (degree 1 = the state)."""

    def __init__(self, n, degree):
        exps = []
        for deg in range(2, degree + 1):
            for combo in itertools.combinations_with_replacement(range(n), deg):
                exps.append(np.bincount(combo, minlength=n))
        self.E = np.array(exps, dtype=int).reshape(-1, n)
        names = ["*".join(f"x{k + 1}" + (f"^{e[k]}" if e[k] > 1 else "") for k in range(n) if e[k])
                 for e in self.E]
        super().__init__(n, names)

    def _extra(self, X):
        return np.prod(X[None, :, :] ** self.E[:, :, None], axis=1)

    def _extra_jacobian(self, x):
        J = np.zeros((len(self.E), self.n))
        for k in range(self.n):
            Ek = self.E.copy()
            Ek[:, k] = np.maximum(Ek[:, k] - 1, 0)
            J[:, k] = self.E[:, k] * np.prod(x[None, :] ** Ek, axis=1)
        return J


class CustomDictionary(Dictionary):
    """State plus user observables psi_k(x) (scalar functions), shifted to vanish
    at 0.  Jacobians by central differences."""

    def __init__(self, n, functions, names=None):
        self.functions = list(functions)
        self._offset = np.array([float(f(np.zeros(n))) for f in self.functions])
        super().__init__(n, names or [f"psi{k + 1}" for k in range(len(self.functions))])

    def _extra(self, X):
        return np.array([[f(X[:, i]) for i in range(X.shape[1])] for f in self.functions]) \
            - self._offset[:, None]

    def _extra_jacobian(self, x, h=1e-6):
        return np.array([[(f(x + h * e) - f(x - h * e)) / (2 * h) for e in np.eye(self.n)]
                         for f in self.functions])


# ---------------------------------------------------------------------------
# Bilinear model and its estimation
# ---------------------------------------------------------------------------

@dataclass
class LiftedSimulationResult:
    """t: (N_t,);  z: (N, N_t) lifted state;  x: (n, N_t) = z[:n];  u: (m, N_t)."""
    t: np.ndarray
    z: np.ndarray
    x: np.ndarray
    u: np.ndarray


class BilinearEdmdModel:
    """dz/dt = A z + B_0 u + sum_j u_j B_j z;   A (N, N), B0 (N, m), B (m, N, N)."""

    def __init__(self, A, B0, B, dictionary, dt):
        self.A, self.B0, self.B, self.dictionary, self.dt = A, B0, B, dictionary, dt
        self.N, self.n, self.m = A.shape[0], dictionary.n, B0.shape[1]

    @property
    def B_tilde(self):
        """[B_1, ..., B_m], (N, N m)."""
        return np.hstack(list(self.B))

    def lift(self, x):
        return self.dictionary(x)

    def rhs(self, z, u):
        u = np.atleast_1d(u)
        return self.A @ z + self.B0 @ u + np.einsum("j,jab,b->a", u, self.B, z)

    def vector_field(self, z, j):
        """Lifted counterpart of f_j:  A z (j = 0),  B_0[:, j-1] + B_j z."""
        return self.A @ z if j == 0 else self.B0[:, j - 1] + self.B[j - 1] @ z

    def simulate(self, x0, t_span, u=None, t_eval=None):
        """Simulate from z(0) = Phi_hat(x0); u: None, constant, or u(t, x) with x = z[:n]."""
        if u is None or not callable(u):
            u_c = np.zeros(self.m) if u is None else np.atleast_1d(np.asarray(u, dtype=float))
            u_fun = lambda t, x: u_c
        else:
            u_fun = u
        sol = solve_ivp(lambda t, z: self.rhs(z, u_fun(t, z[:self.n])), t_span, self.lift(x0),
                        t_eval=t_eval, rtol=1e-8, atol=1e-10)
        U = np.column_stack([np.atleast_1d(u_fun(tk, sol.y[:self.n, k])) for k, tk in enumerate(sol.t)])
        return LiftedSimulationResult(sol.t, sol.y, sol.y[:self.n], U)

    def as_system(self, name="lifted bilinear model"):
        """The lifted model as an InputAffineSystem in z (e.g. for MPC)."""
        from dynamics import InputAffineSystem
        A, B0, B = self.A, self.B0, self.B
        fields = [(lambda z, j=j: B0[:, j] + B[j] @ z) for j in range(self.m)]
        return InputAffineSystem(lambda z: A @ z, fields, self.N, self.m, name=name)

    def empirical_error_bound(self, system, X):
        """
        Sampled estimate of c_r in  ||r(x, u)|| <= c_r (||Phi_hat(x)|| + ||u||)  over the
        states X (n, M), from the generator residuals with the true vector fields,
            r_0 = DPhi(x) f_0(x) - A Phi_hat(x),   r_j = DPhi(x) f_j(x) - (B_0[:, j] + B_j Phi_hat(x));
        c_r = max(max ||r_0|| / ||Phi_hat||,  sqrt(m) max ||r_j||).
        """
        X = np.asarray(X, dtype=float)
        Z = self.lift(X)
        R = np.empty((self.m + 1, self.N, X.shape[1]))
        for i in range(X.shape[1]):
            J = self.dictionary.jacobian(X[:, i])
            for j in range(self.m + 1):
                R[j, :, i] = J @ system.vector_field(X[:, i], j) - self.vector_field(Z[:, i], j)
        nz = np.linalg.norm(Z, axis=0)
        rho0 = np.max(np.linalg.norm(R[0][:, nz > 1e-12], axis=0) / nz[nz > 1e-12])
        rho1 = np.max(np.linalg.norm(R[1:], axis=1))
        return {"c_r": max(rho0, np.sqrt(self.m) * rho1), "rho_drift": rho0, "rho_inputs": rho1}

    def __repr__(self):
        return f"BilinearEdmdModel(n={self.n}, N={self.N}, m={self.m}, dt={self.dt:g})"


def bilinear_edmd(data, dictionary, dt_index=1, report=True):
    """
    Bilinear EDMD from :class:`dynamics.VectorFieldData`, with snapshot pairs
    (x_i, x_i(dt)), dt = t[dt_index]:
        drift:   Phi_hat(x+) ~ K_0 Phi_hat(x)            (no constant column)
        f_j:     Phi_hat(x+) ~ k_j + K_j Phi_hat(x)
    Generators L_j = logm([[1, 0], [k_j, K_j]]) / dt  give  A = L_0,  [B_0[:, j] | B_j] = L_j.
    """
    t0 = time.perf_counter()
    X, k = data.X, int(dt_index)
    m, dt, N = X.shape[0] - 1, float(data.t[k] - data.t[0]), dictionary.N
    L_all = []
    for j in range(m + 1):
        Za, Zb = dictionary(X[j, :, 0, :]), dictionary(X[j, :, k, :])
        Psi = Za if j == 0 else np.vstack([np.ones((1, Za.shape[1])), Za])
        Kj = np.linalg.lstsq(Psi.T, Zb.T, rcond=None)[0].T      # Zb ~ Kj Psi
        Kfull = np.eye(N + 1)
        Kfull[1:, (1 if j == 0 else 0):] = Kj
        Lj = np.real(logm(Kfull)) / dt
        Lj[0, :] = 0.0
        L_all.append(Lj)
    A = L_all[0][1:, 1:]
    B0 = np.column_stack([L[1:, 0] for L in L_all[1:]])
    B = np.stack([L[1:, 1:] for L in L_all[1:]])
    if report:
        print(f"[bilinear EDMD] N = {N}, dt = {dt:g}: total computational time = "
              f"{time.perf_counter() - t0:.4f} s")
    return BilinearEdmdModel(A, B0, B, dictionary, dt)


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------

class _StateFeedback:
    name = "EDMD-based controller"

    def simulate_closed_loop(self, system, x0, t_span, t_eval=None, **kwargs):
        """True system under u = mu(x); prints the computational time."""
        res = system.simulate(x0, t_span, u=lambda t, x: self(x), t_eval=t_eval, report=False, **kwargs)
        res.mode = f"closed loop ({self.name})"
        system._report_time(res)
        return res


class EdmdRobustController(_StateFeedback):
    """
    Strasser et al. (IEEE TAC 2025), Theorem 7.  Uncertainty sets:
        bilinearity  Delta = Phi_hat(x) with [Delta; 1]'[[Q_z, S_z], [S_z', R_z]][Delta; 1] >= 0
        (default Q_z = -I, S_z = 0:  ||Phi_hat(x)||^2 <= R_z),
        remainder    ||r|| <= c_r (||Phi_hat(x)|| + ||u||).
    Solves LMIs (54)-(55) with MOSEK, maximizing log det(P) (the RoA volume).
    flexible=False fixes L_w = 0 (linear lifted feedback; Theorem 6 for m = 1).
    eps is the margin of the strict inequalities (must be small: P can be tiny
    in nearly uncontrollable lifted directions).
    """

    name = "EDMD robust (Strasser et al.)"

    def __init__(self, model, c_r, Q_z=None, S_z=None, R_z=1.0, flexible=True, eps=1e-9, report=True):
        import cvxpy as cp
        self.model, self.c_r, N, m = model, float(c_r), model.N, model.m
        self.Q_z = -np.eye(N) if Q_z is None else np.asarray(Q_z, dtype=float)
        self.S_z = np.zeros(N) if S_z is None else np.asarray(S_z, dtype=float).ravel()
        self.R_z = float(R_z)
        A, B0, Bt = model.A, model.B0, model.B_tilde
        Qz, Sz, Rz = self.Q_z, self.S_z.reshape(N, 1), self.R_z
        Minv = np.linalg.inv(np.block([[Qz, Sz], [Sz.T, np.array([[Rz]])]]))   # [[Qt, St], [St', Rt]]
        Qt, St, Rt = Minv[:N, :N], Minv[:N, N:], float(Minv[N, N])
        Im, IN, ImSt = np.eye(m), np.eye(N), np.kron(np.eye(m), St)

        P = cp.Variable((N, N), symmetric=True)
        L = cp.Variable((m, N))
        Lw = cp.Variable((m, N * m)) if flexible else np.zeros((m, N * m))
        Lam = cp.Variable((m, m), symmetric=True)
        nu, tau = cp.Variable(nonneg=True), cp.Variable(nonneg=True)

        M11 = -A @ P - B0 @ L - P @ A.T - L.T @ B0.T - tau * IN               # LMI (54)
        M21 = -L - cp.kron(Lam, St.T) @ Bt.T - ImSt.T @ Lw.T @ B0.T
        M22 = Rt * Lam - Lw @ ImSt - ImSt.T @ Lw.T
        M31 = -cp.vstack([P, L])
        M32 = -cp.vstack([np.zeros((N, m)), Lw @ ImSt])
        M33 = 0.5 * self.c_r ** -2 * tau * np.eye(N + m)
        M41 = cp.kron(Lam, IN) @ Bt.T + Lw.T @ B0.T
        M42 = Lw.T
        M43 = -cp.hstack([np.zeros((N * m, N)), Lw.T])
        M44 = -cp.kron(Lam, np.linalg.inv(Qt))
        M = cp.bmat([[M11, M21.T, M31.T, M41.T], [M21, M22, M32.T, M42.T],
                     [M31, M32, M33, M43.T], [M41, M42, M43, M44]])
        z = np.zeros((N, 1))
        nu11 = cp.reshape(nu, (1, 1), order="C")
        C55 = cp.bmat([[P, P @ Sz, P, z], [Sz.T @ P, Rz * nu11, z.T, nu11],     # LMI (55)
                       [P, z, -nu * np.linalg.inv(Qz), z], [z.T, nu11, z.T, np.ones((1, 1))]])
        eps = float(eps)
        cons = [0.5 * (M + M.T) >> eps * np.eye(M.shape[0]), 0.5 * (C55 + C55.T) >> 0,
                P >> eps * IN, Lam >> eps * Im, nu >= eps, tau >= eps]
        prob = cp.Problem(cp.Maximize(cp.log_det(P)), cons)
        t0 = time.perf_counter()
        try:
            prob.solve(solver="MOSEK")
            self.status = prob.status
        except cp.error.SolverError:
            self.status = "solver_error"           # MOSEK often reports infeasibility this way
        self.solve_time = time.perf_counter() - t0

        self.P = None
        if self.status in ("optimal", "optimal_inaccurate"):
            self.P, self.L, self.Lam = P.value, L.value, Lam.value
            self.L_w = Lw.value if flexible else Lw
            self.P_inv = np.linalg.inv(self.P)
            self.Lam_inv = np.linalg.inv(self.Lam)
            self.K = self.L @ self.P_inv
            self.K_w = self.L_w @ np.kron(self.Lam_inv, IN)
        if report:
            print(f"[{self.name}] LMI synthesis: status = {self.status}, N = {N}, m = {m}, "
                  f"c_r = {self.c_r:.3g}: computational time = {self.solve_time:.3f} s")

    @property
    def feasible(self):
        return self.P is not None

    def __call__(self, x):
        """mu(x) = (I - L_w (Lambda^{-1} kron z))^{-1} K z,  z = Phi_hat(x)."""
        z = self.model.lift(x)
        W = self.L_w @ np.kron(self.Lam_inv, z[:, None])
        return np.linalg.solve(np.eye(self.model.m) - W, self.K @ z)

    def lyapunov(self, X):
        """V(x) = Phi_hat(x)' P^{-1} Phi_hat(x)."""
        Z = self.model.lift(X).reshape(self.model.N, -1)
        v = np.einsum("im,ij,jm->m", Z, self.P_inv, Z)
        return v[0] if np.ndim(X) == 1 else v

    def in_roa(self, X):
        return self.lyapunov(X) <= 1.0

    def in_delta_set(self, X):
        Z = self.model.lift(X).reshape(self.model.N, -1)
        return np.einsum("im,ij,jm->m", Z, self.Q_z, Z) + 2 * self.S_z @ Z + self.R_z >= 0

    def __repr__(self):
        return f"EdmdRobustController(N={self.model.N}, c_r={self.c_r:.3g}, status={self.status!r})"


class EdmdLQR(_StateFeedback):
    """LQR of (A, B_0) with Q_lift = C' Q C (weight on the state part), u = -K Phi_hat(x)."""

    name = "EDMD LQR (linear part)"

    def __init__(self, model, Q, R):
        C = np.hstack([np.eye(model.n), np.zeros((model.n, model.N - model.n))])
        Q, R = np.atleast_2d(Q).astype(float), np.atleast_2d(R).astype(float)
        self.model, self.Q_lift = model, C.T @ Q @ C
        self.P = solve_continuous_are(model.A, model.B0, self.Q_lift, R)
        self.K = np.linalg.solve(R, model.B0.T @ self.P)

    def __call__(self, x):
        return -self.K @ self.model.lift(x)


class EdmdMPC(_StateFeedback):
    """
    MPC with the bilinear model as the predictor: the measured state is lifted,
    z = Phi_hat(x), and :class:`mpc.ModelPredictiveControl` is applied to the
    lifted system with Q_lift = C' Q C, x-bounds on z[:n], and terminal weight
    P (N x N; 'lqr': CARE of (A, B_0), or None).
    """

    name = "EDMD MPC (bilinear surrogate)"

    def __init__(self, model, horizon, dt, Q=None, R=None, P="lqr", u_min=None, u_max=None,
                 x_min=None, x_max=None, n_substeps=4, tol=1e-8, max_iter=200):
        from mpc import ModelPredictiveControl
        n, N, m = model.n, model.N, model.m
        Q = np.eye(n) if Q is None else np.atleast_2d(Q).astype(float)
        R = np.eye(m) if R is None else np.atleast_2d(R).astype(float)
        C = np.hstack([np.eye(n), np.zeros((n, N - n))])
        if isinstance(P, str):
            P = solve_continuous_are(model.A, model.B0, C.T @ Q @ C, R)
        lift_bound = lambda b, fill: None if b is None else np.r_[
            np.broadcast_to(np.asarray(b, dtype=float), (n,)), np.full(N - n, fill)]
        self.model, self.P, self.dt, self.N = model, P, float(dt), int(horizon)
        self.mpc = ModelPredictiveControl(model.as_system(), horizon, dt, Q=C.T @ Q @ C, R=R, P=P,
                                          u_min=u_min, u_max=u_max, x_min=lift_bound(x_min, -np.inf),
                                          x_max=lift_bound(x_max, np.inf), n_substeps=n_substeps,
                                          tol=tol, max_iter=max_iter)

    def solve(self, x):
        """OCP from the measured state x (sol.x holds the predicted lifted states)."""
        return self.mpc.solve(self.model.lift(x))

    def __call__(self, x):
        return self.solve(x).u[:, 0]

    def simulate_closed_loop(self, system, x0, n_steps, n_fine=10, report=True):
        """Sample-and-hold closed loop with the true system (lifting the measured state)."""
        from mpc import MPCSimulationResult
        t_start = time.perf_counter()
        self.mpc.reset()
        x = np.asarray(x0, dtype=float).ravel()
        t_l, x_l, u_l, xs, us, costs, times, succ = [], [], [], [x.copy()], [], [], [], []
        for k in range(int(n_steps)):
            sol = self.solve(x)
            u = sol.u[:, 0]
            t0 = k * self.dt
            r = system.simulate(x, (t0, t0 + self.dt), u=u, t_eval=np.linspace(t0, t0 + self.dt, n_fine + 1))
            sl = slice(None) if k == 0 else slice(1, None)
            t_l.append(r.t[sl]); x_l.append(r.x[:, sl]); u_l.append(r.u[:, sl])
            x = r.x[:, -1].copy()
            xs.append(x.copy()); us.append(u); costs.append(sol.cost)
            times.append(sol.solve_time); succ.append(sol.success)
        res = MPCSimulationResult(np.concatenate(t_l), np.hstack(x_l), np.hstack(u_l),
                                  self.dt * np.arange(int(n_steps) + 1), np.column_stack(xs),
                                  np.column_stack(us), np.array(costs), np.array(times), np.array(succ),
                                  wall_time=time.perf_counter() - t_start)
        if report:
            print(f"[{self.name}: {system.name}] closed loop, {int(n_steps)} steps: total computational "
                  f"time = {res.wall_time:.4f} s (OCP solves {res.solve_times.sum():.4f} s, "
                  f"mean {res.solve_times.mean():.4f} s per step)")
        return res

    def __repr__(self):
        return f"EdmdMPC(N_obs={self.model.N}, horizon={self.N}, dt={self.dt:g})"
