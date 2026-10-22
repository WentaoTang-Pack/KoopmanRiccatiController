"""
mpc.py
======

Nominal nonlinear MPC for an :class:`dynamics.InputAffineSystem` (in its own
coordinates; after ``shift_to_steady_state`` the target is the origin).

At every sampling instant, with zero-order-hold inputs u_0, ..., u_{N-1}:

    min   sum_{k=0}^{N-1} dt (h(x_k)' Q h(x_k) + u_k' R u_k) + x_N' P x_N
    s.t.  x_0 = x(t),  x_{k+1} = Phi_dt(x_k, u_k),
          u_min <= u_k <= u_max,  x_min <= x_k <= x_max  (k = 1..N),

where h(x) = x by default (or a given output map with h(0) = 0), Phi_dt is
one sampling interval of the model (RK4 with substeps), and P is by default
the LQR Riccati matrix of the linearization (state weight C' Q C, C = Dh(0)).  The NLP is
solved by multiple shooting with CasADi + IPOPT (exact derivatives),
warm-started from the shifted previous primal-dual solution.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_continuous_are


@dataclass
class OCPSolution:
    """u: (m, N) inputs;  x: (n, N+1) predicted states;  t: (N+1,) prediction grid."""
    u: np.ndarray
    x: np.ndarray
    t: np.ndarray
    cost: float
    success: bool
    message: str
    nit: int
    solve_time: float

    def __repr__(self):
        return (f"OCPSolution(cost={self.cost:.6g}, success={self.success}, nit={self.nit}, "
                f"solve_time={self.solve_time:.3f}s)")


@dataclass
class MPCSimulationResult:
    """
    t, x, u : plant trajectory on a fine grid;  t_samples, x_samples (n, K+1),
    u_samples (m, K) : sampling instants;  costs, solve_times, success : per step;
    wall_time : total computational time [s].
    """
    t: np.ndarray
    x: np.ndarray
    u: np.ndarray
    t_samples: np.ndarray
    x_samples: np.ndarray
    u_samples: np.ndarray
    costs: np.ndarray
    solve_times: np.ndarray
    success: np.ndarray
    predictions: list = field(default_factory=list, repr=False)
    wall_time: float = 0.0

    def __repr__(self):
        return (f"MPCSimulationResult(steps={self.u_samples.shape[1]}, "
                f"all_success={bool(np.all(self.success))}, "
                f"mean_solve_time={np.mean(self.solve_times):.3f}s, wall_time={self.wall_time:.3f}s)")


class ModelPredictiveControl:
    """
    Parameters
    ----------
    model : InputAffineSystem (prediction model)
    horizon : N;  dt : sampling time
    Q, R : weights (default identity)
    P : terminal weight: 'lqr' (default; CARE of the linearization), None (no
        terminal cost), or an (n, n) array
    u_min, u_max, x_min, x_max : bounds (scalars or vectors; default none)
    n_substeps : RK4 substeps per sampling interval
    tol, max_iter, solver_options : IPOPT settings
    output : optional output map h(x) -> (p,) with h(0) = 0, written with NumPy
        operations (traced by CasADi like the model); Q is then (p, p)
    """

    def __init__(self, model, horizon, dt, Q=None, R=None, P="lqr", u_min=None, u_max=None,
                 x_min=None, x_max=None, n_substeps=4, tol=1e-8, max_iter=200, solver_options=None,
                 output=None):
        import casadi as ca
        from dynamics import _to_casadi
        n, m, N = model.n, model.m, int(horizon)
        self.model, self.n, self.m, self.N, self.dt = model, n, m, N, float(dt)
        if output is None:
            output = lambda x: x
        p = len(np.atleast_1d(output(np.zeros(n))))
        self.output = output
        self.Q = np.eye(p) if Q is None else np.atleast_2d(Q).astype(float)
        self.R = np.eye(m) if R is None else np.atleast_2d(R).astype(float)
        if isinstance(P, str):
            A, B = model.linearize()
            h = 1e-6                                                    # C = Dh(0) by central differences
            C = np.column_stack([(np.atleast_1d(output(h * e)) - np.atleast_1d(output(-h * e))) / (2 * h)
                                 for e in np.eye(n)])
            P = solve_continuous_are(A, B, C.T @ self.Q @ C + 1e-10 * np.eye(n), self.R)
        self.P = np.zeros((n, n)) if P is None else np.atleast_2d(P).astype(float)
        bound = lambda b, dim, fill: np.full(dim, fill) if b is None else np.broadcast_to(
            np.asarray(b, dtype=float), (dim,)).copy()
        self.u_min, self.u_max = bound(u_min, m, -np.inf), bound(u_max, m, np.inf)
        self.x_min, self.x_max = bound(x_min, n, -np.inf), bound(x_max, n, np.inf)

        # one sampling interval of the model: RK4 with substeps
        rhs = model.casadi_rhs()
        xs, us = ca.SX.sym("x", n), ca.SX.sym("u", m)
        h, xk = self.dt / n_substeps, xs
        for _ in range(n_substeps):
            k1 = rhs(xk, us)
            k2 = rhs(xk + 0.5 * h * k1, us)
            k3 = rhs(xk + 0.5 * h * k2, us)
            k4 = rhs(xk + h * k3, us)
            xk = xk + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._step = ca.Function("Phi", [xs, us], [xk])
        x_obj = np.empty(n, dtype=object)
        for i in range(n):
            x_obj[i] = xs[i]
        self._h = ca.Function("h", [xs], [_to_casadi(output(x_obj), (p, 1))])

        # multiple-shooting NLP, w = [vec(X); vec(U)]
        X, U, p = ca.SX.sym("X", n, N + 1), ca.SX.sym("U", m, N), ca.SX.sym("x0", n)
        Qd, Rd, Pd = ca.DM(self.Q), ca.DM(self.R), ca.DM(self.P)
        J, g = 0, [X[:, 0] - p]
        for k in range(N):
            hk = self._h(X[:, k])
            J += self.dt * (ca.bilin(Qd, hk, hk) + ca.bilin(Rd, U[:, k], U[:, k]))
            g.append(X[:, k + 1] - self._step(X[:, k], U[:, k]))
        J += ca.bilin(Pd, X[:, N], X[:, N])
        nlp = {"x": ca.vertcat(ca.vec(X), ca.vec(U)), "f": J, "g": ca.vertcat(*g), "p": p}
        opts = {"print_time": False, "show_eval_warnings": False, "ipopt.print_level": 0,
                "ipopt.sb": "yes", "ipopt.tol": tol, "ipopt.max_iter": max_iter}
        opts.update(solver_options or {})
        self._solver = ca.nlpsol("mpc", "ipopt", nlp, opts)
        opts_ws = dict(opts, **{"ipopt.warm_start_init_point": "yes", "ipopt.mu_init": 1e-6,
                                "ipopt.warm_start_bound_push": 1e-9,
                                "ipopt.warm_start_mult_bound_push": 1e-9})
        self._solver_ws = ca.nlpsol("mpc_ws", "ipopt", nlp, opts_ws)

        lbX, ubX = np.tile(self.x_min[:, None], (1, N + 1)), np.tile(self.x_max[:, None], (1, N + 1))
        lbX[:, 0], ubX[:, 0] = -np.inf, np.inf                    # x_0 is fixed by the constraint
        self._lbw = np.concatenate([lbX.ravel("F"), np.tile(self.u_min, N)])
        self._ubw = np.concatenate([ubX.ravel("F"), np.tile(self.u_max, N)])
        self._prev = None                                          # (w, lam_x, lam_g)

    def reset(self):
        """Forget the warm start."""
        self._prev = None

    def _split(self, w):
        n, m, N = self.n, self.m, self.N
        return w[:n * (N + 1)].reshape((n, N + 1), order="F"), w[n * (N + 1):].reshape((m, N), order="F")

    def _shift(self, w):
        X, U = self._split(w)
        return np.concatenate([np.hstack([X[:, 1:], X[:, -1:]]).ravel("F"),
                               np.hstack([U[:, 1:], U[:, -1:]]).ravel("F")])

    def solve(self, x0):
        """Solve the OCP from x0; returns an OCPSolution."""
        n, m, N = self.n, self.m, self.N
        x0 = np.asarray(x0, dtype=float).ravel()
        args = dict(p=x0, lbx=self._lbw, ubx=self._ubw, lbg=0, ubg=0)
        if self._prev is None:                    # cold start: roll out zero input
            U0 = np.clip(np.zeros((m, N)), self.u_min[:, None], self.u_max[:, None])
            X0 = [x0]
            for k in range(N):
                X0.append(np.asarray(self._step(X0[-1], U0[:, k])).ravel())
            X0 = np.column_stack(X0)
            if not np.all(np.isfinite(X0)):
                X0 = np.tile(x0[:, None], (1, N + 1))
            args["x0"], solver = np.concatenate([X0.ravel("F"), U0.ravel("F")]), self._solver
        else:                                     # warm start: shifted primal-dual solution
            w, lam_x, lam_g = self._prev
            w0 = self._shift(w)
            w0[:n] = x0
            LG = lam_g.reshape((n, N + 1), order="F")
            args.update(x0=w0, lam_x0=self._shift(lam_x),
                        lam_g0=np.hstack([LG[:, 1:], LG[:, -1:]]).ravel("F"))
            solver = self._solver_ws
        t0 = time.perf_counter()
        sol = solver(**args)
        solve_time = time.perf_counter() - t0
        stats = solver.stats()
        if not stats["success"] and self._prev is not None:      # warm start failed: cold restart
            self._prev = None
            out = self.solve(x0)
            out.solve_time += solve_time
            return out
        w = np.asarray(sol["x"]).ravel()
        self._prev = (w, np.asarray(sol["lam_x"]).ravel(), np.asarray(sol["lam_g"]).ravel())
        X, U = self._split(w)
        if not stats["success"]:
            print(f"warning: IPOPT did not converge ({stats['return_status']})")
        return OCPSolution(U, X, self.dt * np.arange(N + 1), float(sol["f"]), bool(stats["success"]),
                           stats["return_status"], int(stats["iter_count"]), solve_time)

    def __call__(self, x):
        """MPC feedback: the first optimal input."""
        return self.solve(x).u[:, 0]

    def simulate_closed_loop(self, x0, n_steps, plant=None, n_fine=10, store_predictions=False,
                             report=True):
        """
        Sample-and-hold closed loop: at t_k = k dt the plant state is measured,
        the OCP is solved, and the first input is held on [t_k, t_k + dt).
        ``plant`` defaults to the model (nominal closed loop).
        """
        plant = self.model if plant is None else plant
        t_start = time.perf_counter()
        self.reset()
        x = np.asarray(x0, dtype=float).ravel()
        t_l, x_l, u_l = [], [], []
        xs, us, costs, times, succ, preds = [x.copy()], [], [], [], [], []
        for k in range(int(n_steps)):
            sol = self.solve(x)
            u = sol.u[:, 0]
            t0 = k * self.dt
            r = plant.simulate(x, (t0, t0 + self.dt), u=u, t_eval=np.linspace(t0, t0 + self.dt, n_fine + 1))
            sl = slice(None) if k == 0 else slice(1, None)
            t_l.append(r.t[sl]); x_l.append(r.x[:, sl]); u_l.append(r.u[:, sl])
            x = r.x[:, -1].copy()
            xs.append(x.copy()); us.append(u); costs.append(sol.cost)
            times.append(sol.solve_time); succ.append(sol.success)
            if store_predictions:
                preds.append(sol)
        res = MPCSimulationResult(np.concatenate(t_l), np.hstack(x_l), np.hstack(u_l),
                                  self.dt * np.arange(int(n_steps) + 1), np.column_stack(xs),
                                  np.column_stack(us), np.array(costs), np.array(times), np.array(succ),
                                  preds, time.perf_counter() - t_start)
        if report:
            print(f"[MPC: {plant.name}] closed loop, {int(n_steps)} steps: total computational time = "
                  f"{res.wall_time:.4f} s (OCP solves {res.solve_times.sum():.4f} s, mean "
                  f"{res.solve_times.mean():.4f} s, max {res.solve_times.max():.4f} s per step)")
        return res

    def __repr__(self):
        return f"ModelPredictiveControl(model={self.model.name!r}, N={self.N}, dt={self.dt:g})"
