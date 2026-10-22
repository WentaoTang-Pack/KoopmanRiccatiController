"""
dynamics.py
===========

Nonlinear input-affine continuous-time systems

    dx/dt = f_0(x) + sum_{j=1}^m f_j(x) u_j = f_0(x) + G(x) u .

* Simulation of the full system, of the drift f_0 alone, and of each input
  field f_j alone (unit input on channel j, no drift).
* Steady state  f_0(x_nom) + G(x_nom) u_nom = 0  and a *new* system in shifted
  and scaled coordinates  x = x_nom + D_x z,  u = u_nom + D_u v,

      dz/dt = D_x^{-1} [f_0(x) + G(x) u_nom] + D_x^{-1} G(x) D_u v,

  whose origin is a steady state under zero input.
* Linearization at the origin, LQR, closed-loop simulation.
* Data collection: initial states sampled in a box, and the flows of every
  vector field f_0, f_1, ..., f_m from each of them.
* A CasADi version of the right-hand side (for MPC with IPOPT).

Conventions: vector fields are indexed j = 0 (drift), 1..m (inputs), so f_j
is column j-1 of G.  Data matrices hold one sample per column.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
from scipy.optimize import root


@dataclass
class SimulationResult:
    """t: (N,);  x: (n, N);  u: (m, N) or None;  wall_time: computational time [s]."""
    t: np.ndarray
    x: np.ndarray
    u: Optional[np.ndarray]
    success: bool
    mode: str
    wall_time: float = 0.0

    def __repr__(self):
        return (f"SimulationResult(mode={self.mode!r}, t=[{self.t[0]:.4g}, {self.t[-1]:.4g}], "
                f"steps={len(self.t)}, success={self.success}, wall_time={self.wall_time:.4f}s)")


@dataclass
class VectorFieldData:
    """
    Flows of every vector field from sampled initial states (see
    :meth:`InputAffineSystem.collect_data`).

    X0 : (n, N_s) initial states;  t : (N_t,) time grid;
    X  : (m+1, n, N_t, N_s), X[j, :, l, i] = flow of f_j from X0[:, i] at t[l];
    F0 : (m+1, n, N_s),      F0[j, :, i] = f_j(X0[:, i]).
    """
    X0: np.ndarray
    t: np.ndarray
    X: np.ndarray
    F0: np.ndarray
    half_side_lengths: np.ndarray
    tau: float
    wall_time: float = 0.0
    center: np.ndarray = 0.0

    @property
    def n_samples(self):
        return self.X0.shape[1]

    @property
    def n_fields(self):
        return self.X.shape[0]

    def field(self, j):
        """Trajectories of f_j, (n, N_t, N_s)."""
        return self.X[j]

    def trajectory(self, j, i):
        """Trajectory of f_j from sample i, (n, N_t)."""
        return self.X[j, :, :, i]

    def __repr__(self):
        return (f"VectorFieldData(n_samples={self.n_samples}, n_fields={self.n_fields}, "
                f"half_side_lengths={self.half_side_lengths}, tau={self.tau:g}, N_t={len(self.t)}, "
                f"wall_time={self.wall_time:.3f}s)")


def _to_casadi(out, shape):
    """Turn the output of a traced map (NumPy array of CasADi/number entries) into SX."""
    import casadi as ca
    a = np.asarray(out, dtype=object).reshape(shape)
    return ca.vertcat(*[ca.horzcat(*[ca.SX(e) for e in row]) for row in a])


class InputAffineSystem:
    """
    dx/dt = f_0(x) + G(x) u.

    Parameters
    ----------
    f0 : callable, x (n,) -> (n,)
    g : list [f_1, ..., f_m] of callables x -> (n,), or one callable x -> (n, m)
    n_states, n_inputs : int
    name : str
    x_test : point at which f0 and g are evaluated once to check shapes
        (default: the origin).

    A system created by :meth:`transform` / :meth:`shift_to_steady_state`
    stores how its coordinates relate to those of the user-defined system:
    x = x_offset + x_scale * z,  u = u_offset + u_scale * v  (elementwise).
    """

    def __init__(self, f0, g, n_states, n_inputs, name="InputAffineSystem", x_test=None):
        self.n, self.m = int(n_states), int(n_inputs)
        self.name = name
        self._f0 = f0
        self._g = g                                   # raw user maps (used for tracing)
        if callable(g):
            self._G = lambda x: np.asarray(g(x), dtype=float).reshape(self.n, self.m)
        else:
            self._G = lambda x: np.column_stack([np.asarray(fj(x), dtype=float).ravel() for fj in g])
        self.x_offset, self.x_scale = np.zeros(self.n), np.ones(self.n)
        self.u_offset, self.u_scale = np.zeros(self.m), np.ones(self.m)
        self.x_nom = self.u_nom = None
        self.K = self.P = None
        x_test = np.zeros(self.n) if x_test is None else np.asarray(x_test, dtype=float)
        assert self.f0(x_test).shape == (self.n,) and self.G(x_test).shape == (self.n, self.m)

    @property
    def n_states(self):
        return self.n

    @property
    def n_inputs(self):
        return self.m

    # -- evaluation ------------------------------------------------------------

    def f0(self, x):
        return np.asarray(self._f0(np.asarray(x, dtype=float)), dtype=float).ravel()

    def G(self, x):
        return self._G(np.asarray(x, dtype=float))

    def vector_field(self, x, j):
        """f_0 (j = 0) or f_j (j = 1..m)."""
        return self.f0(x) if j == 0 else self.G(x)[:, j - 1]

    def rhs(self, x, u):
        return self.f0(x) + self.G(x) @ np.atleast_1d(np.asarray(u, dtype=float))

    # -- simulation ------------------------------------------------------------

    def simulate(self, x0, t_span, u=None, t_eval=None, solver="Radau", rtol=1e-6, atol=1e-8,
                 report=None, **kwargs):
        """
        Simulate dx/dt = f_0(x) + G(x) u with u = None (zero), a constant array,
        or a callable u(t, x).  The computational time is printed when u is a
        callable (closed loop), unless ``report`` says otherwise.
        """
        t0 = time.perf_counter()
        if u is None:
            u_fun = lambda t, x: np.zeros(self.m)
        elif callable(u):
            u_fun = u
        else:
            u_c = np.atleast_1d(np.asarray(u, dtype=float))
            u_fun = lambda t, x: u_c
        sol = solve_ivp(lambda t, x: self.rhs(x, u_fun(t, x)), t_span, np.asarray(x0, dtype=float),
                        method=solver, t_eval=t_eval, rtol=rtol, atol=atol, **kwargs)
        U = np.column_stack([np.atleast_1d(u_fun(tk, sol.y[:, k])) for k, tk in enumerate(sol.t)])
        mode = "feedback / time-varying input" if callable(u) else "open loop"
        res = SimulationResult(sol.t, sol.y, U, sol.success, mode, time.perf_counter() - t0)
        if report or (report is None and callable(u)):
            self._report_time(res)
        return res

    def simulate_field(self, j, x0, t_span, t_eval=None, solver="Radau", rtol=1e-6, atol=1e-8):
        """Flow of a single vector field, dx/dt = f_j(x)  (j = 0: drift)."""
        t0 = time.perf_counter()
        sol = solve_ivp(lambda t, x: self.vector_field(x, j), t_span, np.asarray(x0, dtype=float),
                        method=solver, t_eval=t_eval, rtol=rtol, atol=atol)
        mode = "drift f_0" if j == 0 else f"input field f_{j}"
        return SimulationResult(sol.t, sol.y, None, sol.success, mode, time.perf_counter() - t0)

    def simulate_drift(self, x0, t_span, **kwargs):
        return self.simulate_field(0, x0, t_span, **kwargs)

    def simulate_input_fields(self, x0, t_span, **kwargs):
        """[flow of f_1, ..., flow of f_m] from x0."""
        return [self.simulate_field(j, x0, t_span, **kwargs) for j in range(1, self.m + 1)]

    def _report_time(self, res):
        print(f"[{self.name}] {res.mode} simulation on t = [{res.t[0]:.4g}, {res.t[-1]:.4g}]: "
              f"total computational time = {res.wall_time:.4f} s")

    # -- data collection -------------------------------------------------------

    def sample_hypercube(self, n_samples, half_side_lengths, rng=None, center=0.0):
        """Uniform samples in prod_k [c_k - L_k, c_k + L_k]; returns (n, n_samples)."""
        L = np.broadcast_to(np.asarray(half_side_lengths, dtype=float), (self.n,))
        c = np.broadcast_to(np.asarray(center, dtype=float), (self.n,))
        rng = np.random.default_rng(rng)
        return (c[None, :] + L[None, :] * rng.uniform(-1, 1, (int(n_samples), self.n))).T

    def collect_data(self, n_samples, half_side_lengths, tau, n_time=101, rng=None,
                     solver="Radau", rtol=1e-6, atol=1e-8, center=0.0, X0=None):
        """
        Sample initial states uniformly in prod_k [c_k - L_k, c_k + L_k] (or use
        the given initial states X0, (n, N_s)) and simulate the flow of every
        vector field f_0, f_1, ..., f_m from each of them on
        np.linspace(0, tau, n_time).  Prints the computational time.
        """
        t0 = time.perf_counter()
        if X0 is None:
            X0 = self.sample_hypercube(n_samples, half_side_lengths, rng, center)
        X0 = np.asarray(X0, dtype=float)
        t = np.linspace(0.0, tau, n_time)
        X = np.full((self.m + 1, self.n, n_time, X0.shape[1]), np.nan)
        F0 = np.empty((self.m + 1, self.n, X0.shape[1]))
        for i in range(X0.shape[1]):
            for j in range(self.m + 1):
                F0[j, :, i] = self.vector_field(X0[:, i], j)
                r = self.simulate_field(j, X0[:, i], (0.0, tau), t_eval=t, solver=solver, rtol=rtol, atol=atol)
                X[j, :, :r.x.shape[1], i] = r.x
        data = VectorFieldData(X0, t, X, F0, np.broadcast_to(np.asarray(half_side_lengths, float), (self.n,)),
                               float(tau), time.perf_counter() - t0,
                               np.broadcast_to(np.asarray(center, float), (self.n,)))
        print(f"[{self.name}] data collection: {X0.shape[1]} samples x {self.m + 1} vector fields: "
              f"total computational time = {data.wall_time:.4f} s")
        return data

    # -- steady state and coordinates ------------------------------------------

    def find_steady_state(self, u_nom, x_guess=None, tol=1e-8):
        """Solve f_0(x) + G(x) u_nom = 0 (the guess selects among several steady states)."""
        u_nom = np.atleast_1d(np.asarray(u_nom, dtype=float))
        x_guess = np.zeros(self.n) if x_guess is None else np.asarray(x_guess, dtype=float)
        sol = root(lambda x: self.rhs(x, u_nom), x_guess, tol=1e-12)
        res = np.max(np.abs(self.rhs(sol.x, u_nom)))
        if res > tol * (1.0 + np.max(np.abs(self.rhs(x_guess, u_nom)))):
            raise RuntimeError(f"Steady-state search failed (residual {res:.2e}); try another x_guess.")
        return sol.x

    def transform(self, x_offset, x_scale, u_offset, u_scale, name=None):
        """
        New system in the coordinates  x = x_offset + x_scale * z,  u = u_offset + u_scale * v:

            f~_0(z) = [f_0(x) + G(x) u_offset] / x_scale,   f~_j(z) = u_scale_j f_j(x) / x_scale.

        The new maps are built on the raw maps of ``self`` so that they can
        still be traced by CasADi.
        """
        n, m = self.n, self.m
        ox, sx = [np.broadcast_to(np.asarray(a, dtype=float), (n,)).copy() for a in (x_offset, x_scale)]
        ou, su = [np.broadcast_to(np.asarray(a, dtype=float), (m,)).copy() for a in (u_offset, u_scale)]
        f0p, gp = self._f0, self._g
        to_x = lambda z: ox + sx * np.asarray(z)
        if callable(gp):
            G_raw = lambda x: np.asarray(gp(x)).reshape(n, m)
        else:
            G_raw = lambda x: np.column_stack([np.asarray(fj(x)).reshape(n) for fj in gp])

        def f0_new(z):
            x = to_x(z)
            return (np.asarray(f0p(x)).reshape(n) + G_raw(x) @ ou) / sx

        g_new = [(lambda z, j=j: su[j] * G_raw(to_x(z))[:, j] / sx) for j in range(m)]
        new = InputAffineSystem(f0_new, g_new, n, m, name=name or f"{self.name} (transformed)")
        new.x_offset, new.x_scale = self.x_offset + self.x_scale * ox, self.x_scale * sx
        new.u_offset, new.u_scale = self.u_offset + self.u_scale * ou, self.u_scale * su
        return new

    def shift_to_steady_state(self, u_nom, x_guess=None, scaling=None, name=None):
        """
        New system whose origin is the steady state x_nom under u_nom, in the
        coordinates  z = (x - x_nom) / D_x,  v = (u - u_nom) / D_u, with

            scaling = None        -> D_x = 1, D_u = 1,
            scaling = 'nominal'   -> D_x = |x_nom|, D_u = |u_nom| (zero entries left unscaled),
            scaling = (dx, du)    -> user-given scales.
        """
        u_nom = np.atleast_1d(np.asarray(u_nom, dtype=float))
        x_nom = self.find_steady_state(u_nom, x_guess)
        if scaling is None:
            dx, du = np.ones(self.n), np.ones(self.m)
        elif isinstance(scaling, str) and scaling == "nominal":
            dx, du = np.abs(x_nom), np.abs(u_nom)
            for label, d in (("state", dx), ("input", du)):
                if np.any(d < 1e-12):
                    print(f"[{self.name}] warning: zero nominal {label} component(s) "
                          f"{np.flatnonzero(d < 1e-12).tolist()} left unscaled.")
                    d[d < 1e-12] = 1.0
        else:
            dx, du = scaling
        new = self.transform(x_nom, dx, u_nom, du, name=name or f"{self.name} (shifted)")
        new.x_nom, new.u_nom = x_nom, u_nom
        return new

    def to_original_state(self, z):
        """x = x_offset + x_scale * z  (z: (n,) or (n, N))."""
        z = np.asarray(z, dtype=float)
        return (self.x_offset + self.x_scale * z.T).T

    def from_original_state(self, x):
        x = np.asarray(x, dtype=float)
        return ((x.T - self.x_offset) / self.x_scale).T

    def to_original_input(self, v):
        v = np.asarray(v, dtype=float)
        return (self.u_offset + self.u_scale * v.T).T

    def from_original_input(self, u):
        u = np.asarray(u, dtype=float)
        return ((u.T - self.u_offset) / self.u_scale).T

    # -- CasADi model ------------------------------------------------------------

    def casadi_rhs(self):
        """
        CasADi Function rhs(x, u) = f_0(x) + G(x) u, obtained by calling the
        NumPy maps with a NumPy object array of CasADi scalars.  This works for
        arithmetic, indexing/unpacking, np.array([...]) and elementwise NumPy
        functions (np.exp, np.sin, ...).
        """
        import casadi as ca
        try:
            ca.GlobalOptions.setNumpyMode(-1)        # silence the numpy-on-casadi notice
        except AttributeError:
            pass
        n, m = self.n, self.m
        x, u = ca.SX.sym("x", n), ca.SX.sym("u", m)
        x_obj = np.empty(n, dtype=object)
        for i in range(n):
            x_obj[i] = x[i]
        f0x = _to_casadi(self._f0(x_obj), (n, 1))
        if callable(self._g):
            Gx = _to_casadi(self._g(x_obj), (n, m))
        else:
            Gx = ca.horzcat(*[_to_casadi(fj(x_obj), (n, 1)) for fj in self._g])
        return ca.Function("rhs", [x, u], [f0x + ca.mtimes(Gx, u)])

    # -- linearization and LQR ---------------------------------------------------

    def linearize(self, eps=1e-6):
        """A = D f_0(0) (central differences), B = G(0)."""
        A = np.column_stack([(self.f0(e) - self.f0(-e)) / (2 * eps) for e in eps * np.eye(self.n)])
        return A, self.G(np.zeros(self.n))

    def design_lqr(self, Q, R):
        """LQR gain K = R^{-1} B' P for the linearization; stores K, P, Q, R."""
        self.Q, self.R = Q, R = np.atleast_2d(Q).astype(float), np.atleast_2d(R).astype(float)
        A, B = self.linearize()
        self.P = solve_continuous_are(A, B, Q, R)
        self.K = np.linalg.solve(R, B.T @ self.P)
        return self.K

    def closed_loop_matrix(self):
        A, B = self.linearize()
        return A - B @ self.K

    def simulate_closed_loop(self, x0, t_span, K=None, **kwargs):
        """Nonlinear system under v = -K z (default: the stored LQR gain); prints the time."""
        K = self.K if K is None else np.atleast_2d(K)
        res = self.simulate(x0, t_span, u=lambda t, z: -K @ z, report=False, **kwargs)
        res.mode = "closed loop (v = -K z)"
        self._report_time(res)
        return res

    def __repr__(self):
        return f"InputAffineSystem(name={self.name!r}, n={self.n}, m={self.m})"

    def __str__(self):
        s = f"System {self.name}: n = {self.n}, m = {self.m}"
        if np.any(self.x_offset != 0) or np.any(self.x_scale != 1):
            s += (f"\n  x = x_offset + x_scale * z,  x_offset = {np.array2string(self.x_offset, precision=5)},"
                  f"  x_scale = {np.array2string(self.x_scale, precision=5)}"
                  f"\n  u = u_offset + u_scale * v,  u_offset = {np.array2string(self.u_offset, precision=5)},"
                  f"  u_scale = {np.array2string(self.u_scale, precision=5)}")
        if self.K is not None:
            s += f"\n  LQR gain K = {np.array2string(self.K, precision=4)}"
        return s
