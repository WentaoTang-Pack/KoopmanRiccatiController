"""
kernel.py
=========

Mercer kernels and kernel ridge regression (KRR).  Data matrices hold one
sample per column: X has shape (d, N); K[i, j] = k(X[:, i], X'[:, j]).

Kernels
-------
    linear_kernel()                      k(x, x') = x' x
    gaussian_kernel(sigma)               exp(-||x - x'||^2 / (2 sigma^2))
    matern_kernel(d, s, l)               Matern, nu = s - d/2;  RKHS = H^s(R^d) (s > d/2)
    wendland_kernel(d, k, rho)           compact support, C^{2k};  RKHS = H^{(d+1)/2 + k}(R^d)
    linear_gaussian_kernel(sigma)        (x' x) * Gaussian
    linear_matern_kernel(d, s, l)        (x' x) * Matern
    linear_wendland_kernel(d, k, rho)    (x' x) * Wendland

Functions in the RKHS of a linear-X kernel vanish at the origin.
Kernels can be added, multiplied, and scaled by c > 0.

Each kernel provides
    k.matrix(X, Xp)       kernel matrix (Xp defaults to X)
    k.grad(X, Xp)         d/dx k(x, x'), shape (d, N, M)
    k.feature_map(X)      phi(x) = [k(x, x_1), ..., k(x, x_N)]'
    k.casadi_row(x, Xp)   CasADi expression of [k(x, x'_j)]_j (all kernels above,
                          Matern only for nu in {1/2, 3/2, 5/2})
    k.fit(X, Y, lam)      KRR:  f(x) = alpha k_X(x),  alpha = Y (K + lam I)^{-1}
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import gamma as _gamma
from scipy.special import kv as _kv


def _cols(X):
    """Data as (d, N); a 1-D array is one sample."""
    X = np.asarray(X, dtype=float)
    return X[:, None] if X.ndim == 1 else X


def _sq_dist(X, Xp):
    """Squared distances between the columns of X (d, N) and Xp (d, M)."""
    sq = np.sum(X ** 2, 0)[:, None] + np.sum(Xp ** 2, 0)[None, :] - 2.0 * X.T @ Xp
    return np.maximum(sq, 0.0)


def _casadi_sq_dist(x, Xp):
    """(1, M) row of ||x - Xp[:, j]||^2 for a CasADi column x."""
    import casadi as ca
    return ca.sum1((ca.repmat(x, 1, Xp.shape[1]) - ca.DM(Xp)) ** 2)


def _casadi_dist(x, Xp, delta=1e-12):
    """sqrt(||x - x'||^2 + delta^2): the tiny delta keeps derivatives finite at x = x'."""
    import casadi as ca
    return ca.sqrt(_casadi_sq_dist(x, Xp) + delta ** 2)


# ---------------------------------------------------------------------------
# Kernel ridge regression model
# ---------------------------------------------------------------------------

class KRRModel:
    """f(x) = alpha k_X(x), fitted by :meth:`MercerKernel.fit`.

    X_train (d, N);  Y_train (p, N) or (N,);  alpha (p, N) or (N,);  gram_matrix (N, N).
    """

    def __init__(self, kernel, X_train, Y_train, alpha, regularization, gram_matrix, cho):
        self.kernel, self.X_train, self.Y_train, self.alpha = kernel, X_train, Y_train, alpha
        self.regularization, self.gram_matrix, self._cho = regularization, gram_matrix, cho

    def predict(self, X):
        """f at the columns of X (d, M): (p, M), or (M,) for a scalar target."""
        return self.alpha @ self.kernel.matrix(self.X_train, _cols(X))

    def predict_gradient(self, X):
        """df/dx at the columns of X: (p, d, M), or (d, M) for a scalar target."""
        G = self.kernel.grad(_cols(X), self.X_train)              # (d, M, N)
        if self.alpha.ndim == 1:
            return np.einsum("dmn,n->dm", G, self.alpha)
        return np.einsum("dmn,pn->pdm", G, self.alpha)

    def loo_residuals(self):
        """Closed-form leave-one-out residuals  r_i / (lam [(K + lam I)^{-1}]_ii)."""
        Cinv = cho_solve(self._cho, np.eye(self.X_train.shape[1]))
        r = self.Y_train - self.alpha @ self.gram_matrix
        return r / (self.regularization * np.diag(Cinv))

    def loo_mse(self):
        """Mean squared leave-one-out error (e.g. to choose lam)."""
        return float(np.mean(self.loo_residuals() ** 2))

    def __repr__(self):
        return (f"KRRModel(kernel={self.kernel.name!r}, N={self.X_train.shape[1]}, "
                f"lambda={self.regularization:.1e})")


# ---------------------------------------------------------------------------
# Feature map
# ---------------------------------------------------------------------------

class KernelFeatureMap:
    """phi(x) = [k(x, x_1), ..., k(x, x_N)]' for the data X_data = [x_1, ..., x_N] (d, N).

    phi(x): (N,) for x (d,), (N, M) for X (d, M);  phi.jacobian(x): (N, d);
    phi.casadi(x): (N, 1) CasADi expression.
    """

    def __init__(self, kernel, X_data):
        self.kernel, self.X_data = kernel, _cols(X_data)

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        Phi = self.kernel.matrix(self.X_data, _cols(x))
        return Phi[:, 0] if x.ndim == 1 else Phi

    def jacobian(self, x):
        return self.kernel.grad(_cols(x), self.X_data)[:, 0, :].T

    def casadi(self, x):
        return self.kernel.casadi_row(x, self.X_data).T

    def __repr__(self):
        return f"KernelFeatureMap(kernel={self.kernel.name!r}, N={self.X_data.shape[1]})"


# ---------------------------------------------------------------------------
# MercerKernel
# ---------------------------------------------------------------------------

class MercerKernel:
    """
    A Mercer kernel given by vectorized functions of data X (d, N), Xp (d, M):

    matrix_func(X, Xp) -> (N, M) kernel matrix;
    grad_func(X, Xp)   -> (d, N, M), d/dx k(x, x') at x = X[:, i], x' = Xp[:, j] (optional);
    casadi_func(x, Xp) -> (1, M) CasADi row for a CasADi column x (optional).
    """

    def __init__(self, matrix_func, grad_func=None, name="MercerKernel", casadi_func=None):
        self._matrix, self._grad, self._casadi, self.name = matrix_func, grad_func, casadi_func, name

    def __call__(self, x, xp):
        return float(self.matrix(np.ravel(x), np.ravel(xp))[0, 0])

    def matrix(self, X, Xp=None):
        X = _cols(X)
        return self._matrix(X, X if Xp is None else _cols(Xp))

    def grad(self, X, Xp=None):
        X = _cols(X)
        return self._grad(X, X if Xp is None else _cols(Xp))

    def casadi_row(self, x, Xp):
        if self._casadi is None:
            raise NotImplementedError(f"{self.name} has no CasADi expression.")
        return self._casadi(x, _cols(Xp))

    @property
    def has_grad(self):
        return self._grad is not None

    @property
    def has_casadi(self):
        return self._casadi is not None

    def feature_map(self, X_data):
        return KernelFeatureMap(self, X_data)

    def fit(self, X, Y, regularization=1e-6):
        """KRR: alpha = Y (K + lam I)^{-1};  X (d, N), Y (p, N) or (N,)."""
        X, Y = _cols(X), np.asarray(Y, dtype=float)
        K = self.matrix(X)
        cho = cho_factor(K + regularization * np.eye(X.shape[1]))
        alpha = cho_solve(cho, Y.T).T
        return KRRModel(self, X, Y, alpha, float(regularization), K, cho)

    # -- arithmetic: k1 + k2, k1 * k2, c * k (c > 0) ------------------------------

    def __add__(self, other):
        if not isinstance(other, MercerKernel):
            return NotImplemented
        a, b = self, other
        g = (lambda X, Xp: a._grad(X, Xp) + b._grad(X, Xp)) if a.has_grad and b.has_grad else None
        c = (lambda x, Xp: a._casadi(x, Xp) + b._casadi(x, Xp)) if a.has_casadi and b.has_casadi else None
        return MercerKernel(lambda X, Xp: a._matrix(X, Xp) + b._matrix(X, Xp), g, f"({a.name} + {b.name})", c)

    def __mul__(self, other):
        a = self
        if isinstance(other, MercerKernel):
            b = other
            g = None
            if a.has_grad and b.has_grad:
                g = lambda X, Xp: a._grad(X, Xp) * b._matrix(X, Xp)[None] + a._matrix(X, Xp)[None] * b._grad(X, Xp)
            c = (lambda x, Xp: a._casadi(x, Xp) * b._casadi(x, Xp)) if a.has_casadi and b.has_casadi else None
            return MercerKernel(lambda X, Xp: a._matrix(X, Xp) * b._matrix(X, Xp), g, f"({a.name} * {b.name})", c)
        if np.isscalar(other):
            s = float(other)
            if s <= 0:
                raise ValueError("A kernel can only be scaled by a positive constant.")
            g = (lambda X, Xp: s * a._grad(X, Xp)) if a.has_grad else None
            c = (lambda x, Xp: s * a._casadi(x, Xp)) if a.has_casadi else None
            return MercerKernel(lambda X, Xp: s * a._matrix(X, Xp), g, f"{s:g} * {a.name}", c)
        return NotImplemented

    __rmul__ = __mul__

    def __repr__(self):
        return f"MercerKernel({self.name!r})"


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

def linear_kernel():
    """k(x, x') = x' x."""
    def K(X, Xp):
        return X.T @ Xp

    def grad(X, Xp):
        return np.broadcast_to(Xp[:, None, :], (X.shape[0], X.shape[1], Xp.shape[1])).copy()

    def cas(x, Xp):
        import casadi as ca
        return ca.mtimes(x.T, ca.DM(Xp))

    return MercerKernel(K, grad, "Linear", cas)


def gaussian_kernel(sigma=1.0):
    """k(x, x') = exp(-||x - x'||^2 / (2 sigma^2))."""
    c = 0.5 / sigma ** 2

    def K(X, Xp):
        return np.exp(-c * _sq_dist(X, Xp))

    def grad(X, Xp):
        return -2.0 * c * (X[:, :, None] - Xp[:, None, :]) * K(X, Xp)[None]

    def cas(x, Xp):
        import casadi as ca
        return ca.exp(-c * _casadi_sq_dist(x, Xp))

    return MercerKernel(K, grad, f"Gaussian(sigma={sigma:g})", cas)


def matern_kernel(d, s, length_scale=1.0):
    """
    Matern kernel  k = 2^{1-nu}/Gamma(nu) r^nu K_nu(r),  r = ||x - x'|| / l,
    nu = s - d/2 > 0, whose RKHS is norm-equivalent to H^s(R^d); k(x, x) = 1.
    For nu = 1/2, 3/2, 5/2: exp(-r), (1 + r) exp(-r), (1 + r + r^2/3) exp(-r).
    The gradient  -2^{1-nu}/Gamma(nu) r^{nu-1} K_{nu-1}(r) (x - x') / l^2  exists for nu > 1/2.
    """
    nu, ell = float(s) - d / 2.0, float(length_scale)
    if nu <= 0:
        raise ValueError(f"Need s > d/2 = {d / 2:g}.")
    c = 2.0 ** (1.0 - nu) / _gamma(nu)

    def K(X, Xp):
        R = np.sqrt(_sq_dist(X, Xp)) / ell
        Rs = np.where(R == 0, 1.0, R)
        return np.where(R == 0, 1.0, c * Rs ** nu * _kv(nu, Rs))

    def grad(X, Xp):
        D = X[:, :, None] - Xp[:, None, :]
        R = np.sqrt(np.sum(D ** 2, 0)) / ell
        Rs = np.where(R == 0, 1.0, R)
        return np.where(R == 0, 0.0, -c * Rs ** (nu - 1) * _kv(nu - 1, Rs) / ell ** 2)[None] * D

    cas = None
    if nu in (0.5, 1.5, 2.5):
        def cas(x, Xp):
            import casadi as ca
            r = _casadi_dist(x, Xp) / ell
            poly = {0.5: 1, 1.5: 1 + r, 2.5: 1 + r + r ** 2 / 3}[nu]
            return poly * ca.exp(-r)

    return MercerKernel(K, grad if nu > 0.5 else None, f"Matern(d={d}, s={float(s):g}, nu={nu:g}, l={ell:g})", cas)


def wendland_kernel(d, k=1, support_radius=1.0):
    """
    Wendland kernel  phi(||x - x'|| / rho),  phi(r) = (1 - r)_+^{l+k} p_k(r),
    l = floor(d/2) + k + 1, normalized to phi(0) = 1 (k = 0, 1, 2, 3).
    E.g. d <= 3, k = 1:  phi(r) = (1 - r)_+^4 (4 r + 1).
    """
    l, rho = d // 2 + k + 1, float(support_radius)
    p = {0: [1.0],
         1: [1.0, l + 1.0],
         2: [3.0, 3 * l + 6.0, l ** 2 + 4 * l + 3.0],
         3: [15.0, 15 * l + 45.0, 6 * l ** 2 + 36 * l + 45.0, l ** 3 + 9 * l ** 2 + 23 * l + 15.0]}[k]
    p = np.array(p) / p[0]
    a = l + k
    P = np.polynomial.polynomial
    phi = lambda r: np.maximum(1 - r, 0) ** a * P.polyval(r, p)
    dphi = lambda r: (-a * np.maximum(1 - r, 0) ** (a - 1) * P.polyval(r, p)
                      + np.maximum(1 - r, 0) ** a * P.polyval(r, P.polyder(p)))

    def grad(X, Xp):
        D = X[:, :, None] - Xp[:, None, :]
        R = np.sqrt(np.sum(D ** 2, 0)) / rho
        Rs = np.where(R == 0, 1.0, R)
        return np.where(R == 0, 0.0, dphi(Rs) / (Rs * rho ** 2))[None] * D

    def cas(x, Xp):
        import casadi as ca
        r = _casadi_dist(x, Xp) / rho
        return ca.fmax(1 - r, 0) ** a * sum(c * r ** i for i, c in enumerate(p))

    return MercerKernel(lambda X, Xp: phi(np.sqrt(_sq_dist(X, Xp)) / rho), grad if k >= 1 else None,
                        f"Wendland(d={d}, k={k}, rho={rho:g})", cas)


def linear_gaussian_kernel(sigma=1.0):
    k = linear_kernel() * gaussian_kernel(sigma)
    k.name = f"Linear-Gaussian(sigma={sigma:g})"
    return k


def linear_matern_kernel(d, s, length_scale=1.0):
    k = linear_kernel() * matern_kernel(d, s, length_scale)
    k.name = f"Linear-Matern(d={d}, s={float(s):g}, l={float(length_scale):g})"
    return k


def linear_wendland_kernel(d, k=1, support_radius=1.0):
    kern = linear_kernel() * wendland_kernel(d, k, support_radius)
    kern.name = f"Linear-Wendland(d={d}, k={k}, rho={float(support_radius):g})"
    return kern
