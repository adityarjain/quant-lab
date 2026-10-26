"""M3 mean-variance optimization. Long-only, fully invested, per-asset cap; rf = 0.

    maximize  w'mu / sqrt(w' Sigma w)      (max Sharpe)
    minimize  w' Sigma w                   (min variance; frontier adds w'mu = target)
    s.t.      sum(w) = 1,  0 <= w_i <= CAP
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252
CAP = 0.4


def annualize(returns: pd.DataFrame):
    """(mu, Sigma) annualized from daily returns."""
    return returns.mean() * TRADING_DAYS, returns.cov() * TRADING_DAYS


def perf(w, mu, cov):
    """(return, vol, Sharpe) of weights w."""
    w = np.asarray(w)
    r, v = float(w @ np.asarray(mu)), float(np.sqrt(w @ np.asarray(cov) @ w))
    return r, v, r / v


def _solve(objective, n: int, cap: float, extra=()):
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}, *extra]
    res = minimize(objective, np.full(n, 1 / n), method="SLSQP", bounds=[(0, cap)] * n,
                   constraints=cons, options={"ftol": 1e-12, "maxiter": 1000})
    # Status 8 ("positive directional derivative") = converged but couldn't verify progress at ftol;
    # accept it only if the point is feasible. Anything else is a real failure.
    feasible = np.isclose(res.x.sum(), 1, atol=1e-8) and (res.x > -1e-9).all() and (res.x < cap + 1e-9).all()
    if not (res.success or (res.status == 8 and feasible)):
        raise RuntimeError(f"optimizer failed: {res.message}")
    return np.clip(res.x, 0, cap)


def min_variance(mu: pd.Series, cov: pd.DataFrame, cap: float = CAP) -> pd.Series:
    c = cov.to_numpy()
    return pd.Series(_solve(lambda w: w @ c @ w, len(mu), cap), index=mu.index)


def max_sharpe(mu: pd.Series, cov: pd.DataFrame, cap: float = CAP) -> pd.Series:
    m, c = mu.to_numpy(), cov.to_numpy()
    return pd.Series(_solve(lambda w: -(w @ m) / np.sqrt(w @ c @ w), len(mu), cap), index=mu.index)


def max_return(mu: pd.Series, cap: float = CAP) -> float:
    """Highest feasible return: fill the best assets up to the cap."""
    left, total = 1.0, 0.0
    for m in mu.sort_values(ascending=False):
        take = min(cap, left)
        total, left = total + take * m, left - take
    return total


def frontier(mu: pd.Series, cov: pd.DataFrame, cap: float = CAP, points: int = 40) -> pd.DataFrame:
    """Efficient frontier: min variance at each target return from min-var up to max feasible."""
    m, c = mu.to_numpy(), cov.to_numpy()
    r_lo = perf(min_variance(mu, cov, cap), mu, cov)[0]
    rows = []
    for t in np.linspace(r_lo, max_return(mu, cap) - 1e-6, points):
        try:
            w = _solve(lambda w: w @ c @ w, len(mu), cap, [{"type": "eq", "fun": lambda w, t=t: w @ m - t}])
        except RuntimeError:
            continue  # numerically infeasible at the very edge; skip the point
        rows.append({"ret": t, "vol": float(np.sqrt(w @ c @ w))})
    return pd.DataFrame(rows)


def random_portfolios(mu: pd.Series, cov: pd.DataFrame, n: int = 10_000, seed: int = 42,
                      cap: float = CAP) -> pd.DataFrame:
    """Uniform random long-only portfolios (Dirichlet(1)) that respect the same cap as the frontier,
    so the frontier is the true upper-left edge of the cloud."""
    w = np.random.default_rng(seed).dirichlet(np.ones(len(mu)), n * 4)
    w = w[(w <= cap).all(axis=1)][:n]
    ret = w @ mu.to_numpy()
    vol = np.sqrt(np.einsum("ij,jk,ik->i", w, cov.to_numpy(), w))
    return pd.DataFrame({"ret": ret, "vol": vol, "sharpe": ret / vol})


def walk_forward(returns: pd.DataFrame, window: int = 756, cap: float = CAP, fit=None) -> pd.DataFrame:
    """Re-fit every quarter end on the trailing `window` days only (no future data).
    `fit(hist_returns) -> weights`; default = sample max-Sharpe with the cap.
    Returns target weights on rebalance dates, forward-filled to daily."""
    fit = fit or (lambda h: max_sharpe(*annualize(h), cap))
    idx = returns.index
    s = idx.to_series()
    rebal = pd.DatetimeIndex(s.groupby(idx.to_period("Q")).max().values)
    rows = {}
    for t in rebal:
        hist = returns.loc[:t].tail(window)
        if len(hist) < window:
            continue
        rows[t] = fit(hist)
    return pd.DataFrame(rows).T.reindex(idx).ffill().fillna(0.0)


# ---------------------------------------------------------------- M12: estimators robust to estimation error
def ledoit_wolf(returns: pd.DataFrame):
    """Ledoit & Wolf (2004) shrinkage of the sample covariance toward a scaled identity.
    Returns (annualized covariance, shrinkage intensity delta in [0, 1])."""
    x = (returns - returns.mean()).to_numpy()
    n, p = x.shape
    s = x.T @ x / n
    m = np.trace(s) / p
    d2 = np.sum((s - m * np.eye(p)) ** 2)
    sq = np.sum(x ** 2, axis=1)  # ||x_k||^2
    b2_bar = (np.sum(sq ** 2) - 2 * np.einsum("ki,ij,kj->", x, s, x) + n * np.sum(s ** 2)) / n ** 2
    delta = float(min(b2_bar, d2) / d2) if d2 > 0 else 1.0
    shrunk = delta * m * np.eye(p) + (1 - delta) * s
    return pd.DataFrame(shrunk * TRADING_DAYS, index=returns.columns, columns=returns.columns), delta


def bayes_stein_mu(returns: pd.DataFrame, cov: pd.DataFrame | None = None):
    """Jorion (1986) Bayes-Stein shrinkage of mean returns toward the minimum-variance portfolio's mean.
    Returns (annualized shrunk mu, shrinkage weight phi in [0, 1]). `cov` annualized; sample if None."""
    n, p = returns.shape
    m = returns.mean().to_numpy()
    c = (cov.to_numpy() / TRADING_DAYS) if cov is not None else returns.cov().to_numpy()
    ci = np.linalg.inv(c)
    one = np.ones(p)
    m0 = one @ ci @ m / (one @ ci @ one)
    gap = m - m0
    phi = float((p + 2) / ((p + 2) + n * gap @ ci @ gap))
    return pd.Series(((1 - phi) * m + phi * m0) * TRADING_DAYS, index=returns.columns), phi


def risk_parity(cov: pd.DataFrame) -> pd.Series:
    """Equal risk contribution: every asset adds the same share of portfolio variance.
    Needs no return forecast. Solves the convex problem min 0.5 w'Sw - (1/p) sum(log w), then normalizes."""
    c = cov.to_numpy() / np.mean(np.diag(cov.to_numpy()))  # weights are scale-free
    p = len(c)
    x = np.full(p, 1 / np.sqrt(p))
    # Cyclical coordinate descent (Griveau-Billion, Richard & Roncalli, 2013): each coordinate's
    # first-order condition c_ii x_i^2 + b_i x_i - 1/p = 0 has one positive root, solved exactly.
    for _ in range(10_000):
        prev = x.copy()
        for i in range(p):
            b = c[i] @ x - c[i, i] * x[i]
            x[i] = (-b + np.sqrt(b * b + 4 * c[i, i] / p)) / (2 * c[i, i])
        if np.abs(x - prev).max() < 1e-14:
            break
    w = pd.Series(x / x.sum(), index=cov.index)
    # Judge by the defining property, not by an iteration count.
    if np.abs(risk_contributions(w, cov) - 1 / p).max() > 1e-6:
        raise RuntimeError("risk parity did not converge")
    return w


def risk_contributions(w: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Share of portfolio variance from each asset: w_i (S w)_i / w'S w."""
    cw = cov.to_numpy() @ w.to_numpy()
    return pd.Series(w.to_numpy() * cw / (w.to_numpy() @ cw), index=w.index)
