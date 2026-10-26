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


def walk_forward(returns: pd.DataFrame, window: int = 756, cap: float = CAP) -> pd.DataFrame:
    """Re-fit max-Sharpe every quarter end on the trailing `window` days only (no future data).
    Returns target weights on rebalance dates, forward-filled to daily."""
    idx = returns.index
    s = idx.to_series()
    rebal = pd.DatetimeIndex(s.groupby(idx.to_period("Q")).max().values)
    rows = {}
    for t in rebal:
        hist = returns.loc[:t].tail(window)
        if len(hist) < window:
            continue
        mu, cov = annualize(hist)
        rows[t] = max_sharpe(mu, cov, cap)
    return pd.DataFrame(rows).T.reindex(idx).ffill().fillna(0.0)
