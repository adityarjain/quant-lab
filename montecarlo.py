"""M4 Monte Carlo risk. Two models of the same daily portfolio return series:

A  Normal:          r_t ~ N(mu, sigma), i.i.d.  For fixed weights this is exactly the multivariate
                    normal of the components projected onto w (w'r ~ N(w'mu, w'Sigma w)).
B  Block bootstrap: resample 21-day blocks of real history, which keeps fat tails, skew and
                    short-run volatility clustering that the normal model throws away.
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252
YEARS = 5
N_PATHS = 10_000
BLOCK = 21


def simulate_normal(returns: pd.Series, n_paths: int = N_PATHS, n_days: int = YEARS * TRADING_DAYS,
                    seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(returns.mean(), returns.std(), size=(n_paths, n_days))


def simulate_bootstrap(returns: pd.Series, n_paths: int = N_PATHS, n_days: int = YEARS * TRADING_DAYS,
                       block: int = BLOCK, seed: int = 42) -> np.ndarray:
    """Moving-block bootstrap: each path is a chain of random `block`-day windows of real history."""
    x = returns.to_numpy()
    rng = np.random.default_rng(seed)
    n_blocks = -(-n_days // block)
    starts = rng.integers(0, len(x) - block + 1, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_paths, -1)[:, :n_days]
    return x[idx]


def var_cvar(r: np.ndarray, level: float):
    """(VaR, CVaR) as positive loss fractions. CVaR = mean loss beyond VaR (expected shortfall)."""
    r = np.ravel(r)
    q = np.quantile(r, 1 - level)
    return float(-q), float(-r[r <= q].mean())


def horizon_returns(daily: np.ndarray, h: int) -> np.ndarray:
    """Non-overlapping compounded h-day returns within each path."""
    n = daily.shape[1] // h
    return np.prod(1 + daily[:, : n * h].reshape(len(daily), n, h), axis=2) - 1


def equity(daily: np.ndarray) -> np.ndarray:
    """Path values starting at 1 (column 0 = start)."""
    return np.hstack([np.ones((len(daily), 1)), np.cumprod(1 + daily, axis=1)])


def max_drawdowns(daily: np.ndarray) -> np.ndarray:
    eq = equity(daily)
    return (eq / np.maximum.accumulate(eq, axis=1) - 1).min(axis=1)


def summary(daily: np.ndarray) -> dict:
    terminal = equity(daily)[:, -1] - 1
    v95, c95 = var_cvar(daily, 0.95)
    v99, c99 = var_cvar(daily, 0.99)
    m95, mc95 = var_cvar(horizon_returns(daily, 21), 0.95)
    mdd = max_drawdowns(daily)
    return {
        "1d VaR95": v95, "1d CVaR95": c95, "1d VaR99": v99, "1d CVaR99": c99,
        "1m VaR95": m95, "1m CVaR95": mc95,
        "5y p5": np.quantile(terminal, 0.05), "5y p50": np.quantile(terminal, 0.5),
        "5y p95": np.quantile(terminal, 0.95), "P(5y loss)": float((terminal < 0).mean()),
        "maxDD median": float(np.median(mdd)), "maxDD p5 (worst 5%)": float(np.quantile(mdd, 0.05)),
    }


def exceedance_rate(realized: pd.Series, var: float) -> float:
    """Share of days whose loss exceeded VaR. A calibrated 95% VaR is exceeded ~5% of days."""
    return float((realized < -var).mean())
