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


# ---------------------------------------------------------------- M9: VaR that adapts to volatility
EWMA_LAMBDA = 0.94  # RiskMetrics daily decay


def ewma_sigma(returns: pd.Series, lam: float = EWMA_LAMBDA) -> pd.Series:
    """Forecast of day-t volatility using returns up to t-1 only:
    sigma^2_t = lam * sigma^2_{t-1} + (1 - lam) * r^2_{t-1}."""
    return np.sqrt(returns.pow(2).ewm(alpha=1 - lam, adjust=False).mean().shift(1))


def var_series(returns: pd.Series, train_end: str, level: float, burn_in: int = 63) -> pd.DataFrame:
    """Daily 1-day VaR (positive loss) from four models, all parameters from data before `train_end`.

    static normal      -(mu + sigma * z)                    fixed train mean/vol
    static historical  -quantile_train(r)                   fixed empirical quantile
    EWMA normal        sigma_t * z                          today's volatility, normal shape
    EWMA filtered hist -sigma_t * quantile_train(r / sigma) today's volatility, train's tail shape (FHS)
    """
    from scipy.stats import norm

    q = 1 - level
    train = returns[returns.index < train_end]
    sig = ewma_sigma(returns)
    resid = (train / sig.reindex(train.index)).iloc[burn_in:].dropna()  # standardized train residuals
    return pd.DataFrame({
        "Static normal": -(train.mean() + train.std() * norm.ppf(q)),
        "Static historical": -np.quantile(train, q),
        "EWMA normal": -sig * norm.ppf(q),
        "EWMA filtered historical": -sig * np.quantile(resid, q),
    }, index=returns.index)


def kupiec(breaches: np.ndarray, p: float) -> float:
    """Kupiec (1995) proportion-of-failures test: p-value of H0 'breach rate = p'. Chi-square(1)."""
    from scipy.special import xlogy
    from scipy.stats import chi2

    b = np.asarray(breaches, dtype=bool)
    t, x = len(b), b.sum()
    ll0 = xlogy(t - x, 1 - p) + xlogy(x, p)
    ll1 = xlogy(t - x, 1 - x / t) + xlogy(x, x / t)
    return float(chi2.sf(-2 * (ll0 - ll1), 1))


def christoffersen(breaches: np.ndarray) -> float:
    """Christoffersen (1998) independence test: p-value of H0 'a breach today doesn't make one
    tomorrow more likely' (no clustering). Chi-square(1)."""
    from scipy.special import xlogy
    from scipy.stats import chi2

    b = np.asarray(breaches, dtype=int)
    prev, cur = b[:-1], b[1:]
    n00, n01 = ((prev == 0) & (cur == 0)).sum(), ((prev == 0) & (cur == 1)).sum()
    n10, n11 = ((prev == 1) & (cur == 0)).sum(), ((prev == 1) & (cur == 1)).sum()
    if n10 + n11 == 0:  # no breaches to cluster
        return 1.0
    pi01, pi11, pi = n01 / (n00 + n01), n11 / (n10 + n11), (n01 + n11) / (n00 + n01 + n10 + n11)
    ll0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll1 = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    return float(chi2.sf(-2 * (ll0 - ll1), 1))
