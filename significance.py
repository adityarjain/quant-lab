"""M8 is a Sharpe ratio real? Inference on daily EXCESS returns (return minus T-bill).

- Block-bootstrap confidence intervals (21-day blocks keep autocorrelation and vol clustering).
- Paired block bootstrap for the difference between two strategies' Sharpe ratios: both series are
  resampled on the SAME dates, so their correlation is preserved (cf. Ledoit & Wolf, 2008).
- Probabilistic Sharpe Ratio (Bailey & Lopez de Prado, 2012): P(true SR > SR*) given the sample's
  length, skew and kurtosis.
- Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014): PSR against the Sharpe you would expect
  from the BEST of N strategies that are all truly zero, i.e. corrected for having tried N things.
"""
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649015329


def _block_index(n: int, n_boot: int, block: int, rng) -> np.ndarray:
    n_blocks = -(-n // block)
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    return (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]


def _annual_sharpe(x: np.ndarray) -> np.ndarray:
    """Row-wise annualized Sharpe of excess returns."""
    return x.mean(axis=-1) / x.std(axis=-1, ddof=1) * np.sqrt(TRADING_DAYS)


def sharpe_ci(excess: pd.Series, level: float = 0.95, n_boot: int = 5000, block: int = 21, seed: int = 42):
    """(point, low, high) annualized Sharpe with a moving-block bootstrap percentile interval."""
    x = np.asarray(excess, dtype=float)
    boot = _annual_sharpe(x[_block_index(len(x), n_boot, block, np.random.default_rng(seed))])
    a = (1 - level) / 2
    return float(_annual_sharpe(x)), float(np.quantile(boot, a)), float(np.quantile(boot, 1 - a))


def sharpe_diff(excess_a: pd.Series, excess_b: pd.Series, level: float = 0.95, n_boot: int = 5000,
                block: int = 21, seed: int = 42):
    """(diff, low, high, p) for Sharpe(A) - Sharpe(B), resampling both on the same dates.
    p = two-sided bootstrap p-value for H0: no difference."""
    a, b = (np.asarray(s, dtype=float) for s in (excess_a, excess_b))
    assert len(a) == len(b), "series must cover the same dates"
    idx = _block_index(len(a), n_boot, block, np.random.default_rng(seed))
    d = _annual_sharpe(a[idx]) - _annual_sharpe(b[idx])
    q = (1 - level) / 2
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return float(_annual_sharpe(a) - _annual_sharpe(b)), float(np.quantile(d, q)), float(np.quantile(d, 1 - q)), p


def psr(excess: pd.Series, sr_star: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio. `sr_star` is per-period (daily), like the estimate it is compared to."""
    x = np.asarray(excess, dtype=float)
    sr = x.mean() / x.std(ddof=1)
    g3, g4 = skew(x), kurtosis(x, fisher=False)  # normal: 0 and 3
    se = np.sqrt((1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) / (len(x) - 1))
    return float(norm.cdf((sr - sr_star) / se))


def expected_max_sharpe(trial_sharpes: np.ndarray) -> float:
    """Per-period Sharpe expected from the best of N trials whose true Sharpe is zero, given the
    observed spread (variance) of the N trials' Sharpe estimates."""
    s = np.asarray(trial_sharpes, dtype=float)
    n = len(s)
    return float(np.sqrt(s.var(ddof=1)) * ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / n)
                                           + EULER_GAMMA * norm.ppf(1 - 1 / (n * np.e))))


def dsr(excess: pd.Series, trial_sharpes: np.ndarray) -> float:
    """Deflated Sharpe Ratio: PSR against the expected maximum of the trials (per-period Sharpes)."""
    return psr(excess, expected_max_sharpe(trial_sharpes))
