"""M1 momentum strategies. Each returns target weights (dates x tickers); no P&L here.

Parameters are the textbook defaults (50/200 MA; Jegadeesh-Titman 12-1), not tuned,
so the train period is a genuine in-sample check and test is out-of-sample.
"""
import pandas as pd


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each month present in the index."""
    s = index.to_series()
    return pd.DatetimeIndex(s.groupby(index.to_period("M")).max().values)


def buy_and_hold(prices: pd.DataFrame, ticker: str = "SPY") -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    w[ticker] = 1.0
    return w


def equal_weight(prices: pd.DataFrame, universe: list[str]) -> pd.DataFrame:
    """Equal weight across `universe`; isolates stock selection from market exposure."""
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    w[universe] = 1.0 / len(universe)
    return w


def ma_crossover(prices: pd.DataFrame, ticker: str = "SPY", fast: int = 50, slow: int = 200) -> pd.DataFrame:
    """Time-series momentum: 100% in `ticker` while fast SMA > slow SMA, else cash."""
    p = prices[ticker]
    on = (p.rolling(fast).mean() > p.rolling(slow).mean()).astype(float)  # NaN warm-up compares False -> cash
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    w[ticker] = on
    return w


def cross_sectional(prices: pd.DataFrame, universe: list[str], lookback: int = 252,
                    skip: int = 21, top: float = 0.2) -> pd.DataFrame:
    """12-1 cross-sectional momentum: at each month end rank `universe` by the return from
    t-lookback to t-skip (skipping the last month avoids short-term reversal), hold the top
    `top` fraction equal-weighted until the next month end. Long-only."""
    p = prices[universe]
    score = p.shift(skip) / p.shift(lookback) - 1
    rebal = month_ends(prices.index)
    score = score.loc[rebal].dropna(how="all")  # first year has no score -> no position
    n = max(1, round(len(universe) * top))
    picks = score.rank(axis=1, ascending=False, method="first") <= n
    w = picks.astype(float).div(n)
    return w.reindex(index=prices.index, columns=prices.columns).ffill().fillna(0.0)
