"""Vectorized backtest engine. The ONLY place P&L is computed.

Contract: a strategy hands over `weights` (dates x tickers) = the target position decided
at the close of each date. The engine holds it from the NEXT day (shift(1)), so a signal
can never earn the return of the day it was computed on.
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def run(weights: pd.DataFrame, prices: pd.DataFrame, cost_bps: float = 10.0) -> pd.DataFrame:
    """Daily gross/net returns and turnover for a weights frame.

    cost = cost_bps per unit of turnover (sum of |change in weight| across tickers).
    # ponytail: holds target weights constant between signal changes (implicit daily
    # rebalance) and ignores drift turnover, so costs are slightly understated; track
    # drifted weights if a high-turnover strategy makes this matter.
    """
    rets = prices.pct_change().fillna(0.0)
    w = weights.reindex(index=prices.index, columns=prices.columns).ffill().fillna(0.0)
    held = w.shift(1).fillna(0.0)
    turnover = (held - held.shift(1).fillna(0.0)).abs().sum(axis=1)
    gross = (held * rets).sum(axis=1)
    net = gross - turnover * cost_bps / 1e4
    return pd.DataFrame({"gross": gross, "net": net, "turnover": turnover})


def drawdown(returns: pd.Series) -> pd.Series:
    equity = (1 + returns).cumprod()
    return equity / equity.cummax() - 1


def metrics(returns: pd.Series, turnover: pd.Series | None = None) -> dict:
    """Annualized summary. Sharpe uses rf = 0 (stated in README)."""
    n = len(returns)
    equity = (1 + returns).prod()
    vol = returns.std() * np.sqrt(TRADING_DAYS)
    out = {
        "CAGR": equity ** (TRADING_DAYS / n) - 1 if n else np.nan,
        "vol": vol,
        "Sharpe": returns.mean() * TRADING_DAYS / vol if vol > 0 else np.nan,
        "maxDD": drawdown(returns).min(),
    }
    if turnover is not None:
        out["turnover"] = turnover.mean() * TRADING_DAYS  # annual one-way turnover, in x of capital
    return out
