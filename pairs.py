"""M2 mean-reversion pairs. Selection and hedge ratios use TRAIN prices only; returns weights.

Spread  s_t = log A_t - beta * log B_t   (beta from OLS on train)
Signal  z_t = (s_t - mean_60(s)) / std_60(s)
Rules   enter short spread z > 2, long spread z < -2; exit when z crosses back inside 0.5;
        stop out at |z| > 4 and stay flat until z returns inside 0.5.
"""
import itertools

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

import data

WINDOW, ENTRY, EXIT, STOP = 60, 2.0, 0.5, 4.0


def hedge_ratio(a: pd.Series, b: pd.Series) -> float:
    """OLS slope of log A on log B (with intercept)."""
    return float(np.polyfit(np.log(b), np.log(a), 1)[0])


def half_life(spread: pd.Series) -> float:
    """Days for a deviation to halve, from ds_t = k * s_{t-1} + c. Inf if not mean-reverting."""
    s = spread.dropna()
    k = np.polyfit(s.shift(1).iloc[1:], s.diff().iloc[1:], 1)[0]
    return float(-np.log(2) / k) if k < 0 else np.inf


def coint_pvalue(a: pd.Series, b: pd.Series) -> float:
    """Engle-Granger p-value, H0 = no cointegration. Order matters: A regressed on B."""
    return float(coint(np.log(a), np.log(b))[1])


def scan(train: pd.DataFrame) -> pd.DataFrame:
    """Every within-sector pair: return correlation vs cointegration, hedge ratio, half-life."""
    rows = []
    rets = train.pct_change()
    for sector, names in data.SECTORS.items():
        for a, b in itertools.combinations([n for n in names if n in train], 2):
            beta = hedge_ratio(train[a], train[b])
            rows.append({
                "sector": sector, "a": a, "b": b,
                "corr": rets[a].corr(rets[b]),
                "pvalue": coint_pvalue(train[a], train[b]),
                "beta": beta,
                "half_life": half_life(np.log(train[a]) - beta * np.log(train[b])),
            })
    return pd.DataFrame(rows).sort_values("pvalue", ignore_index=True)


def select(scanned: pd.DataFrame, n: int = 5, alpha: float = 0.05) -> pd.DataFrame:
    """Top-n pairs by p-value that pass alpha with a positive hedge ratio.
    # ponytail: ~120 tests at 5% means ~6 false positives expected by chance; reported in the
    # README instead of corrected. Use Bonferroni/BH if the selection is meant to be strict.
    """
    ok = scanned[(scanned["pvalue"] < alpha) & (scanned["beta"] > 0)]
    return ok.head(n).reset_index(drop=True)


def positions(z: pd.Series, entry=ENTRY, exit=EXIT, stop=STOP) -> pd.Series:
    """Spread position in {-1, 0, +1} from the z-score path. Stateful, so a plain loop
    (~4k steps per pair); the engine itself stays vectorized."""
    out = np.zeros(len(z))
    cur, stopped = 0, False
    for i, v in enumerate(z.to_numpy()):
        if np.isnan(v):
            cur = 0
        elif cur == 0:
            if stopped:
                stopped = abs(v) >= exit          # re-arm only once the spread has normalized
            elif entry < v < stop:
                cur = -1                          # spread rich: short A, long B
            elif -stop < v < -entry:
                cur = 1                           # spread cheap: long A, short B
        elif abs(v) > stop:
            cur, stopped = 0, True
        elif (cur == 1 and v > -exit) or (cur == -1 and v < exit):
            cur = 0
        out[i] = cur
    return pd.Series(out, index=z.index)


def spread_z(prices: pd.DataFrame, a: str, b: str, beta: float, window: int = WINDOW):
    spread = np.log(prices[a]) - beta * np.log(prices[b])
    roll = spread.rolling(window)
    return spread, (spread - roll.mean()) / roll.std()


def pair_weights(prices: pd.DataFrame, a: str, b: str, beta: float, window: int = WINDOW,
                 entry: float = ENTRY, exit: float = EXIT, stop: float = STOP) -> pd.DataFrame:
    """Long/short weights scaled to gross exposure 1: +1 spread = long $1/(1+beta) A, short $beta/(1+beta) B.
    # ponytail: short borrow fees and rebates ignored; add a borrow-cost bps on the short leg if needed.
    """
    _, z = spread_z(prices, a, b, beta, window)
    pos = positions(z, entry, exit, stop)
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    w[a] = pos / (1 + beta)
    w[b] = -pos * beta / (1 + beta)
    return w


def portfolio(prices: pd.DataFrame, chosen: pd.DataFrame, **rules) -> pd.DataFrame:
    """Equal capital per pair. `rules` = optional window/entry/exit/stop overrides."""
    ws = [pair_weights(prices, r.a, r.b, r.beta, **rules) for r in chosen.itertuples()]
    return sum(ws) / len(ws)
