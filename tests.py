"""Assert-based checks, one per module. Run: python tests.py"""
import numpy as np
import pandas as pd

import backtest
import data
import momentum


def _random_prices(n_days=1000, n_assets=5, seed=data.SEED):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.01, size=(n_days, n_assets))
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=[f"S{i}" for i in range(n_assets)])


def test_backtest():
    p = _random_prices()
    r = p.pct_change().fillna(0.0)
    cheat = np.sign(r)  # "knows" today's return: perfect if it were applied same-day

    same_day = (cheat * r).sum(axis=1)  # what a look-ahead bug would report
    assert backtest.metrics(same_day)["Sharpe"] > 10, "sanity: the cheat must look amazing without the shift"

    res = backtest.run(cheat, p, cost_bps=0)
    assert abs(backtest.metrics(res["gross"])["Sharpe"]) < 1, "engine leaked same-day information"

    res = backtest.run(cheat, p, cost_bps=10)
    assert (res["net"] <= res["gross"] + 1e-15).all() and res["turnover"].sum() > 0

    hold = backtest.run(momentum.buy_and_hold(p, "S0"), p, cost_bps=0)["gross"]
    assert np.isclose((1 + hold).prod(), p["S0"].iloc[-1] / p["S0"].iloc[0])  # bought at day-0 close


def test_momentum():
    p = _random_prices(n_days=600, n_assets=10)
    uni = list(p.columns)
    w = momentum.cross_sectional(p, uni, top=0.2)
    active = w[w.sum(axis=1) > 0]
    assert np.allclose(active.sum(axis=1), 1.0) and ((active > 0).sum(axis=1) == 2).all()

    # No look-ahead inside the strategy: rewriting the future must not change past weights.
    t = p.index[400]
    p2 = p.copy()
    p2.loc[p2.index > t] *= np.linspace(1, 5, (p2.index > t).sum())[:, None] ** np.arange(10)
    assert w.loc[:t].equals(momentum.cross_sectional(p2, uni, top=0.2).loc[:t])

    # Picks the strongest trend: S9 rises 1%/day, so it must be held once ranked.
    p3 = p.copy()
    p3["S9"] = 100 * 1.01 ** np.arange(len(p3))
    assert (momentum.cross_sectional(p3, uni).loc[p3.index[300]:, "S9"] > 0).all()

    m = momentum.ma_crossover(p3, "S9")
    assert m["S9"].iloc[:199].eq(0).all() and m["S9"].iloc[250:].eq(1).all()


def test_data():
    idx = pd.bdate_range("2018-12-24", periods=10)  # straddles the train/test split
    p = pd.DataFrame({"A": np.arange(10.0), "B": np.arange(10.0), "C": np.arange(10.0)}, index=idx)
    p.loc[idx[3:5], "A"] = np.nan   # 2-day gap -> filled
    p.loc[idx[3:7], "B"] = np.nan   # 4-day gap -> ticker dropped
    p.loc[idx[:2], "C"] = np.nan    # missing history at start -> dropped (no back-fill)
    out = data.clean(p)
    assert list(out.columns) == ["A"]
    assert out.loc[idx[4], "A"] == 2.0  # forward-filled from the last real value, never from the future

    train, test = data.split(p)
    assert len(train) and len(test) and len(train) + len(test) == len(p)
    assert train.index.max() < pd.Timestamp(data.SPLIT) <= test.index.min()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
