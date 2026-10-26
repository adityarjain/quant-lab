"""Assert-based checks, one per module. Run: python tests.py"""
import numpy as np
import pandas as pd

import backtest
import data
import momentum
import frontier
import montecarlo as mc
import pairs


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


def test_pairs():
    rng = np.random.default_rng(data.SEED)
    n = 1500
    idx = pd.bdate_range("2012-01-01", periods=n)
    log_b = np.cumsum(rng.normal(0, 0.01, n)) + np.log(50)
    noise = np.zeros(n)                       # AR(1) spread, phi=0.9 -> half-life ~6.6 days
    for t in range(1, n):
        noise[t] = 0.9 * noise[t - 1] + rng.normal(0, 0.01)
    a = pd.Series(np.exp(0.5 + 1.3 * log_b + noise), idx)
    b = pd.Series(np.exp(log_b), idx)
    walk = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, n)) + np.log(50)), idx)

    assert pairs.coint_pvalue(a, b) < 0.01, "true cointegrated pair not detected"
    assert pairs.coint_pvalue(walk, b) > 0.05, "independent random walks flagged as cointegrated"
    assert abs(pairs.hedge_ratio(a, b) - 1.3) < 0.1  # finite-sample OLS: this path gives ~1.25
    assert 4 < pairs.half_life(np.log(a) - 1.3 * np.log(b)) < 10

    # State machine: enter beyond 2, hold, exit inside 0.5; stop beyond 4 and stay out until reset.
    z = pd.Series([0, 2.5, 1.5, 0.4, -2.5, -4.5, -3.0, -2.5, 0.2, -2.5])
    assert pairs.positions(z).tolist() == [0, -1, -1, 0, 1, 0, 0, 0, 0, 1]

    # Weights: gross exposure 1 when active, and no look-ahead.
    px = pd.DataFrame({"A": a, "B": b})
    w = pairs.pair_weights(px, "A", "B", 1.3)
    active = w[w.abs().sum(axis=1) > 0]
    assert len(active) and np.allclose(active.abs().sum(axis=1), 1.0)
    t = idx[1000]
    px2 = px.copy()
    px2.loc[px2.index > t, "A"] *= 3
    assert w.loc[:t].equals(pairs.pair_weights(px2, "A", "B", 1.3).loc[:t])


def test_frontier():
    names = ["A", "B", "C", "D"]
    mu = pd.Series([0.10, 0.06, 0.08, 0.03], index=names)
    vol = np.array([0.20, 0.10, 0.15, 0.05])
    cov = pd.DataFrame(np.diag(vol ** 2), index=names, columns=names)

    # Uncorrelated + no binding cap: tangency weights are proportional to mu / sigma^2 (closed form).
    w = frontier.max_sharpe(mu, cov, cap=1.0)
    exact = (mu / vol ** 2) / (mu / vol ** 2).sum()
    assert np.allclose(w, exact, atol=1e-4)

    # With the 0.4 cap: fully invested, long-only, cap respected, and no random portfolio beats it
    # among those that also respect the cap.
    for fn in (frontier.max_sharpe, frontier.min_variance):
        w = fn(mu, cov)
        assert np.isclose(w.sum(), 1) and (w >= -1e-9).all() and (w <= frontier.CAP + 1e-9).all()
    best = frontier.perf(frontier.max_sharpe(mu, cov), mu, cov)[2]
    rnd = np.random.default_rng(data.SEED).dirichlet(np.ones(4), 20_000)
    rnd = rnd[(rnd <= frontier.CAP).all(axis=1)]
    sharpe = (rnd @ mu.to_numpy()) / np.sqrt(np.einsum("ij,jk,ik->i", rnd, cov.to_numpy(), rnd))
    assert sharpe.max() <= best + 1e-6

    # Frontier: starts at min variance, vol rises with target return.
    f = frontier.frontier(mu, cov, points=15)
    assert len(f) >= 10 and f["vol"].is_monotonic_increasing
    assert np.isclose(f["vol"].iloc[0], frontier.perf(frontier.min_variance(mu, cov), mu, cov)[1], atol=1e-4)

    # Walk-forward uses only past data: changing the future leaves past weights unchanged.
    rets = _random_prices(n_days=1200, n_assets=4).pct_change().dropna()
    wf = frontier.walk_forward(rets, window=500)
    t = rets.index[800]
    rets2 = rets.copy()
    rets2.loc[rets2.index > t] *= -3
    assert np.allclose(wf.loc[:t], frontier.walk_forward(rets2, window=500).loc[:t])
    assert np.allclose(wf.iloc[600:].sum(axis=1), 1)


def test_montecarlo():
    rng = np.random.default_rng(data.SEED)
    # Fat-tailed history: Student-t with 3 dof, scaled to ~1% daily vol.
    hist = pd.Series(rng.standard_t(3, 5000) * 0.01 / np.sqrt(3), index=pd.bdate_range("2000-01-03", periods=5000))
    norm = mc.simulate_normal(hist, n_paths=2000, n_days=252)
    boot = mc.simulate_bootstrap(hist, n_paths=2000, n_days=252)

    # Normal VaR matches the closed form; bootstrap sees the fat tail the normal misses.
    v95 = mc.var_cvar(norm, 0.95)[0]
    assert abs(v95 - -(hist.mean() - 1.6449 * hist.std())) < 0.0005
    assert mc.var_cvar(boot, 0.99)[0] >= mc.var_cvar(norm, 0.99)[0]
    assert mc.var_cvar(boot, 0.99)[1] > mc.var_cvar(norm, 0.99)[1] * 1.1
    for d in (norm, boot):
        v, c = mc.var_cvar(d, 0.95)
        assert c >= v > 0  # expected shortfall is never smaller than VaR

    # Blocks really are contiguous slices of history.
    x = hist.to_numpy()
    start = int(np.flatnonzero(np.isclose(x, boot[0, 0]))[0])
    assert np.allclose(boot[0, :mc.BLOCK], x[start:start + mc.BLOCK])

    s = mc.summary(boot)
    assert s["5y p5"] <= s["5y p50"] <= s["5y p95"]
    assert s["maxDD p5 (worst 5%)"] <= s["maxDD median"] <= 0
    assert mc.max_drawdowns(np.full((3, 10), 0.01)).max() == 0  # always up -> no drawdown
    assert np.isclose(mc.max_drawdowns(np.array([[0.1, -0.5, 0.2]]))[0], -0.5)


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


def test_app():
    """Dashboard smoke test: runs headless, no errors, and its defaults reproduce the published
    test-period Sharpe ratios (so the UI can't silently drift from run.py)."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py", default_timeout=300).run()
    assert not at.exception and not at.error, [e.value for e in at.exception]
    sharpe = lambda i, name: at.dataframe[i].value.query("period == 'test' and strategy == @name")["Sharpe (net)"].item()
    assert round(sharpe(0, "XS momentum 12-1"), 2) == 0.87
    assert round(sharpe(1, "Pairs portfolio"), 2) == 0.27
    assert round(sharpe(3, "Equal-weight 4"), 2) == 0.97 and round(sharpe(3, "Max-Sharpe (train fit)"), 2) == 0.85

    at.sidebar.slider[0].set_value(0).run()  # zero costs -> net == gross through the same engine
    assert not at.exception
    m1 = at.dataframe[0].value
    assert np.allclose(m1["Sharpe (net)"], m1["Sharpe (gross)"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
