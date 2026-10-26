"""M13 many out-of-sample periods instead of one.

For every test year Y from FIRST_TEST_YEAR on, the whole research design is re-run as of Y-01-01:
pairs are re-selected and their hedge ratios re-fit on data before Y only, and the max-Sharpe weights
are re-fit on the strategy streams before Y. That fit is then evaluated on year Y alone. Stitching the
years gives one long, fully out-of-sample track record (anchored, expanding training window).

Momentum, MA 50/200, equal-weight 48 and SPY have no fitted parameters, so one full-history run of
each is simply sliced by year.
"""
import pandas as pd

import backtest
import data
import frontier
import momentum
import pairs

FIRST_TEST_YEAR = 2014  # first year with >= 4 years of training data (momentum needs 1 to warm up)


def fixed_strategies(prices: pd.DataFrame, rf: pd.Series, cost_bps: float):
    """Strategies with no fitted parameters: daily net returns over the full history, and the first
    day momentum holds a position (detected from weights: warm-up cash earns T-bills, so isn't zero)."""
    stocks = [t for t in prices.columns if t not in data.BENCHMARKS]
    run = lambda w: backtest.run(w, prices, cost_bps, rf)["net"]
    mom_w = momentum.cross_sectional(prices, stocks)
    live = prices.index[prices.index.get_loc(mom_w.sum(axis=1).gt(0).idxmax()) + 1]
    return {
        "Buy & hold SPY": run(momentum.buy_and_hold(prices, "SPY")),
        "Equal-weight 48": run(momentum.equal_weight(prices, stocks)),
        "XS momentum 12-1": run(mom_w),
        "MA 50/200 on SPY": run(momentum.ma_crossover(prices, "SPY")),
    }, live


def cutoff_fit(prices: pd.DataFrame, rf: pd.Series, cutoff: str, momentum_net: pd.Series,
               live: pd.Timestamp, cost_bps: float):
    """Re-fit the fitted parts of the design on data strictly before `cutoff`.
    Returns (net daily returns of the fitted strategies over the full history, chosen pairs, weights)."""
    train = prices[prices.index < cutoff]
    chosen = pairs.select(pairs.scan(train))
    prs = backtest.run(pairs.portfolio(prices, chosen), prices, cost_bps, rf)["net"]
    streams = pd.DataFrame({"Momentum": momentum_net, "Pairs": prs, "SPY": prices["SPY"].pct_change(),
                            "BND": prices["BND"].pct_change()}).loc[live:]
    ex = streams.sub(rf.reindex(streams.index).fillna(0.0), axis=0)
    w = frontier.max_sharpe(*frontier.annualize(ex[ex.index < cutoff]))
    equity = (1 + streams).cumprod()
    const = lambda v: pd.DataFrame([v.to_numpy()] * len(streams), index=streams.index, columns=streams.columns)
    out = {"Pairs portfolio (5)": prs,
           "Max-Sharpe (re-fit yearly)": backtest.run(const(w), equity, cost_bps, rf)["net"],
           "Equal-weight 4": backtest.run(const(pd.Series(0.25, index=streams.columns)), equity, cost_bps, rf)["net"]}
    return out, chosen, w


def yearly(prices: pd.DataFrame, rf: pd.Series, cost_bps: float, first_year: int = FIRST_TEST_YEAR):
    """Stitched out-of-sample daily net returns per strategy, plus one row of fit details per year.
    # ponytail: each year's fitted strategies are separate runs, so the turnover of switching pairs or
    # weights at a year boundary (about 1x capital, ~10 bps/yr) is not charged; stitch weights if it matters.
    """
    fixed, live = fixed_strategies(prices, rf, cost_bps)
    pieces, info = {}, []
    for year in range(first_year, prices.index[-1].year + 1):
        start, end = f"{year}-01-01", f"{year + 1}-01-01"
        fitted, chosen, w = cutoff_fit(prices, rf, start, fixed["XS momentum 12-1"], live, cost_bps)
        for k, r in (fixed | fitted).items():
            pieces.setdefault(k, []).append(r[(r.index >= start) & (r.index < end)])
        info.append({"year": year, "train days": int((prices.index < start).sum()),
                     "pairs selected": ", ".join(f"{r.a}/{r.b}" for r in chosen.itertuples()),
                     **{f"w {k}": v for k, v in w.items()}})
    return {k: pd.concat(v) for k, v in pieces.items()}, pd.DataFrame(info)
