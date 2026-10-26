"""M6 interactive dashboard. Run: streamlit run app.py

UI only: every number comes from the same tested modules run.py uses. Defaults = the research
settings in the README, so the dashboard opens on exactly the published results.
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import backtest
import data
import frontier
import momentum
import montecarlo as mc
import pairs
from run import COST_BPS, MUTED, SERIES, metrics_table, sector_streams, strategy_streams

st.set_page_config(page_title="Quant Portfolio Lab", layout="wide")
SPLIT = pd.Timestamp(data.SPLIT)
PCT_COLS = ("CAGR (net)", "vol", "maxDD")


@st.cache_data
def load_prices():
    return data.load_prices()


@st.cache_data
def scanned():
    return pairs.scan(data.split(load_prices())[0])


@st.cache_data
def streams(cost_bps: float):
    return strategy_streams(load_prices(), cost_bps)


def show_table(df: pd.DataFrame, pct=PCT_COLS, digits: int = 2):
    fmt = {c: "{:.1%}" for c in df.columns if c in pct}
    fmt |= {c: f"{{:.{digits}f}}" for c in df.select_dtypes("number").columns if c not in fmt}
    st.dataframe(df.style.format(fmt, na_rep="–"), hide_index=True, width="stretch")


def colors(names):
    return alt.Scale(domain=list(names), range=SERIES[: len(names)])


def line_chart(curves: dict, y_title: str, log: bool = False, split: bool = True, height: int = 360):
    """Up to 4 series in the validated palette, hover tooltips, dashed train|test divider."""
    df = pd.DataFrame(curves)
    df.index.name = "date"
    long = df.reset_index().melt("date", var_name="series", value_name="value").dropna()
    y = alt.Y("value:Q", title=y_title, scale=alt.Scale(type="log", nice=False) if log else alt.Scale(zero=False))
    chart = alt.Chart(long).mark_line(strokeWidth=1.6).encode(
        x=alt.X("date:T", title=None), y=y, color=alt.Color("series:N", scale=colors(curves), title=None),
        tooltip=[alt.Tooltip("date:T"), "series:N", alt.Tooltip("value:Q", format=".3f")])
    if split:
        chart += alt.Chart(pd.DataFrame({"date": [SPLIT]})).mark_rule(strokeDash=[4, 4], color=MUTED).encode(x="date:T")
    return chart.properties(height=height).interactive(bind_y=False)


def equity_curves(results: dict, key: str = "net") -> dict:
    return {k: (1 + r[key]).cumprod() for k, r in results.items()}


prices = load_prices()
stocks = [t for t in prices.columns if t not in data.BENCHMARKS]

st.title("Quant Portfolio Lab")
st.caption("Momentum, pairs, mean-variance and Monte Carlo on 48 US large caps + SPY + BND. "
           f"Train {prices.index[0].date()} → 2018-12-31, test {data.SPLIT} → {prices.index[-1].date()}. "
           "Dashed line in every chart = train | test split.")
with st.sidebar:
    st.header("Settings")
    cost = st.slider("Transaction cost (bps per unit turnover)", 0, 50, COST_BPS)
    st.warning("Defaults are the research settings from the README. A parameter chosen *after* looking at "
               "2019+ results is fit on the test period, so those numbers are no longer out-of-sample.")

tabs = st.tabs(["Verdict", "M1 · Momentum", "M2 · Pairs", "M3 · Frontier", "M4 · Monte Carlo"])

# ---------------------------------------------------------------- Verdict
with tabs[0]:
    st.subheader("Out-of-sample verdict (research defaults, test 2019+, net of 10 bps)")
    st.markdown((Path(__file__).parent / "figures" / "summary.md").read_text())
    st.markdown("**No strategy beat a passive benchmark out of sample.** Momentum didn't beat equal-weighting its own "
                "universe; the pairs screen found chance-level cointegration that vanished after 2018; the optimizer "
                "lost to a naive 25%-each mix on Sharpe (but cut drawdown by two thirds); and both risk models "
                "under-predicted post-2018 tail losses. Use the tabs to see how sensitive each result is.")

# ---------------------------------------------------------------- M1
with tabs[1]:
    c1, c2, c3, c4 = st.columns(4)
    lookback = c1.slider("Lookback (months)", 3, 24, 12)
    skip = c2.slider("Skip most recent (months)", 0, 3, 1)
    top = c3.slider("Hold top fraction", 0.1, 0.5, 0.2, 0.05)
    fast, slow = c4.slider("MA fast / slow (days)", 10, 300, (50, 200), 10)
    if skip >= lookback or fast >= slow:
        st.error("Skip must be shorter than the lookback, and the fast MA shorter than the slow one.")
    else:
        strategies = {
            f"XS momentum {lookback}-{skip}": momentum.cross_sectional(prices, stocks, lookback * 21, skip * 21, top),
            f"MA {fast}/{slow} on SPY": momentum.ma_crossover(prices, "SPY", fast, slow),
            "Equal-weight 48": momentum.equal_weight(prices, stocks),
            "Buy & hold SPY": momentum.buy_and_hold(prices, "SPY"),
        }
        res = {k: backtest.run(w, prices, cost) for k, w in strategies.items()}
        st.altair_chart(line_chart(equity_curves(res), "Growth of $1 (log, net)", log=True), width="stretch")
        show_table(metrics_table(res))

# ---------------------------------------------------------------- M2
with tabs[2]:
    scan = scanned()
    c1, c2, c3, c4, c5 = st.columns(5)
    n_pairs = c1.slider("Pairs to trade", 1, 6, 5)
    window = c2.slider("z-score window (days)", 20, 250, pairs.WINDOW, 10)
    entry = c3.slider("Entry |z|", 1.0, 3.5, pairs.ENTRY, 0.25)
    exit_ = c4.slider("Exit |z|", 0.0, 1.5, pairs.EXIT, 0.25)
    stop = c5.slider("Stop |z|", 2.5, 8.0, pairs.STOP, 0.5)
    rules = dict(window=window, entry=entry, exit=exit_, stop=stop)
    chosen = pairs.select(scan, n=n_pairs)
    labels = [f"{r.a}/{r.b}" for r in chosen.itertuples()]

    left, right = st.columns(2)
    with left:
        st.markdown(f"**120 within-sector pairs (train).** {(scan['pvalue'] < 0.05).sum()} pass at 5%, "
                    f"~{0.05 * len(scan):.0f} expected by pure chance.")
        sc = scan.assign(pair=scan["a"] + "/" + scan["b"],
                         status=np.where(scan["pvalue"] < 0.05, "cointegrated at 5%", "not cointegrated"))
        st.altair_chart(alt.Chart(sc).mark_circle(size=60, opacity=0.9).encode(
            x=alt.X("corr:Q", title="Correlation of daily returns", scale=alt.Scale(zero=False)),
            y=alt.Y("pvalue:Q", title="Engle-Granger p-value (log)", scale=alt.Scale(type="log")),
            color=alt.Color("status:N", scale=alt.Scale(domain=["cointegrated at 5%", "not cointegrated"],
                                                        range=[SERIES[0], MUTED]), title=None),
            tooltip=["pair", "sector", alt.Tooltip("corr:Q", format=".2f"), alt.Tooltip("pvalue:Q", format=".3f"),
                     alt.Tooltip("beta:Q", format=".2f"), alt.Tooltip("half_life:Q", format=".0f")],
        ).properties(height=330), width="stretch")
    with right:
        pick = st.selectbox("Inspect pair", labels)
        row = chosen.iloc[labels.index(pick)]
        _, z = pairs.spread_z(prices, row.a, row.b, row.beta, window)
        zdf = pd.DataFrame({"date": z.index, "z": z.values}).dropna()
        zline = alt.Chart(zdf).mark_line(strokeWidth=0.8, color=SERIES[0]).encode(
            x=alt.X("date:T", title=None), y=alt.Y("z:Q", title=f"z-score of log {row.a} − {row.beta:.2f}·log {row.b}"))
        th = pd.DataFrame({"y": [entry, -entry, exit_, -exit_, stop, -stop],
                           "rule": ["entry"] * 2 + ["exit"] * 2 + ["stop"] * 2})
        rules_chart = alt.Chart(th).mark_rule(strokeDash=[4, 3]).encode(
            y="y:Q", color=alt.Color("rule:N", scale=alt.Scale(domain=["entry", "exit", "stop"],
                                                               range=SERIES[1:4]), title=None))
        split_rule = alt.Chart(pd.DataFrame({"date": [SPLIT]})).mark_rule(strokeDash=[4, 4], color=MUTED).encode(x="date:T")
        st.altair_chart((zline + rules_chart + split_rule).properties(height=300).interactive(bind_y=False),
                        width="stretch")

    res = {"Pairs portfolio": backtest.run(pairs.portfolio(prices, chosen, **rules), prices, cost)}
    res |= {lab: backtest.run(pairs.pair_weights(prices, r.a, r.b, r.beta, **rules), prices, cost)
            for lab, r in zip(labels, chosen.itertuples())}
    curves = {"Portfolio (net)": (1 + res["Pairs portfolio"]["net"]).cumprod(),
              "Portfolio (gross)": (1 + res["Pairs portfolio"]["gross"]).cumprod(),
              f"{pick} (net)": (1 + res[pick]["net"]).cumprod()}
    st.altair_chart(line_chart(curves, "Growth of $1"), width="stretch")
    show_table(metrics_table(res))
    test_px = data.split(prices)[1]
    test_p = [pairs.coint_pvalue(test_px[r.a], test_px[r.b]) for r in chosen.itertuples()]
    st.caption("Test-period cointegration p-values of the selected pairs: "
               + ", ".join(f"{lab} {p:.2f}" for lab, p in zip(labels, test_p)) + " (above 0.05 = relationship gone).")

# ---------------------------------------------------------------- M3
with tabs[3]:
    c1, c2 = st.columns([2, 1])
    universe = c1.radio("Universe", ["Strategies + SPY + BND", "8 sector baskets + BND"], horizontal=True)
    rets = streams(cost) if universe.startswith("Strategies") else sector_streams(prices)
    train_r = data.split(rets)[0]
    cap = c2.slider("Max weight per asset", round(1 / rets.shape[1] + 0.005, 2), 1.0, 0.4, 0.05)
    mu, cov = frontier.annualize(train_r)
    ms_w, mv_w = frontier.max_sharpe(mu, cov, cap), frontier.min_variance(mu, cov, cap)
    cloud = frontier.random_portfolios(mu, cov, n=5000, seed=data.SEED, cap=cap)
    front = frontier.frontier(mu, cov, cap, points=30)
    pts = pd.DataFrame([dict(zip(("ret", "vol", "sharpe"), frontier.perf(w, mu, cov)), name=n)
                        for n, w in (("max Sharpe", ms_w), ("min variance", mv_w))])
    assets = pd.DataFrame({"name": mu.index, "ret": mu.values, "vol": np.sqrt(np.diag(cov))}).sort_values("vol")
    assets["dy"] = np.where(np.arange(len(assets)) % 2, 10, -8)  # alternate above/below so near neighbours don't collide

    left, right = st.columns([3, 2])
    with left:
        base = alt.Chart(cloud).mark_circle(size=6, opacity=0.5).encode(
            x=alt.X("vol:Q", title="Volatility (train, annualized)", axis=alt.Axis(format="%")),
            y=alt.Y("ret:Q", title="Expected return (train, annualized)", axis=alt.Axis(format="%")),
            color=alt.Color("sharpe:Q", scale=alt.Scale(range=["#184f95", "#9ec5f4"]), title="Sharpe"))
        fl = alt.Chart(front).mark_line(color="#ffffff", strokeWidth=2).encode(x="vol:Q", y="ret:Q")
        pt = alt.Chart(pts).mark_point(size=160, filled=True, strokeWidth=0).encode(
            x="vol:Q", y="ret:Q", color=alt.Color("name:N", scale=alt.Scale(domain=["max Sharpe", "min variance"],
                                                                         range=[SERIES[1], SERIES[2]]), title=None),
            tooltip=["name", alt.Tooltip("ret:Q", format=".1%"), alt.Tooltip("vol:Q", format=".1%"),
                     alt.Tooltip("sharpe:Q", format=".2f")])
        ad = alt.Chart(assets).mark_circle(size=40, color="#c3c2b7").encode(x="vol:Q", y="ret:Q")
        at = alt.layer(*[alt.Chart(assets[assets["dy"] == dy]).mark_text(align="left", dx=6, dy=int(dy), color="#c3c2b7")
                         .encode(x="vol:Q", y="ret:Q", text="name") for dy in (-8, 10)])
        st.altair_chart(alt.layer(base, fl, ad, at, pt).resolve_scale(color="independent").properties(height=430),
                        width="stretch")
    with right:
        st.markdown(f"**Train-fitted weights** (cap {cap:.0%})")
        show_table(pd.DataFrame({"asset": mu.index, "max Sharpe": ms_w.values, "min variance": mv_w.values}),
                   pct=("max Sharpe", "min variance"))
        st.metric("Max-Sharpe in-sample Sharpe", f"{frontier.perf(ms_w, mu, cov)[2]:.2f}")

    if universe.startswith("Strategies"):
        equity = (1 + rets).cumprod()
        const = lambda w: pd.DataFrame([w.to_numpy()] * len(rets), index=rets.index, columns=rets.columns)
        wf = frontier.walk_forward(rets, cap=cap)
        live0 = wf.sum(axis=1).gt(0).idxmax()
        allocs = {"Max-Sharpe (train fit)": const(ms_w), "Walk-forward max-Sharpe": wf,
                  "Equal-weight 4": const(pd.Series(0.25, index=rets.columns)),
                  "SPY": const(pd.Series([0, 0, 1, 0], index=rets.columns, dtype=float))}
        res = {k: backtest.run(w, equity, cost).loc[live0:] for k, w in allocs.items()}
        st.altair_chart(line_chart({k: (1 + data.split(v)[1]["net"]).cumprod() for k, v in res.items()},
                                   "Growth of $1 (test only, net)", split=False), width="stretch")
        show_table(metrics_table(res))

# ---------------------------------------------------------------- M4
with tabs[4]:
    c1, c2, c3, c4 = st.columns(4)
    pname = c1.selectbox("Portfolio", ["Max-Sharpe (M3)", "Equal-weight 4", "SPY"])
    n_paths = c2.slider("Paths", 1000, 10000, 5000, 1000)
    years = c3.slider("Horizon (years)", 1, 10, mc.YEARS)
    block = c4.slider("Bootstrap block (days)", 1, 126, mc.BLOCK)
    train_s, test_s = data.split(streams(cost))
    w = {"Max-Sharpe (M3)": frontier.max_sharpe(*frontier.annualize(train_s)),
         "Equal-weight 4": pd.Series(0.25, index=train_s.columns),
         "SPY": pd.Series([0, 0, 1, 0], index=train_s.columns, dtype=float)}[pname]
    r_train, r_test = train_s @ w, test_s @ w
    n_days = years * mc.TRADING_DAYS
    sims = {"Normal": mc.simulate_normal(r_train, n_paths, n_days, seed=data.SEED),
            "Bootstrap": mc.simulate_bootstrap(r_train, n_paths, n_days, block, seed=data.SEED)}

    t = np.arange(n_days + 1) / mc.TRADING_DAYS
    fan = pd.concat([pd.DataFrame({"years": t, "value": np.quantile(mc.equity(d), q, axis=0),
                                   "line": f"{name} p{int(q * 100)}", "model": name})
                     for name, d in sims.items() for q in (0.05, 0.5, 0.95)])
    left, right = st.columns(2)
    with left:
        st.altair_chart(alt.Chart(fan).mark_line(strokeWidth=1.6).encode(
            x=alt.X("years:Q", title="Years"), y=alt.Y("value:Q", title="Portfolio value", scale=alt.Scale(zero=False)),
            color=alt.Color("model:N", scale=colors(sims), title=None), detail="line:N",
            strokeDash=alt.condition("indexof(datum.line, 'p50') >= 0", alt.value([1, 0]), alt.value([5, 3])),
            tooltip=["line:N", alt.Tooltip("years:Q", format=".1f"), alt.Tooltip("value:Q", format=".3f")],
        ).properties(height=340, title="5th / 50th / 95th percentile paths (dashed = 5th/95th)"), width="stretch")
    with right:
        term = pd.concat([pd.DataFrame({"model": n, "ret": mc.equity(d)[:, -1] - 1}) for n, d in sims.items()])
        st.altair_chart(alt.Chart(term).mark_bar(opacity=0.55, binSpacing=0).encode(
            x=alt.X("ret:Q", bin=alt.Bin(maxbins=70), title=f"{years}-year return", axis=alt.Axis(format="%")),
            y=alt.Y("count():Q", stack=None, title="Paths"),
            color=alt.Color("model:N", scale=colors(sims), title=None),
        ).properties(height=340, title="Terminal return distribution"), width="stretch")

    rows = []
    for name, d in sims.items():
        s = mc.summary(d)
        rows.append({"model": name, **{k: s[k] for k in ("1d VaR95", "1d VaR99", "1d CVaR99", "maxDD median",
                                                          "maxDD p5 (worst 5%)", "P(5y loss)")},
                     "VaR95 breached in test (target 5%)": mc.exceedance_rate(r_test, s["1d VaR95"]),
                     "VaR99 breached in test (target 1%)": mc.exceedance_rate(r_test, s["1d VaR99"])})
    real = mc.var_cvar(r_test.to_numpy(), 0.99)
    risk = pd.DataFrame(rows).rename(columns={"P(5y loss)": f"P(loss over {years}y)"})
    show_table(risk, pct=[c for c in risk.columns if c != "model"])
    st.markdown(f"**Realized in test:** 1d VaR99 **{real[0]:.1%}**, 1d CVaR99 **{real[1]:.1%}**, versus the model "
                "forecasts above. Models are fit on train-period portfolio returns; breach rates use the test period.")
