"""Regenerates every figure and results table. Run: python run.py"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

import backtest
import data
import factors
import frontier
import momentum
import montecarlo as mc
import pairs
import significance
import walkforward

FIG = Path(__file__).parent / "figures"
COST_BPS = 10

# Dark chart theme; categorical slots validated (lightness, chroma, CVD, contrast) against the surface.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#1a1a19", "#ffffff", "#c3c2b7", "#898781", "#2c2c2a", "#383835"
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500"]
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID, "axes.grid": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "lines.linewidth": 1.6,
})


def excess(returns: pd.DataFrame, rf: pd.Series) -> pd.DataFrame:
    """Returns minus the daily T-bill rate, row by row."""
    return returns.sub(rf.reindex(returns.index).fillna(0.0), axis=0)


def metrics_table(results: dict[str, pd.DataFrame], rf: pd.Series | None = None) -> pd.DataFrame:
    """Train/test rows per strategy; Sharpe is in excess of `rf`."""
    rows = []
    for name, res in results.items():
        for period, part in zip(("train", "test"), data.split(res)):
            g = backtest.metrics(part["gross"], rf=rf)
            n = backtest.metrics(part["net"], part["turnover"], rf=rf)
            rows.append({"strategy": name, "period": period, "Sharpe (gross)": g["Sharpe"],
                         "Sharpe (net)": n["Sharpe"], "CAGR (net)": n["CAGR"], "vol": n["vol"],
                         "maxDD": n["maxDD"], "turnover/yr": n["turnover"]})
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame, pct=("CAGR (net)", "vol", "maxDD"), digits: int = 2) -> str:
    fmt = lambda c, v: f"{v:.1%}" if c in pct else (f"{v:.{digits}f}" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(c, v) for c, v in row.items()) + " |" for _, row in df.iterrows()]
    return "\n".join(lines) + "\n"


def split_line(*axes):
    """Dashed train|test divider on every axis, labelled on the first."""
    for a in axes:
        a.axvline(pd.Timestamp(data.SPLIT), color=MUTED, linestyle="--", linewidth=1)
    axes[0].text(pd.Timestamp(data.SPLIT), axes[0].get_ylim()[1], "train | test", color=INK2, va="top",
                 ha="center", fontsize=9, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2))


def m1(prices: pd.DataFrame) -> pd.DataFrame:
    stocks = [t for t in prices.columns if t not in data.BENCHMARKS]
    strategies = {
        "XS momentum 12-1": momentum.cross_sectional(prices, stocks),
        "MA 50/200 on SPY": momentum.ma_crossover(prices, "SPY"),
        "Equal-weight 48": momentum.equal_weight(prices, stocks),
        "Buy & hold SPY": momentum.buy_and_hold(prices, "SPY"),
    }
    rf = data.load_rf(prices.index)
    results = {k: backtest.run(w, prices, COST_BPS, rf) for k, w in strategies.items()}

    fig, (ax, dd) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ends = {}
    for (name, res), color in zip(results.items(), SERIES):
        eq = (1 + res["net"]).cumprod()
        ax.plot(eq.index, eq, color=color, label=name)
        ends[name] = eq.iloc[-1]
    # End labels, pushed apart in log space so close finishers don't overlap.
    y_prev = None
    for name, y in sorted(ends.items(), key=lambda kv: kv[1]):
        y_lab = y if y_prev is None else max(y, y_prev * 1.12)
        ax.annotate(f"{name}  {y:.1f}x", (results[name].index[-1], y_lab), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK2, annotation_clip=False)
        y_prev = y_lab
    for name, color in (("XS momentum 12-1", SERIES[0]), ("Buy & hold SPY", SERIES[3])):
        d = backtest.drawdown(results[name]["net"])
        dd.plot(d.index, d * 100, color=color, linewidth=1.2, label=name)
    split_line(ax, dd)
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 3, 5, 10], labels=["1x", "2x", "3x", "5x", "10x"])
    ax.minorticks_off()
    ax.set_ylabel("Growth of $1 (log, net of costs)")
    ax.set_title(f"M1 · Momentum vs benchmarks, net of {COST_BPS} bps per unit turnover", loc="left")
    ax.legend(loc="upper left")
    dd.set_ylabel("Drawdown (%)")
    dd.legend(loc="lower left", fontsize=8)
    fig.subplots_adjust(right=0.82, hspace=0.08)
    fig.savefig(FIG / "m1_momentum.png", dpi=150)
    plt.close(fig)

    table = metrics_table(results, rf)
    (FIG / "m1_metrics.md").write_text(to_markdown(table))
    return table, results


def m2(prices: pd.DataFrame) -> pd.DataFrame:
    train, test = data.split(prices)
    scanned = pairs.scan(train)
    chosen = pairs.select(scanned)
    chosen["p_test"] = [pairs.coint_pvalue(test[r.a], test[r.b]) for r in chosen.itertuples()]

    # Correlation vs cointegration: the scan in one picture.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    passed = scanned["pvalue"] < 0.05
    ax.scatter(scanned.loc[~passed, "corr"], scanned.loc[~passed, "pvalue"], s=28, color=MUTED,
               edgecolor=SURFACE, linewidth=1, label=f"not cointegrated ({(~passed).sum()})")
    ax.scatter(scanned.loc[passed, "corr"], scanned.loc[passed, "pvalue"], s=48, color=SERIES[0],
               edgecolor=SURFACE, linewidth=1, label=f"cointegrated at 5% ({passed.sum()})")
    for r in chosen.itertuples():
        ax.annotate(f"{r.a}/{r.b}", (r.corr, r.pvalue), xytext=(6, -2), textcoords="offset points",
                    fontsize=8, color=INK2)
    ax.axhline(0.05, color=SERIES[1], linestyle="--", linewidth=1)
    ax.text(scanned["corr"].min(), 0.05, " p = 0.05", color=INK2, va="bottom", fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("Correlation of daily returns (train)")
    ax.set_ylabel("Engle-Granger p-value (log, lower = stronger)")
    ax.set_title("M2 · 120 within-sector pairs: high correlation ≠ cointegration", loc="left")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "m2_corr_vs_coint.png", dpi=150)
    plt.close(fig)

    strategies = {"Pairs portfolio (5)": pairs.portfolio(prices, chosen)}
    strategies |= {f"{r.a}/{r.b}": pairs.pair_weights(prices, r.a, r.b, r.beta) for r in chosen.itertuples()}
    rf = data.load_rf(prices.index)
    results = {k: backtest.run(w, prices, COST_BPS, rf) for k, w in strategies.items()}

    # 4-panel walkthrough of the strongest pair + the portfolio equity.
    top = chosen.iloc[0]
    spread, z = pairs.spread_z(prices, top.a, top.b, top.beta)
    roll = spread.rolling(pairs.WINDOW)
    fig, axs = plt.subplots(4, 1, figsize=(11, 11), sharex=True, gridspec_kw={"height_ratios": [2, 2, 1.6, 2]})
    for t, c in ((top.a, SERIES[0]), (top.b, SERIES[1])):
        axs[0].plot(prices.index, prices[t] / prices[t].iloc[0], color=c, linewidth=1.2, label=t)
    axs[0].set_ylabel("Price (normalized)")
    axs[0].set_title(f"M2 · Pairs walkthrough: {top.a}/{top.b}  (β = {top.beta:.2f}, train p = {top.pvalue:.3f}, "
                     f"test p = {top.p_test:.2f})", loc="left")
    axs[0].legend(loc="upper left")
    axs[1].plot(spread.index, spread, color=INK2, linewidth=1, label=f"spread = log {top.a} − β·log {top.b}")
    axs[1].plot(spread.index, roll.mean(), color=SERIES[1], linewidth=1.2, label=f"{pairs.WINDOW}d mean")
    for sign in (1, -1):
        axs[1].plot(spread.index, roll.mean() + sign * pairs.ENTRY * roll.std(), color=MUTED, linewidth=0.8,
                    linestyle="--", label="±2σ" if sign == 1 else None)
    axs[1].set_ylabel("Spread")
    axs[1].legend(loc="upper left", fontsize=8)
    axs[2].plot(z.index, z, color=SERIES[0], linewidth=0.8)
    for lvl, c, lab in ((pairs.ENTRY, SERIES[1], "entry ±2"), (pairs.EXIT, SERIES[2], "exit ±0.5"),
                        (pairs.STOP, SERIES[3], "stop ±4")):
        for sign in (1, -1):
            axs[2].axhline(sign * lvl, color=c, linestyle="--", linewidth=0.9, label=lab if sign == 1 else None)
    axs[2].set_ylabel("z-score")
    axs[2].legend(loc="upper left", fontsize=8, ncol=3)
    for (name, key), c in zip((("Pairs portfolio (5)", "net"), ("Pairs portfolio (5)", "gross"),
                               (f"{top.a}/{top.b}", "net")), SERIES):
        eq = (1 + results[name][key]).cumprod()
        axs[3].plot(eq.index, eq, color=c, linewidth=1.4, label=f"{name} ({key})")
    axs[3].set_ylabel("Growth of $1")
    axs[3].legend(loc="upper left", fontsize=8)
    split_line(*axs)
    fig.tight_layout()
    fig.savefig(FIG / "m2_pairs.png", dpi=150)
    plt.close(fig)

    sel = chosen[["sector", "a", "b", "corr", "pvalue", "p_test", "beta", "half_life"]]
    (FIG / "m2_pairs.md").write_text(to_markdown(sel, pct=(), digits=3))
    table = metrics_table(results, rf)
    (FIG / "m2_metrics.md").write_text(to_markdown(table))
    print(f"scan: {len(scanned)} pairs, {passed.sum()} cointegrated at 5% "
          f"(~{0.05 * len(scanned):.0f} expected by chance); corr>0.7: {(scanned['corr'] > 0.7).sum()}, "
          f"of which not cointegrated: {((scanned['corr'] > 0.7) & ~passed).sum()}")
    print(sel.round(3).to_string(index=False))
    return table, results


def strategy_streams(prices: pd.DataFrame, cost_bps: float = COST_BPS) -> pd.DataFrame:
    """Daily NET returns of the building blocks the M3 optimizer allocates across."""
    stocks = [t for t in prices.columns if t not in data.BENCHMARKS]
    train, _ = data.split(prices)
    rf = data.load_rf(prices.index)
    mom_w = momentum.cross_sectional(prices, stocks)
    mom = backtest.run(mom_w, prices, cost_bps, rf)["net"]
    prs = backtest.run(pairs.portfolio(prices, pairs.select(pairs.scan(train))), prices, cost_bps, rf)["net"]
    streams = pd.DataFrame({"Momentum": mom, "Pairs": prs, "SPY": prices["SPY"].pct_change(),
                            "BND": prices["BND"].pct_change()})
    # Drop momentum's 12-month warm-up (all cash) so its stats aren't diluted. Detected from positions,
    # not returns: idle cash now earns T-bills, so warm-up returns are no longer exactly zero.
    first_signal = mom_w.index.get_loc(mom_w.sum(axis=1).gt(0).idxmax())
    return streams.iloc[first_signal + 1:]  # positions are held from the day after the signal


def sector_streams(prices: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight sector baskets + BND: the 'raw assets' view of the frontier."""
    rets = prices.pct_change()
    out = pd.DataFrame({s: rets[names].mean(axis=1) for s, names in data.SECTORS.items()})
    out["BND"] = rets["BND"]
    return out.dropna()


def frontier_panel(ax, returns: pd.DataFrame, title: str):
    """Random cloud (sequential blue = Sharpe), capped frontier, max-Sharpe/min-var, CML, assets."""
    from matplotlib.colors import LinearSegmentedColormap
    seq = LinearSegmentedColormap.from_list("blue", ["#184f95", "#3987e5", "#9ec5f4"])
    mu, cov = frontier.annualize(returns)
    cloud = frontier.random_portfolios(mu, cov, seed=data.SEED)
    sc = ax.scatter(cloud["vol"], cloud["ret"], c=cloud["sharpe"], cmap=seq, s=3, alpha=0.6, linewidths=0)
    f = frontier.frontier(mu, cov)
    ax.plot(f["vol"], f["ret"], color=INK, linewidth=2, label=f"efficient frontier (cap {frontier.CAP:.0%})")
    ms_w, mv_w = frontier.max_sharpe(mu, cov), frontier.min_variance(mu, cov)
    r, v, s = frontier.perf(ms_w, mu, cov)
    ax.plot([0, v * 1.6], [0, r * 1.6], color=SERIES[3], linestyle="--", linewidth=1, label="capital market line (from T-bills)")
    # Max Sharpe drawn as a larger ring under min variance, so both stay visible if they coincide.
    ax.scatter([v], [r], s=200, facecolor="none", edgecolor=SERIES[1], linewidth=2.5, zorder=5,
               label=f"max Sharpe ({s:.2f})")
    mv = frontier.perf(mv_w, mu, cov)
    ax.scatter([mv[1]], [mv[0]], s=60, color=SERIES[2], edgecolor=SURFACE, linewidth=1.5, zorder=6, label="min variance")
    vols = pd.Series(np.sqrt(np.diag(cov)), index=mu.index)
    ax.scatter(vols, mu, s=22, color=INK2, zorder=4)
    ax.set_xlim(0, vols.max() * 1.25)  # room for the right-most label
    span_x, span_y = ax.get_xlim()[1], mu.max() - min(0, mu.min())
    placed = []
    for name in mu.sort_values().index:  # labels that would collide go below their point instead
        x, y = vols[name], mu[name]
        clash = any(abs(x - px) < 0.06 * span_x and abs(y - py) < 0.04 * span_y for px, py in placed)
        ax.annotate(name, (x, y), xytext=(5, -11 if clash else 3), textcoords="offset points", fontsize=8, color=INK2)
        placed.append((x, y))
    ax.set_xlabel("Volatility (annualized)")
    ax.set_ylabel("Expected excess return over T-bills (annualized)")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left", fontsize=8)
    return sc, ms_w, mv_w


def m3(prices: pd.DataFrame) -> pd.DataFrame:
    streams = strategy_streams(prices)
    rf = data.load_rf(prices.index)
    train_s, test_s = data.split(streams)
    sectors_train, _ = data.split(excess(sector_streams(prices), rf))

    # Optimize on returns in excess of T-bills, so max Sharpe is the true tangency portfolio.
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))
    frontier_panel(a1, sectors_train, "M3a · 8 sector baskets + BND (train)")
    sc, ms_w, mv_w = frontier_panel(a2, excess(train_s, rf), "M3b · Strategies + SPY + BND (train)")
    fig.subplots_adjust(right=0.9, wspace=0.18)
    fig.colorbar(sc, cax=fig.add_axes([0.92, 0.15, 0.012, 0.7]), label="Sharpe of random portfolio (capped)")
    fig.savefig(FIG / "m3_frontier.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Hold train-fitted weights through test; compare with walk-forward re-fitting and naive mixes.
    # Each stream's equity curve is treated as an asset, so the same engine charges rebalancing costs.
    equity = (1 + streams).cumprod()
    const = lambda w: pd.DataFrame([w.to_numpy()] * len(streams), index=streams.index, columns=streams.columns)
    wf = frontier.walk_forward(excess(streams, rf))
    allocs = {
        "Max-Sharpe (train fit)": const(ms_w),
        "Walk-forward max-Sharpe": wf,
        "Equal-weight 4": const(pd.Series(0.25, index=streams.columns)),
        "SPY": const(pd.Series([0, 0, 1, 0], index=streams.columns, dtype=float)),
    }
    live0 = wf.sum(axis=1).gt(0).idxmax()  # walk-forward needs 3 years of history; compare all from then
    results = {k: backtest.run(w, equity, COST_BPS, rf).loc[live0:] for k, w in allocs.items()}

    fig, axs = plt.subplots(3, 1, figsize=(11, 11), gridspec_kw={"height_ratios": [1.2, 1.6, 2]})
    ax = axs[0]
    y = np.arange(len(ms_w))
    ax.barh(y + 0.2, ms_w, height=0.38, color=SERIES[1], label="max Sharpe")
    ax.barh(y - 0.2, mv_w, height=0.38, color=SERIES[2], label="min variance")
    for i, (a, b) in enumerate(zip(ms_w, mv_w)):
        ax.text(a + 0.005, i + 0.2, f"{a:.0%}", va="center", fontsize=8, color=INK2)
        ax.text(b + 0.005, i - 0.2, f"{b:.0%}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, ms_w.index)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlim(0, frontier.CAP + 0.05)
    ax.axvline(frontier.CAP, color=MUTED, linestyle=":", linewidth=1)
    ax.set_title("M3 · Train-fitted weights (cap 40%)", loc="left")
    ax.legend(loc="upper right", fontsize=8)

    ax = axs[1]
    live = wf[wf.sum(axis=1) > 0]
    ax.stackplot(live.index, live.T.to_numpy(), labels=live.columns, colors=SERIES, edgecolor=SURFACE, linewidth=0.5)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title("Walk-forward weights (re-fit quarterly, trailing 3y)", loc="left")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.08), fontsize=8, ncol=4)
    split_line(ax)
    ax.set_title(ax.get_title(loc="left"), loc="left", pad=22)

    ax = axs[2]
    for (name, res), c in zip(results.items(), SERIES):
        part = data.split(res)[1]["net"]
        eq = (1 + part).cumprod()
        ax.plot(eq.index, eq, color=c, label=f"{name}  {eq.iloc[-1]:.2f}x")
    ax.set_ylabel("Growth of $1 (test, net)")
    ax.set_title("Out-of-sample 2019–2026: does optimization beat naive mixes?", loc="left")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "m3_allocation.png", dpi=150)
    plt.close(fig)

    corr = train_s.corr()
    (FIG / "m3_corr.md").write_text(to_markdown(corr.reset_index(names=""), pct=(), digits=2))
    (FIG / "m3_weights.md").write_text(to_markdown(
        pd.DataFrame({"asset": ms_w.index, "max Sharpe": ms_w.values, "min variance": mv_w.values}),
        pct=("max Sharpe", "min variance")))
    table = metrics_table(results, rf)
    (FIG / "m3_metrics.md").write_text(to_markdown(table))
    print("train correlations:\n", corr.round(2).to_string())
    print("weights:\n", pd.DataFrame({"max_sharpe": ms_w, "min_var": mv_w}).round(3).to_string())
    swing = wf.loc[wf.sum(axis=1) > 0].agg(["min", "max"]).round(2)
    print("walk-forward weight range:\n", swing.to_string())
    return table, results


def m4(prices: pd.DataFrame) -> pd.DataFrame:
    """Fit both risk models on TRAIN portfolio returns, simulate 5 years, then check the 1-day VaR
    against what actually happened in test."""
    streams = strategy_streams(prices)
    train_s, test_s = data.split(streams)
    ms_w = frontier.max_sharpe(*frontier.annualize(excess(train_s, data.load_rf(prices.index))))
    weights = {
        "Max-Sharpe (M3)": ms_w,
        "Equal-weight 4": pd.Series(0.25, index=streams.columns),
        "SPY": pd.Series([0, 0, 1, 0], index=streams.columns, dtype=float),
    }
    models = {"Normal": mc.simulate_normal, "Bootstrap": mc.simulate_bootstrap}
    risk, calib, keep = [], [], {}
    for pname, w in weights.items():
        r_train, r_test = train_s @ w, test_s @ w
        for mname, sim in models.items():
            daily = sim(r_train, seed=data.SEED)
            s = mc.summary(daily)
            risk.append({"portfolio": pname, "model": mname, **s})
            calib.append({"portfolio": pname, "model": mname,
                          "VaR95": s["1d VaR95"], "breach95 (target 5%)": mc.exceedance_rate(r_test, s["1d VaR95"]),
                          "VaR99": s["1d VaR99"], "breach99 (target 1%)": mc.exceedance_rate(r_test, s["1d VaR99"])})
            if pname == "Max-Sharpe (M3)":
                keep[mname] = daily
        realized = mc.var_cvar(r_test.to_numpy(), 0.99)
        risk.append({"portfolio": pname, "model": "Realized (test)", "1d VaR99": realized[0], "1d CVaR99": realized[1]})
    risk, calib = pd.DataFrame(risk), pd.DataFrame(calib)

    years = np.arange(mc.YEARS * mc.TRADING_DAYS + 1) / mc.TRADING_DAYS
    c_norm, c_boot, c_real = SERIES[0], SERIES[1], SERIES[2]
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))

    ax = axs[0, 0]
    eq_b, eq_n = mc.equity(keep["Bootstrap"]), mc.equity(keep["Normal"])
    for path in eq_b[:150]:
        ax.plot(years, path, color=c_boot, alpha=0.05, linewidth=0.6)
    for q in (0.05, 0.5, 0.95):
        ax.plot(years, np.quantile(eq_b, q, axis=0), color=c_boot, linewidth=1.8)
    for q in (0.05, 0.95):
        ax.plot(years, np.quantile(eq_n, q, axis=0), color=c_norm, linewidth=1.4, linestyle="--")
    ax.plot([], [], color=c_boot, linewidth=1.8, label="bootstrap 5th / 50th / 95th pct")
    ax.plot([], [], color=c_norm, linewidth=1.4, linestyle="--", label="normal 5th / 95th pct")
    ax.set_xlabel("Years")
    ax.set_ylabel("Portfolio value")
    ax.set_title("M4 · 10,000 simulated 5-year paths (Max-Sharpe portfolio)", loc="left")
    ax.legend(loc="upper left", fontsize=8)

    ax = axs[0, 1]
    for mname, c in (("Normal", c_norm), ("Bootstrap", c_boot)):
        term = mc.equity(keep[mname])[:, -1] - 1
        ax.hist(term, bins=80, histtype="step", linewidth=1.6, color=c, label=mname)
        ax.axvline(np.quantile(term, 0.05), color=c, linestyle=":", linewidth=1.2)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("5-year return (dotted = 5th percentile)")
    ax.set_ylabel("Paths")
    ax.set_title("Terminal 5-year return", loc="left")
    ax.legend(loc="upper right", fontsize=8)

    ax = axs[1, 0]
    for mname, c in (("Normal", c_norm), ("Bootstrap", c_boot)):
        ax.hist(mc.max_drawdowns(keep[mname]), bins=80, histtype="step", linewidth=1.6, color=c, label=mname)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Worst drawdown within the 5 years")
    ax.set_ylabel("Paths")
    ax.set_title("Max drawdown distribution", loc="left")
    ax.legend(loc="upper left", fontsize=8)

    ax = axs[1, 1]
    bars = risk.pivot(index="portfolio", columns="model", values="1d CVaR99").loc[list(weights)]
    x = np.arange(len(bars))
    for i, (mname, c) in enumerate((("Normal", c_norm), ("Bootstrap", c_boot), ("Realized (test)", c_real))):
        vals = bars[mname].to_numpy()
        ax.bar(x + (i - 1) * 0.27, vals, width=0.25, color=c, label=mname)
        for xi, v in zip(x, vals):
            ax.text(xi + (i - 1) * 0.27, v, f"{v:.1%}", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(x, bars.index)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    ax.set_ylabel("Avg loss on the worst 1% of days")
    ax.set_title("1-day CVaR 99%: models (fit on train) vs realized test", loc="left")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "m4_montecarlo.png", dpi=150)
    plt.close(fig)

    pct_cols = [c for c in risk.columns if c not in ("portfolio", "model")]
    (FIG / "m4_risk.md").write_text(to_markdown(risk, pct=pct_cols).replace("nan%", "–"))
    (FIG / "m4_calibration.md").write_text(
        to_markdown(calib, pct=[c for c in calib.columns if c not in ("portfolio", "model")]))
    print(calib.round(4).to_string(index=False))
    return risk


# The out-of-sample verdict set, in README order, and the head-to-head questions M8 tests.
VERDICT = ["Buy & hold SPY", "Equal-weight 48", "XS momentum 12-1", "MA 50/200 on SPY", "Pairs portfolio (5)",
           "Max-Sharpe (train fit)", "Walk-forward max-Sharpe", "Equal-weight 4"]
COMPARISONS = [("Equal-weight 4", "Max-Sharpe (train fit)"), ("Walk-forward max-Sharpe", "Max-Sharpe (train fit)"),
               ("XS momentum 12-1", "Equal-weight 48"), ("Equal-weight 48", "Buy & hold SPY"),
               ("Buy & hold SPY", "Equal-weight 4")]


def m8(results: dict, rf: pd.Series) -> pd.DataFrame:
    """Is any test-period Sharpe, or any difference between two, distinguishable from luck?"""
    ex = {k: excess(data.split(results[k])[1]["net"], rf) for k in VERDICT}
    trials = np.array([e.mean() / e.std() for e in ex.values()])  # per-period Sharpes of the N tried
    sr0 = significance.expected_max_sharpe(trials) * np.sqrt(mc.TRADING_DAYS)
    rows = []
    for k, e in ex.items():
        point, lo, hi = significance.sharpe_ci(e, seed=data.SEED)
        rows.append({"strategy": k, "Sharpe": point, "95% CI low": lo, "95% CI high": hi,
                     "PSR (SR > 0)": significance.psr(e), f"DSR (N = {len(ex)})": significance.dsr(e, trials)})
    table = pd.DataFrame(rows)
    diffs = []
    for a, b in COMPARISONS:
        d, lo, hi, p = significance.sharpe_diff(ex[a], ex[b], seed=data.SEED)
        diffs.append({"comparison": f"{a} − {b}", "ΔSharpe": d, "95% CI low": lo, "95% CI high": hi, "p-value": p})
    diffs = pd.DataFrame(diffs)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    y = np.arange(len(table))[::-1]
    ax.hlines(y, table["95% CI low"], table["95% CI high"], color=SERIES[0], linewidth=2.2)
    ax.scatter(table["Sharpe"], y, s=60, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5, zorder=3)
    for yi, (_, r) in zip(y, table.iterrows()):
        ax.text(r["95% CI high"] + 0.04, yi, f"{r['Sharpe']:.2f}  [{r['95% CI low']:.2f}, {r['95% CI high']:.2f}]",
                va="center", fontsize=8, color=INK2)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.axvline(sr0, color=SERIES[1], linestyle="--", linewidth=1.2)
    ax.text(sr0, len(table) - 0.35, f" best-of-{len(ex)} by luck: {sr0:.2f}", color=SERIES[1], fontsize=8,
            va="bottom")
    ax.set_yticks(y, table["strategy"])
    ax.set_ylim(-0.7, len(table) - 0.1)
    ax.set_xlim(None, table["95% CI high"].max() + 0.55)
    ax.set_xlabel("Annualized Sharpe over T-bills, test 2019–2026 (net), with 95% block-bootstrap CI")
    ax.set_title("M8 · Out-of-sample Sharpe ratios with their uncertainty", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "m8_sharpe_ci.png", dpi=150)
    plt.close(fig)

    (FIG / "m8_significance.md").write_text(to_markdown(table, pct=(), digits=2))
    (FIG / "m8_differences.md").write_text(to_markdown(diffs, pct=(), digits=2))
    print(f"expected best-of-{len(ex)} Sharpe under the null: {sr0:.2f}")
    print(diffs.round(3).to_string(index=False))
    return table


def m9(prices: pd.DataFrame) -> pd.DataFrame:
    """Backtest four 1-day VaR models on 2019–2026; only the EWMA ones are allowed to adapt."""
    streams = strategy_streams(prices)
    rf = data.load_rf(prices.index)
    train_s = data.split(streams)[0]
    weights = {
        "Max-Sharpe (M3)": frontier.max_sharpe(*frontier.annualize(excess(train_s, rf))),
        "Equal-weight 4": pd.Series(0.25, index=streams.columns),
        "SPY": pd.Series([0, 0, 1, 0], index=streams.columns, dtype=float),
    }
    rows, spy = [], {}
    for pname, w in weights.items():
        r = streams @ w
        test = r.index >= data.SPLIT
        per_model = {}
        for level in (0.95, 0.99):
            v = mc.var_series(r, data.SPLIT, level)
            tag = int(round(level * 100))
            for model in v.columns:
                br = (r[test] < -v.loc[test, model]).to_numpy()
                per_model.setdefault(model, {}).update({
                    f"breach{tag}": br.mean(), f"Kupiec p{tag}": mc.kupiec(br, 1 - level),
                    f"Christoffersen p{tag}": mc.christoffersen(br)})
            if pname == "SPY" and level == 0.99:
                spy = {"r": r[test], "var": v[test]}
        rows += [{"portfolio": pname, "model": m, **vals} for m, vals in per_model.items()]
    table = pd.DataFrame(rows)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8.5), gridspec_kw={"height_ratios": [1.6, 1]})
    r, v = spy["r"], spy["var"]
    a1.plot(r.index, r * 100, color=MUTED, linewidth=0.6, label="SPY daily return")
    for model, c in (("Static historical", SERIES[1]), ("EWMA filtered historical", SERIES[3])):  # = bar colours
        a1.plot(v.index, -v[model] * 100, color=c, linewidth=1.3, label=f"−VaR99, {model}")
        hit = r < -v[model]
        a1.scatter(r.index[hit], r[hit] * 100, s=14, color=c, zorder=3)
    a1.set_ylabel("Daily return (%)")
    a1.set_title("M9 · SPY 2019–2026: a static VaR99 stands still, an EWMA-filtered one moves with volatility",
                 loc="left")
    a1.legend(loc="lower left", fontsize=8)
    models = list(table["model"].unique())
    x = np.arange(len(weights))
    for i, (model, c) in enumerate(zip(models, SERIES)):
        vals = table[table["model"] == model].set_index("portfolio").loc[list(weights), "breach99"].to_numpy()
        a2.bar(x + (i - 1.5) * 0.2, vals, width=0.19, color=c, label=model)
    a2.axhline(0.01, color=INK2, linestyle="--", linewidth=1)
    a2.text(x[-1] + 0.45, 0.01, "target 1%", color=INK2, fontsize=8, va="bottom", ha="right")
    a2.set_xticks(x, list(weights))
    a2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    a2.set_ylabel("Share of test days breaching VaR99")
    a2.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "m9_var_backtest.png", dpi=150)
    plt.close(fig)

    pct = [c for c in table.columns if c.startswith("breach")]
    (FIG / "m9_var_backtest.md").write_text(to_markdown(table, pct=pct, digits=3))
    return table


def m11(results: dict, rf: pd.Series) -> pd.DataFrame:
    """Factor attribution of every verdict strategy, train and test, Newey-West t-stats."""
    ff = factors.load_factors()
    rows = []
    for k in VERDICT:
        for period, part in zip(("train", "test"), data.split(results[k]["net"])):
            rows.append({"strategy": k, "period": period, **factors.regress(excess(part, rf), ff)})
    table = pd.DataFrame(rows)
    test = table[table["period"] == "test"].reset_index(drop=True)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.1, 1]}, sharey=True)
    y = np.arange(len(test))[::-1]
    se = (test["alpha/yr"] / test["t(alpha)"]).abs()
    a1.hlines(y, test["alpha/yr"] - 1.96 * se, test["alpha/yr"] + 1.96 * se, color=SERIES[0], linewidth=2.2)
    a1.scatter(test["alpha/yr"], y, s=60, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5, zorder=3)
    for yi, (_, r), s_i in zip(y, test.iterrows(), se):
        a1.text(r["alpha/yr"] + 1.96 * s_i + 0.003, yi, f"{r['alpha/yr']:+.1%}  t = {r['t(alpha)']:.2f}",
                va="center", fontsize=8, color=INK2)
    a1.axvline(0, color=MUTED, linewidth=1)
    a1.set_yticks(y, test["strategy"])
    a1.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    a1.set_xlim(None, (test["alpha/yr"] + 1.96 * se).max() + 0.09)
    a1.set_xlabel("Annual alpha after 6 factors, test (95% Newey-West CI)")
    a1.set_title("M11 · Alpha after Fama-French 5 + momentum", loc="left")
    for i, (col, c) in enumerate((("β Mkt-RF", SERIES[1]), ("β Mom", SERIES[2]))):
        a2.barh(y + (0.18 if i == 0 else -0.18), test[col], height=0.34, color=c, label=col)
    for yi, r2 in zip(y, test["R²"]):
        a2.text(1.32, yi, f"R² {r2:.2f}", va="center", fontsize=8, color=INK2)
    a2.axvline(0, color=MUTED, linewidth=1)
    a2.axvline(1, color=MUTED, linewidth=0.8, linestyle=":")
    a2.set_xlim(-0.35, 1.55)
    a2.set_xlabel("Factor loading, test")
    a2.set_title("Market and momentum loadings", loc="left")
    a2.legend(loc="upper center", bbox_to_anchor=(0.45, -0.12), fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "m11_factors.png", dpi=150)
    plt.close(fig)

    cols = ["strategy", "period", "alpha/yr", "t(alpha)", "p(alpha)"] + [f"β {c}" for c in factors.FACTORS] + ["R²", "days"]
    (FIG / "m11_factors.md").write_text(to_markdown(table[cols], pct=("alpha/yr",), digits=2))
    print(f"factor data end: {ff.index[-1].date()}")
    return table[cols]


def m12(prices: pd.DataFrame) -> pd.DataFrame:
    """Estimators built to survive estimation error, against the sample max-Sharpe and 1/N."""
    streams = strategy_streams(prices)
    rf = data.load_rf(prices.index)
    ex = excess(streams, rf)
    ex_train, _ = data.split(ex)
    mu_s, cov_s = frontier.annualize(ex_train)
    cov_lw, delta = frontier.ledoit_wolf(ex_train)
    mu_bs, phi = frontier.bayes_stein_mu(ex_train, cov_lw)
    ew = pd.Series(0.25, index=streams.columns)
    fitted = {
        "Max-Sharpe (sample)": frontier.max_sharpe(mu_s, cov_s),
        "Max-Sharpe (Bayes-Stein μ, LW Σ)": frontier.max_sharpe(mu_bs, cov_lw),
        "Min-variance (LW Σ)": frontier.min_variance(mu_s, cov_lw),
        "Risk parity": frontier.risk_parity(cov_s),
        "Equal-weight 4": ew,
    }
    wf_rp = frontier.walk_forward(ex, fit=lambda h: frontier.risk_parity(frontier.annualize(h)[1]))
    equity = (1 + streams).cumprod()
    const = lambda w: pd.DataFrame([w.to_numpy()] * len(streams), index=streams.index, columns=streams.columns)
    allocs = {k: const(w) for k, w in fitted.items()} | {"Walk-forward risk parity": wf_rp}
    live0 = wf_rp.sum(axis=1).gt(0).idxmax()
    results = {k: backtest.run(w, equity, COST_BPS, rf).loc[live0:] for k, w in allocs.items()}
    table = metrics_table(results, rf)

    test_ex = {k: excess(data.split(r)[1]["net"], rf) for k, r in results.items()}
    diffs = pd.DataFrame([dict(zip(("comparison", "ΔSharpe", "95% CI low", "95% CI high", "p-value"),
                                   (f"{k} − Equal-weight 4",
                                    *significance.sharpe_diff(test_ex[k], test_ex["Equal-weight 4"], seed=data.SEED))))
                          for k in results if k != "Equal-weight 4"])
    wtab = pd.DataFrame(fitted).T
    risk = pd.DataFrame({k: frontier.risk_contributions(w, cov_s) for k, w in fitted.items()}).T

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 9.5), gridspec_kw={"height_ratios": [1, 1.4]})
    y = np.arange(len(wtab))[::-1]
    left = np.zeros(len(wtab))
    for asset, c in zip(wtab.columns, SERIES):  # same asset colours as the M3 walk-forward panel
        a1.barh(y, wtab[asset], left=left, color=c, edgecolor=SURFACE, linewidth=1.5, label=asset, height=0.62)
        for yi, l, v in zip(y, left, wtab[asset]):
            if v >= 0.06:
                a1.text(l + v / 2, yi, f"{v:.0%}", ha="center", va="center", fontsize=8, color=SURFACE)
        left += wtab[asset].to_numpy()
    a1.set_yticks(y, wtab.index)
    a1.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    a1.set_xlim(0, 1)
    a1.grid(False)
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8, ncol=4)
    a1.set_title(f"M12 · Train-fitted weights (LW δ = {delta:.3f}, Bayes-Stein φ = {phi:.2f})", loc="left")
    same = wtab.iloc[:3].round(6).nunique().max() == 1  # the three mean-variance variants coincide
    show = {"Risk parity": "Risk parity", "Walk-forward risk parity": "Walk-forward risk parity",
            "Max-Sharpe (sample)": "All 3 mean-variance variants (identical weights)" if same else "Max-Sharpe (sample)",
            "Equal-weight 4": "Equal-weight 4"}
    for (name, label), c in zip(show.items(), SERIES):
        eq = (1 + data.split(results[name])[1]["net"]).cumprod()
        a2.plot(eq.index, eq, color=c, label=f"{label}  {eq.iloc[-1]:.2f}x")
    a2.set_ylabel("Growth of $1 (test, net)")
    a2.set_title("Out of sample 2019–2026", loc="left")
    a2.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "m12_allocation.png", dpi=150)
    plt.close(fig)

    (FIG / "m12_metrics.md").write_text(to_markdown(table))
    (FIG / "m12_weights.md").write_text(to_markdown(wtab.reset_index(names="method"), pct=tuple(wtab.columns)))
    (FIG / "m12_risk_contributions.md").write_text(to_markdown(risk.reset_index(names="method"), pct=tuple(risk.columns)))
    (FIG / "m12_vs_equal_weight.md").write_text(to_markdown(diffs, pct=(), digits=2))
    print(f"Ledoit-Wolf delta = {delta:.4f}, Bayes-Stein phi = {phi:.3f}")
    print("weights:\n", wtab.round(3).to_string())
    print("risk contributions:\n", risk.round(3).to_string())
    print(diffs.round(3).to_string(index=False))
    return table


WF_VERDICT = ["Buy & hold SPY", "Equal-weight 48", "XS momentum 12-1", "MA 50/200 on SPY", "Pairs portfolio (5)",
              "Max-Sharpe (re-fit yearly)", "Equal-weight 4"]
WF_COMPARISONS = [("Equal-weight 4", "Max-Sharpe (re-fit yearly)"), ("XS momentum 12-1", "Equal-weight 48"),
                  ("Equal-weight 48", "Buy & hold SPY"), ("Buy & hold SPY", "Equal-weight 4")]


def m13(prices: pd.DataFrame) -> pd.DataFrame:
    """Re-run the design at 13 yearly cutoffs (2014–2026) and pool the out-of-sample years."""
    rf = data.load_rf(prices.index)
    oos, fits = walkforward.yearly(prices, rf, COST_BPS)
    ex = {k: excess(oos[k], rf) for k in WF_VERDICT}
    years = sorted(set(oos["Buy & hold SPY"].index.year))
    yr = lambda s, y: float((1 + s[s.index.year == y]).prod() - 1)
    by_year = pd.DataFrame({k: [yr(oos[k], y) for y in years] for k in WF_VERDICT}, index=years)
    by_year["T-bills"] = [yr(rf.reindex(oos["Buy & hold SPY"].index), y) for y in years]

    rows, recent = [], {}
    for k, e in ex.items():
        s_all, lo_all, hi_all = significance.sharpe_ci(e, seed=data.SEED)
        recent[k] = significance.sharpe_ci(e[e.index >= data.SPLIT], seed=data.SEED)
        rows.append({"strategy": k, "Sharpe 2014–2026": s_all, "95% CI low": lo_all, "95% CI high": hi_all,
                     "CI width": hi_all - lo_all, "CI width 2019–2026 only": recent[k][2] - recent[k][1],
                     "years beating SPY": "–" if k == "Buy & hold SPY"
                     else f"{int((by_year[k] > by_year['Buy & hold SPY']).sum())}/{len(years)}",
                     "worst year": by_year[k].min(), "best year": by_year[k].max()})
    table = pd.DataFrame(rows)
    diffs = pd.DataFrame([dict(zip(("comparison", "ΔSharpe", "95% CI low", "95% CI high", "p-value"),
                                   (f"{a} − {b}", *significance.sharpe_diff(ex[a], ex[b], seed=data.SEED))))
                          for a, b in WF_COMPARISONS])

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 10), gridspec_kw={"height_ratios": [1.15, 1]})
    show = ["Buy & hold SPY", "XS momentum 12-1", "Max-Sharpe (re-fit yearly)", "Equal-weight 4"]
    x = np.arange(len(years))
    for i, (k, c) in enumerate(zip(show, SERIES)):
        a1.bar(x + (i - 1.5) * 0.2, by_year[k], width=0.19, color=c, label=k)
    a1.axhline(0, color=MUTED, linewidth=1)
    a1.set_xticks(x, [str(y) if y != years[-1] else f"{y}*" for y in years])
    a1.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    a1.set_ylabel("Calendar-year net return")
    a1.set_title("M13 · Each year out of sample, with everything re-fit on the years before it", loc="left")
    a1.legend(loc="upper left", fontsize=8, ncol=2)
    a1.text(1, -0.12, "* 2026 to Oct 6", transform=a1.transAxes, ha="right", fontsize=8, color=MUTED)
    y = np.arange(len(table))[::-1]
    for yi, (_, r) in zip(y, table.iterrows()):
        _, lo, hi = recent[r["strategy"]]
        a2.hlines(yi - 0.17, lo, hi, color=MUTED, linewidth=2.2)
        a2.hlines(yi + 0.17, r["95% CI low"], r["95% CI high"], color=INK, linewidth=2.2)
        a2.scatter([r["Sharpe 2014–2026"]], [yi + 0.17], s=50, color=INK, edgecolor=SURFACE, linewidth=1.5,
                   zorder=3)
    a2.plot([], [], color=INK, linewidth=2.2, label="2014–2026, 12.8 years (M13)")
    a2.plot([], [], color=MUTED, linewidth=2.2, label="2019–2026 only, 7.8 years")
    a2.axvline(0, color=MUTED, linewidth=1)
    a2.set_yticks(y, table["strategy"])
    a2.set_xlabel("Annualized Sharpe over T-bills with 95% block-bootstrap CI")
    a2.set_title("More out-of-sample years, narrower intervals", loc="left")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "m13_walkforward.png", dpi=150)
    plt.close(fig)

    (FIG / "m13_by_year.md").write_text(to_markdown(by_year.reset_index(names="year").astype({"year": str}),
                                                    pct=tuple(by_year.columns)))
    (FIG / "m13_summary.md").write_text(to_markdown(table, pct=("worst year", "best year"), digits=2))
    (FIG / "m13_differences.md").write_text(to_markdown(diffs, pct=(), digits=2))
    (FIG / "m13_fits.md").write_text(to_markdown(fits.astype({"year": str, "train days": str}),
                                                 pct=tuple(c for c in fits.columns if c.startswith("w ")), digits=2))
    print(fits.to_string(index=False))
    print(diffs.round(3).to_string(index=False))
    print(by_year.round(3).to_string())
    return table


# Indian costs per unit of one-way turnover (delivery trades, 2024-25 schedules): STT 10 bps on every buy
# and sell, stamp duty 1.5 bps on buys, exchange + SEBI fees + GST ~0.4 bps -> ~11 bps statutory, plus ~9 bps
# of spread/impact for large caps. The pairs' short legs would trade as stock futures (lower STT, monthly rolls);
# the same rate is applied to both legs.
IN_COST_BPS = 20
IN_NAMES = {"Buy & hold SPY": "Buy & hold Nifty", "MA 50/200 on SPY": "MA 50/200 on Nifty"}


def m14(prices_us: pd.DataFrame, tables: dict) -> pd.DataFrame:
    """M14 transfer study: the M1–M3 design on 48 NSE large caps, with Indian costs and short rate."""
    p = data.load_prices_in()
    rf = data.load_rf(p.index, **data.IN_RF)
    stocks = [t for t in p.columns if t not in data.IN_BENCHMARKS]
    train, test = data.split(p)
    scanned = pairs.scan(train, data.IN_SECTORS)
    chosen = pairs.select(scanned)
    chosen["p_test"] = [pairs.coint_pvalue(test[r.a], test[r.b]) for r in chosen.itertuples()]
    mom_w = momentum.cross_sectional(p, stocks)
    weights = {
        "Buy & hold Nifty": momentum.buy_and_hold(p, "NIFTYBEES"),
        "Equal-weight 48": momentum.equal_weight(p, stocks),
        "XS momentum 12-1": mom_w,
        "MA 50/200 on Nifty": momentum.ma_crossover(p, "NIFTYBEES"),
        "Pairs portfolio (5)": pairs.portfolio(p, chosen),
    }
    res = {k: backtest.run(w, p, IN_COST_BPS, rf) for k, w in weights.items()}

    live = p.index[p.index.get_loc(mom_w.sum(axis=1).gt(0).idxmax()) + 1]
    streams = pd.DataFrame({"Momentum": res["XS momentum 12-1"]["net"], "Pairs": res["Pairs portfolio (5)"]["net"],
                            "Nifty": p["NIFTYBEES"].pct_change(), "Gold": p["GOLDBEES"].pct_change()}).loc[live:]
    ms_w = frontier.max_sharpe(*frontier.annualize(data.split(excess(streams, rf))[0]))
    equity = (1 + streams).cumprod()
    const = lambda v: pd.DataFrame([v.to_numpy()] * len(streams), index=streams.index, columns=streams.columns)
    res["Max-Sharpe (train fit)"] = backtest.run(const(ms_w), equity, IN_COST_BPS, rf)
    res["Equal-weight 4"] = backtest.run(const(pd.Series(0.25, index=streams.columns)), equity, IN_COST_BPS, rf)
    table = metrics_table(res, rf)

    # Same strategies, US vs India, test period.
    us = pd.concat([tables[k] for k in ("m1", "m2", "m3")])
    us = us[us["period"] == "test"].drop_duplicates("strategy").set_index("strategy")
    india = table[table["period"] == "test"].set_index("strategy")
    rows = []
    for us_name in ["Buy & hold SPY", "Equal-weight 48", "XS momentum 12-1", "MA 50/200 on SPY", "Pairs portfolio (5)",
                    "Max-Sharpe (train fit)", "Equal-weight 4"]:
        in_name = IN_NAMES.get(us_name, us_name)
        label = us_name.replace("SPY", "SPY / Nifty")
        rows.append({"strategy": label,
                     "US Sharpe": us.loc[us_name, "Sharpe (net)"], "India Sharpe": india.loc[in_name, "Sharpe (net)"],
                     "US CAGR": us.loc[us_name, "CAGR (net)"], "India CAGR": india.loc[in_name, "CAGR (net)"],
                     "US maxDD": us.loc[us_name, "maxDD"], "India maxDD": india.loc[in_name, "maxDD"]})
    compare = pd.DataFrame(rows)

    # Cost sensitivity of the two traded strategies, both markets, test-period Sharpe.
    us_rf = data.load_rf(prices_us.index)
    us_stocks = [t for t in prices_us.columns if t not in data.BENCHMARKS]
    us_w = {"XS momentum 12-1": momentum.cross_sectional(prices_us, us_stocks),
            "Pairs portfolio (5)": pairs.portfolio(prices_us, pairs.select(pairs.scan(data.split(prices_us)[0])))}
    in_w = {k: weights[k] for k in us_w}
    grid = [0, 10, 20, 30, 40, 60]
    costs = []
    for bps in grid:
        row = {"cost (bps)": str(bps)}
        for mkt, (px, w_map, r_f) in {"US": (prices_us, us_w, us_rf), "India": (p, in_w, rf)}.items():
            for k, w in w_map.items():
                net = data.split(backtest.run(w, px, bps, r_f)["net"])[1]
                row[f"{mkt} · {k}"] = backtest.sharpe(net, r_f)
        costs.append(row)
    costs = pd.DataFrame(costs)

    ex_t = {k: excess(data.split(r)[1]["net"], rf) for k, r in res.items()}
    sig_rows = [dict(zip(("strategy", "Sharpe", "95% CI low", "95% CI high"),
                         (k, *significance.sharpe_ci(e, seed=data.SEED)))) for k, e in ex_t.items()]
    for a, b in [("XS momentum 12-1", "Equal-weight 48"), ("Equal-weight 4", "Max-Sharpe (train fit)"),
                 ("Equal-weight 48", "Buy & hold Nifty")]:
        d, lo, hi, pv = significance.sharpe_diff(ex_t[a], ex_t[b], seed=data.SEED)
        sig_rows.append({"strategy": f"{a} − {b}", "Sharpe": d, "95% CI low": lo, "95% CI high": hi, "p-value": pv})
    sig = pd.DataFrame(sig_rows)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    ax = axs[0, 0]
    for k, c in zip(["XS momentum 12-1", "MA 50/200 on Nifty", "Equal-weight 48", "Buy & hold Nifty"], SERIES):
        eq = (1 + data.split(res[k])[1]["net"]).cumprod()
        ax.plot(eq.index, eq, color=c, label=f"{k}  {eq.iloc[-1]:.2f}x")
    ax.set_ylabel("Growth of ₹1 (test, net of 20 bps)")
    ax.set_title("M14 · India, out of sample 2019–2026", loc="left")
    ax.legend(loc="upper left", fontsize=8)

    ax = axs[0, 1]
    for k, c in zip(["XS momentum 12-1", "Pairs portfolio (5)"], SERIES):
        ax.plot(grid, costs[f"India · {k}"], color=c, marker="o", label=f"India · {k}")
        ax.plot(grid, costs[f"US · {k}"], color=c, marker="o", linestyle="--", alpha=0.8, label=f"US · {k}")
    ax.axhline(0, color=MUTED, linewidth=1)
    for x_, lab in ((10, "US base"), (IN_COST_BPS, "India base")):
        ax.axvline(x_, color=MUTED, linestyle=":", linewidth=1)
        ax.text(x_, ax.get_ylim()[1], f" {lab}", color=MUTED, fontsize=8, va="top")
    ax.set_xlabel("Cost per unit of turnover (bps)")
    ax.set_ylabel("Test Sharpe over the local short rate")
    ax.set_title("Cost sensitivity", loc="left")
    ax.legend(loc="lower left", fontsize=8)

    ax = axs[1, 0]
    y = np.arange(len(compare))[::-1]
    ax.barh(y + 0.18, compare["US Sharpe"], height=0.34, color=SERIES[2], label="US (over T-bills, 10 bps)")
    ax.barh(y - 0.18, compare["India Sharpe"], height=0.34, color=SERIES[3], label="India (over call money, 20 bps)")
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(y, compare["strategy"])
    ax.set_xlabel("Test Sharpe (net)")
    ax.set_title("Same design, two markets", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8, ncol=2)

    ax = axs[1, 1]
    single = sig[~sig["strategy"].str.contains(" − ")].reset_index(drop=True)
    y = np.arange(len(single))[::-1]
    ax.hlines(y, single["95% CI low"], single["95% CI high"], color=INK, linewidth=2.2)
    ax.scatter(single["Sharpe"], y, s=50, color=INK, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(y, single["strategy"])
    ax.set_xlabel("India test Sharpe with 95% block-bootstrap CI")
    ax.set_title("…with the same uncertainty", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "m14_india.png", dpi=150)
    plt.close(fig)

    (FIG / "m14_metrics.md").write_text(to_markdown(table))
    (FIG / "m14_pairs.md").write_text(to_markdown(
        chosen[["sector", "a", "b", "corr", "pvalue", "p_test", "beta", "half_life"]], pct=(), digits=3))
    (FIG / "m14_weights.md").write_text(to_markdown(
        pd.DataFrame({"asset": ms_w.index, "max Sharpe": ms_w.values}), pct=("max Sharpe",)))
    (FIG / "m14_us_vs_india.md").write_text(to_markdown(
        compare, pct=("US CAGR", "India CAGR", "US maxDD", "India maxDD")))
    (FIG / "m14_costs.md").write_text(to_markdown(costs, pct=(), digits=2))
    (FIG / "m14_significance.md").write_text(to_markdown(sig, pct=(), digits=2).replace("nan", "–"))
    passed = scanned["pvalue"] < 0.05
    print(f"India scan: {passed.sum()} of {len(scanned)} cointegrated at 5%; corr>0.7: {(scanned['corr'] > 0.7).sum()}, "
          f"of which not cointegrated: {((scanned['corr'] > 0.7) & ~passed).sum()}")
    print("India pairs:\n", chosen.round(3).to_string(index=False))
    print("max-Sharpe weights:", ms_w.round(3).to_dict())
    print(compare.round(3).to_string(index=False))
    print(costs.round(3).to_string(index=False))
    print(sig.round(3).to_string(index=False))
    return table


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    prices = data.load_prices()
    rf = data.load_rf(prices.index)
    pd.set_option("display.width", 200)
    tables, results = {}, {}
    for step in (m1, m2, m3, m4):
        print(f"\n== {step.__name__} ==")
        out = step(prices)
        tables[step.__name__], res = out if isinstance(out, tuple) else (out, {})
        results |= res
        print(tables[step.__name__].round(3).to_string(index=False))

    # One-table verdict: every strategy's out-of-sample (2019+) net result, in the README's order.
    test = pd.concat([tables[k] for k in ("m1", "m2", "m3")])
    summary = test[(test["period"] == "test") & test["strategy"].isin(VERDICT)].drop_duplicates("strategy")
    summary = summary.set_index("strategy").loc[VERDICT].reset_index()
    cols = ["strategy", "Sharpe (net)", "CAGR (net)", "vol", "maxDD", "turnover/yr"]
    (FIG / "summary.md").write_text(to_markdown(summary[cols]))
    print("\n== summary (test, net) ==\n", summary[cols].round(3).to_string(index=False))

    print("\n== m8 ==\n", m8(results, rf).round(3).to_string(index=False))
    print("\n== m9 ==\n", m9(prices).round(3).to_string(index=False))
    print("\n== m11 ==\n", m11(results, rf).round(3).to_string(index=False))
    print("\n== m12 ==\n", m12(prices).round(3).to_string(index=False))
    print("\n== m13 ==\n", m13(prices).round(3).to_string(index=False))
    print("\n== m14 ==\n", m14(prices, tables).round(3).to_string(index=False))
