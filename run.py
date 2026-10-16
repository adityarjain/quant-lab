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
import frontier
import momentum
import montecarlo as mc
import pairs

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


def metrics_table(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        for period, part in zip(("train", "test"), data.split(res)):
            g, n = backtest.metrics(part["gross"]), backtest.metrics(part["net"], part["turnover"])
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
    results = {k: backtest.run(w, prices, COST_BPS) for k, w in strategies.items()}

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

    table = metrics_table(results)
    (FIG / "m1_metrics.md").write_text(to_markdown(table))
    return table


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
    results = {k: backtest.run(w, prices, COST_BPS) for k, w in strategies.items()}

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
    table = metrics_table(results)
    (FIG / "m2_metrics.md").write_text(to_markdown(table))
    print(f"scan: {len(scanned)} pairs, {passed.sum()} cointegrated at 5% "
          f"(~{0.05 * len(scanned):.0f} expected by chance); corr>0.7: {(scanned['corr'] > 0.7).sum()}, "
          f"of which not cointegrated: {((scanned['corr'] > 0.7) & ~passed).sum()}")
    print(sel.round(3).to_string(index=False))
    return table


def strategy_streams(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily NET returns of the building blocks the M3 optimizer allocates across."""
    stocks = [t for t in prices.columns if t not in data.BENCHMARKS]
    train, _ = data.split(prices)
    mom = backtest.run(momentum.cross_sectional(prices, stocks), prices, COST_BPS)["net"]
    prs = backtest.run(pairs.portfolio(prices, pairs.select(pairs.scan(train))), prices, COST_BPS)["net"]
    streams = pd.DataFrame({"Momentum": mom, "Pairs": prs, "SPY": prices["SPY"].pct_change(),
                            "BND": prices["BND"].pct_change()})
    first_live = mom.ne(0).idxmax()  # drop momentum's 12-month warm-up so its stats aren't diluted by zeros
    return streams.loc[first_live:]


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
    ax.plot([0, v * 1.6], [0, r * 1.6], color=SERIES[3], linestyle="--", linewidth=1, label="capital market line (rf = 0)")
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
    ax.set_ylabel("Expected return (annualized)")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left", fontsize=8)
    return sc, ms_w, mv_w


def m3(prices: pd.DataFrame) -> pd.DataFrame:
    streams = strategy_streams(prices)
    train_s, test_s = data.split(streams)
    sectors_train, _ = data.split(sector_streams(prices))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))
    frontier_panel(a1, sectors_train, "M3a · 8 sector baskets + BND (train)")
    sc, ms_w, mv_w = frontier_panel(a2, train_s, "M3b · Strategies + SPY + BND (train)")
    fig.subplots_adjust(right=0.9, wspace=0.18)
    fig.colorbar(sc, cax=fig.add_axes([0.92, 0.15, 0.012, 0.7]), label="Sharpe of random portfolio (capped)")
    fig.savefig(FIG / "m3_frontier.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Hold train-fitted weights through test; compare with walk-forward re-fitting and naive mixes.
    # Each stream's equity curve is treated as an asset, so the same engine charges rebalancing costs.
    equity = (1 + streams).cumprod()
    const = lambda w: pd.DataFrame([w.to_numpy()] * len(streams), index=streams.index, columns=streams.columns)
    wf = frontier.walk_forward(streams)
    allocs = {
        "Max-Sharpe (train fit)": const(ms_w),
        "Walk-forward max-Sharpe": wf,
        "Equal-weight 4": const(pd.Series(0.25, index=streams.columns)),
        "SPY": const(pd.Series([0, 0, 1, 0], index=streams.columns, dtype=float)),
    }
    live0 = wf.sum(axis=1).gt(0).idxmax()  # walk-forward needs 3 years of history; compare all from then
    results = {k: backtest.run(w, equity, COST_BPS).loc[live0:] for k, w in allocs.items()}

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
    table = metrics_table(results)
    (FIG / "m3_metrics.md").write_text(to_markdown(table))
    print("train correlations:\n", corr.round(2).to_string())
    print("weights:\n", pd.DataFrame({"max_sharpe": ms_w, "min_var": mv_w}).round(3).to_string())
    swing = wf.loc[wf.sum(axis=1) > 0].agg(["min", "max"]).round(2)
    print("walk-forward weight range:\n", swing.to_string())
    return table


def m4(prices: pd.DataFrame) -> pd.DataFrame:
    """Fit both risk models on TRAIN portfolio returns, simulate 5 years, then check the 1-day VaR
    against what actually happened in test."""
    streams = strategy_streams(prices)
    train_s, test_s = data.split(streams)
    ms_w = frontier.max_sharpe(*frontier.annualize(train_s))
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


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    prices = data.load_prices()
    pd.set_option("display.width", 200)
    tables = {}
    for step in (m1, m2, m3, m4):
        print(f"\n== {step.__name__} ==")
        tables[step.__name__] = step(prices)
        print(tables[step.__name__].round(3).to_string(index=False))

    # One-table verdict: every strategy's out-of-sample (2019+) net result, in the README's order.
    keep = ["Buy & hold SPY", "Equal-weight 48", "XS momentum 12-1", "MA 50/200 on SPY", "Pairs portfolio (5)",
            "Max-Sharpe (train fit)", "Walk-forward max-Sharpe", "Equal-weight 4"]
    test = pd.concat([tables[k] for k in ("m1", "m2", "m3")])
    summary = test[(test["period"] == "test") & test["strategy"].isin(keep)].drop_duplicates("strategy")
    summary = summary.set_index("strategy").loc[keep].reset_index()
    cols = ["strategy", "Sharpe (net)", "CAGR (net)", "vol", "maxDD", "turnover/yr"]
    (FIG / "summary.md").write_text(to_markdown(summary[cols]))
    print("\n== summary (test, net) ==\n", summary[cols].round(3).to_string(index=False))
