"""Regenerates every figure and results table. Run: python run.py"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import backtest
import data
import momentum

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


def to_markdown(df: pd.DataFrame) -> str:
    pct = {"CAGR (net)", "vol", "maxDD"}
    fmt = lambda c, v: f"{v:.1%}" if c in pct else (f"{v:.2f}" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(c, v) for c, v in row.items()) + " |" for _, row in df.iterrows()]
    return "\n".join(lines) + "\n"


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
    for a in (ax, dd):
        a.axvline(pd.Timestamp(data.SPLIT), color=MUTED, linestyle="--", linewidth=1)
    ax.text(pd.Timestamp(data.SPLIT), ax.get_ylim()[1], "  train | test ", color=MUTED, va="top", ha="center", fontsize=9)
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


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    prices = data.load_prices()
    pd.set_option("display.width", 160)
    print(m1(prices).round(3).to_string(index=False))
