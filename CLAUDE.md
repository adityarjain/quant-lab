# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git: never commit or push

**Do not run `git commit`, `git push`, or any GitHub operation (gh, PRs, releases).** The user makes every commit manually.
When a piece of work is done, give the exact commands for the user to run, e.g.:

```bash
git add data.py backtest.py tests.py
```
```bash
git commit -m "M1: backtest engine + momentum"
```
```bash
git push origin main
```

Read-only git (`status`, `diff`, `log`) is fine.

## Project

Quant Portfolio Lab: one integrated research project (not four separate tutorials). Research question:
> Do momentum and pairs trading survive realistic costs out-of-sample, and does combining them with mean-variance optimization improve risk-adjusted returns and tail risk?

The repo must read like a research note, covering for every module: the Idea, the Math, reproducible Code, Results (including what failed), and a researcher-style README.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```
```bash
python run.py        # regenerates every figure in figures/ + the results table
```
```bash
python tests.py      # assert-based checks, one per module
```
```bash
python -c "import tests; tests.test_momentum()"   # run a single check
```
```bash
streamlit run app.py       # dashboard on http://localhost:8501 (also "dashboard" in .claude/launch.json)
```
```bash
python data.py --refresh   # re-download prices (changes results; only on purpose)
```

## Fixed ground rules (do not change without asking)

- **Universe**: ~40–50 US large caps across ~8 sectors + SPY + BND. Must serve both momentum cross-section and same-sector pairs.
- **Dates**: 2010-01-01 → today. **Train < 2019-01-01, Test ≥ 2019-01-01.** All tuning (parameters, pair selection, hedge ratios, optimizer weights) uses train only. Test is evaluated once at the end.
- **Costs**: 10 bps per unit turnover. Always report gross and net.
- **Fixed random seed** for all random work (random portfolios, Monte Carlo).
- Data from yfinance, downloaded once and cached to `data/prices.parquet` with the download date recorded. Never re-download during development.
- Vectorized pandas/numpy only. No backtrader, no event-driven engine.

## Architecture

Flat layout, one file per module. The contract that ties everything together:

**Every strategy returns a `weights` DataFrame (dates × tickers) of target positions. Strategies never compute P&L.** `backtest.run(weights, prices, cost_bps)` is the single place that applies `weights.shift(1)` (no look-ahead), multiplies by daily returns, and subtracts `cost × turnover`. Any new strategy must plug in through this interface.

Data flow:
```
data.py  → prices
momentum.py, pairs.py  → weights → backtest.py → daily return series per strategy
frontier.py  → optimizes over assets AND over the strategy return series [momentum, pairs, SPY, BND]
montecarlo.py → simulates the final combined portfolio
run.py  → orchestrates all of the above, writes figures/ and results table
```

Module specifics:
- **momentum.py**: (a) 50/200 MA crossover on SPY, (b) cross-sectional 12-1 month ranking, long top quintile, monthly rebalance. Compared against buy-and-hold; equity curve (log) + drawdown panel.
- **pairs.py**: candidates only within the same sector. Show correlation vs cointegration (Engle-Granger); highly correlated pairs that fail cointegration are a key finding. OLS hedge ratio fit on train; 60-day rolling z-score; enter |z|>2, exit |z|<0.5, stop |z|>4. 4-panel figure: prices, spread+bands, z-score+thresholds, equity.
- **frontier.py**: 10k random Dirichlet portfolios colored by Sharpe; exact frontier via `scipy.optimize.minimize` (SLSQP), constraints sum=1, 0≤w≤0.4. Max-Sharpe + min-variance, weights bar chart, capital market line. Fit on train, hold weights on test vs equal-weight and SPY. Rolling re-estimation to show weight instability.
- **montecarlo.py**: 10k paths × 5 years. Method A multivariate normal, method B block bootstrap (keeps fat tails/vol clustering). Fan chart with 5/50/95 percentiles, terminal histogram, VaR/CVaR at 95/99, max-drawdown distribution. Compare A vs B tails.

## Tests (`tests.py`)

Plain asserts, no framework. Must cover:
- A cheating signal using tomorrow's return must NOT be profitable through the engine (look-ahead guard).
- Frontier weights sum to 1 and respect bounds.
- Synthetic cointegrated pair is detected; independent random walks are rejected.
- Monte Carlo percentiles are ordered; bootstrap VaR ≥ normal VaR on a fat-tailed sample.
- `test_app`: the dashboard runs headless (Streamlit `AppTest`) with no errors and its defaults reproduce the published test Sharpe ratios. If README numbers change, update the expected values there too.

`app.py` is UI only: it must call the same module functions as `run.py` (no duplicated strategy or metric logic), and its defaults must equal the research settings. New tunable parameters go in as optional arguments whose defaults are the module constants, so `run.py` output stays byte-identical.

## Milestones (finish each fully before the next)

| ID | Milestone | Status |
|---|---|---|
| M0 | Setup + data (`data.py`) | done |
| M1 | Backtest engine + momentum (`backtest.py`, `momentum.py`) | done |
| M2 | Pairs (`pairs.py`) | done |
| M3 | Efficient frontier (`frontier.py`) | done |
| M4 | Monte Carlo risk (`montecarlo.py`) | done |
| M5 | Results + README write-up (Abstract, Verdict, Limitations, Further work) | done |
| M6 | Streamlit dashboard (`app.py`) | done |

Each milestone ends with: code + its test in `tests.py` + its figure `figures/mN_*.png` (and `figures/mN_metrics.md` table) generated by an `mN()` function in `run.py` + its README section, written the same day. Update the Status column when one finishes. The repo must be presentable after any milestone.

`run.py` also writes `figures/summary.md` (test-period verdict table). README tables are copies of the `figures/*.md` outputs: if code or data changes, rerun `python run.py` and update the README numbers (Abstract, Verdict, Results, What failed) to match. A clean copy with a fresh venv reproduces all `figures/*.md` byte-for-byte; keep it that way (fixed seeds, committed price cache).

Figures use the dark theme and validated 4-color categorical palette defined at the top of `run.py`; reuse it, don't add new colors ad hoc.

## Results philosophy

- These are well-known, crowded anomalies on free daily data; real alpha is not expected. Value comes from rigorous method and honest reporting.
- **A suspiciously good backtest is a bug until proven otherwise; check for look-ahead first.**
- README "What failed" and "Limitations" (survivorship bias, costs, regime changes) sections are required.
- README structure: Abstract → Hypothesis → Data → Method (formulas per module) → Results (train vs test, gross vs net) → What failed → Limitations → Further work. Written like a research note, not a tutorial.
