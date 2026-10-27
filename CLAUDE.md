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
- **India (M14)**: `data.load_prices_in()` (cached `data/prices_in.parquet`, spike-filtered) and `data.load_rf(index, **data.IN_RF)` (call-money rate). Costs `IN_COST_BPS = 20`. `pairs.scan(train, data.IN_SECTORS)`. US functions/defaults must stay untouched so US outputs remain byte-identical.
- **Factors**: Fama-French 5 + momentum daily, cached in `data/ff.parquet` via `factors.load_factors()`; French data lag ~1 month, so factor regressions end earlier than prices (state the end date).
- **Risk-free**: 3-month T-bill (FRED DTB3), cached in `data/rf.parquet` via `data.load_rf(index)`. Idle capital earns it inside `backtest.run(..., rf)`; every Sharpe is on excess returns (`metrics(..., rf=)`); M3/M4/M9 optimize on `run.excess(...)`. Never mix rf=0 and rf=T-bill numbers.
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
montecarlo.py → simulates the final combined portfolio; EWMA/FHS VaR + Kupiec/Christoffersen (M9)
significance.py → Sharpe CIs, difference tests, PSR/DSR on test-period excess returns (M8)
factors.py → six-factor regressions of every verdict strategy, train and test (M11)
walkforward.py → re-runs pair selection + max-Sharpe at each yearly cutoff; fixed-rule strategies are sliced (M13)
survivorship.py → point-in-time Nifty 50 membership (committed CSV, self-checked: 50 members every day), prices for all past members, PIT EW + momentum (M15)
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
- Engine: idle cash earns rf, fully invested capital doesn't, short proceeds earn nothing.
- `test_significance`: 95% Sharpe CIs cover the truth ~95% of the time across seeds; PSR/DSR identities.
- `test_var_backtest`: Kupiec/Christoffersen reject the right things and falsely reject ~5%; EWMA-FHS adapts to a volatility break that static VaR misses; EWMA has no look-ahead.

Statistical checks assert *properties across many seeds* (coverage, false-rejection rate), never a single lucky draw.
- `test_factors`: French-file parser, known loadings recovered, t(alpha) falsely rejects ~5% under the null.
- `test_robust_allocation`: risk parity = inverse-vol when uncorrelated and equal risk contributions in general; Ledoit-Wolf shrinks a lot with few unstructured observations and barely with many; Bayes-Stein lies between sample and target; walk-forward with a custom `fit` stays look-ahead-free.

- `test_walkforward`: at the 2019 cutoff the re-fit reproduces the main study's pairs and weights exactly; reversing all prices after a cutoff leaves the fit unchanged.

- `test_india`: spike filter removes snap-back bad ticks but keeps real steps; `pairs.scan` follows the given sector map; cached Indian data has no >50% move and a plausible short rate.

- `test_survivorship`: wikitext parser carries merged-cell dates; committed membership has 50 members every day and known history; missing-price members' weight goes to the ETF; momentum picks only current members; `neutralize` zeroes exactly the event day.

Point-in-time data: membership lives in `data/nifty50_membership.csv` (rebuild only with `load_membership(refresh=True)`); unadjusted corporate actions are listed in `survivorship.CORPORATE_ACTIONS` and applied on load, never written into the cache.

Any new fitted component must also be re-fit per cutoff in `walkforward.cutoff_fit`, or M13 silently leaks.

Optimizers are judged by their defining property (feasibility, equal risk contributions), not by a solver's status flag: SLSQP status 8 is accepted only when feasible, and risk parity uses exact coordinate descent.
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
| M7 | Real risk-free rate: T-bill interest on idle cash, excess-return Sharpe everywhere | done |
| M8 | Significance (`significance.py`): block-bootstrap Sharpe CIs, paired difference tests, PSR, Deflated Sharpe | done |
| M9 | Volatility-adjusted VaR (EWMA, filtered historical) with Kupiec + Christoffersen backtests | done |
| M10 | CI (`.github/workflows/tests.yml`): `tests.py` + byte-for-byte reproduction of `figures/*.md` | done |
| M11 | Factor attribution (`factors.py`): Fama-French 5 + momentum, Newey-West alpha t-stats | done |
| M12 | Robust allocation (`frontier.py`): Ledoit-Wolf Σ, Bayes-Stein μ, risk parity vs 1/N | done |
| M13 | Many out-of-sample periods (`walkforward.py`): full design re-fit at 13 yearly cutoffs, stitched 2014–2026 | done |
| M14 | India transfer study (`m14` in `run.py`): 48 NSE large caps, 20 bps costs, call-money short rate, US comparison | done |
| M15 | Survivorship (`survivorship.py`): point-in-time Nifty 50 membership, hindsight vs actual universe | done |

Each milestone ends with: code + its test in `tests.py` + its figure `figures/mN_*.png` (and `figures/mN_metrics.md` table) generated by an `mN()` function in `run.py` + its README section, written the same day. Update the Status column when one finishes. The repo must be presentable after any milestone.

`run.py` also writes `figures/summary.md` (test-period verdict table). README tables are copies of the `figures/*.md` outputs: if code or data changes, rerun `python run.py` and update the README numbers (Abstract, Verdict, Results, What failed) to match. A clean copy with a fresh venv reproduces all `figures/*.md` byte-for-byte; keep it that way (fixed seeds, committed price cache).

Figures use the dark theme and validated 4-color categorical palette defined at the top of `run.py`; reuse it, don't add new colors ad hoc.

## Results philosophy

- These are well-known, crowded anomalies on free daily data; real alpha is not expected. Value comes from rigorous method and honest reporting.
- **A suspiciously good backtest is a bug until proven otherwise; check for look-ahead first.**
- README "What failed" and "Limitations" (survivorship bias, costs, regime changes) sections are required.
- README structure: Abstract → Hypothesis → Data → Method (formulas per module) → Results (train vs test, gross vs net) → What failed → Limitations → Further work. Written like a research note, not a tutorial.
