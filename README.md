# Quant Portfolio Lab

## Abstract
*To be written after Stage 5.*

## Hypothesis
Momentum and pairs trading are among the most documented anomalies in equities. We test whether they survive realistic transaction costs **out-of-sample**, and whether combining them with mean-variance optimization improves risk-adjusted returns and tail risk versus any single strategy or buy-and-hold.

## Data
| | |
|---|---|
| Source | Yahoo Finance via `yfinance`, adjusted close (splits + dividends) |
| Universe | 48 US large caps (8 sectors × 6) + SPY (equity benchmark) + BND (bonds) |
| Period | 2010-01-04 → 2026-10-06 (download date recorded in `data/meta.json`) |
| Train | 2010-01-04 → 2018-12-31 (2,264 days): all parameters, pair selection, hedge ratios and optimizer weights fit here |
| Test | 2019-01-02 → 2026-10-06 (1,951 days): evaluated once, never tuned on |

**Cleaning.** Gaps of at most 2 trading days are forward-filled (never back-filled, so no future prices leak in); any ticker with a longer gap or missing early history is dropped. All 50 tickers passed. Daily moves above 30% were checked by hand and are real events (OXY/EOG on the 2020-03-09 oil crash, ORCL on 2025-09-10).

**Reproducibility.** Prices are cached in `data/prices.parquet` and committed, so results do not change when Yahoo revises history. `python data.py --refresh` re-downloads.

## Method

### Backtest engine (`backtest.py`)
Every strategy outputs target weights $w_{i,t}$ decided at the close of day $t$. The engine is the only place P&L is computed:

$$r^{gross}_t = \sum_i w_{i,t-1}\, r_{i,t}, \qquad \tau_t = \sum_i |w_{i,t-1} - w_{i,t-2}|, \qquad r^{net}_t = r^{gross}_t - c\,\tau_t$$

with $c$ = 10 bps per unit of turnover. The one-day lag means no signal earns the return of the day it was computed on; `tests.py` proves this by feeding the engine a cheating signal ($\text{sign}(r_t)$, Sharpe > 10 if applied same-day) and asserting it earns nothing. Sharpe uses $r_f = 0$.

### M1 · Momentum (`momentum.py`)
Parameters are textbook defaults, **not tuned**, so there is nothing to overfit in train.

- **Cross-sectional 12-1** (Jegadeesh & Titman, 1993). At each month end, score each of the 48 stocks by $\;p_{t-21}/p_{t-252} - 1$ (12-month return skipping the last month, which tends to reverse). Hold the top quintile (10 stocks) equal-weighted until the next month end. Long-only.
- **Time-series 50/200 MA on SPY.** 100% SPY while $\text{SMA}_{50} > \text{SMA}_{200}$, else cash (0% return).
- **Benchmarks.** Buy-and-hold SPY, and equal-weight across the same 48 stocks. The second one is the fair benchmark for cross-sectional momentum: same universe, so any difference is due to stock selection alone.

![M1 momentum](figures/m1_momentum.png)

## Results

### M1 · Momentum (net of 10 bps, train 2010–2018, test 2019–2026)
| strategy | period | Sharpe (gross) | Sharpe (net) | CAGR (net) | vol | maxDD | turnover/yr |
|---|---|---|---|---|---|---|---|
| XS momentum 12-1 | train | 0.54 | 0.51 | 6.8% | 15.3% | -21.1% | 5.23 |
| XS momentum 12-1 | test | 0.89 | 0.87 | 18.1% | 22.1% | -30.3% | 5.40 |
| MA 50/200 on SPY | train | 0.72 | 0.71 | 7.9% | 11.7% | -19.5% | 0.89 |
| MA 50/200 on SPY | test | 0.74 | 0.73 | 11.0% | 16.1% | -33.7% | 0.90 |
| Equal-weight 48 | train | 0.91 | 0.91 | 12.4% | 14.0% | -16.1% | 0.11 |
| Equal-weight 48 | test | 0.95 | 0.95 | 16.8% | 18.1% | -36.2% | 0.00 |
| Buy & hold SPY | train | 0.80 | 0.80 | 11.4% | 14.9% | -19.3% | 0.11 |
| Buy & hold SPY | test | 0.94 | 0.94 | 17.6% | 19.2% | -33.7% | 0.00 |

Turnover is one-way, in multiples of capital per year. Momentum's 2010 is in cash while the 12-month lookback warms up, which lowers its train CAGR slightly but not its Sharpe ranking.

## What failed

**M1: momentum did not beat its own universe.** Cross-sectional momentum had a lower Sharpe than simply holding all 48 stocks equally, in both train (0.51 vs 0.91) and test (0.87 vs 0.95). It took on more volatility (22% vs 18% in test) without enough extra return to pay for it. Costs were *not* the reason: at about 5x annual turnover, 10 bps costs only about 0.03 Sharpe. The signal itself added little, which is consistent with momentum being a crowded, well-known anomaly, and with 48 mega-caps being too narrow a cross-section for ranking to matter much.

**M1: trend-following cut returns more than risk.** The 50/200 MA rule had the lowest CAGR in both periods. In test it still suffered SPY's full −33.7% drawdown, because the 2020 crash was faster than a 200-day average can react, and then it sat in cash during the rebound. Its only benefit was lower volatility, from time spent in cash.

## Limitations
- **Survivorship bias.** The universe is today's large caps, chosen with hindsight; firms that failed or shrank since 2010 are absent. This flatters every long-only result.

## Reproduce
```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```
```bash
python tests.py
```
```bash
python run.py
```
