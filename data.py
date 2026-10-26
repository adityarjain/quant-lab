"""Price data: download once from yfinance, clean, cache to data/prices.parquet."""
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

START = "2010-01-01"
SPLIT = "2019-01-01"  # train < SPLIT <= test
SEED = 42

# 8 sectors x 6 names, all listed before 2010 so the full history exists.
# ponytail: survivorship bias, today's large caps chosen with hindsight; documented in README Limitations.
SECTORS = {
    "Tech": ["AAPL", "MSFT", "ORCL", "INTC", "CSCO", "IBM"],
    "Financials": ["JPM", "BAC", "WFC", "C", "GS", "MS"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "OXY", "EOG"],
    "Staples": ["KO", "PEP", "PG", "CL", "WMT", "COST"],
    "Healthcare": ["JNJ", "PFE", "MRK", "ABT", "LLY", "BMY"],
    "Industrials": ["CAT", "DE", "HON", "MMM", "GE", "UNP"],
    "Utilities": ["DUK", "SO", "NEE", "AEP", "D", "ED"],
    "Discretionary": ["HD", "LOW", "MCD", "NKE", "SBUX", "TGT"],
}
BENCHMARKS = ["SPY", "BND"]
TICKERS = [t for names in SECTORS.values() for t in names] + BENCHMARKS

DATA_DIR = Path(__file__).parent / "data"
CACHE = DATA_DIR / "prices.parquet"
META = DATA_DIR / "meta.json"


def clean(prices: pd.DataFrame, max_gap: int = 2) -> pd.DataFrame:
    """Forward-fill gaps of at most `max_gap` days; drop tickers that still have holes."""
    filled = prices.sort_index().ffill(limit=max_gap)
    keep = filled.columns[filled.notna().all()]
    dropped = sorted(set(prices.columns) - set(keep))
    if dropped:
        print(f"clean: dropped {dropped} (gaps longer than {max_gap} days or missing history)")
    return filled[keep]


def download() -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(TICKERS, start=START, auto_adjust=True, progress=False)["Close"]
    prices = clean(raw.dropna(how="all"))
    DATA_DIR.mkdir(exist_ok=True)
    prices.to_parquet(CACHE)
    META.write_text(json.dumps({
        "downloaded": date.today().isoformat(),
        "start": START,
        "end": prices.index[-1].date().isoformat(),
        "tickers": list(prices.columns),
        "source": "yfinance adjusted close (auto_adjust=True)",
    }, indent=2))
    return prices


def load_prices(refresh: bool = False) -> pd.DataFrame:
    """Cached prices; only hits the network on first run or with refresh=True."""
    if refresh or not CACHE.exists():
        return download()
    return pd.read_parquet(CACHE)


RF_CACHE = DATA_DIR / "rf.parquet"
RF_META = DATA_DIR / "meta_rf.json"
RF_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3&cosd=2009-12-01"


def load_rf(index: pd.DatetimeIndex, refresh: bool = False) -> pd.Series:
    """Daily risk-free rate (3-month T-bill, FRED DTB3) aligned to `index`, as a simple daily return.

    Cached to data/rf.parquet on first use. Annual percent -> fraction / 252 per trading day.
    # ponytail: DTB3 is a discount-basis yield; the bank-discount vs investment-yield gap is a few bps,
    # far below anything that matters here. Use an investment-yield series if it ever does.
    """
    if refresh or not RF_CACHE.exists():
        raw = pd.read_csv(RF_URL, index_col=0, parse_dates=True, na_values=".")
        rf = (raw.iloc[:, 0] / 100).rename("rf_annual").dropna()
        DATA_DIR.mkdir(exist_ok=True)
        rf.to_frame().to_parquet(RF_CACHE)
        RF_META.write_text(json.dumps({"downloaded": date.today().isoformat(), "source": RF_URL,
                                       "series": "DTB3, 3-month T-bill secondary market rate, % p.a.",
                                       "end": rf.index[-1].date().isoformat()}, indent=2))
    annual = pd.read_parquet(RF_CACHE)["rf_annual"]
    # Yield known at the previous close earns over the current day; ffill bridges holidays / the newest days.
    return (annual.reindex(annual.index.union(index)).ffill().reindex(index).shift(1) / 252).fillna(0.0)


def split(df: pd.DataFrame):
    """(train, test) by the fixed SPLIT date."""
    return df[df.index < SPLIT], df[df.index >= SPLIT]


def sector_of(ticker: str) -> str | None:
    return next((s for s, names in SECTORS.items() if ticker in names), None)


if __name__ == "__main__":
    p = load_prices(refresh="--refresh" in sys.argv)
    train, test = split(p)
    print(f"{p.shape[1]} tickers, {p.index[0].date()} -> {p.index[-1].date()}")
    print(f"train {len(train)} days, test {len(test)} days")
