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
