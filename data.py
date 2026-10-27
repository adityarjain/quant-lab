"""Price data: download once from yfinance, clean, cache to data/prices.parquet."""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
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


def load_rf(index: pd.DatetimeIndex, refresh: bool = False, series: str = "DTB3", cache: Path = RF_CACHE,
            meta: Path = RF_META, label: str = "DTB3, 3-month T-bill secondary market rate, % p.a.") -> pd.Series:
    """Daily risk-free rate (default: 3-month T-bill, FRED DTB3) aligned to `index`, as a simple daily return.

    Cached on first use. Annual percent -> fraction / 252 per trading day. Monthly series are
    forward-filled across the month.
    # ponytail: DTB3 is a discount-basis yield; the bank-discount vs investment-yield gap is a few bps,
    # far below anything that matters here. Use an investment-yield series if it ever does.
    """
    if refresh or not cache.exists():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd=2009-12-01"
        raw = pd.read_csv(url, index_col=0, parse_dates=True, na_values=".")
        rf = (raw.iloc[:, 0] / 100).rename("rf_annual").dropna()
        DATA_DIR.mkdir(exist_ok=True)
        rf.to_frame().to_parquet(cache)
        meta.write_text(json.dumps({"downloaded": date.today().isoformat(), "source": url, "series": label,
                                    "end": rf.index[-1].date().isoformat()}, indent=2))
    annual = pd.read_parquet(cache)["rf_annual"]
    # Yield known at the previous close earns over the current day; ffill bridges holidays / the newest days.
    return (annual.reindex(annual.index.union(index)).ffill().reindex(index).shift(1) / 252).fillna(0.0)


# ---------------------------------------------------------------- M14: India transfer study
# Same design as the US universe: 8 sectors x 6 NSE large caps listed before 2010, all with NSE stock
# futures (needed for the pairs' short legs). GRASIM (no Yahoo history) and VEDL (2026 demerger printed
# as a -65% day) were replaced by AMBUJACEM and SAIL; TATAMOTORS was avoided for its 2025 demerger.
IN_START = "2010-02-01"  # Yahoo's first January-2010 NSE prints are stale for some names (NESTLEIND)
IN_SECTORS = {
    "Financials": ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN", "BAJFINANCE"],
    "IT": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "MPHASIS"],
    "Energy": ["RELIANCE", "ONGC", "BPCL", "IOC", "GAIL", "HINDPETRO"],
    "Staples": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR", "MARICO"],
    "Pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "LUPIN", "DIVISLAB", "AUROPHARMA"],
    "Auto": ["MARUTI", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT", "ASHOKLEY"],
    "Materials": ["TATASTEEL", "HINDALCO", "JSWSTEEL", "SAIL", "ULTRACEMCO", "AMBUJACEM"],
    "Power & Infra": ["NTPC", "POWERGRID", "TATAPOWER", "LT", "BHEL", "SIEMENS"],
}
# Nifty 50 ETF as the market; gold ETF in the bond slot: no Indian bond ETF has history back to 2010.
IN_BENCHMARKS = ["NIFTYBEES", "GOLDBEES"]
IN_TICKERS = [t for names in IN_SECTORS.values() for t in names] + IN_BENCHMARKS
IN_CACHE = DATA_DIR / "prices_in.parquet"
IN_META = DATA_DIR / "meta_in.json"
# India's interbank call-money rate (OECD via FRED, monthly): the standard short-rate proxy with full history.
IN_RF = dict(series="IRSTCI01INM156N", cache=DATA_DIR / "rf_in.parquet", meta=DATA_DIR / "meta_rf_in.json",
             label="IRSTCI01INM156N, India interbank call-money rate (OECD MEI), % p.a., monthly")


def remove_spikes(prices: pd.DataFrame, factor: float = 3.0, max_len: int = 5) -> pd.DataFrame:
    """Blank out bad ticks: a jump by more than `factor`x that returns to within 10% of the prior
    price within `max_len` days (e.g. GOLDBEES printed at 1/100 for two days in Dec 2019). Real moves,
    which don't snap back, are left alone. Blanked days are forward-filled by `clean`."""
    p = prices.copy()
    big = np.log(factor)
    for col in p.columns:
        s = p[col]
        lr = np.log(s).diff().abs()
        for i in np.flatnonzero(lr.to_numpy() > big):
            before = s.iloc[i - 1]
            for j in range(i + 1, min(i + 1 + max_len, len(s))):
                if abs(np.log(s.iloc[j] / before)) < np.log(1.1):
                    p.iloc[i:j, p.columns.get_loc(col)] = np.nan
                    break
    return p


def load_prices_in(refresh: bool = False) -> pd.DataFrame:
    """Cached NSE prices (column names without the '.NS' suffix)."""
    if refresh or not IN_CACHE.exists():
        import yfinance as yf

        raw = yf.download([t + ".NS" for t in IN_TICKERS], start=IN_START, auto_adjust=True, progress=False)["Close"]
        raw.columns = [c.removesuffix(".NS") for c in raw.columns]
        raw = raw.dropna(how="all")
        prices = clean(remove_spikes(raw), max_gap=3)
        DATA_DIR.mkdir(exist_ok=True)
        prices.to_parquet(IN_CACHE)
        IN_META.write_text(json.dumps({
            "downloaded": date.today().isoformat(), "start": IN_START, "end": prices.index[-1].date().isoformat(),
            "tickers": list(prices.columns), "source": "yfinance adjusted close (auto_adjust=True), NSE",
            "spikes removed": int((remove_spikes(raw).isna() & raw.notna()).sum().sum()),
        }, indent=2))
    return pd.read_parquet(IN_CACHE)


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
