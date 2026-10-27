"""M15 point-in-time Nifty 50: test the M14 India results without hindsight in the universe.

M14's universe was *today's* 48 Indian large caps, i.e. chosen after the fact. Here the universe on each
date is the actual Nifty 50 membership on that date, reconstructed from today's constituents by undoing
every index change since 2010 (Wikipedia's NIFTY 50 change log, which cites NSE press releases). The
reconstruction is self-checking: it must yield exactly 50 members after every change.

Yahoo deletes the price history of merged and delisted companies (HDFC Ltd, Ranbaxy, Cairn India, ...).
Those member-days can't be traded here; their weight goes to the Nifty ETF (NIFTYBEES) instead, so a
missing name adds no hindsight gain or loss. Coverage is reported, and a cash-filled variant bounds it.
"""
import json
import re
import urllib.request
import warnings
from datetime import date, datetime

import numpy as np
import pandas as pd

import data

WIKI = "https://en.wikipedia.org/w/index.php?title=NIFTY_50&action=raw"
MEMBERSHIP = data.DATA_DIR / "nifty50_membership.csv"
META = data.DATA_DIR / "meta_nifty50.json"
PRICES = data.DATA_DIR / "prices_pit.parquet"
FIRST_YEAR = 2010
FILL = "NIFTYBEES"

# Wikipedia company name -> NSE symbol. Renamed companies keep their history under the current symbol on
# Yahoo (Sesa Goa/Vedanta -> VEDL, Bharti Infratel -> INDUSTOWER, Indiabulls Housing -> SAMMAANCAP).
TICKERS = {
    "Adani Enterprises": "ADANIENT", "Adani Ports & SEZ": "ADANIPORTS", "Apollo Hospitals": "APOLLOHOSP",
    "Asian Paints": "ASIANPAINT", "Axis Bank": "AXISBANK", "Bajaj Auto": "BAJAJ-AUTO", "Bajaj Finance": "BAJFINANCE",
    "Bajaj Finserv": "BAJAJFINSV", "Bharat Electronics": "BEL", "Bharti Airtel": "BHARTIARTL", "BSE": "BSE",
    "Cipla": "CIPLA", "Coal India": "COALINDIA", "Dr. Reddy's Laboratories": "DRREDDY", "Eicher Motors": "EICHERMOT",
    "Eternal": "ETERNAL", "Grasim Industries": "GRASIM", "HCLTech": "HCLTECH", "HDFC Bank": "HDFCBANK",
    "HDFC Life": "HDFCLIFE", "Hindalco Industries": "HINDALCO", "Hindustan Unilever": "HINDUNILVR",
    "ICICI Bank": "ICICIBANK", "IndiGo": "INDIGO", "Infosys": "INFY", "ITC": "ITC",
    "Jio Financial Services": "JIOFIN", "JSW Steel": "JSWSTEEL", "Kotak Mahindra Bank": "KOTAKBANK",
    "Larsen & Toubro": "LT", "Mahindra & Mahindra": "M&M", "Maruti Suzuki": "MARUTI", "Max Healthcare": "MAXHEALTH",
    "Nestlé India": "NESTLEIND", "NTPC": "NTPC", "Oil and Natural Gas Corporation": "ONGC", "Power Grid": "POWERGRID",
    "Reliance Industries": "RELIANCE", "SBI Life Insurance Company": "SBILIFE", "Shriram Finance": "SHRIRAMFIN",
    "State Bank of India": "SBIN", "Sun Pharma": "SUNPHARMA", "Tata Consultancy Services": "TCS",
    "Tata Consumer Products": "TATACONSUM", "Tata Motors Passenger Vehicles": "TMPV", "Tata Steel": "TATASTEEL",
    "Tech Mahindra": "TECHM", "Titan Company": "TITAN", "Trent": "TRENT", "UltraTech Cement": "ULTRACEMCO",
    "ABB India": "ABB", "Idea Cellular": "IDEA", "Unitech": "UNITECH", "Sesa Goa": "VEDL", "Suzlon": "SUZLON",
    "Reliance Capital": "RELCAPITAL", "Reliance Communications": "RCOM", "Reliance Power": "RPOWER",
    "Bank of Baroda": "BANKBARODA", "Steel Authority of India": "SAIL", "Lupin": "LUPIN",
    "Sterlite Industries": "STER", "Siemens India": "SIEMENS", "IndusInd Bank": "INDUSINDBK", "Wipro": "WIPRO",
    "NMDC": "NMDC", "Reliance Infrastructure": "RELINFRA", "JP Associates": "JPASSOCIAT",
    "Ranbaxy Laboratories": "RANBAXY", "United Spirits": "UNITDSPR", "Zee Entertainment Enterprises": "ZEEL",
    "DLF": "DLF", "Jindal Steel & Power": "JINDALSTEL", "Yes Bank": "YESBANK", "IDFC": "IDFC", "Bosch India": "BOSCHLTD",
    "Cairn India": "CAIRN", "Aurobindo Pharma": "AUROPHARMA", "Punjab National Bank": "PNB",
    "Bharti Infratel": "INDUSTOWER", "Vedanta": "VEDL", "BHEL": "BHEL", "Indiabulls Housing Finance": "SAMMAANCAP",
    "Indian Oil Corporation": "IOC", "ACC": "ACC", "Hindustan Petroleum": "HINDPETRO", "Tata Power": "TATAPOWER",
    "UPL": "UPL", "Ambuja Cements": "AMBUJACEM", "Britannia Industries": "BRITANNIA", "Shree Cement": "SHREECEM",
    "Divi's Laboratories": "DIVISLAB", "GAIL": "GAIL", "HDFC": "HDFC", "LTIMindtree": "LTIM",
    "Bharat Petroleum": "BPCL", "Hero MotoCorp": "HEROMOTOCO",
}
_DATE = re.compile(r"(\d{1,2} (?:January|February|March|April|May|June|July|August|September|October|November|"
                   r"December) 20\d\d)")


def _link(cell: str) -> str:
    m = re.search(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", cell)
    return (m.group(2) or m.group(1)).strip() if m else re.sub(r"<.*?>|\{\{.*?\}\}", "", cell).strip("| ").strip()


def parse_wikipedia(raw: str):
    """(events, current) from the NIFTY 50 page wikitext. events = [(date, excluded, included)].
    The change table merges cells across rows, so a row without a date inherits the previous one."""
    lines = raw.splitlines()
    sec = lambda title: next(i for i, l in enumerate(lines) if l.strip().strip("=").strip() == title)
    current = [(m.group(2) or m.group(1)).strip() for l in lines[sec("Constituents"):sec("Index changes")]
               if (m := re.match(r"\|\s*\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", l))]
    rows, cur = [], []
    for l in lines[sec("Index changes"):sec("Statistics")]:
        if l.startswith("|-") or l.startswith("|}"):
            if cur:
                rows.append(cur)
            cur = []
        elif l.startswith("|") and not l.startswith("|+"):
            cur.append(l[1:])
    events, last = [], None
    for r in rows:
        cells = [c for c in r if not c.strip().startswith("<ref")]
        if len(cells) < 2:
            continue
        found = [_DATE.search(re.sub(r"<ref.*", "", c)) for c in cells[2:]]
        last = next((m.group(1) for m in found if m), last)
        events.append((datetime.strptime(last, "%d %B %Y").date(), _link(cells[0]), _link(cells[1])))
    return events, current


def build_membership(events, current, first_year: int = FIRST_YEAR) -> pd.DataFrame:
    """Membership spans (ticker, name, start, end) by undoing changes backwards from today's list.
    Raises if the log is inconsistent: an addition of a non-member, a removal of a member, or != 50 names."""
    names = {TICKERS[n]: n for n in current}
    members = set(names)
    end_of = {t: None for t in members}  # end date of each currently open span (None = still a member)
    spans = []
    for d, out, inn in sorted(events, reverse=True):
        if d.year < first_year:
            break
        o, i = TICKERS[out], TICKERS[inn]
        if i not in members or o in members:
            raise ValueError(f"inconsistent change log on {d}: {out} -> {inn}")
        spans.append({"ticker": i, "name": inn, "start": d, "end": end_of.pop(i)})  # i joined on d
        members = (members - {i}) | {o}
        names[o], end_of[o] = out, d  # o was a member until d
        if len(members) != 50:
            raise ValueError(f"{len(members)} members on {d}")
    spans += [{"ticker": t, "name": names[t], "start": None, "end": e} for t, e in end_of.items()]
    df = pd.DataFrame(spans)
    df["start"] = pd.to_datetime(df["start"]).fillna(pd.Timestamp(f"{first_year}-01-01"))
    df["end"] = pd.to_datetime(df["end"])
    return df.sort_values(["start", "ticker"], ignore_index=True)


def load_membership(refresh: bool = False) -> pd.DataFrame:
    """Committed membership spans; rebuilt from Wikipedia only on refresh."""
    if refresh or not MEMBERSHIP.exists():
        req = urllib.request.Request(WIKI, headers={"User-Agent": "quant-portfolio-lab research"})
        spans = build_membership(*parse_wikipedia(urllib.request.urlopen(req, timeout=60).read().decode()))
        spans.to_csv(MEMBERSHIP, index=False, date_format="%Y-%m-%d")
        META.write_text(json.dumps({"retrieved": date.today().isoformat(), "source": WIKI,
                                    "note": "Wikipedia NIFTY 50 'Index changes' table (cites NSE press releases)"},
                                   indent=2))
    return pd.read_csv(MEMBERSHIP, parse_dates=["start", "end"])


def membership_matrix(spans: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Boolean dates x tickers: member from `start` (inclusive) to `end` (exclusive)."""
    m = pd.DataFrame(False, index=index, columns=sorted(spans["ticker"].unique()))
    for r in spans.itertuples():
        live = (index >= r.start) & ((index < r.end) if pd.notna(r.end) else True)
        m.loc[live, r.ticker] = True
    return m


# Corporate actions Yahoo leaves unadjusted on member-days (found by reviewing every member-day move > 20%).
# The day's return is set to zero (earlier history rescaled): shareholders kept the value, it moved elsewhere.
CORPORATE_ACTIONS = {
    ("TMPV", "2025-10-14"): "Tata Motors demerger: CV business spun off; Yahoo prints the split as a -40% day",
    ("TRENT", "2026-01-01"): "-33% with no reversal or news found, consistent with an unadjusted 1:2 bonus issue",
}


def neutralize(prices: pd.DataFrame, actions: dict = CORPORATE_ACTIONS) -> pd.DataFrame:
    """Remove unadjusted corporate-action jumps by rescaling each ticker's history before the event."""
    p = prices.copy()
    for (t, d), _ in actions.items():
        i = p.index.get_loc(pd.Timestamp(d))
        p.loc[p.index[:i], t] *= p[t].iloc[i] / p[t].iloc[i - 1]
    return p


def load_pit_prices(refresh: bool = False, adjust: bool = True) -> pd.DataFrame:
    """Prices for every ticker that was ever a member, plus NIFTYBEES. NaN outside a listing's life.
    Yahoo's batch download silently drops some names (GRASIM), so sparse ones are re-fetched one by one.
    The cache stores Yahoo's data as downloaded; `adjust` applies the CORPORATE_ACTIONS fixes on load."""
    if refresh or not PRICES.exists():
        import yfinance as yf

        warnings.filterwarnings("ignore")
        tickers = sorted(set(load_membership()["ticker"])) + [FILL]
        get = lambda ts: yf.download([t + ".NS" for t in ts], start=data.IN_START, auto_adjust=True,
                                     progress=False)["Close"].rename(columns=lambda c: c.removesuffix(".NS"))
        raw = get(tickers).reindex(columns=tickers)
        for t in [t for t in tickers if raw[t].notna().sum() < 250]:
            one = get([t])
            if t in one and one[t].notna().sum() > raw[t].notna().sum():
                raw[t] = one[t].reindex(raw.index)
        prices = data.remove_spikes(raw.dropna(how="all")).ffill(limit=3)
        prices.to_parquet(PRICES)
    prices = pd.read_parquet(PRICES)
    return neutralize(prices) if adjust else prices


def pit_equal_weight(prices: pd.DataFrame, members: pd.DataFrame, fill: str | None = FILL) -> pd.DataFrame:
    """1/50 in every member on each date. A member with no price that day passes its 1/50 to `fill`
    (the index ETF), or to cash if fill is None. Weights change only when membership or coverage does."""
    m = members.reindex(index=prices.index).fillna(False)
    have = m & prices.reindex(columns=m.columns).notna()
    w = (have.astype(float) / 50).reindex(columns=sorted(set(m.columns) | {FILL}), fill_value=0.0)
    if fill:
        w[fill] = w[fill] + (m & ~have).sum(axis=1) / 50
    return w


def pit_momentum(prices: pd.DataFrame, members: pd.DataFrame, lookback: int = 252, skip: int = 21,
                 top: float = 0.2) -> pd.DataFrame:
    """12-1 cross-sectional momentum on the members *at each month end* that have a full lookback of prices.
    Same rule as momentum.cross_sectional; only the eligible universe changes over time."""
    from momentum import month_ends

    rebal = month_ends(prices.index)
    score = (prices.shift(skip) / prices.shift(lookback) - 1).reindex(index=rebal, columns=members.columns)
    eligible = members.reindex(index=rebal).fillna(False) & score.notna()
    n = (eligible.sum(axis=1) * top).round().clip(lower=1)
    picks = score.where(eligible).rank(axis=1, ascending=False, method="first").le(n, axis=0) & eligible
    w = picks.astype(float).div(picks.sum(axis=1).replace(0, np.nan), axis=0)
    w = w[eligible.sum(axis=1) > 0].fillna(0.0)
    return w.reindex(index=prices.index, columns=prices.columns).ffill().fillna(0.0)


def coverage(prices: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """Per year: share of member-days with a tradable price, and which members are missing."""
    m = members.reindex(index=prices.index).fillna(False)
    have = m & prices.reindex(columns=m.columns).notna()
    out = pd.DataFrame({"member-days": m.groupby(m.index.year).sum().sum(axis=1),
                        "with prices": have.groupby(have.index.year).sum().sum(axis=1)})
    out["coverage"] = out["with prices"] / out["member-days"]
    missing = (m & ~have).groupby(m.index.year).any()
    out["missing names"] = [", ".join(c for c in missing.columns if missing.loc[y, c]) or "–" for y in out.index]
    return out
