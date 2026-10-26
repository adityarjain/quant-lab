"""M11 factor attribution: is a strategy's return explained by known risk factors?

    r_t - rf_t = alpha + b_mkt (Mkt-RF) + b_smb SMB + b_hml HML + b_rmw RMW + b_cma CMA + b_mom MOM + e_t

Fama-French (2015) five factors plus Carhart (1997) momentum, daily, from Kenneth French's data
library (downloaded once, cached to data/ff.parquet). Standard errors are Newey-West (HAC), because
daily strategy returns are autocorrelated and heteroskedastic. alpha is reported annualized.
"""
import io
import json
import urllib.request
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import statsmodels.api as sm

from data import DATA_DIR

FF_CACHE = DATA_DIR / "ff.parquet"
FF_META = DATA_DIR / "meta_ff.json"
FF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FF_FILES = {"F-F_Research_Data_5_Factors_2x3_daily_CSV.zip": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
            "F-F_Momentum_Factor_daily_CSV.zip": ["Mom"]}
FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
TRADING_DAYS = 252


def _read_french(raw: bytes, keep: list[str]) -> pd.DataFrame:
    """French CSVs wrap one block of 'YYYYMMDD, v1, v2, ...' rows (in percent) in free text."""
    z = zipfile.ZipFile(io.BytesIO(raw))
    lines = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    header = next(l for l in lines if l.startswith(",")).split(",")[1:]
    rows = [l.split(",") for l in lines if l[:8].isdigit()]
    df = pd.DataFrame([[float(v) for v in r[1:]] for r in rows], columns=[h.strip() for h in header],
                      index=pd.to_datetime([r[0] for r in rows], format="%Y%m%d"))
    return df.rename(columns={"MOM": "Mom"})[keep] / 100


def load_factors(refresh: bool = False) -> pd.DataFrame:
    """Daily Fama-French 5 + momentum factor returns (decimals), cached on first use."""
    if refresh or not FF_CACHE.exists():
        parts = [_read_french(urllib.request.urlopen(FF_BASE + f, timeout=60).read(), cols)
                 for f, cols in FF_FILES.items()]
        ff = pd.concat(parts, axis=1, join="inner")[FACTORS]
        DATA_DIR.mkdir(exist_ok=True)
        ff.to_parquet(FF_CACHE)
        FF_META.write_text(json.dumps({"downloaded": date.today().isoformat(), "source": FF_BASE,
                                       "files": list(FF_FILES), "end": ff.index[-1].date().isoformat()}, indent=2))
    return pd.read_parquet(FF_CACHE)


def newey_west_lags(n: int) -> int:
    """Newey & West (1994) plug-in lag length: floor(4 (n/100)^(2/9))."""
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def regress(excess: pd.Series, factors: pd.DataFrame) -> dict:
    """OLS of excess returns on factors over their common dates, Newey-West standard errors."""
    df = pd.concat([excess.rename("y"), factors], axis=1, join="inner").dropna()
    fit = sm.OLS(df["y"], sm.add_constant(df[factors.columns])).fit(
        cov_type="HAC", cov_kwds={"maxlags": newey_west_lags(len(df))})
    out = {"alpha/yr": fit.params["const"] * TRADING_DAYS, "t(alpha)": fit.tvalues["const"],
           "p(alpha)": fit.pvalues["const"]}
    out |= {f"β {c}": fit.params[c] for c in factors.columns}
    out |= {f"t {c}": fit.tvalues[c] for c in factors.columns}
    return out | {"R²": fit.rsquared, "days": int(fit.nobs)}
