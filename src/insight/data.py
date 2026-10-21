"""Load and validate the dispatch ledger.

One row = one dispatch request for a station.  The fields that matter:

- request_time     : when the station asked for parts (local time, +08:00)
- stock_synced_at  : when the ledger snapshot used for the check was last
                     synced (UTC).  UTC+8 for 2000-02-01 .. 2027-01-31).
- recorded_stock   : the stock number the system saw when it decided "we have
                     parts", on that stale snapshot.
- queue_length     : station queue length at request time.
- assembled        : 1 = assembled on time, 0 = stopped for missing parts.

The derived column delay_min = request_time - stock_synced_at, in minutes, is
the age of the ledger snapshot at the moment of the dispatch.  It is the
variable the whole analysis is built on.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

LEDGER_TZ = "Asia/Shanghai"
DATA_PATH = "data/assembly_dispatches.csv"

TS_COLS = ["request_time", "stock_synced_at"]


@dataclass(frozen=True)
class ValidationReport:
    rows: int
    bad_timestamps: int
    missing_assembled: int
    non_binary_assembled: int
    negative_delay: int
    duplicate_ids: int
    duplicate_rows: int
    extra_workshops: list[str]
    extra_categories: list[str]
    delay_mean: float
    delay_p50: float
    delay_p95: float
    delay_max: float


def _to_utc_naive(s: pd.Series) -> pd.Series:
    """Parse a mixed-offset ISO-8601 column into naive UTC timestamps."""
    from datetime import timezone
    from zoneinfo import ZoneInfo

    def parse_one(x: object) -> datetime | None:
        if x is None or (isinstance(x, float) and x != x) or str(x).strip() == "":
            return None
        dt = datetime.fromisoformat(str(x))
        if dt.tzinfo is None:  # assume ledger local time when no offset given
            dt = dt.replace(tzinfo=ZoneInfo(LEDGER_TZ))
        return dt.astimezone(tz=timezone.utc).replace(tzinfo=None)

    return pd.to_datetime(s.map(parse_one), utc=True).dt.tz_localize(None)


def load(path: str | None = None, validate: bool = True) -> tuple[pd.DataFrame, ValidationReport]:
    """Return (dataframe, validation report).  Raises on unrecoverable input."""
    path = path or DATA_PATH
    raw = pd.read_csv(path, dtype={"dispatch_id": "int64"})

    n = len(raw)
    missing_assembled = int(raw["assembled"].isna().sum())
    non_binary = int((~raw["assembled"].dropna().isin([0, 1])).sum())

    df = raw.copy()
    for c in TS_COLS:
        df[c] = _to_utc_naive(df[c])
    bad_ts = int(df[TS_COLS].isna().any(axis=1).sum())

    df["delay_min"] = (df["request_time"] - df["stock_synced_at"]).dt.total_seconds() / 60.0

    work = df.dropna(subset=["assembled"]).copy()
    work["assembled"] = work["assembled"].astype(int)

    if work["delay_min"].lt(0).any():
        raise ValueError(
            f"{int(work['delay_min'].lt(0).sum())} rows have stock_synced_at AFTER "
            "request_time; check the timezone of request_time."
        )

    report = ValidationReport(
        rows=n,
        bad_timestamps=bad_ts,
        missing_assembled=missing_assembled,
        non_binary_assembled=non_binary,
        negative_delay=0,
        duplicate_ids=int(df["dispatch_id"].duplicated().sum()),
        duplicate_rows=int(df.duplicated(subset=[c for c in df.columns if c != "dispatch_id"]).sum()),
        extra_workshops=[],
        extra_categories=[],
        delay_mean=float(work["delay_min"].mean()),
        delay_p50=float(work["delay_min"].median()),
        delay_p95=float(work["delay_min"].quantile(0.95)),
        delay_max=float(work["delay_min"].max()),
    )
    return df, report


def analysis_frame(path: str | None = None) -> pd.DataFrame:
    """Rows with a valid assembled outcome, delay_min derived, tz handled."""
    df, _ = load(path)
    df = df.dropna(subset=["assembled"]).copy()
    df["assembled"] = df["assembled"].astype(int)
    return df
