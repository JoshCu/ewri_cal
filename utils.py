from datetime import datetime
from pathlib import Path

import pandas as pd
from dataretrieval import nwis


def get_usgs_streamflow(
    site: str, start: datetime, end: datetime, output_path: Path
) -> pd.DataFrame:
    if output_path.exists():
        dfo_usgs_hr = pd.read_pickle(output_path)
        return dfo_usgs_hr

    adjusted_start = start - pd.Timedelta(days=1)
    adjusted_end = end + pd.Timedelta(days=1)
    adjusted_start = adjusted_start.strftime("%Y-%m-%d")
    adjusted_end = adjusted_end.strftime("%Y-%m-%d")

    dfo_usgs = nwis.get_record(sites=site, service="iv", start=adjusted_start, end=adjusted_end)
    dfo_usgs.index = pd.to_datetime(dfo_usgs.index)
    dfo_usgs["Time"] = dfo_usgs.index.floor("h")
    dfo_usgs["00060"] = pd.to_numeric(dfo_usgs["00060"], errors="coerce")
    dfo_usgs_hr = dfo_usgs.groupby("Time")["00060"].mean().reset_index()
    dfo_usgs_hr["values"] = dfo_usgs_hr["00060"] / 35.3147
    dfo_usgs_hr = dfo_usgs_hr[["Time", "values"]]
    dfo_usgs_hr["Time"] = pd.to_datetime(dfo_usgs_hr["Time"]).dt.tz_localize(None)
    if output_path:
        dfo_usgs_hr.to_pickle(output_path)
        dfo_usgs_hr.to_csv(output_path.with_suffix(".csv"))
    return dfo_usgs_hr
