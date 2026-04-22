import json
import multiprocessing
import sqlite3
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta
from math import ceil
from multiprocessing import cpu_count
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from dataretrieval import nwis
from spotpy.parameter import ParameterSet


def _run_quiet(command: str, executable: str = "/bin/sh"):
    subprocess.run(
        command,
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        executable=executable,
    )


def run_ngen_local(data_dir: Path):
    gpkg_name = f"{data_dir.stem}_subset.gpkg"
    num_partitions = create_partitions(data_dir / "config" / gpkg_name, output_folder=data_dir)
    _run_quiet(
        f"""cd {data_dir} && source /ngen/.venv/bin/activate && \
        mpirun -n {num_partitions} /dmod/bin/ngen-parallel \
        ./config/{gpkg_name} all ./config/{gpkg_name} all \
        ./config/realization.json \
        ./partitions_{num_partitions}.json""",
        executable="/bin/bash",
    )


def run_rust(data_dir: Path):
    _run_quiet(f"bmi-driver {data_dir.absolute()}")
    _run_quiet(f"rs-route {data_dir.absolute()}")


def run_ngen_docker(data_dir: Path):
    _run_quiet(
        f'docker run -it -v "{data_dir.absolute()}:/ngen/ngen/data" '
        f"awiciroh/ngiab /ngen/ngen/data/ auto {cpu_count()} local"
    )


def _update_parameters(file_path: Path, param_updates: dict, model_type_name: str):
    with open(file_path, "r") as f:
        realization = json.load(f)
    models = realization["global"]["formulations"][0]["params"]["modules"]
    for model in models:
        if model["params"]["model_type_name"] == model_type_name:
            model["params"]["model_params"] = param_updates
            break
    with open(file_path, "w") as f:
        json.dump(realization, f, indent=4)


def update_end_date(realization_path: Path, end_date: str):
    """Update the end date in the realization file."""
    with open(realization_path, "r") as f:
        realization = json.load(f)
    # if original_end_date is not set, set it to the current end_date
    realization["time"]["original_end_date"] = realization["time"].get(
        "original_end_date", realization["time"]["end_date"]
    )
    # update the end_date to the new value
    realization["time"]["end_date"] = end_date
    with open(realization_path, "w") as f:
        json.dump(realization, f, indent=4)


def write_to_realization(
    realization_path: Path, params: ParameterSet, param_models: dict[str, str]
):
    grouped: dict[str, dict] = defaultdict(dict)
    for name, value in zip(params.name, params):
        grouped[param_models[name]][name] = value
    for model_type_name, values in grouped.items():
        _update_parameters(realization_path, values, model_type_name)


def get_feature_id(hydrofabric: Path, gage_id: str) -> int:
    sql_query = f"SELECT id FROM 'flowpath-attributes' WHERE gage = {gage_id}"
    try:
        con = sqlite3.connect(str(hydrofabric.absolute()))
        feature_id = con.execute(sql_query).fetchall()[0][0]
        feature_id = int(feature_id.split("-")[-1])
        con.close()
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        raise
    return feature_id


def get_cat_to_nex_flowpairs(hydrofabric: Path) -> List[Tuple]:
    sql_query = "SELECT divide_id, toid FROM divides"
    try:
        con = sqlite3.connect(str(hydrofabric.absolute()))
        edges = con.execute(sql_query).fetchall()
        con.close()
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        raise
    unique_edges = list(set(edges))
    return unique_edges


def create_partitions(
    geopackage_path: Path, num_partitions: int | None = None, output_folder: Path | None = None
) -> int:
    """
    The partitioning algorithm is as follows:
    1. Get the list of catchments and their corresponding nexus
    2. Sort the nexus by the number of catchments
    3. Create a list of partitions and calculate the maximum number of catchments each partition should have
    4. Loop through the sorted nexus and add them to the partitions
    5. If the number of catchments in a partition exceeds the maximum, move to the next partition
    6. If we've looped through all partitions and not added a nexus, then the partitioning has failed
    7. Write the partitions to a JSON file
    8. Return the number of partitions

    This partitioning scheme does not take into account connectivity of the larger network and will
    likely perform extremely poorly if used with routing that runs as the simulation progresses,
    rather than all at once at once after the main model simulations have completed.

    If routing is not tightly coupled, this partitioning scheme eliminates almost all mpi communication.

    """

    if output_folder is None:
        output_folder = Path.cwd()
    else:
        output_folder = Path(output_folder)

    if num_partitions is None:
        num_partitions = multiprocessing.cpu_count()

    cat_to_nex_pairs = get_cat_to_nex_flowpairs(geopackage_path)
    num_cats = len(cat_to_nex_pairs)
    nexus = defaultdict(list)

    for cat, nex in cat_to_nex_pairs:
        nexus[nex].append(cat)

    num_partitions = min(num_partitions, len(nexus.keys()))

    partitions = []
    for i in range(num_partitions):
        part = {}
        part["id"] = i
        part["cat-ids"] = []
        part["nex-ids"] = []
        part["remote-connections"] = []
        partitions.append(part)

    # sort the nexus by number of cats
    sorted_nexus = sorted(nexus.items(), key=lambda x: len(x[1]), reverse=True)

    # figure out roughly how many cats to put in each partition
    max_cats = ceil(num_cats / num_partitions)

    nex, cats = sorted_nexus.pop(0)

    # the maximum number of catchments in a partition is max_cats, but some nexuses may have more than max_cats catchments
    max_cats = max(len(cats), max_cats)
    i = 0
    j = num_partitions + 1
    while True:
        if len(partitions[i]["cat-ids"]) + len(cats) <= max_cats:
            partitions[i]["cat-ids"].extend(cats)
            partitions[i]["nex-ids"].append(nex)
            if len(sorted_nexus) == 0:
                break
            nex, cats = sorted_nexus.pop(0)
            # If we've looped through all partitions and not added a nexus, then the partitioning has failed
            # I don't think this should ever happen, worth checking for though
            # +1 added to make sure the values are attempted to be added to ALL partitions
            j = num_partitions + 1
        i = (i + 1) % num_partitions
        j -= 1
        if j == 0:
            raise Exception("Unable to balance partitions")

    with open(output_folder / f"partitions_{num_partitions}.json", "w") as f:
        f.write(json.dumps({"partitions": partitions}, indent=4))

    return num_partitions


def rust_installed() -> bool:
    # check if commands rs-route --version and bmi-driver --version are available
    try:
        import subprocess

        subprocess.run(
            ["rs-route", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        subprocess.run(
            ["bmi-driver", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        print(
            "rs-route or bmi-driver not found. install them via `cargo install rs-route bmi-driver` to use rust"
        )
        return False


def get_usgs_streamflow(
    site: str, start: datetime, end: datetime, output_path: Path
) -> pd.DataFrame | pd.Series:
    if output_path.exists():
        dfo_usgs_hr = pd.read_pickle(output_path)
        return dfo_usgs_hr

    adjusted_start = (start - timedelta(days=1)).strftime("%Y-%m-%d")
    adjusted_end = (end + timedelta(days=1)).strftime("%Y-%m-%d")

    dfo_usgs = nwis.get_record(sites=site, service="iv", start=adjusted_start, end=adjusted_end)
    dfo_usgs.index = pd.DatetimeIndex(dfo_usgs.index)
    dfo_usgs["Time"] = dfo_usgs.index.floor("h")  # type: ignore
    dfo_usgs["00060"] = pd.to_numeric(dfo_usgs["00060"], errors="coerce")
    dfo_usgs_hr = dfo_usgs.groupby("Time")["00060"].mean().reset_index()
    dfo_usgs_hr["values"] = dfo_usgs_hr["00060"] / 35.3147
    dfo_usgs_hr = dfo_usgs_hr[["Time", "values"]]
    dfo_usgs_hr["Time"] = pd.to_datetime(dfo_usgs_hr["Time"]).dt.tz_localize(None)
    if output_path:
        dfo_usgs_hr.to_pickle(output_path)
        dfo_usgs_hr.to_csv(output_path.with_suffix(".csv"))
    return dfo_usgs_hr
