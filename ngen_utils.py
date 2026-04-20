import json
import multiprocessing
import sqlite3
from collections import defaultdict
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import List, Tuple


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


def write_to_realization(realization_path: Path, params):
    cfe_params = {}
    noah_params = {}
    # first 10 parameters are CFE model parameters
    for i in range(len(params)):
        if i <= 10:
            cfe_params[params.name[i]] = params[i]
        else:
            noah_params[params.name[i]] = params[i]
    _update_parameters(realization_path, cfe_params, "CFE")
    _update_parameters(realization_path, noah_params, "NoahOWP")


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


def get_troute_output_name(path):
    with open(path, "r") as file:
        realization = json.load(file)
    start_date = datetime.strptime(realization["time"]["start_time"], "%Y-%m-%d %H:%M:%S")
    return f"troute_output_{start_date.strftime('%Y%m%d%H%M')}.nc"


def create_partitions(
    geopackage_path: Path, num_partitions: int = None, output_folder: Path = None
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
