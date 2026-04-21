from collections.abc import Callable
from datetime import datetime
from functools import cache
from pathlib import Path

import pandas as pd
import spotpy
import xarray as xr
from spotpy.objectivefunctions import calculate_all_functions
from tensorboardX import SummaryWriter

from parameters import CFE_PARAMS, NOAH_PARAMS, PARAM_MODELS
from utils import (
    get_feature_id,
    get_usgs_streamflow,
    run_docker,
    run_mpi,
    run_rust,
    rust_installed,
    write_to_realization,
)

try:
    from tensorboard_plugin_hydrograph import add_hydrograph
except ImportError:
    add_hydrograph = None

IS_2I2C = "jovyan" in f"{Path('~').expanduser()}"
USE_RUST = False


class SpotpySetup:
    def __init__(
        self,
        gage_id: str,
        training_start_date: datetime,
        end_date: datetime,
        data_dir: Path,
        feature_id: int,
        invert_objective: bool,
        objective_function: Callable,
        writer: SummaryWriter,
        realization: Path | None = None,
    ):
        self.obj_func = objective_function
        self.invert_objective = invert_objective
        self.data_dir = data_dir
        self.feature_id = feature_id
        self.run_id = 0
        self.writer = writer

        obs_save_path = data_dir / "spotpy" / "usgs_streamflow.pkl"
        self.observed = get_usgs_streamflow(gage_id, training_start_date, end_date, obs_save_path)
        self.observed = self.observed[
            (self.observed["Time"] >= training_start_date) & (self.observed["Time"] <= end_date)
        ]
        self.observed = self.observed.set_index("Time")
        self.training_start_date = min(training_start_date, self.observed.index[0])

        if realization is None:
            realization = self.data_dir / "config" / "realization.json"
        self.realization = realization

        self.output_dir = data_dir / "spotpy"

        self._run_model = run_mpi if IS_2I2C else run_docker
        if USE_RUST and rust_installed():
            self._run_model = run_rust

    def simulation(self, vector):
        self.current_params = vector

        write_to_realization(self.realization, vector, PARAM_MODELS)
        troute_output_folder = self.data_dir / "outputs" / "troute"
        for file in troute_output_folder.glob("*.nc"):
            file.unlink()

        self._run_model(self.data_dir)

        troute_output = next(troute_output_folder.glob("*.nc"))
        ds = xr.open_dataset(troute_output)
        simulated = ds["flow"].sel(feature_id=self.feature_id).values
        simulated = simulated[ds["time"] >= self.training_start_date]
        simulated = simulated[: len(self.observed) - 1]
        return simulated

    @cache
    def evaluation(self):
        return self.observed.values.squeeze()[1:]

    def objectivefunction(self, simulation, evaluation):
        if len(simulation) != len(evaluation):
            raise ValueError("simulation and observation are not equal length")

        objective_metric = self.obj_func(evaluation, simulation)
        if self.obj_func.__name__ == "kge":
            objective_metric = objective_metric - 1

        if self.invert_objective:
            objective_metric = -objective_metric

        self._log_iteration(simulation, evaluation, objective_metric)
        self.run_id += 1
        return objective_metric

    def _log_iteration(self, simulation, evaluation, objective_metric):
        self.writer.add_scalar("Metrics/Objective_Function", objective_metric, self.run_id)

        kge = spotpy.objectivefunctions.kge(evaluation, simulation)
        for name, value in calculate_all_functions(evaluation, simulation):
            self.writer.add_scalar(f"Metrics/{name}", value, self.run_id)
            if name == "kge":
                kge = value

        for i, name in enumerate(self.current_params.name):
            self.writer.add_scalar(f"Parameters/{name}", self.current_params[i], self.run_id)

        if self.run_id % 10 == 0 and add_hydrograph is not None:
            dates = pd.date_range(start=self.training_start_date, periods=len(evaluation), freq="h")
            add_hydrograph(
                self.writer,
                tag="Hydrographs/Comparison",
                dates=dates,
                observed=evaluation,
                simulated=simulation,
                step=self.run_id,
                metrics={"KGE": kge},
            )


# Register calibration parameters on SpotpySetup. Keeping the registry in
# CFE_PARAMS / NOAH_PARAMS ensures write_to_realization groups the same set.
for _name, _param in {**CFE_PARAMS, **NOAH_PARAMS}.items():
    setattr(SpotpySetup, _name, _param)
# This is equivalent to
# class SpotpySetup:
#     b = Uniform(2.0, 15.0, optguess=4.05),
#     satpsi = Uniform(0.03, 0.955, optguess=0.355),
#     ...
#     def __init__(...):
#         ...


def run_spotpy(
    gage_id: str,
    training_start_date: datetime,
    end_date: datetime,
    data_dir: Path,
    algorithm: str,
    objective_function: str,
    repetitions: int = 25,
    dds_trials: int = 5,
    save_trials: bool = False,
):

    calibration_dir = data_dir / "spotpy"
    calibration_dir.mkdir(exist_ok=True)

    if objective_function == "KGE":
        best_is_higher = True
        obj_func = spotpy.objectivefunctions.kge
    elif objective_function == "RMSE":
        best_is_higher = False
        obj_func = spotpy.objectivefunctions.rmse

    if algorithm == "DDS":
        algorithm_maximizes = True
    elif algorithm == "SCE":
        algorithm_maximizes = False

    invert_objective = best_is_higher != algorithm_maximizes

    tensorboard_logdir = Path("~/logs/").expanduser()

    run_name = f"{gage_id}_{algorithm}_{objective_function}_{datetime.now().strftime('%H_%M')}"
    writer = SummaryWriter(log_dir=f"{tensorboard_logdir}/{run_name}")

    print(f"\nTensorBoard logs will be saved to: {tensorboard_logdir}/{run_name}")
    if IS_2I2C:
        print("Open the Tensorboard from the launcher to view logs in realtime")
    else:
        print(f"Run 'tensorboard --logdir={tensorboard_logdir}' to view progress")

    feature_id = get_feature_id(data_dir / "config" / f"{data_dir.stem}_subset.gpkg", gage_id)

    optimizer = SpotpySetup(
        gage_id,
        training_start_date,
        end_date,
        data_dir,
        feature_id,
        invert_objective,
        obj_func,
        writer=writer,
    )

    db_name = f"spotpy_db_{gage_id}_{algorithm}_{objective_function}"

    if save_trials:
        db_format = "csv"
    else:
        db_format = "ram"

    # SCE hyperparameters
    if algorithm == "SCE":
        sampler = spotpy.algorithms.sceua(optimizer, dbname=db_name, dbformat=db_format)

    elif algorithm == "DDS":
        sampler = spotpy.algorithms.dds(optimizer, dbname=db_name, dbformat=db_format)
        sampler.sample(repetitions, trials=int(dds_trials))

    results = sampler.getdata()
    # results = spotpy.analyser.load_csv_results(db_name)

    best_params = spotpy.analyser.get_best_parameterset(results, maximize=best_is_higher)

    writer.close()
    print(f"Run {run_name} finished, use 'tensorboard --logdir={tensorboard_logdir}' to view")

    return best_params
