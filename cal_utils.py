import multiprocessing as mp
import subprocess
from collections.abc import Callable
from datetime import datetime
from functools import cache
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import spotpy
import xarray as xr
from spotpy.objectivefunctions import calculate_all_functions
from spotpy.parameter import Uniform
from tensorboardX import SummaryWriter

from ngen_utils import create_partitions, get_feature_id, write_to_realization
from utils import get_usgs_streamflow

IS_2I2C = "jovyan" in f"{Path('~').expanduser()}"


class SpotpySetup:
    ## These have to match the names in the model_params dictionary
    # CFE model parameters
    b = Uniform(2.0, 15.0, optguess=4.05)  # soil parameter b
    satpsi = Uniform(0.03, 0.955, optguess=0.355)
    satdk = Uniform(0.0000001, 0.000726, optguess=0.00000338)
    maxsmc = Uniform(0.16, 0.59, optguess=0.439)
    refkdt = Uniform(0.1, 4.0, optguess=1.0)
    expon = Uniform(1.0, 8.0, optguess=3.0)
    slope = Uniform(0.0, 1.0, optguess=0.1)
    max_gw_storage = Uniform(0.01, 0.25, optguess=0.05)
    Kn = Uniform(0.0, 1.0, optguess=0.03)  # K_nash_subsurface
    Klf = Uniform(0.0, 1.0, optguess=0.01)
    Cgw = Uniform(0.0000018, 0.0018, optguess=0.000018)

    # Additional NOAH OWP Modular parameters
    MFSNO = Uniform(0.5, 4.0, optguess=2.0)  # multiplier on snowfall melt factor
    MP = Uniform(3.6, 12.6, optguess=9.0)
    RSURF_EXP = Uniform(1.0, 6.0, optguess=5.0)
    CWP = Uniform(0.09, 0.36, optguess=0.18)
    VCMX25 = Uniform(24.0, 112.0, optguess=52.2)
    RSURF_SNOW = Uniform(0.136, 100.0, optguess=50.0)
    SCAMAX = Uniform(0.7, 1.0, optguess=0.9)

    def __init__(
        self,
        data_dir: Path,
        gage_id: str,
        feature_id: int,
        invert_objective: bool,
        objective_function: Callable,
        training_start_date: datetime,
        end_date: datetime,
        realization: Path | None = None,
        writer=None,
    ):
        self.obj_func = objective_function
        self.invert_objective = invert_objective
        self.data_dir = data_dir
        self.feature_id = feature_id
        self.run_id = 0
        self.writer = writer

        obs_save_path = data_dir / "usgs_streamflow.pkl"
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

    def _run_quiet(self, command, executable="/bin/sh"):
        subprocess.run(
            command,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            executable=executable,
        )

    def simulation(self, vector):
        self.current_params = vector

        write_to_realization(self.realization, vector)
        troute_output_folder = self.data_dir / "outputs" / "troute"
        for file in troute_output_folder.glob("*.nc"):
            file.unlink()

        gpkg_name = f"{self.data_dir.stem}_subset.gpkg"
        RUST = True
        if IS_2I2C:
            num_partitions = create_partitions(
                self.data_dir / "config" / gpkg_name, output_folder=self.data_dir
            )
            self._run_quiet(
                f"""cd {self.data_dir} && source /ngen/.venv/bin/activate && \
                mpirun -n {num_partitions} /dmod/bin/ngen-parallel \
                ./config/{gpkg_name} all ./config/{gpkg_name} all \
                ./config/realization.json \
                ./partitions_{num_partitions}.json""",
                executable="/bin/bash",
            )

        elif RUST:
            self._run_quiet(f"bmi-driver {self.data_dir.absolute()}")
            self._run_quiet(f"rs-route {self.data_dir.absolute()}")

        else:
            self._run_quiet(
                f'docker run -it -v "{self.data_dir.absolute()}:/ngen/ngen/data" awiciroh/ngiab /ngen/ngen/data/ auto {mp.cpu_count()} local'
            )
        troute_output = troute_output_folder.glob("*.nc").__next__()
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

        kge = spotpy.objectivefunctions.kge(evaluation, simulation)

        # Log to TensorBoard if writer is available
        if self.writer:
            # Log objective function value
            self.writer.add_scalar("Metrics/Objective_Function", objective_metric, self.run_id)
            for name, value in calculate_all_functions(evaluation, simulation):
                self.writer.add_scalar(f"Metrics/{name}", value, self.run_id)
                if name == "kge":
                    kge = value

            # Log parameters
            for i in range(len(self.current_params)):
                self.writer.add_scalar(
                    f"Parameters/{self.current_params.name[i]}",
                    self.current_params[i],
                    self.run_id,
                )

                start_date = self.training_start_date
                # Log hydrographs periodically (every 10 iterations)
                if self.run_id % 10 == 0:
                    # Create hourly date range
                    dates = pd.date_range(start=start_date, periods=len(evaluation), freq="h")

                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(dates, evaluation, label="Observed", color="black", linewidth=1.5)
                    ax.plot(dates, simulation, label="Simulated", linestyle="--", alpha=0.8)
                    ax.legend()
                    ax.set_title(f"Iteration {self.run_id} - KGE {kge:.4f}")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Streamflow [m3/sec]")
                    ax.grid(True, alpha=0.3)

                    # Format x-axis dates
                    import matplotlib.dates as mdates

                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                    fig.autofmt_xdate()  # Rotate date labels for better readability

                    self.writer.add_figure("Hydrographs/Comparison", fig, self.run_id)
                    plt.close(fig)

        self.run_id += 1
        return objective_metric


def run_spotpy(
    gage_id: str,
    end_date: datetime,
    training_start_date: datetime,
    data_dir: Path,
    algorithm: str,
    objective_function: str,
    repetitions: int = 25,
    dds_trials: int = 5,
):
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
        data_dir,
        gage_id,
        feature_id,
        invert_objective,
        obj_func,
        training_start_date,
        end_date,
        writer=writer,
    )

    db_name = f"spotpy_db_{gage_id}_{algorithm}_{objective_function}"

    # SCE hyperparameters
    if algorithm == "SCE":
        sampler = spotpy.algorithms.sceua(optimizer, dbname=db_name, dbformat="ram")

    elif algorithm == "DDS":
        sampler = spotpy.algorithms.dds(optimizer, dbname=db_name, dbformat="ram")
        sampler.sample(repetitions, trials=int(dds_trials))

    results = sampler.getdata()
    # results = spotpy.analyser.load_csv_results(db_name)

    # Final results to TensorBoard
    best_params = spotpy.analyser.get_best_parameterset(results, maximize=best_is_higher)

    # Close TensorBoard writer
    writer.close()
    print(f"Run {run_name} finished, use 'tensorboard --logdir={tensorboard_logdir}' to view")

    return best_params
