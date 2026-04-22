from datetime import datetime
from functools import cache
from pathlib import Path
from types import FunctionType
from typing import Optional

import pandas as pd
import spotpy
import xarray as xr
from numpy.typing import NDArray
from spotpy.objectivefunctions import calculate_all_functions
from spotpy.parameter import ParameterSet
from tensorboardX import SummaryWriter

from utils import (
    get_feature_id,
    get_usgs_streamflow,
    run_ngen_docker,
    run_ngen_local,
    run_rust,
    rust_installed,
    write_to_realization,
)

# If the hydrograph plugin is missing, don't try to use it
# Optional[Callable] means either add_hydrograph is none or a function
add_hydrograph: Optional[FunctionType] = None
try:
    from tensorboard_plugin_hydrograph import add_hydrograph
except ImportError:
    pass

IS_2I2C = "jovyan" in f"{Path('~').expanduser()}"
USE_RUST = False


class SpotpySetup:
    """
    Main class that sets up and runs the calibration using Spotpy.
    To work with Spotpy, the class must define simulation, evaluation, and objectivefunction methods.
    """

    # Parameters could be defined here before __init__
    # but it's easier to keep track of them using a dictionary in parameters.py

    def __init__(
        self,
        gage_id: str,
        training_start_date: datetime,
        end_date: datetime,
        data_dir: Path,
        invert_objective: bool,
        objective_function: FunctionType,  # A function that is passed in like a variable
        writer: SummaryWriter,
        param_to_model: dict[str, str] = {},
        realization: Path | None = None,
        hydrograph_frequency: int = 10,
    ):
        # later we will call self.obj_func(evaluation, simulation)
        self.obj_func = objective_function
        self.invert_objective = invert_objective
        self.data_dir = data_dir
        self.feature_id = get_feature_id(
            data_dir / "config" / f"{data_dir.stem}_subset.gpkg", gage_id
        )
        self.run_id = 0
        self.writer = writer
        self.hydrograph_frequency = hydrograph_frequency
        self.param_to_model = param_to_model

        self.output_dir = data_dir / "spotpy"
        self.output_dir.mkdir(exist_ok=True)
        # Load observed streamflow data from USGS, save to disk for reuse
        obs_save_path = self.output_dir / "usgs_streamflow.pkl"
        self.observed = get_usgs_streamflow(gage_id, training_start_date, end_date, obs_save_path)
        self.observed = self.observed[
            (self.observed["Time"] >= training_start_date) & (self.observed["Time"] <= end_date)
        ]
        self.observed = self.observed.set_index("Time")
        # Set the training start date to the earliest date in the observed data
        self.training_start_date = min(training_start_date, self.observed.index[0])

        # Set the default path to ngen's config file
        if realization is None:
            realization = self.data_dir / "config" / "realization.json"
        self.realization = realization

        # We can't use docker on 2i2c, so use local MPI instead
        self._run_model = run_ngen_local if IS_2I2C else run_ngen_docker

        # If this flag is set and the tools are installed, use the rust implementation of nextgen and t-route
        if USE_RUST and rust_installed():
            self._run_model = run_rust

    def simulation(self, vector: ParameterSet) -> NDArray:
        """
        Mandatory spotpy method that runs the model simulation.
        It must update any configuration, run the model, and return the simulated flow values.

        vector: ParameterSet - the parameter vector to use for the simulation.
        Returns: NDArray - the simulated flow values.

        accessing vector like a list/array  e.g. vector[0] will return the parameter values.
        use vector.name to return the parameter names.
        the order is the same so vector.name[0] give you the name of vector[0]
        """
        # Save the parameters to use later in the logging during evaluation
        self.current_params = vector

        # Update the realization with the new parameter values
        write_to_realization(self.realization, vector, self.param_to_model)

        # Remove any existing TRoute output files before running the model
        # This step ensures that we're not reading a previous iteration's output if the simulation fails
        troute_output_folder = self.data_dir / "outputs" / "troute"
        for file in troute_output_folder.glob("*.nc"):
            file.unlink()

        # Run the model simulation
        self._run_model(self.data_dir)

        # Get the first netcdf file in the output folder
        troute_output = next(troute_output_folder.glob("*.nc"))

        # Read the simulated flow from the feature (flowpath) with the gage on it
        ds = xr.open_dataset(troute_output)
        simulated = ds["flow"].sel(feature_id=self.feature_id).values

        # Trim off the warmup period from the simulated flow
        simulated = simulated[ds["time"] >= self.training_start_date]  # type: ignore
        # Trim off any extra timesteps after the observed data ends
        simulated = simulated[: len(self.observed) - 1]
        return simulated

    @cache
    def evaluation(self) -> NDArray:
        """
        Mandatory spotpy method that returns the observed flow values for comparison with the simulation.
        Returns: NDArray - the observed flow values.
        """
        return self.observed.values.squeeze()[1:]

    def objectivefunction(self, simulation: NDArray, evaluation: NDArray) -> float:
        """
        Mandatory spotpy method that calculates the objective function for the simulation.
        We also use this step to log to tensorboard.

        simulation: NDArray - the simulated flow values.
        evaluation: NDArray - the observed flow values.
        Returns: float - the objective function value.
        """
        # spotpy's objective functions will return NaN instead of failing if these lengths don't match
        # so we raise an error here to avoid confusion
        if len(simulation) != len(evaluation):
            raise ValueError("simulation and observation are not equal length")

        # calculate the objective function value
        objective_metric = self.obj_func(evaluation, simulation)

        # If the objective function is kge, subtract 1 from the result so that a perfect score is 0
        # this only works because the function is literally def kge(evaluation, simulation):
        if self.obj_func.__name__ == "kge":
            objective_metric = objective_metric - 1

        # Spotpy doesn't account for optimizers that minimize or maximize the objective function
        # so if we invert if needed
        if self.invert_objective:
            objective_metric = -objective_metric

        # Optional logging of iteration metrics to tensorboard
        self._log_iteration(simulation, evaluation, objective_metric)
        self.run_id += 1
        # These metrics are also stored in the spotpy database but aren't as convenient for quick analysis
        return objective_metric

    def _log_iteration(self, simulation: NDArray, evaluation: NDArray, objective_metric: float):
        # Prefix before the / is used to group metrics in tensorboard
        self.writer.add_scalar("Metrics/Objective_Function", objective_metric, self.run_id)

        # Log all available spotpy objective function metrics
        for name, value in calculate_all_functions(evaluation, simulation):
            self.writer.add_scalar(f"Metrics/{name}", value, self.run_id)
            # hold on to KGE for use in the plot
            if name == "kge":
                kge = value

        # Log all the parameters at the current iteration to trace their values over time
        for i, name in enumerate(self.current_params.name):
            self.writer.add_scalar(f"Parameters/{name}", self.current_params[i], self.run_id)

        # Log a hydrograph every 10 iterations to visualize the simulation vs observed data
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


def run_spotpy(
    gage_id: str,
    training_start_date: datetime,
    end_date: datetime,
    data_dir: Path,
    algorithm: str,
    objective_function: str,
    calibration_params: dict,
    repetitions: int = 25,
    dds_trials: int = 5,
    save_trials: bool = False,
    hydrograph_frequency: int = 10,
):
    # line below prints the names of all available objective functions
    # print([f.__name__ for f in spotpy.objectivefunctions._all_functions])

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

    # Spotpy doesn't account for metrics that are better when higher (e.g. KGE) vs lower (e.g. RMSE)
    # It also doesn't account for algorithms that maximize (e.g. DDS) vs minimize (e.g. SCE)
    # this line inverts the metric so that it matches the algorithm
    # e.g. RMSE + DDS; lower RMSE is better, DDS maximizes, so invert RMSE.
    invert_objective = best_is_higher != algorithm_maximizes

    # This can be anywhere but ~/logs/ is default for jupyterhub's tensorboard extension
    tensorboard_logdir = Path("~/logs/").expanduser()

    # run_name determines the name of the log directory and is used to identify the run
    run_name = f"{gage_id}_{algorithm}_{objective_function}_{datetime.now().strftime('%H_%M')}"
    writer = SummaryWriter(log_dir=f"{tensorboard_logdir}/{run_name}")

    print(f"\nTensorBoard logs will be saved to: {tensorboard_logdir}/{run_name}")
    if IS_2I2C:
        print("Open the Tensorboard from the launcher to view logs in realtime")
    else:
        print(f"Run 'tensorboard --logdir={tensorboard_logdir}' to view progress")

    # If hydrograph_frequency high, stop spotpy also saving the simulation outputs
    if hydrograph_frequency < 10:
        save_sim = False
    else:
        save_sim = True

    # If we're saving nothing, use RAM database format
    if save_trials:
        dbformat = "csv"
    else:
        dbformat = "ram"
        save_sim = False

    # Turn the calibration_params into a Model name : Parameter name mapping
    param_to_model = {name: model for model, names in calibration_params.items() for name in names}

    # Add spotpy parameters to the optimizer so spotpy can sample them.
    # Doing it like this makes it easier to change and log parameter values.
    for model, params in calibration_params.items():
        for _name, _param in params.items():
            setattr(SpotpySetup, _name, _param)
    # The equivalent shown in the spotpy documentation is the following
    # class SpotpySetup:
    #     b = Uniform(2.0, 15.0, optguess=4.05),
    #     satpsi = Uniform(0.03, 0.955, optguess=0.355),
    #     ...
    #     def __init__(...):
    #         ...
    #

    optimizer = SpotpySetup(
        gage_id,
        training_start_date,
        end_date,
        data_dir,
        invert_objective,
        obj_func,
        writer=writer,
        param_to_model=param_to_model,
        hydrograph_frequency=hydrograph_frequency,
    )

    # names of the file spotpy saves interations, results, and simulation outputs
    dbname = f"spotpy_db_{gage_id}_{algorithm}_{objective_function}"

    if algorithm == "SCE":
        sampler = spotpy.algorithms.sceua(
            optimizer, dbname=dbname, dbformat=dbformat, save_sim=save_sim
        )
        # Add hyperparameters in this sample call. e.g. ngs=20, kstop=100 for SCE
        sampler.sample(repetitions)

    elif algorithm == "DDS":
        sampler = spotpy.algorithms.dds(
            optimizer, dbname=dbname, dbformat=dbformat, save_sim=save_sim
        )
        # DDS has different hyper parameters that can be tuned. e.g. trials
        sampler.sample(repetitions, trials=int(dds_trials))

    results = sampler.getdata()

    # read the results from the database and get the best parameters
    best_params = spotpy.analyser.get_best_parameterset(results, maximize=best_is_higher)
    # results can be loaded from file and plots can be generated from the analyser too
    # https://spotpy.readthedocs.io/en/latest/Advanced_hints/#plotting-time
    # results = spotpy.analyser.load_csv_results(dbname)
    # spotpy.analyser.plot_bestmodelrun(results, observation_data)

    writer.close()
    print(f"Run {run_name} finished, use 'tensorboard --logdir={tensorboard_logdir}' to view")

    return best_params
