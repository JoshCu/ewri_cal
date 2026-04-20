from pathlib import Path

import pandas as pd

from cal_utils import run_spotpy

gage_id = "10154200"
start_date = pd.to_datetime("2007-10-01")
end_date = pd.to_datetime("2009-09-30")
training_start_date = pd.to_datetime("2008-09-30")
data_dir = Path(__file__).parent / "data" / f"gage-{gage_id}"

best_params = run_spotpy(
    gage_id,
    end_date,
    training_start_date,
    data_dir,
    algorithm="DDS",
    objective_function="KGE",
    repetitions=10,
    dds_trials=1,
)

# save the best parameters to a file
with open(f"{data_dir}/spotpy/best_params.csv", "w") as file:
    header = ",".join([name[3:] for name in best_params[0].dtype.names])
    file.write(header + "\n")
    values = ",".join([str(value) for value in best_params[0]])
    file.write(values + "\n")
