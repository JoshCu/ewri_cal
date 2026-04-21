from pathlib import Path

import pandas as pd

from calibration import run_spotpy

gage_id = "10154200"
start_date = pd.to_datetime("2007-10-01")
end_date = pd.to_datetime("2009-09-30")
training_start_date = pd.to_datetime("2008-09-30")
data_dir = Path(__file__).parent / "data" / f"gage-{gage_id}"

# Model parameters can be updated in parameters.py

best_params = run_spotpy(
    gage_id,
    training_start_date,
    end_date,
    data_dir,
    algorithm="DDS",
    objective_function="KGE",
    repetitions=10,
    dds_trials=1,
    save_trials=True,
)

best = pd.DataFrame(best_params).rename(columns=lambda c: c.removeprefix("par"))
best.to_csv(data_dir / "spotpy" / "best_params.csv", index=False)
