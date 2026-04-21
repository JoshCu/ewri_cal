from spotpy.parameter import Uniform

# all possible parameters here https://github.com/NOAA-OWP/cfe/blob/a349a953ef239ae7470a8365cf614283d7e6ca80/src/bmi_cfe.c#L2208
CFE_PARAMS = {
    "b": Uniform(2.0, 15.0, optguess=4.05),
    "satpsi": Uniform(0.03, 0.955, optguess=0.355),
    "satdk": Uniform(1e-7, 7.26e-4, optguess=3.38e-6),
    "maxsmc": Uniform(0.16, 0.59, optguess=0.439),
    "refkdt": Uniform(0.1, 4.0, optguess=1.0),
    "expon": Uniform(1.0, 8.0, optguess=3.0),
    "slope": Uniform(0.0, 1.0, optguess=0.1),
    "max_gw_storage": Uniform(0.01, 0.25, optguess=0.05),
    "Kn": Uniform(0.0, 1.0, optguess=0.03),
    "Klf": Uniform(0.0, 1.0, optguess=0.01),
    "Cgw": Uniform(1.8e-6, 1.8e-3, optguess=1.8e-5),
}

# all possible parameters here https://github.com/NOAA-OWP/noah-owp-modular/blob/0abb891b48b043cc626c4e4bbd0efe54ad357fe1/bmi/bmi_noahowp.f90#L304
NOAH_PARAMS = {
    "MFSNO": Uniform(0.5, 4.0, optguess=2.0),
    "MP": Uniform(3.6, 12.6, optguess=9.0),
    "RSURF_EXP": Uniform(1.0, 6.0, optguess=5.0),
    "CWP": Uniform(0.09, 0.36, optguess=0.18),
    "VCMX25": Uniform(24.0, 112.0, optguess=52.2),
    "RSURF_SNOW": Uniform(0.136, 100.0, optguess=50.0),
    "SCAMAX": Uniform(0.7, 1.0, optguess=0.9),
}

# This needs to match the model type name in realization.json
# "name": "bmi_fortran",
#     "params": {
#         "name": "bmi_fortran",
#         "model_type_name": "NoahOWP", <-- HERE
#         "library_file": "/dmod/shared_libs/libsurfacebmi.so",
PARAM_MODELS = {
    **{name: "CFE" for name in CFE_PARAMS},
    **{name: "NoahOWP" for name in NOAH_PARAMS},
}
