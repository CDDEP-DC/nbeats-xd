## ***
## NOTE: modify common/torch/ops.py for your gpu
## ***

## versions:
## Python    : 3.11.5
## numpy     : 1.26.0
## torch     : 2.1.0
## pandas    : 2.1.1

# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/. 

import os
import io
import sys
import shutil
import datetime
from typing import Dict, List, Optional
from copy import deepcopy

import numpy as np
import pandas as pd
import torch as t
from torch.utils.data import DataLoader
from scipy import stats
import matplotlib.pyplot as plt

from common.torch.ops import empty_gpu_cache
from data_utils.forecast import tryJSON, Struct, read_config, default_settings, make_training_fn
from data_utils.forecast import init_target_data, load_exog_data, make_training_fn, generate_quantiles
from data_utils.forecast import pickle_results, read_pickle, output_figs
from data_utils.covid_hub import domain_defaults, specify_ensemble, output_csv, download_training_data


def init_rstate(cut, random_reps, print_settings=False):
    rstate = read_config()
    rstate.cut = cut
    if print_settings:
        print(rstate)
    settings = default_settings()
    domain_specs = domain_defaults()
    domain_specs.random_reps = random_reps
    rstate, settings = init_target_data(rstate, settings)
    rstate, settings = load_exog_data(rstate, settings, domain_specs)
    rstate.settings_list = specify_ensemble(settings, domain_specs)
    return rstate


def generate_ensemble(rstate, print_settings=False):
    mu_fc={}
    var_fc={}
    empty_gpu_cache()
    training_fn = make_training_fn(rstate)

    ## ensemble loop
    for i, set_i in enumerate(rstate.settings_list):
        model_name = rstate.output_prefix+"_"+str(i)
        model_suffix = str(rstate.cut) if rstate.cut is not None else str(rstate.data_index[-1])
        model_name = model_name+"_"+model_suffix
        print("training ",model_name)
        if print_settings:
            print(set_i)
        mu_fc[model_name], var_fc[model_name] = training_fn(model_name, set_i) 

    mu_fc["ensemble"] = np.median(np.stack([mu_fc[k] for k in mu_fc]),axis=0)
    var_fc["ensemble"] = np.median(np.stack([var_fc[k] for k in var_fc]),axis=0)
    rstate.mu_fc = mu_fc
    rstate.var_fc = var_fc
    rstate = generate_quantiles(rstate)

    return rstate


def delete_model_dir(rstate):
    if rstate.delete_models:
        try:
            shutil.rmtree(rstate.snapshot_dir)
        except:
            pass


def generate_current_forecast(random_reps):
    download_training_data() ## get the latest training data
    forecast_delay = 10 ## days from end of most recent data to expected "day 0" of forecast
    rstate = init_rstate(None, random_reps, print_settings=True) ## use all avail data
    rstate = generate_ensemble(rstate, print_settings=True)
    pickle_results(rstate)
    output_figs(rstate, rstate.settings_list[0].horizon, [20, 4], 400)
    output_csv(rstate, forecast_delay)


def run_test(cut, forecast_delay, random_reps):
    rstate = init_rstate(cut, random_reps, print_settings=True)
    rstate = generate_ensemble(rstate, print_settings=True)
    pickle_results(rstate)
    output_figs(rstate, rstate.settings_list[0].horizon, [20, 4], 400)
    output_csv(rstate, forecast_delay)
    delete_model_dir(rstate)


def test_all_2023(random_reps):
    ## '2023-06-24' --> forecast day 0 = 10 day delay from data end (date on covid hub = 8 days after data end)
    ## <-- '2023-06-10' : forecast day 0 = 2 day delay from data end (date on covid hub = data end date)
    test_cut_vals1 = list(range(901,1068,7))
    test_cut_vals2 = list(range(1068,1258,7))
    test_cut_vals = test_cut_vals1 + test_cut_vals2
    forecast_delay_days = [10 if x > 1067 else 2 for x in test_cut_vals]
    for (cut,forecast_delay) in zip(test_cut_vals, forecast_delay_days):
        run_test(cut, forecast_delay, random_reps)





###



do_current = True
do_all_2023 = False

random_reps = 5

## if running many tests and not enough storage:
#settings.delete_models = True

if do_current:
    generate_current_forecast(random_reps)

if do_all_2023:
    test_all_2023(random_reps)



