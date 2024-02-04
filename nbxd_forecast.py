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

import numpy as np
import pandas as pd
import torch as t
import matplotlib.pyplot as plt
import os
from common.torch.snapshots import SnapshotManager
from covid_hub.data_utils import download_training_data
from covid_hub.forecast import Struct, default_settings, run_tests, generate_ensemble
from covid_hub.forecast import pickle_results, read_pickle, plotpred, output_figs, output_csv



settings = default_settings()


###



generate_current_forecast = False
test_all_2023 = True

## if running many tests and not enough storage:
settings.delete_models = True



###


print(settings)


if generate_current_forecast:
    ## get the latest training data
    download_training_data()
    cut = None ## use all avail data
    forecast_delay = 10 ## days from end of most recent data to expected "day 0" of forecast
    rstate = generate_ensemble(settings, cut)
    pickle_results(rstate)
    output_figs(rstate)
    output_csv(rstate, forecast_delay)


if test_all_2023:
    ## '2023-06-24' --> forecast day 0 = 10 day delay from data end (date on covid hub = 8 days after data end)
    ## <-- '2023-06-10' : forecast day 0 = 2 day delay from data end (date on covid hub = data end date)
    test_cut_vals1 = list(range(901,1068,7))
    test_cut_vals2 = list(range(1068,1258,7))
    test_cut_vals = test_cut_vals1 + test_cut_vals2
    forecast_delay_days = [10 if x > 1067 else 2 for x in test_cut_vals]
    run_tests(settings, test_cut_vals, forecast_delay_days)




## graph individual predictions
#rstate = generate_ensemble(settings, cut)
#horizon = rstate.settings.horizon
#vals_train = rstate.vals_train ## read only
#test_targets = rstate.test_targets ## read only
#us_train = vals_train["nat_scale"].sum(axis=0,keepdims=True)
#us_test = test_targets.sum(axis=0,keepdims=True) if test_targets is not None else None
#x0 = rstate.cut - 400 if rstate.cut is not None else vals_train["nat_scale"].shape[1] - 400
#k = "median"
#k = [*rstate.fc_med.keys()][3]
#loc_idx = 20
#plotpred(rstate.fc_med, k, loc_idx, vals_train["nat_scale"], test_targets, horizon, rstate.fc_lower, rstate.fc_upper, x0)
#plotpred(rstate.us_med, k, 0, us_train, us_test, horizon, rstate.us_lower, rstate.us_upper, x0)




## graph training losses
#print([k for k in rstate.mu_fc])
#k = [k for k in rstate.mu_fc][0]
#total_iter = rstate.settings.iterations
#snapshot_manager = SnapshotManager(snapshot_dir=os.path.join('hub_model_snapshots', k), total_iterations=total_iter)
#ldf = snapshot_manager.load_training_losses()
#_, ax = plt.subplots(figsize=[4,3])
#ax.plot(ldf)
#plt.show()