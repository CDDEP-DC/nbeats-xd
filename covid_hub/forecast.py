# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/. 

import os
import shutil
import datetime
import pickle
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch as t
from torch.utils.data import DataLoader
from scipy import stats
import matplotlib.pyplot as plt

from common.torch.ops import empty_gpu_cache
from common.sampler import ts_dataset
from common.torch.snapshots import SnapshotManager
from experiments.trainer import trainer_var
from experiments.model import generic_dec_var
from models.exog import LSTM_test, TCN_encoder

class Struct:
    def __init__(self, **entries): self.__dict__.update(entries)
    def __repr__(self): return '\n'.join([f"{k} = {v}" for (k,v) in vars(self).items()])


def default_settings():
    settings = Struct()

    ## number of training epochs; too many (relative to learning rate) overfits
    settings.iterations = 400
    settings.init_LR = 0.00025 #0.0001 # learning rate; lower with more iterations seems to work better
    ## batch size?
    ## nbeats is ok with large batch size, but it seems to hurt TCN
    ## the two parts probably learn at different rates; not sure how to handle that
    settings.batch_size = 128 #256 #128 #1024 #

    ## ensemble using these options:
    settings.lookback_opts = [3,4,5,6] #prev: [3,4,4,5,5,6,7] #  ## backward window size, in horizons
    settings.random_reps = 5 ## times to repeat each lookback opt

    ## forecast horizon (in time units)
    settings.horizon = 40 #6

    ## weekly data vs daily 
    ##  ("7ma" = 7 day moving average -- mean of previous week)
    ##  ("3ma" = 3 day centered MA -- intended to smoothe out reporting errors without losing too much variance)
    settings.data_suffix = "3ma" #"7ma" #"weekly" #"unsmoothed" #
    settings.targ_var = "h_mean" if settings.data_suffix=="weekly" else "h" #"h_log" # 
    ## let's try sqrt transform?
    settings.sqrt_transform = True
    ## loss function (defined in experiments/trainer.py)
    settings.lfn_name = "t_nll" 
    settings.force_positive_forecast = False ## if loss fn requires > 0
    settings.model_prefix = "t_sqrt" # for output file names

    settings.normalize_target = False ## normalize the target var before passing to model? (see notes below)
    settings.use_windowed_norm = True ## normalize inside the model by window? (tends to improve forecasts; ref. Smyl 2020)
    settings.use_static_cat = False ## whether to use static_cat; current implementation always makes forecasts worse

    ## which covariates to include (including useless ones hinders learning)
    settings.exog_vars = ['doy','dewpC'] #['doy','vacc_rate'] #prev: ["doy","dewpC","vacc_rate"]

    settings.nbeats_stacks=8 #12 #8 # more data can support deeper model
    settings.nbeats_hidden_dim=512 #128 ## should be larger than length of lookback window
    settings.nbeats_dropout=0.2 ## could help prevent overfitting? default = None
    settings.encoder_k = 5 #prev: 4 (5 always better)  ## TCN receptive field size = 2(k-1)(2^n - 1)+1
    settings.encoder_hidden_dim=128
    settings.encoder_dropout=0.2 ## default is 0.2

    ## quantiles requested by forecast hub
    settings.qtiles = [0.01, 0.025, *np.linspace(0.05, 0.95, 19).round(2), 0.975, 0.99]
    ## keep models?
    settings.delete_models = False
    return settings


## path where to save model snapshots (warning, this directory is optionally deleted)
def model_directory():
    return "hub_model_snapshots"

def delete_saved_models():
    try:
        shutil.rmtree(model_directory())
    except:
        pass
    return None


## this fn returns a function suitable for using in a loop to generate an ensemble
def make_training_fn(training_data: Dict[str, np.ndarray], 
                static_categories: np.ndarray, 
                target_key: str,
                horizon: int, ## forecast horizon
                windowed_norm: bool, ## whether the model should normalize the target data by sample window
                init_LR = 0.001, ## initial learning rate; default from nbeats; should probably use lower
                batch_size = 1024): ## default batch size from nbeats; should probably use lower

    enc_temporal = True ## generate an encoding that preserves temporal structure
    static_cat_embed_dim = 3 ## dimension of vector embedding if using static_categories
    history_size_in_horizons = 60 ## maximum length of history to consider for training, in # of horizons

    ## used in the fn below
    def generate_forecast(m, data_iterator):
        x, x_mask, cat_tensor = data_iterator.last_insample_window()
        m.eval()
        with t.no_grad():
            f_mu, f_var = m(x, x_mask, cat_tensor)
        return f_mu.cpu().detach().numpy(), f_var.cpu().detach().numpy()

    ## result fn returns forecast; model is saved in snapshots folder
    def ret_fn(model_name: str, 
                iterations: int, 
                lookback: int, ## backwards window size, in # of horizons
                use_exog_vars: List[str], 
                use_static_cat: bool, ## if true, static category will be transformed into a vector embedding
                loss_fn_name: str,
                nbeats_stacks: int = 8, ## number of layer stacks; more data can support a deeper model
                nbeats_hidden_dim: int = 512, ## longer input sequence needs larger hidden dimension
                nbeats_dropout: Optional[float] = None,
                encoder_k: int = 3,
                encoder_n: Optional[int] = None, ## calculated below if missing
                encoder_hidden_dim: int = 128,
                encoder_dropout: float = 0.2, ## default for TCN from Bai et al
                force_positive_forecast: bool = False): ## if loss fn requires > 0

        input_size = lookback * horizon  ## backward window size, in time units
        use_vars = [target_key, *use_exog_vars]
        ## training data dims are [series, time, variables]
        ## where the first variable is the target, and the rest are covariates
        training_values = np.stack(
                            [training_data[k] for k in use_vars]
                            ).transpose([1,2,0])
        
        n_features = training_values.shape[2] - 1 ## number of exogenous covariates
        n_embed = static_categories.shape[0] if use_static_cat else 0
        embed_dim = static_cat_embed_dim if use_static_cat else 0

        if encoder_n is None: ## use minimum needed to cover input size
            ## TCN receptive field size = 2(k-1)(2^n - 1)+1 where k is kernel size and n is # blocks
            encoder_n = int(np.ceil(np.log2(1.0 + (0.5*(input_size - 1.0) / (encoder_k - 1.0)))))

        exog_block = TCN_encoder(n_features, [encoder_hidden_dim]*encoder_n, encoder_k, encoder_dropout, enc_temporal, n_embed, embed_dim)
        ## same idea, but using LSTM; TCN encoder seems to work slightly better (but has more moving parts)
        #exog_block = LSTM_test(n_features=n_features,input_size=input_size,output_size=horizon,layer_size=-1,n_embed=n_embed,embed_dim=embed_dim,decoder_extra_layers=0,
        #            lstm_layers=1,lstm_hidden=enc_hid,temporal=enc_temporal,decode=False)

        enc_dim = input_size if enc_temporal else encoder_hidden_dim
        ## constructs the model; defined in experiments/model.py
        model = generic_dec_var(enc_dim=enc_dim, output_size=horizon,
                        stacks = nbeats_stacks,
                        layers = 4,  ## 4 per stack, from the nbeats paper
                        layer_size = nbeats_hidden_dim,
                        exog_block = exog_block,
                        use_norm=windowed_norm,
                        dropout=nbeats_dropout,
                        force_positive=force_positive_forecast)
        
        ## dataset iterator; defined in common/sampler.py
        train_ds = ts_dataset(timeseries=training_values, static_cat=static_categories,
                                        insample_size=input_size,
                                        outsample_size=horizon,
                                        window_sampling_limit=int(history_size_in_horizons * horizon))

        training_set = DataLoader(train_ds,batch_size=batch_size)

        snapshot_manager = SnapshotManager(snapshot_dir=os.path.join(model_directory(), model_name),
                                            total_iterations=iterations)

        ## training loop, including loss fn; defined in experiments/trainer.py
        model = trainer_var(snapshot_manager=snapshot_manager,
                        model=model,
                        training_set=iter(training_set),
                        timeseries_frequency=0,  ## not used
                        loss_name=loss_fn_name,
                        iterations=iterations,
                        learning_rate=init_LR)
        
        # training done; generate forecasts
        return generate_forecast(model, train_ds)

    return ret_fn


def ensemble_loop(rstate, settings):

    ## create training function
    training_fn = make_training_fn(training_data = rstate.vals_train,
                            static_categories = rstate.static_cat,
                            target_key = rstate.target_key,
                            horizon = settings.horizon,
                            windowed_norm = settings.use_windowed_norm,
                            init_LR = settings.init_LR,
                            batch_size = settings.batch_size)

    mu_fc={}
    var_fc={}
    for i,lookback in enumerate(settings.lookback_opts):
        for j in range(settings.random_reps):
            model_name = settings.model_prefix+"_"+str(lookback)+"_"+str(j)
            model_suffix = str(rstate.cut) if rstate.cut is not None else str(rstate.data_index[-1])
            model_name = model_name+"_"+model_suffix
            print("training ",model_name)
            mu_fc[model_name], var_fc[model_name] = training_fn(model_name = model_name,
                                                                iterations = settings.iterations,
                                                                lookback = lookback,
                                                                use_exog_vars = settings.exog_vars,
                                                                use_static_cat = settings.use_static_cat,
                                                                loss_fn_name = settings.lfn_name,
                                                                nbeats_stacks = settings.nbeats_stacks,
                                                                nbeats_hidden_dim = settings.nbeats_hidden_dim,
                                                                nbeats_dropout = settings.nbeats_dropout,
                                                                encoder_k = settings.encoder_k,
                                                                #encoder_n=None, ## auto calculated
                                                                encoder_hidden_dim = settings.encoder_hidden_dim,
                                                                encoder_dropout = settings.encoder_dropout,
                                                                force_positive_forecast = settings.force_positive_forecast) 

    ## forecast shape for each model is [series, time]
    ## ensemble using median across models
    mu_fc["median"] = np.median(np.stack([mu_fc[k] for k in mu_fc]),axis=0)
    var_fc["median"] = np.median(np.stack([var_fc[k] for k in var_fc]),axis=0)

    ## write results to state
    rstate.mu_fc = mu_fc
    rstate.var_fc = var_fc
    return None


def generate_ensemble(settings, cut):

    empty_gpu_cache()
    ## clean state for current run:
    rstate = Struct()
    ## record a copy of settings used for this run
    rstate.settings = deepcopy(settings)

    load_training(rstate, settings, cut)
    normalize_training(rstate, settings)
    ensemble_loop(rstate, settings)
    generate_quantiles(rstate, settings)

    ## return struct containing results
    return rstate


def run_tests(settings, test_cut_vals, forecast_delay_days):
    for (cut,forecast_delay) in zip(test_cut_vals, forecast_delay_days):
        rstate = generate_ensemble(settings, cut)
        pickle_results(rstate)
        output_figs(rstate)
        output_csv(rstate, forecast_delay)
        if settings.delete_models:
            delete_saved_models()
    empty_gpu_cache()
    return None


def load_training(rstate, settings, cut):
    targ_var = settings.targ_var
    data_suffix = settings.data_suffix

    rstate.data_is_log = (targ_var=="h_log")
    ## if we're using weekly or 7-day moving average to forecast daily values,
    ## the daily variance is 7 * that of the weekly means or smoothed data
    ## (set this to 1.0 if we used actual daily data, or if we want confidence intervals for weekly means)
    ##  (also: this relationship breaks down for log-transformed data)
    rstate.variance_scale = 7.0 if (data_suffix=="weekly" or data_suffix=="7ma") else 1.0

    ## load training data
    df_targ_all = pd.read_csv("storage/training_data/"+targ_var+"_"+data_suffix+".csv",index_col=0)
    df_targ = df_targ_all.iloc[:cut,:]
    data_index = df_targ.index
    data_columns = df_targ.columns

    if cut is not None:
        test_targets = df_targ_all.iloc[cut:,:].to_numpy(dtype=np.float32).transpose()
    else:
        test_targets = None

    vals_train = {}
    vals_train[targ_var] = df_targ.to_numpy(dtype=np.float32).transpose() ## dims are [series, time]

    if rstate.data_is_log: # read in log data
        vals_train["nat_scale"] = np.exp(vals_train[targ_var]) - 1.0
        test_targets = np.exp(test_targets) - 1.0 if test_targets is not None else None
    else:
        vals_train["nat_scale"] = vals_train[targ_var]

    ## use 7-day MA for daily weather data
    covar_suffix = "weekly" if data_suffix=="weekly" else "7ma"
    ## load covar files
    load_co_vars = ["tempC","dewpC"]
    for f in load_co_vars:
        df = pd.read_csv("storage/training_data/"+f+"_"+covar_suffix+".csv",index_col=0).iloc[:cut,:]
        assert df.index.equals(data_index), "data index mismatch"
        assert df.columns.equals(data_columns), "data columns mismatch"
        vals_train[f] = df.to_numpy(dtype=np.float32).transpose() ## dims are [series, time]

    ## include travel data?
    read_tsa_data = False
    if read_tsa_data:
        travel_file = "tsa_by_pop_weekly" if data_suffix=="weekly" else "tsa_by_pop_daily"
        df = pd.read_csv("storage/training_data/"+travel_file+".csv",index_col=0).iloc[:cut,:]
        assert df.index.equals(data_index), "data index mismatch"
        assert df.columns.equals(data_columns), "data columns mismatch"
        vals_train["tsa_by_pop"] = df.to_numpy(dtype=np.float32).transpose()

    ## add time as predictor; same scale as other predictors
    vals_train["t"] = np.tile(np.linspace(-2,2,vals_train[targ_var].shape[1],dtype=np.float32),
                              (vals_train[targ_var].shape[0],1))
    vals_train["t_decay"] = np.tile(2.0*np.exp((-1/720)*np.arange(0,vals_train[targ_var].shape[1],dtype=np.float32)),
                                (vals_train[targ_var].shape[0],1))

    ## also day of year
    vals_train["doy"] = np.tile(np.array(-2.0 + 4.0 * pd.to_datetime(data_index).dayofyear.values / 366.0, dtype=np.float32), 
                (vals_train[targ_var].shape[0],1))

    ## vaccination data
    df = pd.read_csv("storage/training_data/vacc_full_pct_to_may23.csv",index_col=0)
    df.index = pd.to_datetime(df.index)
    assert df.columns.equals(data_columns), "data columns mismatch"
    ## merge in estimated vaccination rates at the dates indexed in the target data
    ## vaccination data is only avail to 2023-05-10, so assume it's constant after that, using ffill()
    ## then fill data before 2021-01-12 with 0's
    vacc_pct_fake = pd.DataFrame(index=pd.to_datetime(df_targ.index)).join(df).ffill().fillna(0.0)
    vals_train["vacc_rate"] = vacc_pct_fake.to_numpy(dtype=np.float32).transpose()

    ## add predictor: time since last variant of concern (dates from cdc website)
    ## first wk of available target data = 7/14-7/20 (19-20 weeks from 3/1)
    ## a/b/g: 12/29/20  (24 weeks from 7/14)
    ## e: 3/19/21  (wk 35)
    ## d: 6/15/21  (wk 48)
    ## o: 11/26/21  (71)
    data_start = pd.to_datetime(df_targ.index[0])
    time_a = pd.to_datetime("2020-12-29") 
    time_e = pd.to_datetime("2021-03-19")
    time_d = pd.to_datetime("2021-06-15")
    time_o = pd.to_datetime("2021-11-26")
    time_unit = 7 if data_suffix=="weekly" else 1
    delta_a = (time_a - data_start).days // time_unit
    delta_e = (time_e - data_start).days // time_unit
    delta_d = (time_d - data_start).days // time_unit
    delta_o = (time_o - data_start).days // time_unit
    data_offset = (data_start - pd.to_datetime("2020-03-01")).days // time_unit
    timepoints = np.arange(vals_train[targ_var].shape[1])
    t_a = timepoints - delta_a; t_a[t_a < 0] = 99999
    t_e = timepoints - delta_e; t_e[t_e < 0] = 99999
    t_d = timepoints - delta_d; t_d[t_d < 0] = 99999
    t_o = timepoints - delta_o; t_o[t_o < 0] = 99999
    time_since_voc = np.stack([timepoints+data_offset,t_a,t_e,t_d,t_o]).min(axis=0)
    time_since_voc = (2.0 * time_since_voc / np.max(time_since_voc))
    vals_train["t_voc"] = np.tile(time_since_voc, (vals_train[targ_var].shape[0],1))

    ## static predictors
    load_static_real = ["pop_density_2020","med_age_2023"]

    for f in load_static_real:
        df = pd.read_csv("storage/training_data/"+f+".csv",dtype={"fips":str}).set_index("fips").sort_index()
        assert df.index.equals(data_columns), "static data mismatch"
        ## repeat the same value across time steps
        vals_train[f] = np.tile(df.to_numpy(dtype=np.float32),(1,vals_train[targ_var].shape[1]))

    ## just one static categorical covariate for now, identifying which time series each window comes from
    ## categorical vars should either be one-hot encoded, or converted to a learned "embedding" vector
    static_cat = np.arange(vals_train[targ_var].shape[0],dtype=int)

    rstate.cut = cut
    rstate.data_index = data_index
    rstate.data_columns = data_columns
    rstate.vals_train = vals_train
    rstate.test_targets = test_targets
    rstate.static_cat = static_cat
    return None


def normalize_training(rstate, settings):
    normalize_target = settings.normalize_target
    targ_var = settings.targ_var
    vals_train = rstate.vals_train ## modifying

    ## for weather, normalize using overall mean instead of by series? (otherwise losing info)
    for k in ["tempC","dewpC","AH"]:
        if k in vals_train:
            vals_train[k] = (vals_train[k] - np.nanmean(vals_train[k])) / np.nanstd(vals_train[k])

    for k in ["tsa_by_pop"]:
        if k in vals_train:
            ## normalize by series, losing pop size info but preserving relative change?
            u = np.nanmean(vals_train[k],axis=1,keepdims=True)
            vals_train[k] = vals_train[k] / u
            # or normalize by global mean?
            #vals_train[k] = vals_train[k] / np.nanmean(vals_train[k])

    ## log-transform, then normalize using z score
    for k in ["pop_density_2020"]:
        if k in vals_train:
            log_v = np.log(vals_train[k])
            ## z score across series (axis 0)
            u = np.nanmean(log_v,axis=0,keepdims=True)
            s = np.nanstd(log_v,axis=0,keepdims=True)
            vals_train[k] = (log_v - u) / s

    ## normalize using z score
    for k in ["med_age_2023"]:
        if k in vals_train:
            ## z score across series (axis 0)
            u = np.nanmean(vals_train[k],axis=0,keepdims=True)
            s = np.nanstd(vals_train[k],axis=0,keepdims=True)
            vals_train[k] = (vals_train[k] - u) / s

    ## normalize by global max? (by series doesn't make sense)
    for k in ["vacc_rate"]:
        if k in vals_train:
            vals_train[k] = 2.0 * vals_train[k] / np.nanmax(vals_train[k])

    ## read in nat scale data, but want to sqrt transform it
    if settings.sqrt_transform and not rstate.data_is_log: ## ignore setting if data is log-transformed
        vals_train[targ_var] = np.sqrt(vals_train[targ_var])
        rstate.data_is_sqrt = True
    else:
        rstate.data_is_sqrt = False

    ## optionally normalize the target
    if normalize_target:
        ## try transforming series to common scale
        ## NOTE: in the unscaled data, series with small values contribute less to the weight gradients
        ##  scaling makes the model learn better from states with small populations, whose data is noisier and more error prone
        ##  this could make the overall forecast worse; can maybe be compensated with more training iterations
        inv_scale = np.nanmedian(vals_train[targ_var], axis=1, keepdims=True)
        target_key = "scaled_" + targ_var
        vals_train[target_key] = vals_train[targ_var] / inv_scale
    else:
        inv_scale = np.ones((vals_train[targ_var].shape[0],1))
        target_key = targ_var

    rstate.inv_scale = inv_scale
    rstate.target_key = target_key
    return None
  


def inverse_cdf(mu, s2, qtiles, lfn_name):
    if lfn_name == "t_nll": ## df hardcoded in experiments/trainer.py
        return [stats.t.ppf(q=x,loc=mu,scale=np.sqrt(s2),df=5) for x in qtiles]
    elif lfn_name == "norm_nll":
        return [stats.norm.ppf(q=x,loc=mu,scale=np.sqrt(s2)) for x in qtiles]
    elif lfn_name == "gamma_nll":
        ## gamma for counts breaks down near 0
        m = np.maximum(mu, 0.5)
        v = np.maximum(s2, 0.1)
        return [stats.gamma.ppf(q=x, a=(m*m/v) , scale=(v/m)) for x in qtiles]

def generate_quantiles(rstate, settings):
    qtiles = settings.qtiles
    data_is_log = rstate.data_is_log ## read only
    data_is_sqrt = rstate.data_is_sqrt ## read only
    variance_scale = rstate.variance_scale ## read only
    inv_scale = rstate.inv_scale ## read only
    mu_fc = rstate.mu_fc ## read only
    var_fc = rstate.var_fc ## read only

    ## estimated correlations, to calculate var(sum)
    ##  (because var(sum) = sum(var) only for independent variables)
    corr_mat = np.corrcoef(rstate.vals_train["nat_scale"])  

    fc_quantiles = {}
    fc_med = {}
    fc_upper = {}
    fc_lower = {}
    us_quantiles = {}
    us_med = {}
    us_upper = {}
    us_lower = {}

    for k in mu_fc:
        mu = mu_fc[k] * inv_scale
        s2 = var_fc[k] * variance_scale * inv_scale * inv_scale

        ## inverse-transform the estimates
        ## don't need to inv-trans the variance, because we'll inv-trans the quantiles directly
        ##   (that won't work for the sum though, see below)
        if data_is_log:
            mu_nat = np.exp(mu) - 1.0
        elif data_is_sqrt:
            mu_nat = np.square(mu)
        else:
            mu_nat = mu

        sum_mu = np.nansum(mu_nat, axis=0, keepdims=True)  ## sum(natural scale mu)
        st_devs = np.sqrt(s2) ## needed for var(sum) below

        if data_is_log:
            ## approximate variance on natural scale: var(x) ~~ mu_x * var(log(x)) * mu_x
            sum_s2 = np.array([np.nansum(corr_mat * st_devs[:,i,None] * st_devs[:,i] * mu_nat[:,i,None] * mu_nat[:,i]) for i in range(st_devs.shape[1])])
        elif data_is_sqrt:
            ## delta method: var(f(x)) ~~ f'(mu) * var(x) * f'(mu) -> var(x^2) ~~ 2mu * var(x) * 2mu
            sum_s2 = np.array([np.nansum(corr_mat * st_devs[:,i,None] * st_devs[:,i] * 4.0 * mu[:,i,None] * mu[:,i]) for i in range(st_devs.shape[1])])
        else:
            ## var(sum) = sum(covar) at each timepoint for which variance was forecast
            ## covar_i_j = correlation_i_j * std_i * std_j
            sum_s2 = np.array([np.nansum(corr_mat * st_devs[:,i,None] * st_devs[:,i]) for i in range(st_devs.shape[1])])

        ##
        ## try using gamma err dist, even if trained on a different error fn?
        ##
        if data_is_log:
            err_name = settings.lfn_name
        else:
            err_name = "gamma_nll"

        ## note, sum(mu) and var(sum) were converted to nat scale above
        fc_quantiles[k] = np.stack(inverse_cdf(mu, s2, qtiles, err_name), axis=2) ## [series, time, quantiles]
        us_quantiles[k] = np.stack(inverse_cdf(sum_mu, sum_s2, qtiles, err_name), axis=2)
        fc_med[k],fc_upper[k],fc_lower[k] = inverse_cdf(mu, s2, [0.5,0.975,0.025], err_name)
        us_med[k],us_upper[k],us_lower[k] = inverse_cdf(sum_mu, sum_s2, [0.5,0.975,0.025], err_name)

        if data_is_log:
            fc_quantiles[k] = np.exp(fc_quantiles[k]) - 1.0
            fc_med[k] = np.exp(fc_med[k]) - 1.0
            fc_upper[k] = np.exp(fc_upper[k]) - 1.0
            fc_lower[k] = np.exp(fc_lower[k]) - 1.0
        elif data_is_sqrt:
            fc_quantiles[k] = np.square(fc_quantiles[k])
            fc_med[k] = np.square(fc_med[k])
            fc_upper[k] = np.square(fc_upper[k])
            fc_lower[k] = np.square(fc_lower[k])

    ## write results to state
    rstate.fc_quantiles = fc_quantiles
    rstate.fc_med = fc_med
    rstate.fc_upper = fc_upper
    rstate.fc_lower = fc_lower
    rstate.us_quantiles = us_quantiles
    rstate.us_med = us_med
    rstate.us_upper = us_upper
    rstate.us_lower = us_lower
    return None



def plotpred(forecasts, name, ser, training_targets, test_targets, horizon, lower_fc=None, upper_fc=None, x_start = 0, date0 = "2020-07-14"):
    x_end = training_targets.shape[1]
    dates=pd.date_range(pd.to_datetime(date0),periods=x_end+horizon,freq="D")
    colors = ["black","orangered"]
    #colors = ["white","yellow"]
    _, ax = plt.subplots(figsize=(7,5))
    ax.grid(alpha=0.2)
    pd.Series(training_targets[ser,x_start:x_end],index=dates[x_start:x_end]).plot(ax=ax,grid=True,color=colors[0],linewidth=0.5)
    if test_targets is not None:
        test_end = min(test_targets.shape[1], horizon)
        pd.Series(test_targets[ser,0:test_end],index=dates[x_end:x_end+test_end]).plot(ax=ax,grid=True,color=colors[0],linewidth=0.5)
    pd.Series(forecasts[name][ser],index=dates[x_end:x_end+horizon]).plot(ax=ax,grid=True,color=colors[1],linewidth=1.5,alpha=0.8)
    if upper_fc is not None:
        ax.fill_between(dates[x_end:x_end+horizon],lower_fc[name][ser],upper_fc[name][ser],color=colors[1],alpha=0.4)
    #plt.show()

def output_figs(rstate):
    horizon = rstate.settings.horizon
    vals_train = rstate.vals_train ## read only
    test_targets = rstate.test_targets ## read only
    model_prefix = rstate.settings.model_prefix ## read only

    us_train = vals_train["nat_scale"].sum(axis=0,keepdims=True)
    us_test = test_targets.sum(axis=0,keepdims=True) if test_targets is not None else None
    x0 = rstate.cut - 400 if rstate.cut is not None else vals_train["nat_scale"].shape[1] - 400
    model_suffix = str(rstate.cut) if rstate.cut is not None else str(rstate.data_index[-1])

    plotpred(rstate.fc_med, "median", 20, vals_train["nat_scale"], test_targets, horizon, rstate.fc_lower, rstate.fc_upper, x0)
    plt.savefig("storage/"+model_prefix+"_MD_"+model_suffix+".png")
    plotpred(rstate.fc_med, "median", 4, vals_train["nat_scale"], test_targets, horizon, rstate.fc_lower, rstate.fc_upper, x0)
    plt.savefig("storage/"+model_prefix+"_CA_"+model_suffix+".png")
    plotpred(rstate.us_med, "median", 0, us_train, us_test, horizon, rstate.us_lower, rstate.us_upper, x0)
    plt.savefig("storage/"+model_prefix+"_US_"+model_suffix+".png")
    return None


def pickle_results(rstate):
    model_prefix = rstate.settings.model_prefix
    model_suffix = str(rstate.cut) if rstate.cut is not None else str(rstate.data_index[-1])
    path = "storage/"+model_prefix+"_"+model_suffix+".pickle"
    with open(path,"wb") as f:
        pickle.dump(rstate, f)

def read_pickle(path):
    with open(path,"rb") as f:
        r = pickle.load(f)
    return r


## process data for forecast hub
def output_csv(rstate, forecast_delay):
    qtiles = rstate.settings.qtiles
    data_index = rstate.data_index
    data_columns = rstate.data_columns
    fc_quantiles = rstate.fc_quantiles
    us_quantiles = rstate.us_quantiles

    ## forecast date: output file will contain forecast for this day forward; default = current local date
    ##
    ## NOTE: model generates a forecast starting with the day after the training data ends,
    ##   which may be in the past. But only forecast_date onward is written to the output file.
    train_end_date = pd.to_datetime(data_index[-1])

    if forecast_delay is not None:
        forecast_date = pd.to_datetime(train_end_date + pd.Timedelta(days=forecast_delay))
    else:
        forecast_date = pd.to_datetime(datetime.date.today()) 

    ## use ensembled forecasts; append US forecast derived above
    q_ensemble = np.concatenate([fc_quantiles["median"], us_quantiles["median"]],axis=0)

    location_codes = data_columns.to_list() + ["US"] ## fips codes
    quantile_labels = [f'{x:.3f}' for x in qtiles]
    date_indices = pd.date_range(train_end_date + pd.Timedelta(days=1), train_end_date + pd.Timedelta(days=q_ensemble.shape[1]))

    dfs = []
    ## loop through each location in q_ensemble and make a dataframe with shape [date, value at each quantile]
    for i in range(q_ensemble.shape[0]):
        df = pd.DataFrame(q_ensemble[i,:,:])
        df.columns = quantile_labels
        df.index = date_indices
        dfs.append(df.loc[forecast_date:,:].melt(ignore_index=False,var_name="quantile").reset_index(names="target_end_date"))

    ## concatenate the location dataframes and set index to location code
    df_hub = pd.concat(dfs,keys=location_codes).droplevel(1).reset_index(names="location")

    ## add the rest of the columns required by forecast hub
    df_hub.loc[:,"type"] = "quantile"
    df_hub.loc[:,"forecast_date"] = forecast_date
    df_hub.loc[:,"target"] = df_hub.target_end_date.map(lambda d: str((d - forecast_date).days) + " day ahead inc hosp")
    df_hub.loc[:,"value"] = df_hub.loc[:,"value"].round(2)

    ## if using error dist allows negative values, set them to 0
    df_hub.loc[df_hub["value"]<0.0,"value"] = 0.0

    # write to csv
    hub_name = "OHT_JHU-nbxd"
    filename = "storage/"+ forecast_date.strftime("%Y-%m-%d") + "-" + hub_name + ".csv"
    print("writing ",filename)
    df_hub.to_csv(filename, index=False)
    return None


## not the actual loss fn, just for comparing different forecasts
def calc_loss(forecasts, name, test_targets, horizon):
    #lfn = t.nn.functional.l1_loss
    #def lfn(a,b):
    #    x = ((a-b)/b)
    #    return x.abs().nanmean()
    def lfn(a,b):
        return 2.0 * t.nn.functional.l1_loss(a,b) / (a.nanmean() + b.nanmean())
    return lfn(t.tensor(np.array(forecasts[name],dtype=np.float32))[:,:horizon], 
               t.tensor(np.array(test_targets,dtype=np.float32))[:,:horizon])


