
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
import dill as pickle
from typing import Dict, List, Optional
from copy import deepcopy
import json

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
from models.exog import TCN_encoder


## I didn't write this class, Peter Norvig did
## it is the only class you need
class Struct:
    def __init__(self, **entries): self.__dict__.update(entries)
    def __repr__(self): return '\n'.join([f"{k} = {v}" for (k,v) in vars(self).items()])


## returns json file as dict; empty dict on fail
def tryJSON(filename):
    try:
        with open(filename, 'r') as f:
            d = json.load(f)
    except Exception as e:
        print("warning: ",e)
        d = {}
    return d


## default model settings; entries in settings.json take precedence
## this data structure is meant to hold settings that can change between models within an ensemble
def default_settings():
    settings = Struct()
    d = tryJSON("settings.json")

    ## which covariates to include (will be set later if not in settings.json)
    settings.exog_vars = d.get("exog_columns",None)

    ## number of training epochs; too many (relative to learning rate) overfits
    settings.iterations = d.get("iterations",400)
    settings.init_LR = d.get("init_LR",0.00025) #0.0001 # learning rate; lower with more iterations seems to work better
    ## batch size?
    ## nbeats is ok with large batch size, but it seems to hurt TCN
    ## the two parts probably learn at different rates; not sure how to handle that
    settings.batch_size = d.get("batch_size",128) #256 #128 #1024 #

    settings.lookback = d.get("lookback",4)  ## backward window size, in horizons
    ## forecast horizon (in time units)
    settings.horizon = d.get("horizon",40) #6

    ## loss function (defined in experiments/trainer.py)
    settings.lfn_name = d.get("lfn_name","t_nll")
    settings.force_positive_forecast = bool(d.get("force_positive_forecast",False)) ## if loss fn requires > 0
    settings.normalize_target = bool(d.get("normalize_target",False)) ## normalize the target var before passing to model? (see notes below)
    settings.use_windowed_norm = bool(d.get("use_windowed_norm",True)) ## normalize inside the model by window? (tends to improve forecasts; ref. Smyl 2020)
    settings.use_static_cat = bool(d.get("use_static_cat",False)) ## whether to use static_cat; current implementation always makes forecasts worse

    settings.nbeats_stacks=d.get("nbeats_stacks",8) #12 #8 # more data can support deeper model
    settings.nbeats_hidden_dim=d.get("nbeats_hidden_dim",512) #128 ## should be larger than length of lookback window
    settings.nbeats_dropout=d.get("nbeats_dropout",0.2) ## could help prevent overfitting? default = None
    settings.encoder_k = d.get("encoder_k",5) #prev: 4 (5 always better)  ## TCN receptive field size = 2(k-1)(2^n - 1)+1
    settings.encoder_n = None ## auto-calculated
    settings.encoder_hidden_dim=d.get("encoder_hidden_dim",128)
    settings.encoder_dropout=d.get("encoder_dropout",0.2) ## default is 0.2

    settings.enc_temporal = d.get("enc_temporal",True) ## encode exog predictors in a way that preserves temporal structure ?
    settings.static_cat_embed_dim = d.get("static_cat_embed_dim",3) ## dimension of vector embedding if using static_categories
    settings.history_size_in_horizons = d.get("history_size_in_horizons",60) ## maximum length of history to consider for training, in # of horizons

    return settings


## default config settings; entries in config.json take precedence
## this struct is meant to hold settings that don't change between models within an ensemble
def read_config():
    rstate = Struct()
    d = tryJSON("config.json")
    rstate.target_file = d["target_file"] ## required
    rstate.cut = d.get("cut",None) ## train/test cut-off (integer value)
    rstate.output_prefix = d.get("output_prefix","nbxd")
    rstate.data_dir = d.get("data_dir", os.path.join("storage","training_data"))
    rstate.snapshot_dir = d.get("snapshot_dir", "model_snapshots")
    ## delete models after saving forecasts?
    rstate.delete_models = bool(d.get("delete_models",False))
    ## whether target data was transformed (for making predictions on natural scale):
    rstate.input_is_log = bool(d.get("input_is_log",False))
    rstate.input_is_sqrt = bool(d.get("input_is_sqrt",False))
    ## transform target data?
    rstate.log_transform = bool(d.get("log_transform",False))
    ## sqrt should be good for counts
    rstate.sqrt_transform = bool(d.get("sqrt_transform",True))
    ## if we're using weekly or 7-day moving average to forecast daily values,
    ## the daily variance is 7 * that of the weekly means or smoothed data
    ## (set this to 1.0 if we used actual daily data, or if we want confidence intervals for weekly means)
    ##  (also: this relationship breaks down for log-transformed data)
    rstate.variance_scale = d.get("variance_scale", 1.0)
    ## default quantiles
    rstate.qtiles = [0.01, 0.025, *np.linspace(0.05, 0.95, 19).round(2), 0.975, 0.99]

    return rstate


## read in and optionally transform target data
## set timepoint indices and series identifiers; write results to rstate
def init_target_data(rstate, settings):

    ## whether the target data is already transformed
    target_is_log = rstate.input_is_log
    target_is_sqrt = rstate.input_is_sqrt

    ## whether to transform the target data
    ## TODO: these are currently stored in config rather than per-model settings -- can't ensemble different transforms
    ##  because per-series quantiles are generated without inverse-transforming the variances
    ##  could change this if needed
    sqrt_transform = rstate.sqrt_transform
    log_transform = rstate.log_transform

    var_dfs = {}
    cut = rstate.cut ## train/test split

    ## first column is a string/date index; columns are series identifiers
    df_targ_all = pd.read_csv(os.path.join(rstate.data_dir,rstate.target_file), index_col=0, dtype={0:object})
    series_names = df_targ_all.columns
    ## indexes up to "cut" are training data
    data_index = df_targ_all.index[:cut] if cut is not None else df_targ_all.index
    df_train = df_targ_all.loc[data_index, :].copy()
    ## after cut is test/validation data
    test_index = None if cut is None else df_targ_all.index[cut:]
    ## expected to have dims [series, time]
    test_targets = None if test_index is None else df_targ_all.loc[test_index, :].to_numpy(dtype=np.float32).transpose()

    ## transforming/un-transforming the target is tricky because of variance and sum-of-variances for predictions
    ## this is fine for now:
    if target_is_log:
        transform = None
        reverse_transform = (lambda x: np.exp(x) - 1.0)
        var_dfs["_LOG"] = df_train
        var_dfs["_NAT_SCALE"] = df_train.apply(reverse_transform)
        test_targets = reverse_transform(test_targets) if test_targets is not None else None
        target_var = "_LOG"
    elif target_is_sqrt:
        transform = None
        reverse_transform = (lambda x: np.square(x))
        var_dfs["_SQRT"] = df_train
        var_dfs["_NAT_SCALE"] = df_train.apply(reverse_transform)
        test_targets = reverse_transform(test_targets) if test_targets is not None else None
        target_var = "_SQRT"
    elif log_transform:
        transform = (lambda x: np.log(x + 1.0))
        reverse_transform = (lambda x: np.exp(x) - 1.0)
        var_dfs["_NAT_SCALE"] = df_train
        var_dfs["_LOG"] = df_train.apply(transform)
        target_var = "_LOG"
        target_is_log = True ## needed for special treament (variance forecasts)
    elif sqrt_transform:
        transform = (lambda x: np.sqrt(x))
        reverse_transform = (lambda x: np.square(x))
        var_dfs["_NAT_SCALE"] = df_train
        var_dfs["_SQRT"] = df_train.apply(transform)
        target_var = "_SQRT"
        target_is_sqrt = True ## needed for special treament (variance forecasts)
    else:
        transform = None
        reverse_transform = None
        var_dfs["_NAT_SCALE"] = df_train
        target_var = "_NAT_SCALE" 

    ## save nat scale target for plotting etc.
    nat_targets = var_dfs["_NAT_SCALE"].to_numpy(dtype=np.float32).transpose() ## dims [series, time]

    ## pre-generate normalized target values (for when settings.normalize_target = True)
    ## NOTE: in the unscaled data, series with small values contribute less to the weight gradients
    ##  scaling makes the model learn better from states with small populations, whose data is noisier and more error prone
    ##  this could make the overall forecast worse; can maybe be compensated with more training iterations
    var_dfs["_SCALED"] = var_dfs[target_var] / var_dfs[target_var].apply(np.nanmedian)
    ## save the scaling separately in case we need to un-scale variance predictions
    optional_inv_scale = var_dfs[target_var].apply(np.nanmedian).values[:,None] ## dims [series, 1]

    ## in theory, series could have different lengths
    ## (TODO: sampler currently requries same length)
    ## convert to dict keyed by series name, each entry is a df with rows = time and cols = variables
    ## for input to training fn
    series_dfs = {s:pd.DataFrame(index=data_index).join([var_dfs[k].loc[:,s].rename(k) for k in var_dfs]) for s in series_names}

    ## write global state
    rstate.series_dfs = series_dfs
    rstate.target_var = target_var
    rstate.series_names = series_names
    rstate.data_index = data_index
    rstate.nat_targets = nat_targets
    rstate.test_targets = test_targets
    rstate.target_is_log = target_is_log
    rstate.target_is_sqrt = target_is_sqrt
    rstate.reverse_transform = reverse_transform
    rstate.optional_inv_scale = optional_inv_scale

    return rstate, settings


## read exogenous predictors, append to rstate.series_dfs
## rstate must contain time index and series names, set in init_target_data()
## domain_specs contains file locations and processing instructions
##   for example, see data_utils/covid_hub.py/domain_defaults()
def load_exog_data(rstate, settings, domain_specs):

    var_dfs = {}
    series_dfs = rstate.series_dfs
    data_index = rstate.data_index
    series_names = rstate.series_names
    
    for (var,file,fn,norm) in zip(domain_specs.var_names,
                                domain_specs.var_files,
                                domain_specs.var_fns,
                                domain_specs.var_norm):
        ## first column is a string/date index
        df = None if file is None else pd.read_csv(os.path.join(rstate.data_dir,file),index_col=0,dtype={0:object})

        ## if no post-processing fn, assume df's index is a superset of target index
        ## otherwise the fn is responsible for indexing
        if fn is None:
            assert df is not None, "var_file and var_fn can't both be None"
            df = pd.DataFrame(index=data_index).join(df) ## keep only rows in data_index
        else:
            df = fn(df, data_index, series_names)

        ## normalize
        if norm is not None:
            df = norm(df)

        ## make sure df has data for all series
        assert series_names.isin(df.columns).all(), "missing series in "+file
        ## double check index
        assert df.index.equals(data_index), "data index mismatch in"+var

        ## save in dict
        var_dfs[var] = df

    ## append to series dfs
    series_dfs = {s:series_dfs[s].join([var_dfs[k].loc[:,s].rename(k) for k in var_dfs]) for s in series_names}

    ## just one static categorical covariate for now, identifying which time series each window comes from
    ## categorical vars should either be one-hot encoded, or converted to a learned "embedding" vector
    static_cat = np.arange(len(series_names),dtype=int)

    ## global state
    rstate.series_dfs = series_dfs
    rstate.static_cat = static_cat

    ## which exog vars to use in the model
    ## value from domain_specs can be overridden in settings.json
    if settings.exog_vars is None:
        settings.exog_vars = domain_specs.exog_vars

    return rstate, settings



## forecasts from the last available data window in data_iterator (the sampler)
## m is the trained model
## used in make_training_fn() below
##  note, this version is meant for mean-variance models
def generate_forecast(m, data_iterator):
    x, x_mask, cat_tensor = data_iterator.last_insample_window()
    m.eval()
    with t.no_grad():
        f_mu, f_var = m(x, x_mask, cat_tensor)
    return f_mu.cpu().detach().numpy(), f_var.cpu().detach().numpy()



## this fn returns a function that trains a model given a settings struct
## (it closes over training data and config settings)
## the return function can be used on its own 
## or called in a loop with different settings to generate an ensemble
def make_training_fn(rstate):

    ## result fn returns forecast; model is saved in snapshots folder
    def ret_fn(model_name, settings):
        ## switch target columns if using normalized target
        if settings.normalize_target:
            target_key = "_SCALED"
        else:
            target_key = rstate.target_var

        input_size = settings.lookback * settings.horizon  ## backward window size, in time units
        use_vars = [target_key, *(settings.exog_vars)]

        ## training data dims are [series, time, variables]
        ## where the first variable is the target, and the rest are covariates
        training_values = np.stack([rstate.series_dfs[s].loc[:,use_vars] for s in rstate.series_names])
        
        n_features = training_values.shape[2] - 1 ## number of exogenous covariates
        n_embed = rstate.static_cat.shape[0] if settings.use_static_cat else 0
        embed_dim = settings.static_cat_embed_dim if settings.use_static_cat else 0

        encoder_n = settings.encoder_n
        if encoder_n is None: ## use minimum needed to cover input size
            ## TCN receptive field size = 2(k-1)(2^n - 1)+1 where k is kernel size and n is # blocks
            encoder_n = int(np.ceil(np.log2(1.0 + (0.5*(input_size - 1.0) / (settings.encoder_k - 1.0)))))

        exog_block = TCN_encoder(n_features, [settings.encoder_hidden_dim]*encoder_n, 
                                 settings.encoder_k, settings.encoder_dropout, 
                                 settings.enc_temporal, n_embed, embed_dim)

        enc_dim = input_size if settings.enc_temporal else settings.encoder_hidden_dim
        ## constructs the model; defined in experiments/model.py
        model = generic_dec_var(enc_dim=enc_dim, output_size=settings.horizon,
                        stacks = settings.nbeats_stacks,
                        layers = 4,  ## 4 per stack, from the nbeats paper
                        layer_size = settings.nbeats_hidden_dim,
                        exog_block = exog_block,
                        use_norm= settings.use_windowed_norm,
                        dropout= settings.nbeats_dropout,
                        force_positive= settings.force_positive_forecast)
        
        ## dataset iterator; defined in common/sampler.py
        train_ds = ts_dataset(timeseries=training_values, static_cat=rstate.static_cat,
                                        insample_size=input_size,
                                        outsample_size=settings.horizon,
                                        window_sampling_limit=int(settings.history_size_in_horizons * settings.horizon))

        training_set = DataLoader(train_ds,batch_size=settings.batch_size)

        snapshot_manager = SnapshotManager(snapshot_dir=os.path.join(rstate.snapshot_dir, model_name),
                                            total_iterations=settings.iterations)

        ## training loop, including loss fn; defined in experiments/trainer.py
        model = trainer_var(snapshot_manager=snapshot_manager,
                        model=model,
                        training_set=iter(training_set),
                        timeseries_frequency=0,  ## not used
                        loss_name=settings.lfn_name,
                        iterations=settings.iterations,
                        learning_rate=settings.init_LR)
        
        # training done; generate forecasts
        f_mu, f_var = generate_forecast(model, train_ds)
        ## denormalize
        f_var = f_var * rstate.variance_scale
        if settings.normalize_target:
            f_mu = f_mu * rstate.optional_inv_scale
            f_var = f_var * rstate.optional_inv_scale * rstate.optional_inv_scale

        return f_mu, f_var

    return ret_fn



## used by generate_quantiles() below
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



## generates quantiles from mean and var forecasts and write results to rstate
##
## rstate.mu_fc and rstate.var_fc must be dicts containing mean and variance forecasts
##  (place the forecasts returned by the training function into these dicts, keyed by model name)
##
## err_name is the value passed to inverse_cdf() above to generate the quantiles
##  it could be the same as the loss fn used in training, but doesn't have to be
##  (gamma error is reasonable to use for counts)
##
## note this also generates mean/median/quantiles for the sum of the target series
##  (e.g., if target series are states and we want national forecast)
##
def generate_quantiles(rstate, err_name = "gamma_nll"):

    ## estimated correlations, to calculate var(sum)
    ##  (because var(sum) = sum(var) only for independent variables)
    corr_mat = np.corrcoef(rstate.nat_targets)  

    fc_quantiles = {} ## quantiles for individual series forecasts
    fc_mean = {}
    fc_med = {}
    fc_upper = {}
    fc_lower = {}
    sum_quantiles = {} ## quantiles for sum of series
    sum_mean = {}
    sum_med = {}
    sum_upper = {}
    sum_lower = {}

    for k in rstate.mu_fc:
        ## mean and variance forecasts; these have been de-scaled but not un-transformed
        mu = rstate.mu_fc[k] 
        s2 = rstate.var_fc[k] 

        ## inverse-transform the estimates
        ## don't need to inv-trans the variance, because we'll inv-trans the quantiles directly
        ##   (that won't work for the sum though, see below)
        mu_nat = mu if rstate.reverse_transform is None else rstate.reverse_transform(mu)

        sum_mu = np.nansum(mu_nat, axis=0, keepdims=True)  ## sum(natural scale mu)
        st_devs = np.sqrt(s2) ## needed for var(sum) below

        if rstate.target_is_log:
            ## approximate variance on natural scale: var(x) ~~ mu_x * var(log(x)) * mu_x
            ## note to self: yes, this is right, you checked
            sum_s2 = np.array([np.nansum(corr_mat * st_devs[:,i,None] * st_devs[:,i] * mu_nat[:,i,None] * mu_nat[:,i]) for i in range(st_devs.shape[1])])
        elif rstate.target_is_sqrt:
            ## delta method: var(f(x)) ~~ f'(mu) * var(x) * f'(mu) -> var(x^2) ~~ 2mu * var(x) * 2mu
            sum_s2 = np.array([np.nansum(corr_mat * st_devs[:,i,None] * st_devs[:,i] * 4.0 * mu[:,i,None] * mu[:,i]) for i in range(st_devs.shape[1])])
        else:
            ## var(sum) = sum(covar) at each timepoint for which variance was forecast
            ## covar_i_j = correlation_i_j * std_i * std_j
            sum_s2 = np.array([np.nansum(corr_mat * st_devs[:,i,None] * st_devs[:,i]) for i in range(st_devs.shape[1])])

        ## note, sum(mu) and var(sum) were converted to nat scale above
        fc_quantiles[k] = np.stack(inverse_cdf(mu, s2, rstate.qtiles, err_name), axis=2) ## [series, time, quantiles]
        sum_quantiles[k] = np.stack(inverse_cdf(sum_mu, sum_s2, rstate.qtiles, err_name), axis=2)
        fc_med[k],fc_upper[k],fc_lower[k] = inverse_cdf(mu, s2, [0.5,0.975,0.025], err_name)
        sum_med[k],sum_upper[k],sum_lower[k] = inverse_cdf(sum_mu, sum_s2, [0.5,0.975,0.025], err_name)
        ## also save mean on nat scale
        fc_mean[k] = mu_nat
        sum_mean[k] = sum_mu

        ## inverse-transform the series quantiles (sum-of-series was already generated on nat scale)
        if rstate.reverse_transform is not None:
            fc_quantiles[k] = rstate.reverse_transform(fc_quantiles[k])
            fc_med[k] = rstate.reverse_transform(fc_med[k])
            fc_upper[k] = rstate.reverse_transform(fc_upper[k])
            fc_lower[k] = rstate.reverse_transform(fc_lower[k])

    ## write results to state
    rstate.fc_quantiles = fc_quantiles
    rstate.fc_mean = fc_mean
    rstate.fc_med = fc_med
    rstate.fc_upper = fc_upper
    rstate.fc_lower = fc_lower
    rstate.sum_quantiles = sum_quantiles
    rstate.sum_mean = sum_mean
    rstate.sum_med = sum_med
    rstate.sum_upper = sum_upper
    rstate.sum_lower = sum_lower

    return rstate


## saves rstate to ./storage in python pickle format
## (the model itself was saved by common/torch/snapshots.py/SnapshotManager to the folder specified in rstate.snapshot_dir)
## (note, model settings are not automatically archived, but you can write them to rstate before saving)
def pickle_results(rstate):
    model_prefix = rstate.output_prefix
    model_suffix = str(rstate.cut) if rstate.cut is not None else str(rstate.data_index[-1])
    path = "storage/"+model_prefix+"_"+model_suffix+".pickle"
    with open(path,"wb") as f:
        pickle.dump(rstate, f)

def read_pickle(path):
    with open(path,"rb") as f:
        r = pickle.load(f)
    return r



## some crude plotting functions
## note, "ser" is the series integer index, not the series name
def plotpred(forecasts, ser, training_targets, test_targets, horizon, lower_fc=None, upper_fc=None, x_start = 0, date0 = None, x_freq="D"):
    x_end = training_targets.shape[1]
    x_idx = pd.date_range(pd.to_datetime(date0),periods=x_end+horizon,freq=x_freq) if date0 is not None else np.arange(x_end+horizon)
    colors = ["black","orangered"]
    #colors = ["white","yellow"]
    _, ax = plt.subplots(figsize=(7,5))
    ax.grid(alpha=0.2)
    pd.Series(training_targets[ser,x_start:x_end],index=x_idx[x_start:x_end]).plot(ax=ax,grid=True,color=colors[0],linewidth=0.5)
    if test_targets is not None:
        test_end = min(test_targets.shape[1], horizon)
        pd.Series(test_targets[ser,0:test_end],index=x_idx[x_end:x_end+test_end]).plot(ax=ax,grid=True,color=colors[0],linewidth=0.5)
    pd.Series(forecasts[ser],index=x_idx[x_end:x_end+horizon]).plot(ax=ax,grid=True,color=colors[1],linewidth=1.5,alpha=0.8)
    if upper_fc is not None:
        ax.fill_between(x_idx[x_end:x_end+horizon],lower_fc[ser],upper_fc[ser],color=colors[1],alpha=0.4)
    #plt.show()

## calls plotpred on series_idxs and on the sum-of-series forecasts
## expects rstate to contain the quantiles provided in generate_quantiles()
## expects the forecast dict to have an entry keyed "ensemble"
def output_figs(rstate, horizon, series_idxs, x_width):
    train_targets = rstate.nat_targets
    test_targets = rstate.test_targets ## read only
    model_prefix = rstate.output_prefix ## read only
    forecasts = rstate.fc_med["ensemble"]
    fc_upper = rstate.fc_upper["ensemble"]
    fc_lower = rstate.fc_lower["ensemble"]
    sum_forecasts = rstate.sum_med["ensemble"]
    sum_upper = rstate.sum_upper["ensemble"]
    sum_lower = rstate.sum_lower["ensemble"]
    sum_train = train_targets.sum(axis=0,keepdims=True)
    sum_test = test_targets.sum(axis=0,keepdims=True) if test_targets is not None else None
    x0 = rstate.cut - x_width if rstate.cut is not None else train_targets.shape[1] - x_width
    model_suffix = str(rstate.cut) if rstate.cut is not None else str(rstate.data_index[-1])
    date0 = rstate.data_index[0]

    for i in series_idxs:
        sname = rstate.series_names[i]
        plotpred(forecasts, i, train_targets, test_targets, horizon, fc_lower, fc_upper, x0, date0)
        plt.savefig("storage/"+model_prefix+"_"+sname+"_"+model_suffix+".png")

    plotpred(sum_forecasts, 0, sum_train, sum_test, horizon, sum_lower, sum_upper, x0, date0)
    plt.savefig("storage/"+model_prefix+"_SUM_"+model_suffix+".png")
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


##
## helper functions for processing, generating, and normalizing data
##  currently just used on exog vars; target var requires special treatment
## 

## if df's index doesn't cover all of data_index,
##  assume the values are constant after the last data point, and 0 before the first
def proc_fwd_const(df, data_index, series_names):
    return pd.DataFrame(index=data_index).join(df).ffill().fillna(0.0)

## generate a df of repeated constant values from single-column data indexed by series name
def proc_const(df, data_index, series_names):
    val_dict = df.loc[series_names,:].iloc[:,0].to_dict()
    return pd.DataFrame({s:val_dict[s] for s in series_names},index=data_index)

## generates a data frame filled with time value as a float from -2 to 2
def proc_t(df, data_index, series_names):
    return pd.DataFrame({s:np.linspace(-2,2,len(data_index),dtype=np.float32) for s in series_names},
                        index=data_index)

## generates a data frame filled with exponentially decaying time value from 2 -> 0
def proc_tdecay(df, data_index, series_names):
    return pd.DataFrame({s:2.0*np.exp((-1/720)*np.arange(0,len(data_index),dtype=np.float32)) for s in series_names},
                        index=data_index)

## day of year as a float from -2 to 2
## assumes date index
def proc_doy(df, data_index, series_names):
    return pd.DataFrame({s:np.array(-2.0 + 4.0 * pd.to_datetime(data_index).dayofyear.values / 366.0, dtype=np.float32) 
                         for s in series_names}, index=data_index)

## normalize by global mean and std to avoid losing per-series information
def norm_global_Z(df):
    return (df - np.nanmean(df.values)) / np.nanstd(df.values)

## scale by series mean
def norm_mean_scale(df):
    return df.apply(lambda s: s / s.mean())

## Z by series
def norm_Z(df):
    return df.apply(lambda s: (s - s.mean()) / s.std())

## log-transform, then Z by series
def norm_logZ(df):
    return norm_Z(df.apply(np.log))

## Z across series
def norm_Z_across(df):
    return df.apply(lambda s: (s - s.mean()) / s.std(), axis=1)

## log-transform, then Z across series
def norm_logZ_across(df):
    return norm_Z_across(df.apply(np.log))

## scale by half global maximum
def norm_global_max(df):
    return 2.0 * df / np.nanmax(df.values)

