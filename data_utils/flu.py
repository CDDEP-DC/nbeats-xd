
import os
import io
import sys
import shutil
import datetime
import requests
import json
from typing import Dict
from copy import deepcopy
import tarfile
import zipfile

import numpy as np
import pandas as pd
from scipy.spatial import distance

from data_utils.forecast import tryJSON, Struct
from data_utils.forecast import proc_t, proc_tdecay, proc_doy, proc_days_from_Jul_1, proc_fwd_const, proc_const, proc_repeat_across
from data_utils.forecast import norm_Z, norm_global_Z, norm_mean_scale, norm_95_scale, norm_global_max, norm_logZ_across, norm_Z_across


## additional functions for processing or generating exogenous predictors
## domain_defaults() below determines when these are used
## called in forecast.py/load_exog_data()
## must have signature (df, data_index, series_names)



## this struct contains domain-specific information used in specify_ensemble()
## and data_utils/forecast.py/load_exog_data()
## see comments below for details
def domain_defaults():
    x = Struct()
    
    ##  which exogenous predictors to use by default
    x.exog_vars = ["dfhy","dewpC","tempC","surveil","outp"]#,"pop_density_2020","med_age_2023"]
    
    ##  information needed to generate a model ensemble, used in specify_ensemble() below
    x.lookback_opts = [2,3,4,5]#[3,4,5,6]#
    x.random_reps = 3

    ##
    ## information for reading exogenous predictors, used by data_utils/forecast.py/load_exog_data()
    ##
    ## arbitrary names
    x.var_names = ["surveil", "outp",
                   "tempC", "dewpC", "AH",
                   "doy", "dfhy",
                   "pop_density_2020","med_age_2023"]
    ## filename that each of the above variables is read from
    ## (directory is specified in config settings)
    ## is this is None, var_fns must specify a function for generating the variable
    x.var_files = ["flu_surveil_weekly.csv", "flu_outp_weekly.csv",
                   "tempC_weekly.csv", "dewpC_weekly.csv", "AH_weekly.csv",
                   None, None,
                   "pop_density_2020.csv","med_age_2023.csv"]
    ## function for processing each of the above files (or generating if file is None)
    x.var_fns = [None, None,
                 None, None, None,
                 proc_doy, proc_days_from_Jul_1,
                 proc_const, proc_const]
    ## function for normalizing each variable (or None to leave as is)
        ## preserve differences between series? ("norm_global")
        ## or not, when target series are normalized on a per-series or windowed basis?
    x.var_norm = [
                    norm_mean_scale, norm_mean_scale, #norm_global_max, norm_global_max, #
                    norm_Z, norm_Z, norm_Z, #norm_global_Z, norm_global_Z, norm_global_Z, #
                    None, None,
                    None, None ## already normalized
                  ]

    return x


## return a list of settings, one for each model to be ensembled
def specify_ensemble(template, specs):
    settings_list = []
    for j in range(specs.random_reps):
        for opt in specs.lookback_opts:
            x = deepcopy(template)
            x.lookback = opt
            settings_list.append(x)
    return settings_list

## setting size of hidden layer based on size of lookback window:
def custom_ensemble(template, specs):
    settings_list = []
    for j in range(specs.random_reps):
        for opt in specs.lookback_opts:
            x = deepcopy(template)
            x.lookback = opt
            x.nbeats_hidden_dim = opt * 2 * 6 * 8
            #x.nbeats_hidden_dim = opt * 2 * 6 * 7
            settings_list.append(x)
    return settings_list



## state names <-> fips codes, etc.
def code_xwalk(a,b):
    return pd.read_csv("series_codes.csv",dtype=str).set_index(a)[b].to_dict()


## process data for forecast hub
## expects rstate to contain quantile forecasts returned by data_utils/forecast.py/generate_quantiles()
## expects the forecast dict to have an entry keyed "ensemble"
## "forecast_delay" is the number of days between end of data and the "reference date"
def output_df(rstate, forecast_delay_days):

    fips = code_xwalk("name","fips")
    
    qtiles = rstate.qtiles
    data_index = rstate.data_index
    fc_quantiles = rstate.fc_quantiles["ensemble"]
    us_quantiles = rstate.sum_quantiles["ensemble"]
    fc_mean = rstate.fc_mean["ensemble"]
    us_mean = rstate.sum_mean["ensemble"]

    time_freq = pd.infer_freq(data_index)
    train_end_date = pd.to_datetime(data_index[-1])
    forecast_date = pd.to_datetime(train_end_date + pd.Timedelta(days=forecast_delay_days))

    ## use ensembled forecasts; append US forecast derived above
    q_ensemble = np.concatenate([fc_quantiles, us_quantiles],axis=0)
    ## use mean as forecast point
    point_ensemble = np.concatenate([fc_mean, us_mean],axis=0)

    series_names = rstate.series_names.to_list() + ["US"] 
    quantile_labels = [f'{x:.3f}' for x in qtiles]

    date_indices = pd.date_range(start=train_end_date,inclusive="right",periods=1+q_ensemble.shape[1],freq=time_freq)
    
    dfs = []
    ## loop through each location in q_ensemble and make a dataframe with shape [date, value at each quantile]
    for i in range(q_ensemble.shape[0]):
        df = pd.DataFrame(q_ensemble[i,:,:], index=date_indices, columns=quantile_labels)
        df["mean"] = point_ensemble[i,:]
        dfs.append(df.melt(ignore_index=False,var_name="quantile").reset_index(names="target_end_date"))

    ## concatenate the location dataframes and set index to location code
    df_hub = pd.concat(dfs,keys=series_names).droplevel(1).reset_index(names="series_name")

    ## add the rest of the columns required by forecast hub
    df_hub["target"] = "wk inc flu hosp"
    df_hub["reference_date"] = forecast_date
    df_hub["horizon"] = df_hub["target_end_date"].dt.to_period(time_freq).view(dtype="int64") - df_hub["reference_date"].dt.to_period(time_freq).view(dtype="int64")
    df_hub["output_type"] = "quantile"
    df_hub.loc[df_hub["quantile"]=="mean", "output_type"] = "point"
    df_hub["output_type_id"] = df_hub["quantile"]
    df_hub["location"] = df_hub["series_name"].apply(lambda x: fips.get(x, x))

    ## if using error dist allows negative values, set them to 0
    df_hub.loc[df_hub["value"]<0.0,"value"] = 0.0

    return df_hub, forecast_date


## append subset of output df to file used for plotting
def append_forecasts(df, filepath):

    df_plt = df.query("quantile=='0.025' or quantile=='0.975' or quantile=='0.500' or quantile=='mean'")[["reference_date","series_name","quantile","target_end_date","value"]].copy()
    df_plt["location"] = df_plt["series_name"]
    ## forecast was generated when data became availalble
    ## this is one week after data end date and 1 week before the forecast hub "reference date"
    ## (at this point, the first forecast week is already in the past)
    df_plt["forecast_date"] = df_plt["reference_date"].map(lambda x: pd.to_datetime(x) + pd.Timedelta(days=-7))

    df_plt = df_plt[["forecast_date","location","quantile","target_end_date","value"]].copy()
    df_plt.loc[df_plt["location"]=="US","location"] = "United States"
    df_plt["value"] = df_plt["value"].round(2)

    try:
        x = pd.read_csv(filepath)
        x["forecast_date"] = pd.to_datetime(x["forecast_date"])
        x["target_end_date"] = pd.to_datetime(x["target_end_date"])
    except:
        x = None
        
    df_plt = pd.concat([x,df_plt],ignore_index=True)
    df_plt.to_csv(filepath,index=False)
    return None



## mmwr week 1 is the first week of the year with 4 or more days
## starts on sunday
def mmwr_start(y):
    j1 = pd.Timestamp(y,1,1)
    dow = j1.isoweekday()
    return j1 + pd.Timedelta(days=7*(dow>3) - dow)

## days to end of week w (saturday)
def mmwr_delta(w):
    return pd.Timedelta(days=(7*w)-1)

## pairwise distance between columns of df
def dfDist(df):
    #return pd.DataFrame(distance.cdist(df.T,df.T,'seuclidean',V=None),index=df.columns,columns=df.columns)
    return pd.DataFrame(distance.cdist(df.T,df.T,'minkowski', p=1.5),index=df.columns,columns=df.columns)

## turns a daily-indexed df into weekly, using reducing function fn
## each date in date_idx is the end of a week
def weekly_reduce(df, fn, date_idx):
    data = []
    for d1 in date_idx:
        d0 = d1 - pd.Timedelta(days=6)
        data.append(df.loc[d0:d1,:].apply(fn))
    return pd.DataFrame(data,index=date_idx)

## replaces zeros and NaNs in input_df with noise
## if thresh is not None, interpolates between the surrounding non-zero points
## - inserts a value equal to noise_scale at the midpoint if the surrounding points are farther apart than thresh
## - then adds noise on the scale of +/- noise_scale
## otherwise, just sets values to 0.5*noise_scale to 2*noise_scale
## rounds the final result to data_precision decimal points
## any final values of 0 are replaced with 0.5 * noise_scale
def zeros_to_noise(input_df, noise_scale, thresh, data_precision):

    ## don't modify input df
    df = input_df.copy(deep=True)

    ## locations of zeros or nans:
    df0 = (input_df.fillna(0) < (0.5*noise_scale))
    ## changes zeros to nans
    df[df0] = np.nan
    
    if thresh is not None:
        zero_idxs = df0.apply(lambda s:np.nonzero(s)[0]).to_list()
        zero_spans = [list(zip(x[(np.diff(x,prepend=-99))>1], x[(np.diff(x[::-1],prepend=99+max(x,default=0))[::-1])<-1])) for x in zero_idxs]
        long_zero_spans = [[x for x in v if ((x[1]-x[0])>(thresh-1))] for v in zero_spans]
        zero_midpoints = [[int(np.round(np.mean(p))) for p in v] for v in long_zero_spans]

        ## before interpolation, set midpoints of long spans to a small number
        for (col_idx,v) in enumerate(zero_midpoints):
            for idx in v:
                df.iloc[idx,col_idx] = noise_scale
    
        ## only interpolate between
        df = df.interpolate(method="linear",limit_area="inside").ffill().bfill()

    else:
        df = df.fillna(noise_scale)

    ## add noise where the data were zero or missing
    rng = np.random.default_rng()
    df_rand = pd.DataFrame((-1.0 * noise_scale) + (2.0 * noise_scale * rng.random(df.shape)), index=df.index, columns=df.columns)
    df[df0] = df[df0] + df_rand[df0]
    ## round to the precision of the original data
    df = df.round(data_precision)
    ## don't actually allow zero values though
    df = df.apply(lambda s: np.maximum(s,(0.5*noise_scale)))
    return df


## preprocess downloaded flu data
## output csv's with columns = series and rows = dates
def read_flu_data():

    data_dir = os.path.join("storage","download")
    output_dir = os.path.join("storage","flu_training")

    ## data series available in flusurv-net; used to pretrain models
    series_names = ['California',
                    'Colorado',
                    'Connecticut',
                    'Georgia',
                    'Maryland',
                    'Michigan',
                    'Minnesota',
                    #'New Mexico',  ## missing surveillance
                    #'New York - Albany',
                    #'New York - Rochester', ## averaged in data.cdc.gov source
                    'New York',
                    'Ohio',
                    'Oregon',
                    'Tennessee',
                    #'Utah'  ## missing surveillance
                    ]

    abbr_to_name = code_xwalk("abbr","name")
    census = pd.read_csv(os.path.join(data_dir,"census2023.csv")).set_index("NAME")["POPESTIMATE2023"]
    ## assume rochester represents 25% of ny state
    #census["New York - Rochester"] = census["New York"] * 0.25
    #census["New York - Albany"] = census["New York"] * 0.75

    ##
    ## flusight forecast-hub targets
    ## from https://data.cdc.gov/Public-Health-Surveillance/Weekly-Hospital-Respiratory-Data-HRD-Metrics-by-Ju/ua7e-t2fy/about_data
    ##
    df = pd.read_csv(os.path.join(data_dir,"NHSN_hosp.csv"),usecols=["weekendingdate","jurisdiction","totalconfc19newadm","totalconfflunewadm","totalconfrsvnewadm"])
    df["date"] = pd.to_datetime(df["weekendingdate"])
    flu_true_count = df.pivot(index="date",columns='jurisdiction',values='totalconfflunewadm')
    flu_true_count = flu_true_count[[k for k in abbr_to_name]].rename(columns=abbr_to_name).fillna(0.0)
    ## drop useless training data
    flu_true_count = flu_true_count.loc["2021-01-01":]
    ## per capita
    flu_true = flu_true_count.apply(lambda s: 100000.0 * s / census[s.name])

    ## pairwise distances between series
    series_distances = dfDist(flu_true)

    ## replace zeros and missing data with noise
    flu_true = zeros_to_noise(flu_true,0.02,None,6)
    ## scale per-capita noise back up to counts
    flu_true_count = flu_true.apply(lambda s: (census[s.name]/100000.0) * s).round(1)

    ##
    ## flusurv-net hospitalization rates per 100k (participating regions from 2009-)
    ## from fluview: https://gis.cdc.gov/GRASP/Fluview/FluHospRates.html
    ##
    #flu_hosp = pd.read_csv(os.path.join(data_dir,"flu_surv_current.csv"),dtype=str).query(
    #    "`AGE CATEGORY`=='Overall' and `SEX CATEGORY`=='Overall' and `RACE CATEGORY`=='Overall' and `VIRUS TYPE CATEGORY`=='Overall'")
    #flu_hosp = flu_hosp[['CATCHMENT', 'YEAR.1', 'WEEK', 'WEEKLY RATE']].copy()
    ## append to data from previous years
    #flu_hosp = pd.concat([
    #    pd.read_csv(os.path.join(data_dir,"flu_surv_hist.csv"),dtype=str), 
    #    flu_hosp
    #    ], ignore_index=True)
    ## process columns
    #flu_hosp["MMWR-YEAR"] = flu_hosp["YEAR.1"].astype(int)
    #flu_hosp["MMWR-WEEK"] = flu_hosp["WEEK"].astype(int)
    #flu_hosp["WEEKLY RATE"] = flu_hosp["WEEKLY RATE"].astype(float)
    #flu_hosp["date"] = flu_hosp["MMWR-YEAR"].map(mmwr_start) + flu_hosp["MMWR-WEEK"].map(mmwr_delta)

    ##
    ## flusurv-net hospitalization rates per 100k 
    ## from https://data.cdc.gov/Public-Health-Surveillance/Rates-of-Laboratory-Confirmed-RSV-COVID-19-and-Flu/kvib-3txy/about_data
    ##
    df = pd.read_csv(os.path.join(data_dir,"kvib-3txy.csv"))
    df["MMWR-YEAR"] = df["mmwr_year"].astype(int)
    df["MMWR-WEEK"] = df["mmwr_week"].astype(int)
    df['CATCHMENT'] = df["site"]
    df["WEEKLY RATE"] = df["weekly_rate"].astype(float)
    df = df[['CATCHMENT', 'MMWR-YEAR', 'MMWR-WEEK', 'WEEKLY RATE']].copy()
    df["date"] = df["MMWR-YEAR"].map(mmwr_start) + df["MMWR-WEEK"].map(mmwr_delta)
    flunet = df.pivot(index="date",columns="CATCHMENT",values="WEEKLY RATE")[series_names]
    ## append to data from previous years
    flunet_hist = pd.read_csv(os.path.join(data_dir,"flunet_samples_hist.csv"),index_col=0)
    flunet_hist.index = pd.to_datetime(flunet_hist.index)
    flunet = pd.concat([flunet_hist,flunet])

    ##
    ## who/nrevss clinical lab surveillance data (all US states from 2015-)
    ## from fluview: https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html
    ##
    surveil = pd.concat([
                pd.read_csv(os.path.join(data_dir,"NREVSS_Clinical_hist.csv"),usecols=['REGION', 'YEAR', 'WEEK', 'TOTAL SPECIMENS','PERCENT POSITIVE'],dtype=str),
                pd.read_csv(os.path.join(data_dir,"WHO_NREVSS_Clinical_Labs.csv"),skiprows=1,usecols=['REGION', 'YEAR', 'WEEK', 'TOTAL SPECIMENS','PERCENT POSITIVE'],dtype=str)
                ], ignore_index=True)
    ## process columns
    surveil["YEAR"] = surveil["YEAR"].astype(int)
    surveil["WEEK"] = surveil["WEEK"].astype(int)
    surveil["TOTAL SPECIMENS"] = pd.to_numeric(surveil["TOTAL SPECIMENS"],errors="coerce")
    surveil["PERCENT POSITIVE"] = pd.to_numeric(surveil["PERCENT POSITIVE"],errors="coerce")
    surveil["date"] = surveil["YEAR"].map(mmwr_start) + surveil["WEEK"].map(mmwr_delta)
    surveil = surveil.pivot(index="date",columns="REGION",values="PERCENT POSITIVE").sort_index()
    ## assume these are the same
    #surveil['New York - Albany'] = surveil["New York"]
    #surveil['New York - Rochester'] = surveil["New York"]
    ## this one is actually all 0's
    surveil.loc[:,"Puerto Rico"] = np.nan

    ## states that are missing more than 6 weeks of surveil data, when it should almost certainly be present
    check_dates = surveil.index[surveil.index.month.isin([11,12,1,2])]
    missing_surveil = surveil.loc[check_dates].isna().sum()
    fill_states = (missing_surveil[missing_surveil > 6]).index.drop(["New York City", "Virgin Islands"])
    ## these get nan's filled with an average of the 2 most similar states (based on series dists calculated above)
    df_tmp = pd.DataFrame(index=surveil.index)
    for k in fill_states:
        use_states = series_distances[k].sort_values().drop(fill_states).index[0:2]
        df_tmp[k] = surveil[use_states].mean(axis=1)
    ## remaining nans get filled with 0
    surveil = surveil.drop(columns=["New York City", "Virgin Islands"]).fillna(df_tmp).fillna(0.0)

    ## dates and locations in surveillance df
    data_index = surveil.index
    data_cols = surveil.columns

    ##
    ## ilinet outpatient surveillance data (all US states from 2010-)
    ## from fluview: https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html
    ##
    outp = pd.concat([
            pd.read_csv(os.path.join(data_dir,"ILINet_hist.csv"),usecols=['REGION', 'YEAR', 'WEEK', '%UNWEIGHTED ILI', 'ILITOTAL', 'TOTAL PATIENTS'],dtype=str),
            pd.read_csv(os.path.join(data_dir,"ILINet.csv"),skiprows=1,usecols=['REGION', 'YEAR', 'WEEK', '%UNWEIGHTED ILI', 'ILITOTAL', 'TOTAL PATIENTS'],dtype=str)
            ], ignore_index=True)
    ## process columns
    outp["YEAR"] = outp["YEAR"].astype(int)
    outp["WEEK"] = outp["WEEK"].astype(int)
    outp["%UNWEIGHTED ILI"] = pd.to_numeric(outp["%UNWEIGHTED ILI"],errors="coerce")
    outp["ILITOTAL"] = pd.to_numeric(outp["ILITOTAL"],errors="coerce")
    outp["TOTAL PATIENTS"] = pd.to_numeric(outp["TOTAL PATIENTS"],errors="coerce")
    outp["date"] = outp["YEAR"].map(mmwr_start) + outp["WEEK"].map(mmwr_delta)
    outp = outp.pivot(index="date",columns="REGION",values="%UNWEIGHTED ILI").sort_index()
    ## assume these are the same
    #outp['New York - Albany'] = outp["New York"]
    #outp['New York - Rochester'] = outp["New York"]
    ## keep rows/cols in lab surveil data
    outp = pd.DataFrame(index=data_index).join(outp).loc[:,data_cols]

    ## use flu-net data only from the time period when surveillance data is available
    flunet = pd.DataFrame(index=data_index).join(flunet)
    ## don't allow 0's
    #flunet = zeros_to_noise(flunet, 0.1, 6, 1)
    flunet = zeros_to_noise(flunet, 0.1, None, 1)
    ## assume each flu-surv sampling area represents its state
    ## (produces an expected state-wide count, for possible use as forecast target)
    flunet_count = flunet.apply(lambda s: s * census[s.name]  / 100000.0)

    ## write csv's
    flunet.round(2).to_csv(os.path.join(output_dir,"flunet_samples_per100k.csv"))
    flunet_count.round(2).to_csv(os.path.join(output_dir,"flunet_samples_count.csv"))

    surveil.round(3).to_csv(os.path.join(output_dir,"flu_surveil_weekly.csv"))
    outp.round(6).to_csv(os.path.join(output_dir,"flu_outp_weekly.csv"))

    flu_true.to_csv(os.path.join(output_dir,"flusight_truth_per100k.csv"))
    flu_true_count.to_csv(os.path.join(output_dir,"flusight_truth_count.csv"))

    ds = census[series_names]
    ds.to_csv(os.path.join(output_dir,"flunet_populations.csv"))
    ## to get national per-capita from regional per-capita forecasts
    pd.DataFrame((ds / ds.sum()).rename("weight")).round(6).to_csv(os.path.join(output_dir,"flunet_weights.csv"))
    ## for all states
    ds = census[flu_true.columns]
    ds.to_csv(os.path.join(output_dir,"flusight_populations.csv"))
    pd.DataFrame((ds / ds.sum()).rename("weight")).round(6).to_csv(os.path.join(output_dir,"flusight_weights.csv"))
    ## for converting per 100k capita forecasts to statewide totals
    #(ds / 100000.0).rename("weight").round(6).to_csv(os.path.join(output_dir,"per100k_to_state.csv"))

    return (data_index, data_cols)


## pop dens and med age; Z-scores across US states
def read_pop_data():

    data_dir = os.path.join("storage","download")
    output_dir = os.path.join("storage","flu_training")

    series_names = pd.read_csv(os.path.join(output_dir,"flusight_truth_count.csv"),index_col=0).columns

    df = pd.read_csv(os.path.join(data_dir,"apportionment.csv"),thousands=",")
    df = df.loc[(df.Year==2020),['Name','Resident Population Density']].copy()
    df["density"] = df['Resident Population Density']
    ds = df.set_index("Name").loc[series_names,"density"]
    ## log Z normalize
    ds = ds.apply(np.log)
    ds = (ds - ds.mean()) / ds.std()
    ## ???
    #ds["New York - Rochester"] = ds["New York"]
    #ds["New York - Albany"] = ds["New York"]
    ds.round(6).to_csv(os.path.join(output_dir,"pop_density_2020.csv"))

    df = pd.read_csv(os.path.join(data_dir,"ACSST1Y2022.S0101-2023-12-18T014857.csv"))
    df = df.iloc[34,1:].reset_index(name="median_age")
    df["labels"] = df["index"].str.split("!!")
    df["loc"] = df.labels.map(lambda x:x[0])
    df["istotal"] = df.labels.map(lambda x:x[1]=="Total")
    df["isest"] = df.labels.map(lambda x:x[2]=="Estimate")
    df = df.loc[(df.istotal & df.isest),:].copy()
    df["median_age"] = df["median_age"].astype(float)
    ds = df.set_index("loc").loc[series_names,"median_age"]
    ## Z normalize
    ds = (ds - ds.mean()) / ds.std()
    ## ???
    #ds["New York - Rochester"] = ds["New York"]
    #ds["New York - Albany"] = ds["New York"]
    ds.round(6).to_csv(os.path.join(output_dir,"med_age_2023.csv"))

    return None


## processes weather data from downloaded files
## saves each variable to a file in training data folder
def read_weather_data(date_idx):

    data_dir = os.path.join("storage","download")
    output_dir = os.path.join("storage","flu_training")
    id_column = "series_name"
    years = range(date_idx[0].year, 1 + date_idx[-1].year)

    weatherdata = None
    for year in years:
        f = os.path.join(data_dir, "weather" + str(year) + ".csv")
        df = pd.read_csv(f,dtype={"STATION":str,"FRSHTT":str,"fips":str})
        weatherdata = pd.concat([weatherdata,df],ignore_index=True)

    ## noaa represents missing data with a bunch of 9's
    ## fill with last reported value
    weatherdata.loc[(weatherdata.TEMP > 200.0), "TEMP"] = np.nan
    weatherdata.TEMP = weatherdata.TEMP.ffill()
    weatherdata.loc[(weatherdata.DEWP > 200.0), "DEWP"] = np.nan
    weatherdata.DEWP = weatherdata.DEWP.ffill()

    weatherdata['tempC'] = (weatherdata['TEMP'].astype(float) - 32) * (5/9)
    weatherdata['dewpC'] = (weatherdata['DEWP'].astype(float) - 32) * (5/9)
    #following RH equation from https://bmcnoldy.rsmas.miami.edu/Humidity.html
    weatherdata['RH'] = 100 * np.exp((17.625*weatherdata['dewpC'])/(243.04+weatherdata['dewpC']))/np.exp((17.625*weatherdata['tempC'])/(243.04+weatherdata['tempC']))
    #following AH equation from https://carnotcycle.wordpress.com/2012/08/04/how-to-convert-relative-humidity-to-absolute-humidity/
    weatherdata['AH'] = (6.112 * np.exp((17.67 * weatherdata['tempC'])/(weatherdata['tempC']+243.5)) * weatherdata['RH'] * 2.1674) / (273.15+weatherdata['tempC'])

    for c in ["tempC","dewpC","RH","AH"]:
        weatherdata[c] = weatherdata[c].round(2)

    weatherdata.DATE = pd.to_datetime(weatherdata.DATE)

    for c in ["tempC","dewpC","AH"]:
        df_by_loc = weatherdata.pivot(columns=id_column,values=c,index="DATE")
        weekly_mean = weekly_reduce(df_by_loc, np.nanmean, date_idx)
        weekly_mean.interpolate(method="time",limit_direction="both").round(2).to_csv(os.path.join(output_dir, c+"_weekly.csv"))
        df_7ma = df_by_loc.rolling(7, min_periods=1).mean().loc[date_idx[0]:date_idx[-1],:]
        df_7ma.interpolate(method="time",limit_direction="both").round(2).to_csv(os.path.join(output_dir, c+"_7ma.csv"))
    return None



## forecast hub targets
## from https://data.cdc.gov/Public-Health-Surveillance/Weekly-Hospital-Respiratory-Data-HRD-Metrics-by-Ju/ua7e-t2fy/about_data
def download_forecast_hub(dest):

    u = "https://data.cdc.gov/resource/ua7e-t2fy.csv?$limit=500000"
    response = requests.get(u)

    with open(os.path.join(dest,"NHSN_hosp.csv"), "wb") as file:
        file.write(response.content)


## flu hospitalization from flusurv-net
## https://data.cdc.gov/Public-Health-Surveillance/Rates-of-Laboratory-Confirmed-RSV-COVID-19-and-Flu/kvib-3txy/about_data
## this link downloads 2024-25 season
## (previous seasons are in flunet_samples_hist.csv)
##
## WARNING: query api silently truncates at 1000 rows
##
def download_flusurv_net(dest):
    u = "https://data.cdc.gov/resource/kvib-3txy.csv?$query=SELECT%0A%20%20%60surveillance_network%60%2C%0A%20%20%60season%60%2C%0A%20%20%60mmwr_year%60%2C%0A%20%20%60mmwr_week%60%2C%0A%20%20%60age_group%60%2C%0A%20%20%60sex%60%2C%0A%20%20%60race_ethnicity%60%2C%0A%20%20%60site%60%2C%0A%20%20%60weekly_rate%60%2C%0A%20%20%60cumulative_rate%60%2C%0A%20%20%60_weekenddate%60%2C%0A%20%20%60type%60%0AWHERE%0A%20%20caseless_one_of(%60season%60%2C%20%222024-25%22)%0A%20%20AND%20(caseless_one_of(%60age_group%60%2C%20%22Overall%22)%0A%20%20%20%20%20%20%20%20%20AND%20(caseless_one_of(%60sex%60%2C%20%22Overall%22)%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20AND%20(caseless_one_of(%60race_ethnicity%60%2C%20%22Overall%22)%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20AND%20caseless_one_of(%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%60surveillance_network%60%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%22FluSurv-NET%22%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20))))"
    response = requests.get(u)

    with open(os.path.join(dest,"kvib-3txy.csv"), "wb") as file:
        file.write(response.content)


## flu surveillance data
## scraped from: https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html
## this request downloads 2024-25 season
## (previous seasons are in NREVSS_Clinical_hist.csv and ILINet_hist.csv)
def download_flu_surveil(dest):
    u = "https://gis.cdc.gov/grasp/flu2/PostPhase02DataDownload"
    d = '{"AppVersion":"Public","DatasourceDT":[{"ID":0,"Name":"WHO_NREVSS"},{"ID":1,"Name":"ILINet"}],"RegionTypeId":5,"SubRegionsDT":[{"ID":1,"Name":"1"},{"ID":2,"Name":"2"},{"ID":3,"Name":"3"},{"ID":4,"Name":"4"},{"ID":5,"Name":"5"},{"ID":6,"Name":"6"},{"ID":7,"Name":"7"},{"ID":8,"Name":"8"},{"ID":9,"Name":"9"},{"ID":10,"Name":"10"},{"ID":11,"Name":"11"},{"ID":12,"Name":"12"},{"ID":13,"Name":"13"},{"ID":14,"Name":"14"},{"ID":15,"Name":"15"},{"ID":16,"Name":"16"},{"ID":17,"Name":"17"},{"ID":18,"Name":"18"},{"ID":19,"Name":"19"},{"ID":20,"Name":"20"},{"ID":21,"Name":"21"},{"ID":22,"Name":"22"},{"ID":23,"Name":"23"},{"ID":24,"Name":"24"},{"ID":25,"Name":"25"},{"ID":26,"Name":"26"},{"ID":27,"Name":"27"},{"ID":28,"Name":"28"},{"ID":29,"Name":"29"},{"ID":30,"Name":"30"},{"ID":31,"Name":"31"},{"ID":32,"Name":"32"},{"ID":33,"Name":"33"},{"ID":34,"Name":"34"},{"ID":35,"Name":"35"},{"ID":36,"Name":"36"},{"ID":37,"Name":"37"},{"ID":38,"Name":"38"},{"ID":39,"Name":"39"},{"ID":40,"Name":"40"},{"ID":41,"Name":"41"},{"ID":42,"Name":"42"},{"ID":43,"Name":"43"},{"ID":44,"Name":"44"},{"ID":45,"Name":"45"},{"ID":46,"Name":"46"},{"ID":47,"Name":"47"},{"ID":48,"Name":"48"},{"ID":49,"Name":"49"},{"ID":50,"Name":"50"},{"ID":51,"Name":"51"},{"ID":52,"Name":"52"},{"ID":54,"Name":"54"},{"ID":55,"Name":"55"},{"ID":56,"Name":"56"},{"ID":58,"Name":"58"},{"ID":59,"Name":"59"}],"SeasonsDT":[{"ID":64,"Name":"64"}]}'
    response = requests.post(u, json=json.loads(d))

    b = io.BytesIO(response.content)
    with zipfile.ZipFile(b, 'r') as z:
        z.extractall(dest)




## process weather file downloaded from bigquery
def read_bq_local(year,stations):
    df = pd.read_csv('storage/weather/bq'+str(year)+'.zip',dtype={"stn":str,"wban":str,"date":str})
    df["station"] = df["stn"] + df["wban"]
    df = df.rename(columns=str.upper).rename(columns={"MXPSD":"MXSPD"})

    for k in ['LATITUDE','LONGITUDE','ELEVATION']:
        df[k] = np.nan

    for k in ['NAME','FRSHTT']:
        df[k] = ""

    df = df[['STATION','DATE','LATITUDE','LONGITUDE','ELEVATION','NAME','TEMP','DEWP','SLP','STP','VISIB','WDSP','MXSPD','GUST','MAX','MIN','PRCP','SNDP','FRSHTT']]

    weatherdata = None
    for k in stations:
        x = stations[k]
        ds = df.loc[(df["STATION"] == x) & (df["DATE"] >= str(year)+'-01-01')].sort_values("DATE")
        ds["series_name"] = k
        weatherdata = pd.concat([weatherdata,ds],ignore_index=True)
    return weatherdata

## tar file from noaa
def read_weather_local(year,station):
    with tarfile.open('storage/weather/'+str(year)+'.tar.gz','r') as t:
        with t.extractfile(station+'.csv') as f:
            ds = pd.read_csv(f,usecols=['STATION', 'DATE', 'LATITUDE', 'LONGITUDE', 
                                'ELEVATION', 'NAME', 'TEMP', 'DEWP', 'SLP',
                                'STP', 'VISIB','WDSP', 'MXSPD', 'GUST',
                                'MAX', 'MIN', 'PRCP', 'SNDP', 'FRSHTT'],
                                dtype={"STATION":str,"FRSHTT":str})
    return ds


## reads specified year from specified weather stations, returns dataframe
## stations keyed by state abbrev.
def read_noaa_dir(year,stations,fips,abbr_to_name,local_archive=False):

    usecols=['STATION', 'DATE', 'LATITUDE', 'LONGITUDE', 
            'ELEVATION', 'NAME', 'TEMP', 'DEWP', 'SLP',
            'STP', 'VISIB','WDSP', 'MXSPD', 'GUST',
            'MAX', 'MIN', 'PRCP', 'SNDP', 'FRSHTT']
    dtype={"STATION":str,"FRSHTT":str}

    weatherdata = None
    for k in stations:
        x = stations[k]

        if local_archive:
             ds = read_weather_local(year,x)
        else:
            url = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/" + str(year) + "/" + x + ".csv"
            s = requests.get(url).content
            ds = pd.read_csv(io.StringIO(s.decode('utf-8')),usecols=usecols,dtype=dtype)

        ds["state"] = k
        ds["fips"] = fips[k]
        ds["series_name"] = abbr_to_name[k]
        weatherdata = pd.concat([weatherdata,ds],ignore_index=True)
    return weatherdata


def download_weather(year, dest, local_archive=False):

    stations = {"AL": "72228013876",
                "AK": "70273026451",
                "AZ": "72274023160",
                "AR": "72340313963",
                "CA": "72295023174",
                "CO": "72565003017",
                "CT": "72508014740",
                "DE": "72418013781",
                "FL": "72202012839",
                "GA": "72219013874",
                "HI": "91182022521",
                "ID": "72681024131",
                "IL": "72530094846",
                "IN": "72438093819",
                "IA": "72546014933",
                "KS": "72450003928",
                "KY": "72423093821",
                "LA": "72231012916",
                "ME": "72606014764",
                "MD": "72406093721",
                "MA": "72509014739",
                "MI": "72537094847",
                "MN": "72658014922",
                "MS": "72235003940",
                "MO": "72434013994",
                "MT": "72677024033",
                "NE": "72550014942",
                "NV": "72386023169",
                "NH": "72605014745",
                "NJ": "72502014734",
                "NM": "72365023050",
                "NY": "74486094789",
                "NC": "72306013722",
                "ND": "72753014914",
                "OH": "72524014820",
                "OK": "72353013967",
                "OR": "72698024229",
                "PA": "72408013739",
                "RI": "72507014765",
                "SC": "72208013880",
                "SD": "72651014944",
                "TN": "72327013897",
                "TX": "72243012960",
                "UT": "72572024127",
                "VT": "72617014742",
                "VA": "72403093738",
                "WA": "72793024233",
                "WV": "72414013866",
                "WI": "72640014839",
                "WY": "72564024018",
                "DC": "72405013743",
                "PR": "78526011641"
                }
                #'New York - Albany': '72518014735',
                #'New York - Rochester': '72529014768'

    df = read_noaa_dir(year, stations, code_xwalk("abbr","fips"), code_xwalk("abbr","name"), local_archive)
    df.to_csv(os.path.join(dest,"weather"+str(year)+".csv"),index=False)



