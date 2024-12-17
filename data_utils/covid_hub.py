
import os
import io
import sys
import shutil
import datetime
import requests
import json
from typing import Dict
from copy import deepcopy
import zipfile
import tarfile

import numpy as np
import pandas as pd

from data_utils.forecast import tryJSON, Struct, str_indexed_csv
from data_utils.forecast import proc_t, proc_tdecay, proc_doy, proc_fwd_const, proc_const, proc_repeat_across, proc_days_from_Jul_1
from data_utils.forecast import norm_Z, norm_global_Z, norm_mean_scale, norm_global_max, norm_logZ_across, norm_Z_across


## additional functions for processing or generating exogenous predictors
## domain_defaults() below determines when these are used
## called in forecast.py/load_exog_data()
## must have signature (df, data_index, series_names)

## custom fn for backfilling and normalizing wastewater data
def proc_wastewater(df, data_index, series_names):
    ## before data is available, set each series to its median
    ##   (is this better than setting to a nonsense value?)
    x = pd.DataFrame(index=data_index).join(df)
    x = x.apply(lambda s: s.fillna(np.nanmedian(s)))
    ## wastewater percentile data is already normalized
    return x

## generate time since last variant of concern as a float from 0 -> 2
## (dates from cdc website)
## first wk of available target data = 7/14-7/20 (19-20 weeks from 3/1)
## a/b/g: 12/29/20  (24 weeks from 7/14)
## e: 3/19/21  (wk 35)
## d: 6/15/21  (wk 48)
## o: 11/26/21  (71)
def proc_tvoc(df, data_index, series_names):
    data_start = pd.to_datetime(data_index[0])
    time_a = pd.to_datetime("2020-12-29") 
    time_e = pd.to_datetime("2021-03-19")
    time_d = pd.to_datetime("2021-06-15")
    time_o = pd.to_datetime("2021-11-26")
    delta_a = (time_a - data_start).days
    delta_e = (time_e - data_start).days
    delta_d = (time_d - data_start).days
    delta_o = (time_o - data_start).days
    data_offset = (data_start - pd.to_datetime("2020-03-01")).days
    timepoints = np.arange(len(data_index))
    t_a = timepoints - delta_a; t_a[t_a < 0] = 99999
    t_e = timepoints - delta_e; t_e[t_e < 0] = 99999
    t_d = timepoints - delta_d; t_d[t_d < 0] = 99999
    t_o = timepoints - delta_o; t_o[t_o < 0] = 99999
    time_since_voc = np.stack([timepoints+data_offset,t_a,t_e,t_d,t_o]).min(axis=0)
    return pd.DataFrame({s:(2.0 * time_since_voc / np.max(time_since_voc))
                        for s in series_names}, index=data_index)

## take sqrt, then scale by series mean
def norm_sqrt_mean(df):
    return df.apply(np.sqrt).apply(lambda s: s / s.mean())


## this struct contains domain-specific information used in specify_ensemble()
## and data_utils/forecast.py/load_exog_data()
## see comments below for details
def domain_defaults():
    x = Struct()
    
    ##  which exogenous predictors to use by default
    x.exog_vars = ["dfhy","dewpC","surveil"]
    
    ##  information needed to generate a model ensemble, used in specify_ensemble() below
    x.lookback_opts = [3,4,5,6]
    x.random_reps = 5

    ##
    ## information for reading exogenous predictors, used by data_utils/forecast.py/load_exog_data()
    ##
    ## arbitrary names
    x.var_names = ["surveil", "dewpC", "doy", "dfhy"]
    ## filename that each of the above variables is read from
    ## (directory is specified in config settings)
    ## is this is None, var_fns must specify a function for generating the variable
    x.var_files = ["surveil_weekly.csv", "dewpC_weekly.csv", None, None]
    ## function for processing each of the above files (or generating if file is None)
    x.var_fns = [None, None, proc_doy, proc_days_from_Jul_1]
    ## function for normalizing each variable (or None to leave as is)
    x.var_norm = [norm_sqrt_mean, norm_Z, None, None]

    return x

## pretraining dataset (aggregated by hhs region)
def domain_defaults_pretrain():
    x = Struct()
    
    ##  which exogenous predictors to use by default
    x.exog_vars = ["dfhy","dewpC","surveil"] 
    
    ##  information needed to generate a model ensemble, used in specify_ensemble() below
    x.lookback_opts = [3,4,5,6]
    x.random_reps = 5

    ##
    ## information for reading exogenous predictors, used by data_utils/forecast.py/load_exog_data()
    ##
    ## arbitrary names
    x.var_names = ["surveil", "dewpC", "doy", "dfhy"]
    ## filename that each of the above variables is read from
    ## (directory is specified in config settings)
    ## is this is None, var_fns must specify a function for generating the variable
    x.var_files = ["surveil_weekly_by_hhs.csv", "dewpC_weekly_by_hhs.csv", None, None]
    ## function for processing each of the above files (or generating if file is None)
    x.var_fns = [None, None, proc_doy, proc_days_from_Jul_1]
    ## function for normalizing each variable (or None to leave as is)
    x.var_norm = [norm_sqrt_mean, norm_Z, None, None]

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



## state names <-> fips codes, etc.
def code_xwalk(a,b):
    return pd.read_csv("series_codes.csv",dtype=str).set_index(a)[b].to_dict()


## process data for covid-19 forecast hub
## expects rstate to contain quantile forecasts returned by data_utils/forecast.py/generate_quantiles()
## expects the forecast dict to have an entry keyed "ensemble"
## "forecast_delay" is the # of days between the last day of data and the "reference date"
def output_df(rstate, forecast_delay_days):

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
    df_hub["target"] = "wk inc covid hosp"
    df_hub["reference_date"] = forecast_date
    df_hub["horizon"] = df_hub["target_end_date"].dt.to_period(time_freq).view(dtype="int64") - df_hub["reference_date"].dt.to_period(time_freq).view(dtype="int64")
    df_hub["output_type"] = "quantile"
    df_hub.loc[df_hub["quantile"]=="mean", "output_type"] = "point"
    df_hub["output_type_id"] = df_hub["quantile"]
    df_hub["location"] = df_hub["series_name"]

    ## if using error dist allows negative values, set them to 0
    df_hub.loc[df_hub["value"]<0.0,"value"] = 0.0

    return df_hub, forecast_date



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


def get_hhs_regions():
    return {
    "Region 1": ["Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island", "Vermont"],
    "Region 2": ["New Jersey", "New York", "Puerto Rico", "Virgin Islands"],
    "Region 3": ["Delaware", "District of Columbia", "Maryland", "Pennsylvania", "Virginia", "West Virginia"],
    "Region 4": ["Alabama", "Florida", "Georgia", "Kentucky", "Mississippi", "North Carolina", "South Carolina", "Tennessee"],
    "Region 5": ["Illinois", "Indiana", "Michigan", "Minnesota", "Ohio", "Wisconsin"],
    "Region 6": ["Arkansas", "Louisiana", "New Mexico", "Oklahoma", "Texas"],
    "Region 7": ["Iowa", "Kansas", "Missouri", "Nebraska"],
    "Region 8": ["Colorado", "Montana", "North Dakota", "South Dakota", "Utah", "Wyoming"],
    "Region 9": ["Arizona", "California", "Hawaii", "Nevada", "American Samoa", "Commonwealth of the Northern Mariana Islands", "Federated States of Micronesia", "Guam", "Marshall Islands", "Republic of Palau"],
    "Region 10": ["Alaska", "Idaho", "Oregon", "Washington"]
    }

## preprocess downloaded (weekly) covid data 
## output csv's with columns = series and rows = dates
def read_covid_weekly():

    data_dir = os.path.join("storage","download")
    output_dir = os.path.join("storage","training_data")

    hhs_regions = get_hhs_regions()
    fips = code_xwalk("name","fips")
    fips_abbr = code_xwalk("abbr","fips")

    census = pd.read_csv(os.path.join(data_dir,"census2023.csv")).set_index("NAME")["POPESTIMATE2023"]
    census = pd.Series({fips[k]:census[k] for k in fips}).rename("pop2023")
    hhs_to_fips = {k:[fips[x] for x in hhs_regions[k] if x in fips] for k in hhs_regions}
    census_by_hhs = pd.Series({k:census[hhs_to_fips[k]].sum() for k in hhs_to_fips}).rename("pop2023")

    ##
    ## weekly hospitalizations for forecast hub
    ## from https://data.cdc.gov/Public-Health-Surveillance/Weekly-Hospital-Respiratory-Data-HRD-Metrics-by-Ju/ua7e-t2fy/about_data
    ##
    df = pd.read_csv(os.path.join(data_dir,"NHSN_hosp.csv"),usecols=["weekendingdate","jurisdiction","totalconfc19newadm","totalconfflunewadm","totalconfrsvnewadm"])
    df["date"] = pd.to_datetime(df["weekendingdate"])
    h_covid_counts = df.pivot(index="date",columns='jurisdiction',values='totalconfc19newadm')
    ## counts for listed fips
    h_covid_counts = h_covid_counts[[k for k in fips_abbr]].rename(columns=fips_abbr).fillna(0.0)
    ## per capita
    h_covid_per100k = h_covid_counts.apply(lambda s: 100000.0 * s / census[s.name])
    ## aggregate counts by hhs region
    covid_by_hhs = pd.DataFrame({k:h_covid_counts[hhs_to_fips[k]].sum(axis=1) for k in hhs_to_fips}, index=h_covid_counts.index)
    ## hhs region per capita
    covid_hhs_per100k = covid_by_hhs.apply(lambda s: 100000.0 * s / census_by_hhs[s.name])
    ## replace zeros and missing data with noise
    h_covid_per100k = zeros_to_noise(h_covid_per100k,0.05,None,6)
    ## scale per-capita noise back up to counts
    h_covid_counts = h_covid_per100k.apply(lambda s: (census[s.name]/100000.0) * s).round(1)

    ##
    ## weekly covid laboratory surveillance by hhs region, from:
    ## https://data.cdc.gov/Laboratory-Surveillance/Percent-Positivity-of-COVID-19-Nucleic-Acid-Amplif/gvsb-yw6g/about_data
    ##
    df = pd.read_csv(os.path.join(data_dir,"NREVSS_covid.csv"))
    df["date"] = pd.to_datetime(df["mmwrweek_end"])
    df["posted"] = pd.to_datetime(df["posted"])
    df = df[["level","percent_pos","posted","date"]].copy()
    ## same date is posted multiple times; keep only the most recently posted
    df = df.sort_values(["level","date","posted"],ascending=[True,True,False])
    df = df.drop_duplicates(subset=["level","date"])
    surveil_weekly_by_hhs = df.pivot(index="date",columns="level",values="percent_pos")[[k for k in hhs_regions]]

    ## set each state's surveillance data to that of its region
    state_to_code = {x:k for k in hhs_regions for x in hhs_regions[k]}
    fips_to_code = {fips[k]:state_to_code[k] for k in fips}
    surveil_weekly = pd.DataFrame({k:surveil_weekly_by_hhs[fips_to_code[k]] for k in fips_to_code},index=surveil_weekly_by_hhs.index)

    ## write csv's
    h_covid_counts.to_csv(os.path.join(output_dir,"covid_truth_weekly_counts.csv"))
    h_covid_per100k.to_csv(os.path.join(output_dir,"covid_truth_weekly_per100k.csv"))
    covid_by_hhs.round(1).to_csv(os.path.join(output_dir,"covid_hhs_weekly_counts.csv"))
    covid_hhs_per100k.round(6).to_csv(os.path.join(output_dir,"covid_hhs_weekly_per100k.csv"))

    census.to_csv(os.path.join(output_dir,"fips_pops.csv"))
    census_by_hhs.to_csv(os.path.join(output_dir,"hhs_pops.csv"))
    (census / census.sum()).rename("weight").round(8).to_csv(os.path.join(output_dir,"fips_weights.csv"))
    (census_by_hhs / census_by_hhs.sum()).rename("weight").round(8).to_csv(os.path.join(output_dir,"hhs_weights.csv"))

    surveil_weekly_by_hhs.to_csv(os.path.join(output_dir,"surveil_weekly_by_hhs.csv"))
    surveil_weekly.to_csv(os.path.join(output_dir,"surveil_weekly.csv"))
    
    ## currently not using:
    #surveil_daily = pd.DataFrame(index=pd.date_range(surveil_weekly.index[0],surveil_weekly.index[-1],freq="D")).join(surveil_weekly).interpolate(method="pchip").round(2)
    #surveil_daily_by_hhs = pd.DataFrame(index=pd.date_range(surveil_weekly_by_hhs.index[0],surveil_weekly_by_hhs.index[-1],freq="D")).join(surveil_weekly_by_hhs).interpolate(method="pchip").round(2)
    #surveil_daily.to_csv(os.path.join(output_dir,"surveil_daily.csv"))
    #surveil_daily_by_hhs.to_csv(os.path.join(output_dir,"surveil_daily_by_hhs.csv"))

    return (surveil_weekly.index, surveil_weekly.columns)



## processes weather data from downloaded files
## saves each variable to a file in training data folder
def read_weather_data(weekly_idx=None):

    data_dir = os.path.join("storage","download")
    output_dir = os.path.join("storage","training_data")
    id_column = "fips"

    if weekly_idx is not None:
        years = range(weekly_idx[0].year, 1 + weekly_idx[-1].year)
    else:
        current_year = (datetime.date.today() - datetime.timedelta(days=3)).year ## data is published on a delay
        years = range(2020, 1+current_year)

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
        if weekly_idx is not None:
            weekly_mean = weekly_reduce(df_by_loc, np.nanmean, weekly_idx)
            weekly_mean.interpolate(method="time",limit_direction="both").round(2).to_csv(os.path.join(output_dir, c+"_weekly.csv"))
        df_7ma = df_by_loc.rolling(7, min_periods=1).mean()
        df_7ma.interpolate(method="time",limit_direction="both").round(2).to_csv(os.path.join(output_dir, c+"_7ma.csv"))
    
    ## population-weighted average of weather data by hhs region
    hhs_regions = get_hhs_regions()
    fips = code_xwalk("name","fips")
    hhs_to_fips = {k:[fips[x] for x in hhs_regions[k] if x in fips] for k in hhs_regions}
    census = str_indexed_csv(os.path.join(output_dir,"fips_pops.csv")).iloc[:,0]
    census_by_hhs = str_indexed_csv(os.path.join(output_dir,"hhs_pops.csv")).iloc[:,0]

    f = "dewpC_weekly" ## currently only using this one
    df = pd.read_csv(os.path.join(output_dir, f+".csv"),index_col=0)
    df_WT = df.apply(lambda s: s * census[s.name])
    df_by_hhs = pd.DataFrame({k:df_WT[hhs_to_fips[k]].sum(axis=1)/census_by_hhs[k] for k in hhs_to_fips}, index=df.index)
    df_by_hhs.round(2).to_csv(os.path.join(output_dir, f+"_by_hhs.csv"))

    return None


## saves target data to training_data folder
def read_old_daily_data(file_loc):

    df = pd.read_csv(file_loc,dtype={"location":str})
    df = df[(df.location.str.len() == 2)]
    df.date = pd.to_datetime(df.date)
    df_by_loc = df.pivot(columns="location",values="value",index="date")
    df_by_loc = df_by_loc.drop(columns=["60","78","US"])

    data_start = pd.to_datetime("2020-07-14")
    data_end = df_by_loc.index[-1]

    os.makedirs("storage/training_data",exist_ok=True)

    ## if using log-transformed data, should probably do that before moving average
    df_log = df_by_loc.apply(lambda s: np.log(s + 1.0))
    df_log.loc[data_start:,:].interpolate(method="time",limit_direction="forward").fillna(0.0).round(6).to_csv("storage/training_data/h_log_unsmoothed.csv")
    ## 1 week non-centered moving average of log(daily) -- each day is mean of previous week
    df_log_7ma = df_log.rolling(7, min_periods=1).mean()
    df_log_7ma.loc[data_start:,:].interpolate(method="time",limit_direction="forward").fillna(0.0).round(6).to_csv("storage/training_data/h_log_7ma.csv")
    ## 3 day centered MA of log -- intended to smoothe out reporting errors without losing too much variance
    df_log_3ma = df_log.rolling(3, center=True, min_periods=1).mean()
    df_log_3ma.loc[data_start:,:].interpolate(method="time",limit_direction="forward").fillna(0.0).round(6).to_csv("storage/training_data/h_log_3ma.csv")

    ## 7 day non-centered ma -- each day is mean of previous week
    df_7ma = df_by_loc.rolling(7, min_periods=1).mean()
    df_7ma.loc[data_start:,:].interpolate(method="time",limit_direction="forward").fillna(0.0).round(2).to_csv("storage/training_data/h_7ma.csv")
    ## 3 day centered MA -- intended to smoothe out reporting errors without losing too much variance
    df_3ma = df_by_loc.rolling(3, center=True, min_periods=1).mean()
    df_3ma.loc[data_start:,:].interpolate(method="time",limit_direction="forward").fillna(0.0).round(2).to_csv("storage/training_data/h_3ma.csv")

    return (data_start, data_end)



## forecast hub targets (weekly hospitalizations)
## from https://data.cdc.gov/Public-Health-Surveillance/Weekly-Hospital-Respiratory-Data-HRD-Metrics-by-Ju/ua7e-t2fy/about_data
def download_forecast_hub(dest):

    u = "https://data.cdc.gov/resource/ua7e-t2fy.csv?$limit=500000"
    response = requests.get(u)

    with open(os.path.join(dest,"NHSN_hosp.csv"), "wb") as file:
        file.write(response.content)


## weekly covid lab surveil by hhs
## https://data.cdc.gov/Laboratory-Surveillance/Percent-Positivity-of-COVID-19-Nucleic-Acid-Amplif/gvsb-yw6g/about_data
def download_covid_surveil(dest):
    u = "https://data.cdc.gov/resource/gvsb-yw6g.csv?$limit=5000000"
    response = requests.get(u)

    with open(os.path.join(dest,"NREVSS_covid.csv"), "wb") as file:
        file.write(response.content)


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

    df = read_noaa_dir(year, stations, code_xwalk("abbr","fips"), code_xwalk("abbr","name"), local_archive)
    df.to_csv(os.path.join(dest,"weather"+str(year)+".csv"),index=False)







def download_training_data_old():
    f = "https://media.githubusercontent.com/media/reichlab/covid19-forecast-hub/master/data-truth/truth-Incident%20Hospitalizations.csv"
    (h_data_start, h_data_end) = read_old_daily_data(f)
    read_weather_data()
    #read_travel_data()
    return None


def fips_names():
    return code_xwalk("name","fips")

def read_travel_data():

    ##
    ## TODO: scrape the TSA page
    ##  https://www.tsa.gov/travel/passenger-volumes
    ##
    tsa_daily = pd.read_csv("storage/other/tsa.csv",index_col=0)

    tsa_daily.index = pd.to_datetime(tsa_daily.index)
    template = pd.read_csv("storage/training_data/h_7ma.csv",index_col=0)
    template.index = pd.to_datetime(template.index)

    ## don't have travel data by state, so scale national data by state pop
    df = pd.read_csv("storage/other/apportionment.csv",thousands=",")
    df = df.loc[(df.Year==2020),['Name','Resident Population']]
    fips = fips_names()

    df["fips"] = df["Name"].map(lambda x: fips.get(x,""))
    df["population"] = df['Resident Population']
    df = df.loc[df.fips != "",:]
    df = df[["fips","population"]].set_index("fips").sort_index()

    tsa_by_pop = template.apply(lambda s: tsa_daily.loc[template.index,"Numbers"] * df.loc[s.name,"population"].item() / df["population"].sum())
    ## use raw daily instead of moving average; hopefully the model can detect spikes
    tsa_by_pop.round(2).to_csv("storage/training_data/tsa_by_pop_daily.csv")
    ## for weekly, use each week's max
    #weekly_reduce(tsa_by_pop, np.nanmax).round(2).to_csv("storage/training_data/tsa_by_pop_weekly.csv")




def read_pop_density():
    df = pd.read_csv("storage/other/apportionment.csv",thousands=",")
    df = df.loc[(df.Year==2020),['Name','Resident Population Density']]
    fips = fips_names()
    df["fips"] = df["Name"].map(lambda x: fips.get(x,""))
    df["density"] = df['Resident Population Density']
    df = df.loc[df.fips != "",:]
    df[["fips","density"]].sort_values("fips").to_csv("storage/training_data/pop_density_2020.csv",index=False)

def read_median_age():
    df = pd.read_csv("storage/other/ACSST1Y2022.S0101-2023-12-18T014857.csv")
    df = df.iloc[34,1:].reset_index(name="median_age")
    df["labels"] = df["index"].str.split("!!")
    df["loc"] = df.labels.map(lambda x:x[0])
    df["istotal"] = df.labels.map(lambda x:x[1]=="Total")
    df["isest"] = df.labels.map(lambda x:x[2]=="Estimate")
    df = df.loc[(df.istotal & df.isest),:].copy()
    fips = fips_names()
    df["fips"] = df["loc"].map(lambda x: fips.get(x,""))
    df.loc[(df.fips != ""),["fips","median_age"]].sort_values("fips").to_csv("storage/training_data/med_age_2023.csv",index=False)

def read_vaccine_hist():
    df = pd.read_csv("storage/other/us_state_vaccinations.csv")
    df.loc[df["location"]=='New York State',"location"] = 'New York'
    fips = fips_names()
    df["fips"] = df["location"].map(lambda x: fips.get(x,""))
    df = df.loc[df.fips != "",:].copy()
    df["date"] = pd.to_datetime(df["date"])
    df_by_loc = df.pivot(columns="fips",index="date",values="people_fully_vaccinated_per_hundred")
    df_int = df_by_loc.interpolate(method="time")
    df_int = df_int.fillna(0.0)
    df_int.to_csv("storage/training_data/vacc_full_pct_to_may23.csv",float_format="%g")


def read_wastewater_data():
    ##
    ## TODO: automate download from https://www.cdc.gov/nwss/rv/COVID19-nationaltrend.html
    ##
    df = pd.read_csv("storage/other/wastewater.csv")
    df = df.loc[df["date_period"]=="All Results", :]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date",verify_integrity=True).sort_index()
    ## for each US state, pull data from the corresponding column of wastewater.csv
    fips = fips_names()
    d = tryJSON("storage/other/wastewater.json")
    state_to_code = {x:k for k in d for x in d[k]}
    fips_to_code = {fips[k]:state_to_code[k] for k in fips}
    ## result is df with column = fips and row = weekly timestamp
    df_fips_weekly = pd.DataFrame({k:df.loc[:,fips_to_code[k]] for k in fips_to_code})
    ## interpolate for daily
    idx_daily = pd.date_range(df_fips_weekly.index[0], df_fips_weekly.index[-1],freq="D")
    df_fips_daily = pd.DataFrame(index=idx_daily).join(df_fips_weekly).interpolate()
    df_fips_weekly.to_csv("storage/training_data/wastewater_weekly.csv")
    df_fips_daily.to_csv("storage/training_data/wastewater_daily.csv")


def read_variant_proportions():
    ## TODO: automate download from
    ## https://data.cdc.gov/Laboratory-Surveillance/SARS-CoV-2-Variant-Proportions/jr58-6ysp/about_data
    ##
    df = pd.read_csv("storage/other/vars_p.csv")
    vars_ord = ['other', 'alpha', 'beta', 'delta', 'gamma', 'oba', 'obq1', 'oxbb',
                'och11', 'oeg51', 'ohk3', 'ojn', 'okp3']
    df = df.pivot(columns="variant",index="date",values="p").fillna(0.0).loc[:,vars_ord]
    df.index = pd.to_datetime(df.index)

    ## interpolate for daily
    daily_idx = pd.date_range(df.index[0],df.index[-1])
    dfi = pd.DataFrame(index=daily_idx).join(df).interpolate(method="pchip")

    ## lumped into 5
    dfi[['alpha']].round(6).to_csv("storage/training_data/variant_alpha.csv")
    dfi[['delta']].round(6).to_csv("storage/training_data/variant_delta.csv")
    dfi[['gamma']].round(6).to_csv("storage/training_data/variant_gamma.csv")
    pd.DataFrame(dfi[['other','beta']].sum(axis=1).rename("other")).round(6).to_csv("storage/training_data/variant_other.csv")
    pd.DataFrame(dfi[['oba', 'obq1', 'oxbb', 'och11', 'oeg51', 'ohk3', 'ojn', 'okp3']].sum(axis=1).rename("omicron")).round(6).to_csv("storage/training_data/variant_omicron.csv")

    ## n highest at each timepoint
    ## "most recent 3" won't work; another way to do this?
    n = 3
    largestn = pd.DataFrame(
        dfi.apply(lambda r: [np.round(x,4) for x in r if x > 0.0001] ,axis=1) \
        .map(lambda v: np.pad(v,(0,max(n-len(v),0)))) \
        .map(lambda x: np.array(x)[(np.argsort(x)[-n:])]) \
        .rename("p"))

    for i in range(n):
        largestn["variant_p"+str(n-i)] = largestn["p"].map(lambda v: v[i])
        largestn[["variant_p"+str(n-i)]].to_csv("storage/training_data/"+"variant_p"+str(n-i)+".csv")

 