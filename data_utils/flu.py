
import os
import io
import sys
import shutil
import datetime
import requests
from typing import Dict
from copy import deepcopy
import tarfile

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
            settings_list.append(x)
    return settings_list



## expects rstate to contain quantile forecasts returned by data_utils/forecast.py/generate_quantiles()
## expects the forecast dict to have an entry keyed "ensemble"
## "forecast_delay" is because the last day of available data is already in the past
def output_df(rstate, forecast_delay_days):
    qtiles = rstate.qtiles
    data_index = rstate.data_index
    data_columns = rstate.series_names
    fc_quantiles = rstate.fc_quantiles["ensemble"]
    us_quantiles = rstate.sum_quantiles["ensemble"]
    fc_mean = rstate.fc_mean["ensemble"]
    us_mean = rstate.sum_mean["ensemble"]

    time_freq = pd.infer_freq(data_index)

    ## forecast date: output file will contain forecast for this day forward; default = current local date
    ##
    ## NOTE: model generates a forecast starting with the day after the training data ends,
    ##   which may be in the past. But only forecast_date onward is written to the output file.
    train_end_date = pd.to_datetime(data_index[-1])

    if forecast_delay_days is not None:
        forecast_date = pd.to_datetime(train_end_date + pd.Timedelta(days=forecast_delay_days))
    else:
        forecast_date = pd.to_datetime(datetime.date.today()) 

    ## use ensembled forecasts; append US forecast derived above
    q_ensemble = np.concatenate([fc_quantiles, us_quantiles],axis=0)
    ## use mean as forecast point
    point_ensemble = np.concatenate([fc_mean, us_mean],axis=0)

    location_codes = data_columns.to_list() + ["US"] 
    quantile_labels = [f'{x:.3f}' for x in qtiles]
    date_indices = pd.date_range(start=train_end_date,inclusive="right",periods=1+q_ensemble.shape[1],freq=time_freq)

    dfs = []
    ## loop through each location in q_ensemble and make a dataframe with shape [date, value at each quantile]
    for i in range(q_ensemble.shape[0]):
        df = pd.DataFrame(q_ensemble[i,:,:], index=date_indices, columns=quantile_labels)
        df["mean"] = point_ensemble[i,:]
        dfs.append(df.loc[forecast_date:,:].melt(ignore_index=False,var_name="quantile").reset_index(names="target_end_date"))

    ## concatenate the location dataframes and set index to location code
    df_hub = pd.concat(dfs,keys=location_codes).droplevel(1).reset_index(names="location")

    ## add the rest of the columns required by forecast hub
    df_hub.loc[:,"type"] = "quantile"
    df_hub.loc[df_hub["quantile"]=="mean","type"] = "point"
    df_hub.loc[:,"forecast_date"] = forecast_date
    df_hub.loc[:,"target"] = df_hub.target_end_date.map(lambda d: str((d - forecast_date).days) + " day ahead inc hosp")
    df_hub.loc[:,"value"] = df_hub.loc[:,"value"].round(2)

    ## if using error dist allows negative values, set them to 0
    df_hub.loc[df_hub["value"]<0.0,"value"] = 0.0

    return df_hub, forecast_date

## append subset of output df to file used for plotting
def append_forecasts(df, filepath):

    df_plt = df.query("quantile=='0.025' or quantile=='0.975' or quantile=='0.500' or quantile=='mean'")[["forecast_date","location","quantile","target_end_date","value"]]
    df_plt.loc[df_plt["location"]=="US","location"] = "United States"

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


## preprocess downloaded flu data
## output csv's with columns = series and rows = dates
def read_flu_data():

    data_dir = "fluview"

    ## data series available in flusurv-net; used to pretrain models
    series_names = ['California',
                    'Colorado',
                    'Connecticut',
                    'Georgia',
                    'Maryland',
                    'Michigan',
                    'Minnesota',
                    #'New Mexico',  ## missing surveillance
                    'New York - Albany',
                    'New York - Rochester',
                    'Ohio',
                    'Oregon',
                    'Tennessee',
                    #'Utah'  ## missing surveillance
                    ]

    census = pd.read_csv(os.path.join(data_dir,"census2023.csv")).set_index("NAME")["POPESTIMATE2023"]
    ## assume rochester represents 25% of ny state
    census["New York - Rochester"] = census["New York"] * 0.25
    census["New York - Albany"] = census["New York"] * 0.75

    ##
    ## flusight forecast-hub targets (all US states from 2022-)
    ##
    df = pd.read_csv(os.path.join(data_dir,"flusight-hospital-admissions.csv"))
    ## per 100k capita
    flu_true = df.pivot(index="date",columns="location_name",values="weekly_rate")
    flu_true = flu_true.drop(columns=["US"])
    flu_true.index = pd.to_datetime(flu_true.index)
    ## counts
    flu_true_count = df.pivot(index="date",columns="location_name",values="value")
    flu_true_count = flu_true_count.drop(columns=["US"])
    flu_true_count.index = pd.to_datetime(flu_true_count.index)

    ## pairwise distances between series
    series_distances = dfDist(flu_true)

    ## don't allow zero (model can't handle an input window of all 0's; also it's probably not true)
    ## fill zero or missing counts with 0.5
    flu_true_count[flu_true_count < 0.5] = np.nan
    flu_true_count = flu_true_count.fillna(0.5)
    ## fill zero or missing per-capita with equivalent value
    small_values = census.map(lambda x: 50000.0 / x)
    flu_true[flu_true < 0.0001] = np.nan
    flu_true = flu_true.apply(lambda s: s.fillna(small_values[s.name]))

    ##
    ## flusurv-net hospitalization rates per 100k (participating regions from 2009-)
    ## from fluview: https://gis.cdc.gov/GRASP/Fluview/FluHospRates.html
    ##
    flu_hosp = pd.read_csv(os.path.join(data_dir,"flu_surv_net.csv"),dtype=str).query(
        "`AGE CATEGORY`=='Overall' and `SEX CATEGORY`=='Overall' and `RACE CATEGORY`=='Overall' and `VIRUS TYPE CATEGORY`=='Overall'")
    flu_hosp = flu_hosp[['CATCHMENT', 'YEAR.1', 'WEEK', 'WEEKLY RATE']].copy()
    flu_hosp["MMWR-YEAR"] = flu_hosp["YEAR.1"].astype(int)
    flu_hosp["MMWR-WEEK"] = flu_hosp["WEEK"].astype(int)
    flu_hosp["WEEKLY RATE"] = flu_hosp["WEEKLY RATE"].astype(float)
    flu_hosp["date"] = flu_hosp["MMWR-YEAR"].map(mmwr_start) + flu_hosp["MMWR-WEEK"].map(mmwr_delta)

    ## assume each flu-surv sampling area represents its state
    ## (produces an expected state-wide count, for possible use as forecast target)
    flu_hosp = flu_hosp.merge(census,left_on="CATCHMENT",right_index=True)
    flu_hosp["count"] = flu_hosp["WEEKLY RATE"] * flu_hosp["POPESTIMATE2023"] / 100000.0

    ##
    ## who/nrevss clinical lab surveillance data (all US states from 2015-)
    ## from fluview: https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html
    ##
    surveil = pd.read_csv(os.path.join(data_dir,"NREVSS_Clinical.csv"),usecols=['REGION', 'YEAR', 'WEEK', 'TOTAL SPECIMENS','PERCENT POSITIVE'],dtype=str)
    surveil["YEAR"] = surveil["YEAR"].astype(int)
    surveil["WEEK"] = surveil["WEEK"].astype(int)
    surveil["TOTAL SPECIMENS"] = pd.to_numeric(surveil["TOTAL SPECIMENS"],errors="coerce")
    surveil["PERCENT POSITIVE"] = pd.to_numeric(surveil["PERCENT POSITIVE"],errors="coerce")
    surveil["date"] = surveil["YEAR"].map(mmwr_start) + surveil["WEEK"].map(mmwr_delta)
    surveil = surveil.pivot(index="date",columns="REGION",values="PERCENT POSITIVE").sort_index()
    ## assume these are the same
    surveil['New York - Albany'] = surveil["New York"]
    surveil['New York - Rochester'] = surveil["New York"]
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
    outp = pd.read_csv(os.path.join(data_dir,"ILINet.csv"),usecols=['REGION', 'YEAR', 'WEEK', '%UNWEIGHTED ILI', 'ILITOTAL', 'TOTAL PATIENTS'],dtype=str)
    outp["YEAR"] = outp["YEAR"].astype(int)
    outp["WEEK"] = outp["WEEK"].astype(int)
    outp["%UNWEIGHTED ILI"] = pd.to_numeric(outp["%UNWEIGHTED ILI"],errors="coerce")
    outp["ILITOTAL"] = pd.to_numeric(outp["ILITOTAL"],errors="coerce")
    outp["TOTAL PATIENTS"] = pd.to_numeric(outp["TOTAL PATIENTS"],errors="coerce")
    outp["date"] = outp["YEAR"].map(mmwr_start) + outp["WEEK"].map(mmwr_delta)
    outp = outp.pivot(index="date",columns="REGION",values="%UNWEIGHTED ILI").sort_index()
    ## assume these are the same
    outp['New York - Albany'] = outp["New York"]
    outp['New York - Rochester'] = outp["New York"]
    ## keep rows/cols in lab surveil data
    outp = pd.DataFrame(index=data_index).join(outp).loc[:,data_cols]

    ## use flu-net data only from the time period when surveillance data is available
    ## per 100k capita
    flunet = pd.DataFrame(index=data_index).join(flu_hosp.pivot(index="date",columns="CATCHMENT",values="WEEKLY RATE")[series_names])
    ## expected state-wide counts
    flunet_count = pd.DataFrame(index=data_index).join(flu_hosp.pivot(index="date",columns="CATCHMENT",values="count")[series_names])

    ## don't allow zero (model can't handle an input window of all 0's; also it's probably not true)
    flunet[flunet < 0.05] = np.nan
    ## fill zero or missing per-capita with a small random number?
    #rng = np.random.default_rng()
    #small_rand = pd.DataFrame(0.02 + 0.02 * rng.random(flunet.shape) ,index=flunet.index, columns=flunet.columns)
    #flunet = flunet.fillna(small_rand)
    ## or just with 1/2 the smallest reported value?
    flunet = flunet.fillna(0.05)

    flunet_count[flunet_count < 0.5] = np.nan
    ## fill zero or missing counts with a small random number?
    #rng = np.random.default_rng()
    #small_rand = pd.DataFrame(0.5 + 0.2 * rng.random(flunet_count.shape) ,index=flunet_count.index, columns=flunet_count.columns)
    #flunet_count = flunet_count.fillna(small_rand)
    ## or just with 0.5?
    flunet_count = flunet_count.fillna(0.5)

    ## write csv's
    flunet.round(6).to_csv(os.path.join(data_dir,"flunet_samples_per100k.csv"))
    flunet_count.round(2).to_csv(os.path.join(data_dir,"flunet_samples_count.csv"))

    surveil.round(3).to_csv(os.path.join(data_dir,"flu_surveil_weekly.csv"))
    outp.round(6).to_csv(os.path.join(data_dir,"flu_outp_weekly.csv"))

    flu_true.round(6).to_csv(os.path.join(data_dir,"flusight_truth_per100k.csv"))
    flu_true_count.astype(float).round(2).to_csv(os.path.join(data_dir,"flusight_truth_count.csv"))

    ds = census[series_names]
    ds.to_csv(os.path.join(data_dir,"flunet_populations.csv"))
    ## to get national per-capita from regional per-capita forecasts
    pd.DataFrame((ds / ds.sum()).rename("weight")).round(6).to_csv(os.path.join(data_dir,"flunet_weights.csv"))
    ## for converting per 100k capita forecasts to statewide totals
    (census[flu_true.columns] / 100000.0).rename("weight").round(6).to_csv(os.path.join(data_dir,"per100k_to_state.csv"))

    return (data_index, data_cols)


## pop dens and med age; Z-scores across US states
def read_pop_data():

    data_dir = "fluview"

    series_names = pd.read_csv(os.path.join(data_dir,"flusight_truth_count.csv"),index_col=0).columns

    df = pd.read_csv(os.path.join(data_dir,"apportionment.csv"),thousands=",")
    df = df.loc[(df.Year==2020),['Name','Resident Population Density']].copy()
    df["density"] = df['Resident Population Density']
    ds = df.set_index("Name").loc[series_names,"density"]
    ## log Z normalize
    ds = ds.apply(np.log)
    ds = (ds - ds.mean()) / ds.std()
    ## ???
    ds["New York - Rochester"] = ds["New York"]
    ds["New York - Albany"] = ds["New York"]
    ds.round(6).to_csv(os.path.join(data_dir,"pop_density_2020.csv"))

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
    ds["New York - Rochester"] = ds["New York"]
    ds["New York - Albany"] = ds["New York"]
    ds.round(6).to_csv(os.path.join(data_dir,"med_age_2023.csv"))

    return None


## reads specified year from specified weather stations, returns dataframe
def read_noaa_dir(year,stations,local_archive=False):

    weatherdata = None
    usecols=['STATION', 'DATE', 'LATITUDE', 'LONGITUDE', 
                                'ELEVATION', 'NAME', 'TEMP', 'DEWP', 'SLP',
                                'STP', 'VISIB','WDSP', 'MXSPD', 'GUST',
                                'MAX', 'MIN', 'PRCP', 'SNDP', 'FRSHTT']
    dtype={"STATION":str,"FRSHTT":str}

    for k in stations:
        x = stations[k]
        if local_archive:
            with tarfile.open('fluview/weather/'+str(year)+'.tar.gz','r') as t:
                with t.extractfile(x+'.csv') as f:
                    ds = pd.read_csv(f,usecols=usecols,dtype=dtype)
        else:
            url = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/" + str(year) + "/" + x + ".csv"
            s = requests.get(url).content
            ds = pd.read_csv(io.StringIO(s.decode('utf-8')),usecols=usecols,dtype=dtype)
        ds["series_name"] = k
        weatherdata = pd.concat([weatherdata,ds],ignore_index=True)
    return weatherdata

## process file downloaded from bigquery
def read_bq_local(year,stations):
    df = pd.read_csv('fluview/weather/bq'+str(year)+'.zip',dtype={"stn":str,"wban":str,"date":str})
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

## reads and processes weather data into local files
## uses saved local files for previous years, but always re-reads current year
## saves each variable to a file in training data folder
def read_weather_data(date_idx,local_archive=False):

    data_dir = "fluview"

    current_year = (datetime.date.today() - datetime.timedelta(days=3)).year ## data is published on a delay
    previous_years = range(date_idx[0].year, current_year)

    os.makedirs(os.path.join(data_dir,"weather"),exist_ok=True)

    stations = {'Alabama': '72228013876',
                'Alaska': '70273026451',
                'Arizona': '72274023160',
                'Arkansas': '72340313963',
                'California': '72295023174',
                'Colorado': '72565003017',
                'Connecticut': '72508014740',
                'Delaware': '72418013781',
                'Florida': '72202012839',
                'Georgia': '72219013874',
                'Hawaii': '91182022521',
                'Idaho': '72681024131',
                'Illinois': '72530094846',
                'Indiana': '72438093819',
                'Iowa': '72546014933',
                'Kansas': '72450003928',
                'Kentucky': '72423093821',
                'Louisiana': '72231012916',
                'Maine': '72606014764',
                'Maryland': '72406093721',
                'Massachusetts': '72509014739',
                'Michigan': '72537094847',
                'Minnesota': '72658014922',
                'Mississippi': '72235003940',
                'Missouri': '72434013994',
                'Montana': '72677024033',
                'Nebraska': '72550014942',
                'Nevada': '72386023169',
                'New Hampshire': '72605014745',
                'New Jersey': '72502014734',
                'New Mexico': '72365023050',
                'New York': '74486094789',
                'North Carolina': '72306013722',
                'North Dakota': '72753014914',
                'Ohio': '72524014820',
                'Oklahoma': '72353013967',
                'Oregon': '72698024229',
                'Pennsylvania': '72408013739',
                'Rhode Island': '72507014765',
                'South Carolina': '72208013880',
                'South Dakota': '72651014944',
                'Tennessee': '72327013897',
                'Texas': '72243012960',
                'Utah': '72572024127',
                'Vermont': '72617014742',
                'Virginia': '72403093738',
                'Washington': '72793024233',
                'West Virginia': '72414013866',
                'Wisconsin': '72640014839',
                'Wyoming': '72564024018',
                'District of Columbia': '72405013743',
                'Puerto Rico': '78526011641',
                'New York - Albany': '72518014735',
                'New York - Rochester': '72529014768'}

    weatherdata = None
    for year in previous_years:
        f = os.path.join(data_dir, "weather", "weather" + str(year) + ".csv")
        if os.path.isfile(f):
            df = pd.read_csv(f,dtype={"STATION":str,"FRSHTT":str,"series_name":str})
        else:
            print("downloading weather ",year)
            df = read_noaa_dir(year,stations,local_archive)
            df.to_csv(f,index=False)
        weatherdata = pd.concat([weatherdata,df],ignore_index=True)

    ## re-read current year weather every time
    if local_archive:
        df = read_bq_local(current_year,stations)
    else:
        print("downloading weather ",current_year)
        df = read_noaa_dir(current_year,stations,local_archive)
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

    ## save a local snapshot
    weatherdata.to_csv(os.path.join(data_dir,"weather","weatherdata.csv"),index=False)

    weatherdata.DATE = pd.to_datetime(weatherdata.DATE)

    for c in ["tempC","dewpC","AH"]:
        df_by_loc = weatherdata.pivot(columns="series_name",values=c,index="DATE")
        weekly_mean = weekly_reduce(df_by_loc, np.nanmean, date_idx)
        weekly_mean.interpolate(method="time",limit_direction="both").round(2).to_csv(os.path.join(data_dir, c+"_weekly.csv"))
        df_7ma = df_by_loc.rolling(7, min_periods=1).mean().loc[date_idx[0]:date_idx[-1],:]
        df_7ma.interpolate(method="time",limit_direction="both").round(2).to_csv(os.path.join(data_dir, c+"_7ma.csv"))
    return None


