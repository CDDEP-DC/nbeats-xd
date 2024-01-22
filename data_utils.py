
import os
import io
import sys
import shutil
import datetime
import requests
from typing import Dict

import numpy as np
import pandas as pd


## turns a daily-indexed df into weekly, using reducing function fn
## preserves the end date, not the start date
def weekly_reduce(df, fn, s_date = "2020-07-14", e_date = None):
    d1 = pd.to_datetime(e_date) if e_date is not None else df.index[-1]
    d0 = d1 - pd.Timedelta(days=6)
    data = []
    idxs = []
    min_date = pd.to_datetime(s_date)
    while d0 >= min_date:
        data.append(df.loc[d0:d1,:].apply(fn))
        idxs.append(d1)
        d1 = d1 - pd.Timedelta(days=7)
        d0 = d0 - pd.Timedelta(days=7)
    data.reverse()
    idxs.reverse()
    df_weekly = pd.DataFrame(data)
    df_weekly.index = idxs
    return df_weekly


## saves target data to training_data folder
def read_target_data(file_loc):

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

    ## weekly mean; weeks divided so last week ends on last day of data
    ## interpolate missing data here so we have all daily values (assume 0 before data starts)
    df_by_loc = df_by_loc.interpolate(method="time",limit_direction="forward").fillna(0.0)
    weekly_mean = weekly_reduce(df_by_loc, np.mean).round(2)
    weekly_var = weekly_reduce(df_by_loc, np.var).round(2)
    weekly_log = weekly_reduce(df_by_loc.apply(lambda s: np.log(s + 1.0)), np.mean).round(6)
    ## to group each week's values (as a list) by that week's index:
    #daily_by_week = weekly_reduce(df_by_loc, (lambda x: {"item":x.values.round(2)} )).map(lambda x: list(x["item"]))

    weekly_mean.to_csv("storage/training_data/h_mean_weekly.csv")
    weekly_var.to_csv("storage/training_data/h_var_weekly.csv")
    weekly_log.to_csv("storage/training_data/h_log_weekly.csv")
    #daily_by_week.to_csv("storage/training_data/daily_vals_weekly.csv")

    return (data_start, data_end)


## reads specified year from specified weather stations, returns dataframe
def read_noaa_dir(year,stations,fips):
    weatherdata = None
    for k in stations:
        x = stations[k]
        url = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/" + str(year) + "/" + x + ".csv"
        s = requests.get(url).content
        ds = pd.read_csv(io.StringIO(s.decode('utf-8')),
                        usecols=['STATION', 'DATE', 'LATITUDE', 'LONGITUDE', 
                                'ELEVATION', 'NAME', 'TEMP', 'DEWP', 'SLP',
                                'STP', 'VISIB','WDSP', 'MXSPD', 'GUST',
                                'MAX', 'MIN', 'PRCP', 'SNDP', 'FRSHTT'],
                        dtype={"STATION":str,"FRSHTT":str})
        ds["state"] = k
        ds["fips"] = fips[k]
        weatherdata = pd.concat([weatherdata,ds],ignore_index=True)
    return weatherdata


## reads and processes weather data into local files
## uses saved local files for previous years, but always re-reads current year
## saves each variable to a file in training data folder
def read_weather_data(targ_data_start, targ_data_end):

    current_year = (datetime.date.today() - datetime.timedelta(days=3)).year ## data is published on a delay
    previous_years = range(2020,current_year)

    os.makedirs("storage/weather",exist_ok=True)

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

    fips = {'AL':'01', 'AK':'02', 'AZ':'04', 'AR':'05', 'CA':'06', 'CO':'08', 'CT':'09', 'DE':'10', 'DC':'11', 
            'FL':'12', 'GA':'13', 'HI':'15', 'ID':'16', 'IL':'17', 'IN':'18', 'IA':'19', 'KS':'20', 'KY':'21', 
            'LA':'22', 'ME':'23', 'MD':'24', 'MA':'25', 'MI':'26', 'MN':'27', 'MS':'28', 'MO':'29', 'MT':'30', 
            'NE':'31', 'NV':'32', 'NH':'33', 'NJ':'34', 'NM':'35', 'NY':'36', 'NC':'37', 'ND':'38', 'OH':'39', 
            'OK':'40', 'OR':'41', 'PA':'42', 'RI':'44', 'SC':'45', 'SD':'46', 'TN':'47', 'TX':'48', 'UT':'49', 
            'VT':'50', 'VA':'51', 'WA':'53', 'WV':'54', 'WI':'55', 'WY':'56', 'PR':'72'}

    weatherdata = None
    for year in previous_years:
        f = "storage/weather/weather" + str(year) + ".csv"
        if os.path.isfile(f):
            df = pd.read_csv(f,dtype={"STATION":str,"FRSHTT":str,"fips":str})
        else:
            print("downloading weather ",year)
            df = read_noaa_dir(year,stations,fips)
            df.to_csv(f,index=False)
        weatherdata = pd.concat([weatherdata,df],ignore_index=True)

    ## re-read current year weather every time
    print("downloading weather ",current_year)
    df = read_noaa_dir(current_year,stations,fips)
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
    weatherdata.to_csv('storage/weather/weatherdata.csv',index=False)

    weatherdata.DATE = pd.to_datetime(weatherdata.DATE)

    for c in ["tempC","dewpC","AH"]:
        df_by_loc = weatherdata.pivot(columns="fips",values=c,index="DATE")
        weekly_mean = weekly_reduce(df_by_loc, np.nanmean, targ_data_start, targ_data_end)
        weekly_mean.interpolate(method="time",limit_direction="both").round(2).to_csv("storage/training_data/"+c+"_weekly.csv")
        df_7ma = df_by_loc.rolling(7, min_periods=1).mean().loc[targ_data_start:targ_data_end,:]
        df_7ma.interpolate(method="time",limit_direction="both").round(2).to_csv("storage/training_data/"+c+"_7ma.csv")
    return None


def fips_names():
    return {'Alabama':'01', 'Alaska':'02', 'Arizona':'04', 'Arkansas':'05', 'California':'06', 'Colorado':'08', 
            'Connecticut':'09', 'Delaware':'10', 'District of Columbia':'11', 'Florida':'12', 'Georgia':'13', 
            'Hawaii':'15', 'Idaho':'16', 'Illinois':'17', 'Indiana':'18', 'Iowa':'19', 'Kansas':'20', 'Kentucky':'21', 
            'Louisiana':'22', 'Maine':'23', 'Maryland':'24', 'Massachusetts':'25', 'Michigan':'26', 'Minnesota':'27', 
            'Mississippi':'28', 'Missouri':'29', 'Montana':'30', 'Nebraska':'31', 'Nevada':'32', 'New Hampshire':'33', 
            'New Jersey':'34', 'New Mexico':'35', 'New York':'36', 'North Carolina':'37', 'North Dakota':'38', 
            'Ohio':'39', 'Oklahoma':'40', 'Oregon':'41', 'Pennsylvania':'42', 'Rhode Island':'44', 'South Carolina':'45', 
            'South Dakota':'46', 'Tennessee':'47', 'Texas':'48', 'Utah':'49', 'Vermont':'50', 'Virginia':'51', 
            'Washington':'53', 'West Virginia':'54', 'Wisconsin':'55', 'Wyoming':'56', 'Puerto Rico':'72'}

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
    weekly_reduce(tsa_by_pop, np.nanmax).round(2).to_csv("storage/training_data/tsa_by_pop_weekly.csv")


def download_training_data():
    f = "https://media.githubusercontent.com/media/reichlab/covid19-forecast-hub/master/data-truth/truth-Incident%20Hospitalizations.csv"
    (h_data_start, h_data_end) = read_target_data(f)
    read_weather_data(h_data_start,h_data_end)
    #read_travel_data()
    return None


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

