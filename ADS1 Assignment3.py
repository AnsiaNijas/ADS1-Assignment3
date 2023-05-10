# -*- coding: utf-8 -*-
"""
Created on Tue May  9 19:05:51 2023

@author: ansia
"""

from sklearn.cluster import KMeans
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import scipy.optimize as opt
import errors as err
import cluster_tools as ct
import sklearn.cluster as cluster
import sklearn.metrics as skmet


def read_Data(indicator):
    """ 
    Function which takes an indicator name as argument, read the file into a dataframe and returns two dataframes: one with years as columns and one with
    countries as columns.
    """
    country_code = ['DZA','AUS','CHN','FRA','DEU','ISL','LUX','MEX','IND','USA']
    years = ['Country Name', '1960 [YR1960]', '1970 [YR1970]', '1980 [YR1980]', '1990 [YR1990]',
             '2000 [YR2000]', '2010 [YR2010]','2020 [YR2020]']
    yearsN=['1960','1970','1980','1990','2000','2010','2020']
    rawdata = pd.read_csv('World_Develoment_indicator_CO2_GDP.csv')
    rawdata = rawdata.set_index('Country Code')
    rawdata.replace("..", 0, inplace=True)
    data = rawdata[rawdata['Series Name'] ==
                   indicator].loc[:, years]
    data.iloc[:, 1:] = data.iloc[:, 1:].astype(float)
    data_T = data.transpose()
    data_T = data_T.drop('Country Name')
    data_T.reset_index(drop=False, inplace=True)
    data_T = data_T.rename(columns={'index': 'Years'})
    data_T['Years']=yearsN
    return data, data_T

def exp_growth(t, scale, growth):
    """ Computes exponential function with scale and growth as free parameters
    """
    f = scale * np.exp(growth * (t-1990))
    return f

def logistics(t, a, k, t0):
    """ Computes logistics function with scale and incr as free parameters
    """
    f = a / (1.0 + np.exp(-k * (t - t0)))
    return f


def data_Fitting_plot(indicator):
    data, data_T=read_Data(indicator) 
    data_T["Years"] = pd.to_numeric(data_T["Years"])
    popt, pcorr = opt.curve_fit(logistics, data_T["Years"], data_T["CHN"],
    p0=(1.2e8, 0.2, 2003.0))
    print("Fit parameter", popt)
    data_T["CO2_exp"] = logistics(data_T["Years"], *popt)
    plt.figure()
    plt.plot(data_T["Years"], data_T["CHN"], label="CHN")
    plt.plot(data_T["Years"], data_T["CO2_exp"], label="fit")
    plt.legend()
    plt.show()
    print("Population in")
    print("2030:", logistics(2030, *popt) / 1.0e6, "Mill.")
    print("2040:", logistics(2040, *popt) / 1.0e6, "Mill.")
    print("2050:", logistics(2050, *popt) / 1.0e6, "Mill.")
    
    
def data_prediction(indicator):
    data, data_T=read_Data(indicator) 
    data_T["Years"] = pd.to_numeric(data_T["Years"])
    popt, pcorr = opt.curve_fit(logistics, data_T["Years"], data_T["CHN"],
    p0=(1.2e8, 0.2, 2003.0))
    year = np.arange(1960, 2031)
    forecast = logistics(year, *popt)
    plt.figure()
    plt.plot(data_T["Years"], data_T["CHN"], label="CHINA")
    plt.plot(year, forecast, label="forecast")
    plt.xlabel("Year")
    plt.ylabel("GDP per capita (current US$)")
    plt.legend()
    plt.show()

    
def data_ErrorRange_plot(indicator):
    data, data_T=read_Data(indicator) 
    data_T["Years"] = pd.to_numeric(data_T["Years"])
    popt, pcorr = opt.curve_fit(logistics, data_T["Years"], data_T["CHN"],
    p0=(1.2e8, 0.2, 2003.0))
    print("Fit parameter", popt)
    # extract variances and calculate sigmas
    sigmas = np.sqrt(np.diag(pcorr))
    data_T["pop_logistics"] = logistics(data_T["Years"], *popt)
    # call function to calculate upper and lower limits with extrapolation
    # create extended year range
    years = np.arange(1960, 2030)
    lower, upper = err.err_ranges(years, logistics, popt, sigmas)
    plt.figure()
    plt.title("logistics function")
    plt.plot(data_T["Years"], data_T["CHN"], label="data")
    plt.plot(data_T["Years"], data_T["pop_logistics"], label="fit")
    # plot error ranges with transparency
    plt.fill_between(years, lower, upper, alpha=0.5)
    plt.legend(loc="upper left")
    plt.show()
    
    
if __name__ == "__main__":
    #calling function to visualize all the plots
    data_Fitting_plot('GDP per capita (current US$)')
    data_prediction('GDP per capita (current US$)')
    data_ErrorRange_plot('GDP per capita (current US$)')
    data_Fitting_plot('CO2 emissions (kt)')
    data_prediction('CO2 emissions (kt)')
    data_ErrorRange_plot('CO2 emissions (kt)')
    