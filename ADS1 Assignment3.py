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
    rawdata = pd.read_csv('World_Develoment_indicator_POP_GDP.csv')
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
    data_T["POP_exp"] = logistics(data_T["Years"], *popt)
    plt.figure()
    plt.plot(data_T["Years"], data_T["CHN"], label="CHN")
    plt.plot(data_T["Years"], data_T["POP_exp"], label="fit")
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
    
    
def clustering(indicator1,indicator2):
    data_POP, data_T_POPPOP=read_Data(indicator1)
    data_GDP, data_T_GDP=read_Data(indicator2)
    df_GDP_POP= pd.merge(data_GDP['2020 [YR2020]'], data_POP['2020 [YR2020]'], on='Country Code')
    df_GDP_POP=df_GDP_POP.rename(columns={'2020 [YR2020]_x': '2020_GDP','2020 [YR2020]_y':'2020_POP'})
    df_GDP_POP=df_GDP_POP.astype(float)
    print(df_GDP_POP)
        
    # normalise
    df_GDP_POP_fit = df_GDP_POP[["2020_GDP", "2020_POP"]].copy()
    df_GDP_POP_fit, df_min, df_max = ct.scaler(df_GDP_POP_fit)
    print(df_GDP_POP_fit.describe())
    print()
    print("n score")
    # loop over trial numbers of clusters calculating the silhouette
    for ic in range(2, 7):
        # set up kmeans and fit
        kmeans = cluster.KMeans(n_clusters=ic)
        kmeans.fit(df_GDP_POP_fit)
        # extract labels and calculate silhoutte score
        labels = kmeans.labels_
        print (ic, skmet.silhouette_score(df_GDP_POP_fit, labels))

    # Plot for four clusters
    nc = 4 # number of cluster centres
    kmeans = cluster.KMeans(n_clusters=nc)
    kmeans.fit(df_GDP_POP_fit)
    # extract labels and cluster centres
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_
    plt.figure(figsize=(6.0, 6.0))
    # scatter plot with colours selected using the cluster numbers
    plt.scatter(df_GDP_POP_fit["2020_GDP"], df_GDP_POP_fit["2020_POP"], c=labels, cmap="tab10")
    # plt.scatter(df_GDP_POP_fit["1990"], df_GDP_POP_fit["2015"], c=labels, cmap="tab10")
    # colour map Accent selected to increase contrast between colours
    # show cluster centres
    xc = cen[:,0]
    yc = cen[:,1]
    plt.scatter(xc, yc, c="k", marker="d", s=80)
    # c = colour, s = size
    plt.xlabel("2020_GDP")
    plt.ylabel("2020_POP")
    plt.title("GDP vs POP in 2015")
    plt.savefig('Data.jpg')
    plt.show()
     
    # move the cluster centres to the original scale
    cen = ct.backscale(cen, df_min, df_max)
    xcen = cen[:, 0]
    ycen = cen[:, 1]
    # cluster by cluster
    plt.figure(figsize=(8.0, 8.0))
    cm = plt.cm.get_cmap('tab10')
    plt.scatter(df_GDP_POP_fit["2020_GDP"], df_GDP_POP_fit["2020_POP"], 10, labels, marker="o",cmap=cm)
    plt.scatter(xcen, ycen, 45, "k", marker="d")
     
    plt.xlabel("2020_GDP")
    plt.ylabel("2020_POP")
    plt.savefig('Data.jpg')
    plt.show()

if __name__ == "__main__":
    #calling function to visualize all the plots
    data_Fitting_plot('GDP per capita (current US$)')
    data_prediction('GDP per capita (current US$)')
    data_ErrorRange_plot('GDP per capita (current US$)')
    data_Fitting_plot('Population, total')
    data_prediction('Population, total')
    data_ErrorRange_plot('Population, total')
    clustering('Population, total','GDP per capita (current US$)')
    