### Quantiacs Trend Following Trading System Example
#Import necessary Packages below:
from datetime import date
import numpy as np
from scipy.signal import lfilter
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.tsa.holtwinters
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf
from random import random
from warnings import filterwarnings
import quantiacsToolbox

#Settings for chosen market and importing data
market = ['F_ES']
data = quantiacsToolbox.loadData(marketList = market, dataToLoad = ['DATE', 'CLOSE', 'USA_CPI'], beginInSample = '20190101', endInSample = '2020930')

#Creating our dataframe
market_closing_prices = data['CLOSE']
time = data['DATE']
exog = data['USA_CPI']

#Setting order from data exploration done in r
order = (4,1,3)
seasonal_order = (0,0,0,0)

#Function to parse time for assignment to series index
def parse_time(time_array):
    new_time_array = []
    for date_object in time_array:
        parsed_date = date(year=int(str(date_object)[0:4]), month=int(str(date_object)[4:6]), day=int(str(date_object)[6:8]))
        new_time_array.append(parsed_date)
    return new_time_array
time = parse_time(time)

#Storing our endogenous and exogenous data for use
information_df = pd.Series(market_closing_prices.flatten(), index = time)
exog_df = pd.Series(exog.flatten(), index = time)
information_df.index.freq = 'D'

#Trading algorithmn now that data preparation is ready
def myTradingSystem(DATE, CLOSE, USA_CPI, settings):
    ''' This system uses trend following techniques to allocate capital into the desired equities'''
    nMarkets=CLOSE.shape[1] #Number of markets we are looking at (should we use > 1)
    x_1 = CLOSE[-1].flatten() #Latest endogenous data
    z_1 = USA_CPI[-1].flatten() #Latest exogenous data

    #Parsing time and setting indexes to that both series are aligned
    new_time = parse_time([DATE[-1],])
    new_data = pd.Series(x_1, index = new_time)
    new_exog = pd.Series(z_1, index = new_time)
    new_data.index.freq = 'D'
    new_exog.index.freq = 'D'
    global information_df
    global exog_df

    #Code to add in latest data to our respective storages
    information_df = information_df.append(new_data)
    exog_df = exog_df.append(new_exog)
    #print(set(information_df.index) == set(exog_df.index)) #test if indexes are equal

    #Creating and fitting model
    model = SARIMAX(information_df, exog_df ,order = order, seasonal_order = seasonal_order)
    model_fit = model.fit()

    #Forecasting next predicted CLOSE value
    pred = model_fit.forecast(exog = exog_df[-1])
    pred = pred.values[0]

    #Trading decisions
    #If predicted CLOSE > current CLOSE, long
    #If predicted CLOSE < current CLOSE, short
    pos=np.zeros(nMarkets)
    if (pred>x_1[0]):
        pos[0] = 1
    elif (pred< x_1[0]):
        pos[0] = -1

    weights = pos/np.nansum(abs(pos))
    return weights, settings


def mySettings():
    settings= {}
    #Market used is E-mini for S&P 500
    settings['markets'] = ['F_ES']
    settings['beginInSample'] = '20201001'
    settings['endInSample'] = '20201231'
    settings['lookback']= 2
    settings['budget']= 10**6
    settings['slippage']= 0.05
    return settings

# Evaluate trading system defined in current file.
if __name__ == '__main__':
    filterwarnings("ignore")
    results = quantiacsToolbox.runts(__file__)
