### Quantiacs Trend Following Trading System Example
# import necessary Packages below:
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
import quantiacsToolbox
from warnings import filterwarnings

#beginInSample to endInSample is train data, beginInSampleAfter to endInSampleAfter is test data.
beginInSample = '20190101'
endInSample = '20201231'
beginInSampleAfter = '20210101' 
endInSampleAfter = '20210331'
#import data for original test data
#Settings for market and importing data
market = ['F_ES']
data = quantiacsToolbox.loadData(marketList = market, dataToLoad = ['DATE', 'CLOSE'], beginInSample = beginInSample, endInSample = endInSample)

#Creating our dataframe
market_closing_prices = data['CLOSE']
time = data['DATE']
order = (4,1,3)
seasonal_order = (0,0,0,0)


def parse_time(time_array):
    new_time_array = []
    for date_object in time_array:
        parsed_date = date(year=int(str(date_object)[0:4]), month=int(str(date_object)[4:6]), day=int(str(date_object)[6:8]))
        new_time_array.append(parsed_date)
    return new_time_array
time = parse_time(time)
information_df = pd.Series(market_closing_prices.flatten(), index = time)

def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    ''' This system uses trend following techniques to allocate capital into the desired equities'''
    
    nMarkets=CLOSE.shape[1]
    #add new data at every iteration to train a new model
    x_1 = CLOSE[-1].flatten()
    #print(x_1)
    #print(DATE[-1])
    new_time = parse_time([DATE[-1],])
    new_data = pd.Series(x_1, index = new_time)
    global information_df
    information_df = information_df.append(new_data)
    model = SARIMAX(information_df, order = order, seasonal_order = seasonal_order)
    model_fit = model.fit() 
    #predict new value
    pred = model_fit.forecast()
    pred = pred.values[0]
    pos=np.zeros(nMarkets)
    if (pred>x_1[0]):
        pos[0] =1
    elif (pred< x_1[0]):
        pos[0] =-1  
    weights = pos/np.nansum(abs(pos))
    return weights, settings


def mySettings():
    ''' Define your trading system settings here '''
    print("mySettings called")

    settings= {}
    settings['markets'] = ['F_ES']
    settings['beginInSample'] = beginInSampleAfter
    settings['endInSample'] = endInSampleAfter
    settings['lookback']= 3
    settings['budget']= 10**6
    settings['slippage']= 0.05

    return settings

# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox
    filterwarnings("ignore")
    results = quantiacsToolbox.runts(__file__)