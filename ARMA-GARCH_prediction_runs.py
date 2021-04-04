### Quantiacs Trend Following Trading System Example
# import necessary Packages below:
from datetime import date
import math
import numpy as np
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from random import random
from warnings import filterwarnings
import quantiacsToolbox

beginInSample = '20190101'
endInSample = '20201231'

#import data for original test data
#Settings for market and importing data
market = ['F_ES']
data = quantiacsToolbox.loadData(marketList = market, dataToLoad = ['DATE', 'CLOSE'], beginInSample = beginInSample, endInSample = endInSample)

#Creating our dataframe
market_closing_prices = data['CLOSE']
time = data['DATE']

def parse_time(time_array):
    new_time_array = []
    for date_object in time_array:
        parsed_date = date(year=int(str(date_object)[0:4]), month=int(str(date_object)[4:6]), day=int(str(date_object)[6:8]))
        new_time_array.append(parsed_date)
    return new_time_array
time = parse_time(time)
information_df = pd.Series(market_closing_prices.flatten(), index = time)

def arma_garch(data):
    p,q,m,r = 4,3,2,2
    M = max(m,r)
    mu = 11.73781
    ar1 = 0.341187
    ar2 = 1.329728
    ar3 = 0.1526004
    ar4 = -0.8227023
    ma1 = 0.5542155
    ma2 = -0.6992497
    ma3 = -0.8270069
    omega = 8.944490*(10**(-6))
    alpha1 = 0.2609367
    alpha2 = 0.3550827
    beta1 = 3.610682*(10**(-8))
    beta2 = 0.3829806

    # initialize ground truth values to zero
    last_p_x = [0] * p
    last_q_z = [0] * q
    last_m_z2 = [0] * M
    last_r_w = [0] * r

    predictions_x = [0] * (len(data)+1)
    predictions_z2 = [0] * (len(data)+1)

    for i in range(len(data)+1):
        zhat2 = omega + last_m_z2[M-1]*(alpha1 + beta1) + last_m_z2[M-2]*(alpha2 + beta2)\
                + last_r_w[r-1]*beta1 + last_r_w[r-2]*beta2
        predictions_z2[i] = zhat2

        xhat = mu + last_p_x[p-1]*ar1 + last_p_x[p-2]*ar2 + last_p_x[p-3]*ar3 + last_p_x[p-4]*ar4\
                + last_q_z[q-1]*ma1 + last_q_z[q-2]*ma2 + last_q_z[q-3]*ma3
        predictions_x[i] = xhat

        # update ground truth values to use for prediction
        if i < len(data):
            x = data[i] 
            z = x - xhat
            z2 = z**2
            w = z2 - zhat2

            last_p_x = last_p_x[1:] + [x]
            last_q_z = last_q_z[1:] + [z]
            last_m_z2 = last_m_z2[1:] + [z2]
            last_r_w = last_r_w[1:] + [w]

    prediction = predictions_x[len(data)]
    return (prediction)

def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    ''' This system uses trend following techniques to allocate capital into the desired equities'''
    nMarkets=CLOSE.shape[1]

    # importing cached information from above
    global information_df

    # make predictions on all timesteps before current
    predicted = arma_garch(information_df)

    # update cached information
    new_time = parse_time([DATE[-1],])
    new_data = pd.Series(CLOSE[0], index = new_time)
    information_df = information_df.append(new_data)
    x1 = information_df[-1]

    pos=np.zeros(nMarkets)

    # check if prediction is larger than or smaller than ground truth at previous time step
    # long if prediction is larger
    # short if prediction is smaller
    if (predicted > x1):
        pos[0] = (predicted - x1)/x1
    elif (predicted < x1):
        pos[0] = (predicted - x1)/x1

    weights = pos/np.nansum(abs(pos))
    return weights, settings


def mySettings():
    ''' Define your trading system settings here '''

    settings= {}
    settings['markets'] = ['F_ES']
    settings['beginInSample'] = '20190101'
    settings['endInSample'] = '20201231'
    settings['lookback']= 5
    settings['budget']= 10**6
    settings['slippage']= 0.05

    return settings
# Evaluate trading system defined in current file.
if __name__ == '__main__':
    results = quantiacsToolbox.runts(__file__)
