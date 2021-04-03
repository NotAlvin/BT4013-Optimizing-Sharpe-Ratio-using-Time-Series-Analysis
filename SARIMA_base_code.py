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

#Settings for market and importing data
market = ['F_ES']
data = quantiacsToolbox.loadData(marketList = market, dataToLoad = ['DATE', 'CLOSE'], beginInSample = '20190101', endInSample = '20201231')

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
#information_df = information_df.asfreq(pd.infer_freq(information_df.index))

#plotting our dataframe
#plt.plot(information_df)
#plt.show()

#differencing to remove trend
first_diff = information_df.diff()[1:]
#plt.plot(first_diff)
#plt.show()

#checking acf graph
acf_vals = acf(first_diff)
num_lags = 20
#plt.bar(range(num_lags),acf_vals[:num_lags])
#plt.show()

#checking pacf graph
pacf_vals = pacf(first_diff)
num_lags = 15
#plt.bar(range(num_lags),pacf_vals[:num_lags])
#plt.show()

#fitting sarima model
order = (4,1,3)
seasonal_order = (0,0,0,0)

#splitting into train and test to verify model
train_end = date(2020,9,1)
test_end = date(2020,12,31)

train_data = information_df[:train_end]
test_data = information_df[train_end:test_end]

model = SARIMAX(train_data, order = order, seasonal_order = seasonal_order)
model_fit = model.fit()

#print(model_fit.summary())

#predicting to verify
predictions = list(model_fit.forecast(len(test_data)))
predictions = pd.Series(predictions, index = test_data.index)
print(predictions)
residuals = test_data - predictions
#plt.plot(residuals)
#plt.show()

#Comparison of predictions and actual values
plt.plot(information_df)
plt.plot(predictions)
plt.legend(('Data', 'Predictions'), fontsize=16)
plt.show()
