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
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf
from random import random
from warnings import filterwarnings
import quantiacsToolbox

#beginInSample to endInSample is train data, beginInSampleAfter to endInSampleAfter is test data.
beginInSample = '20190101'
endInSample = '20201231'
beginInSampleAfter = '20210101' 
endInSampleAfter = '20210331'
#import data for original test data
#Settings for market and importing data
market = ['F_ES']
data = quantiacsToolbox.loadData(marketList = market, dataToLoad = ['DATE', 'CLOSE'], beginInSample = beginInSample ,endInSample = endInSample)

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
    model = HWES(information_df, seasonal_periods=12, trend='add', seasonal='mul')
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

    # S&P 100 stocks
    # settings['markets']=['CASH','AAPL','ABBV','ABT','ACN','AEP','AIG','ALL',
    # 'AMGN','AMZN','APA','APC','AXP','BA','BAC','BAX','BK','BMY','BRKB','C',
    # 'CAT','CL','CMCSA','COF','COP','COST','CSCO','CVS','CVX','DD','DIS','DOW',
    # 'DVN','EBAY','EMC','EMR','EXC','F','FB','FCX','FDX','FOXA','GD','GE',
    # 'GILD','GM','GOOGL','GS','HAL','HD','HON','HPQ','IBM','INTC','JNJ','JPM',
    # 'KO','LLY','LMT','LOW','MA','MCD','MDLZ','MDT','MET','MMM','MO','MON',
    # 'MRK','MS','MSFT','NKE','NOV','NSC','ORCL','OXY','PEP','PFE','PG','PM',
    # 'QCOM','RTN','SBUX','SLB','SO','SPG','T','TGT','TWX','TXN','UNH','UNP',
    # 'UPS','USB','UTX','V','VZ','WAG','WFC','WMT','XOM']

    # Futures Contracts
    settings['markets'] = ['F_ES']
    # settings['markets'] = ['CASH', 'F_AD','F_BO','F_BP','F_C','F_CC','F_CD','F_CL','F_CT','F_DX',
    # 'F_EC','F_ED','F_ES','F_FC','F_FV','F_GC','F_HG','F_HO','F_JY','F_KC','F_LB',
    # 'F_LC','F_LN','F_MD','F_MP','F_NG','F_NQ','F_NR','F_O','F_OJ','F_PA','F_PL',
    # 'F_RB','F_RU','F_S','F_SB','F_SF','F_SI','F_SM','F_TU','F_TY','F_US','F_W',
    # 'F_XX','F_YM','F_AX','F_CA','F_DT','F_UB','F_UZ','F_GS','F_LX','F_SS','F_DL',
    # 'F_ZQ','F_VX','F_AE','F_BG','F_BC','F_LU','F_DM','F_AH','F_CF','F_DZ','F_FB',
    # 'F_FL','F_FM','F_FP','F_FY','F_GX','F_HP','F_LR','F_LQ','F_ND','F_NY','F_PQ',
    # 'F_RR','F_RF','F_RP','F_RY','F_SH','F_SX','F_TR','F_EB','F_VF','F_VT','F_VW',
    # 'F_GD','F_F']
    # settings['markets'] = ['CASH','F_AD', 'F_BO', 'F_BP', 'F_C', 'F_CC', 'F_CD',
    # 'F_CL', 'F_CT', 'F_DX', 'F_EC', 'F_ED', 'F_ES', 'F_FC','F_FV', 'F_GC',
    # 'F_HG', 'F_HO', 'F_JY', 'F_KC', 'F_LB', 'F_LC', 'F_LN', 'F_MD', 'F_MP',
    # 'F_NG', 'F_NQ', 'F_NR', 'F_O', 'F_OJ', 'F_PA', 'F_PL', 'F_RB', 'F_RU',
    # 'F_S','F_SB', 'F_SF', 'F_SI', 'F_SM', 'F_TU', 'F_TY', 'F_US','F_W', 'F_XX',
    # 'F_YM']
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