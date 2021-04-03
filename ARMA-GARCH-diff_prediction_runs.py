### Quantiacs Trend Following Trading System Example
# import necessary Packages below:
import numpy as np
import math
import pmdarima
import arch
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from matplotlib import pyplot
# from arch import arch_model
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
        # print(last_r_w)
        # print(last_m_z2)
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
            # print('last_r_w')
            # print(last_r_w)
            last_r_w = last_r_w[1:] + [w]
            # print('last_r_w')
            # print(last_r_w)

    prediction = predictions_x[len(data)] # - 25.696 + 0.00073
    
    return (prediction)

def arma_garch2(data):
    arima_model = pmdarima.auto_arima(data)
    p, d, q = arima_model.order
    arima_residuals = arima_model.arima_res_.resid

    # fit a GARCH(1,1) model on the residuals of the ARIMA model
    garch = arch.arch_model(arima_residuals, p=2, q=2)
    garch_model = garch.fit()

    # Use ARIMA to predict mu
    predicted_mu = arima_model.predict(n_periods=1)[0]
    # Use GARCH to predict residual
    garch_forecast = garch_model.forecast(horizon=1)
    predicted_et = garch_forecast.mean['h.1'].iloc[-1]
    # Combine both models' output: yt: mu + et
    prediction = predicted_mu + predicted_et
    return prediction

def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    ''' This system uses trend following techniques to allocate capital into the desired equities'''
    #print("myTradingSystem caleld")

    nMarkets=CLOSE.shape[1]
    # data = np.apply_along_axis(math.log, 0, CLOSE[:,0])
    
    data = CLOSE[:-1,0].tolist()
    data = [math.log(x) for x in data]
    # data = CLOSE[:,0].apply(math.log)
    predicted = arma_garch(data)

    log_x1 = data[-1]

    pos=np.zeros(nMarkets)
    f = open("storage.txt", 'a')
    f.write(str(predicted - log_x1) + "\n")
    f.close()
    if (predicted > log_x1):
        pos[0] = (predicted - log_x1)/log_x1
    elif (predicted < log_x1):
        pos[0] = (predicted - log_x1)/log_x1
    

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
    # settings['markets'] = ['CASH','F_AD', 'F_BO', 'F_BP', 'F_C', 'F_CC', 'F_CD',
    # 'F_CL', 'F_CT', 'F_DX', 'F_EC', 'F_ED', 'F_ES', 'F_FC','F_FV', 'F_GC',
    # 'F_HG', 'F_HO', 'F_JY', 'F_KC', 'F_LB', 'F_LC', 'F_LN', 'F_MD', 'F_MP',
    # 'F_NG', 'F_NQ', 'F_NR', 'F_O', 'F_OJ', 'F_PA', 'F_PL', 'F_RB', 'F_RU',
    # 'F_S','F_SB', 'F_SF', 'F_SI', 'F_SM', 'F_TU', 'F_TY', 'F_US','F_W', 'F_XX',
    # 'F_YM']
    settings['beginInSample'] = '20190101'
    settings['endInSample'] = '20201231'
    settings['lookback']= 5
    settings['budget']= 10**6
    settings['slippage']= 0.05

    return settings
# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox
    results = quantiacsToolbox.runts(__file__)
