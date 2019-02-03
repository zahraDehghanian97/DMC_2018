import matplotlib.pyplot as plt

import DataFinder as df

returnable = df.data_finder(75, 70)
ts = returnable["count"]
print(ts)
###to show remove comment
###plt.plot(ts)
###plt.show()

###################testing stationary
from statsmodels.tsa.stattools import adfuller
import pandas as pd


def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()

    # Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    # to show remove comment
    plt.show()

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


###test_stationarity(ts)

#################################eliminating trend
import numpy as np

ts_log = np.log(ts)

########################logarithm
moving_avg = ts_log.rolling(12).mean()
###plt.plot(ts_log)
###plt.plot(moving_avg, color='red')
###plt.show()
ts_log_moving_avg_diff = ts_log - moving_avg
###print("raw data")
###print(ts_log_moving_avg_diff.head(12))
###ts_log_moving_avg_diff.dropna(inplace=True)
###print(ts_log_moving_avg_diff.head())
###test_stationarity(ts_log_moving_avg_diff)

####################exponential
expwighted_avg = ts_log.ewm(halflife=12).mean()
###plt.plot(ts_log)
###plt.plot(expwighted_avg, color='red')
###plt.show()
###ts_log_ewma_diff = ts_log - expwighted_avg
###test_stationarity(ts_log_ewma_diff)

#############################################Eliminating Trend and Seasonality

##########difrencing
ts_log_diff = ts_log - ts_log.shift()
###plt.plot(ts_log_diff)
###plt.show()
###ts_log_diff.dropna(inplace=True)
###test_stationarity(ts_log_diff)

##########Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(ts_log, model="additive", filt=None, freq=45)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
##yeshanbe


plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
# plt.show()
