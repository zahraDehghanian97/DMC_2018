import warnings
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

def find_min_pdq(train_data):
    q = d = range(0, 2)
    p = range(0, 4)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    print('Examples of parameter combinations for Seasonal ARIMA...')
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
    AIC = []
    SARIMAX_model = []
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(train_data,
                                                order=param,
                                                seasonal_order=param_seasonal)

                results = mod.fit()

                print('SARIMAX{}x{} - AIC:{}'.format(param, param_seasonal, results.aic), end='\r')
                AIC.append(results.aic)
                SARIMAX_model.append([param, param_seasonal])
            except:
                continue
    print('The smallest AIC is {} for model SARIMAX{}x{}'.format(min(AIC), SARIMAX_model[AIC.index(min(AIC))][0],SARIMAX_model[AIC.index(min(AIC))][1]))
    return [SARIMAX_model[AIC.index(min(AIC))][0],SARIMAX_model[AIC.index(min(AIC))][1]]


def predict(predict_date):
    data = pd.read_csv('ts.csv', engine='python', skipfooter=3)
    data['Month'] = pd.to_datetime(data['Month'], format='%m/%d/%Y')
    data.set_index(['Month'], inplace=True)
    train_data = data['2016-03-21':'2018-03-20']
    #test_data = data['2017-03-21':'2018-03-20']
    mod = sm.tsa.statespace.SARIMAX(train_data,
                                    order=(2,1,1),
                                    seasonal_order=(2,1,1,24))
    results = mod.fit()
    print(results.summary())
    pred2 = results.get_forecast(steps=predict_date)
    z=pred2.predicted_mean[0:]
    return round(z[-1])
    # ax = data.plot(figsize=(20, 16))
    # pred0.plot(ax=ax, label='1-step-ahead Forecast (get_predictions, dynamic=False)')
    # pred1.plot(ax=ax, label='Dynamic Forecast (get_predictions, dynamic=True)')
    # pred2.predicted_mean.plot(ax=ax, label='Dynamic Forecast (get_forecast)')
    # plt.ylabel('Monthly airline passengers (x1000)')
    # plt.xlabel('Date')
    # plt.legend()
    # plt.show()

print(predict('2019-03-20'))
