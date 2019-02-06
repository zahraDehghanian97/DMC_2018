import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import plotter
from pyramid.arima import auto_arima
import plotly as pi
from plotly.plotly import plot_mpl
# # AR MODEL
# model = ARIMA(plotter.ts_log, order=(2, 1, 0))
# results_AR = model.fit(disp=-1)
# plt.plot(plotter.ts_log_diff)
# plt.plot(results_AR.fittedvalues, color='red')
# plt.title('AR RSS: %.4f' % sum((results_AR.fittedvalues - plotter.ts_log_diff) ** 2))
# plt.show()


stepwise_model = auto_arima(plotter.ts_log_ewma_diff, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                          start_P=0, seasonal=True,
                          d=1, D=1, trace=True,
                         error_action='ignore',
                        suppress_warnings=True,
                          stepwise=True)

train = plotter.ts_log_ewma_diff.loc['3/21/2016':'3/21/2017']
test= plotter.ts_log_ewma_diff.loc['3/21/2017':]

# print(test)
# print(train)
stepwise_model.fit(train)
future_forecast=stepwise_model.predict(n_periods=620)
future_forecast = pd.DataFrame(future_forecast,index=test.index ,columns=['count'])
pd.concat([test,future_forecast],axis=1).plot()
plt.show()
print('this is the auto arima')
print(stepwise_model.aic())
#
# # MA MODEL
# try:
#     model = ARIMA(plotter.ts_log, order=(0, 1, 2))
#     results_MA = model.fit(disp=-1)
#     plt.plot(plotter.ts_log_diff)
#     plt.plot(results_MA.fittedvalues, color='red')
#     plt.title('MA RSS: %.4f'% sum((results_MA.fittedvalues-plotter.ts_log_diff)**2))
#     plt.show()
# except:
#     pass
#
# # combined models
# try:
#     model = ARIMA(plotter.ts_log, order=(2, 1, 2))
#     results_ARIMA = model.fit(disp=-1)
#     plt.plot(plotter.ts_log_diff)
#     plt.plot(results_ARIMA.fittedvalues, color='red')
#     plt.title('Combined RSS: %.4f' % sum((results_ARIMA.fittedvalues - plotter.ts_log_diff) ** 2))
#     plt.show()
# except:
#     pass
# # Taking it back to original scale
# predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
# print(predictions_ARIMA_diff.head())
#
# predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
# print(predictions_ARIMA_diff_cumsum.head())
#
# predictions_ARIMA_log = pd.Series(plotter.ts_log.ix[0], index=plotter.ts_log.index)
# predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
# predictions_ARIMA_log.head()/
#
# predictions_ARIMA = np.exp(predictions_ARIMA_log)
# plt.plot(plotter.ts)
# plt.plot(predictions_ARIMA)
# plt.title('RMSE: %.4f' % np.sqrt(sum((predictions_ARIMA - plotter.ts) ** 2) / len(plotter.ts)))
# plt.show()