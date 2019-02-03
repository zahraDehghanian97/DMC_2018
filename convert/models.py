import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA

import plotter

# AR MODEL
# model = ARIMA(plotter.ts_log, order=(2, 1, 0))
# results_AR = model.fit(disp=-1)
# plt.plot(plotter.ts_log_diff)
# plt.plot(results_AR.fittedvalues, color='red')
# plt.title('RSS: %.4f' % sum((results_AR.fittedvalues - plotter.ts_log_diff) ** 2))
# plt.show()


# MA MODEL
# model = ARIMA(plotter.ts_log, order=(0, 1, 2))
# results_MA = model.fit(disp=-1)
# plt.plot(plotter.ts_log_diff)
# plt.plot(results_MA.fittedvalues, color='red')
# plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-plotter.ts_log_diff)**2))
# plt.show()


# combined models
model = ARIMA(plotter.ts_log, order=(2, 1, 2))
results_ARIMA = model.fit(disp=-1)
plt.plot(plotter.ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f' % sum((results_ARIMA.fittedvalues - plotter.ts_log_diff) ** 2))
plt.show()

# Taking it back to original scale
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())

predictions_ARIMA_log = pd.Series(plotter.ts_log.ix[0], index=plotter.ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
predictions_ARIMA_log.head()

predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(plotter.ts)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f' % np.sqrt(sum((predictions_ARIMA - plotter.ts) ** 2) / len(plotter.ts)))
