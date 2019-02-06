import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf

import plotter

print("here ")
print(np.where(np.isnan(plotter.ts_log_diff)))
pd.DataFrame(plotter.ts_log_diff).nan

lag_acf = acf(plotter.ts_log_diff, nlags=20)
lag_pacf = pacf(plotter.ts_log_diff, nlags=20, method='ols')

# Plot ACF:
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96 / np.sqrt(len(plotter.ts_log_diff)), linestyle='--', color='gray')
plt.axhline(y=1.96 / np.sqrt(len(plotter.ts_log_diff)), linestyle='--', color='gray')
plt.title('Autocorrelation Function')
plt.show()

# Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96 / np.sqrt(len(plotter.ts_log_diff)), linestyle='--', color='gray')
plt.axhline(y=1.96 / np.sqrt(len(plotter.ts_log_diff)), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()
