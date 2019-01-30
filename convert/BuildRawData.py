import pandas as pd

store_data = pd.read_csv('data.csv', header=None)
pd.options.mode.chained_assignment = None
z = pd.DataFrame({'count': store_data.groupby([1, 3, 4]).size()}).reset_index()
print(z)
z.to_csv(r'airline.csv')