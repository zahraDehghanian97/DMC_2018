import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


def data_finder(origin, destination):
    print(origin ,destination)
    dateparse = lambda dates: pd.datetime.strptime(dates, '%m/%d/%Y')
    data = pd.read_csv('newairline.csv', parse_dates=['Log_Date'], index_col='Log_Date', date_parser=dateparse)
    z = []
    returnable = []
    z.append(data.loc[(data['FROM'] == int(origin)) & (data['TO'] == int(destination))])
    for v in z:
        if v.empty:
            print("empty")
        else:
            return v

    #plt.show()
    return z

