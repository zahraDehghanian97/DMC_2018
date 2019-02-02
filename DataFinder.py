import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


def data_finder(origin, destination):
    dateparse = lambda dates: pd.datetime.strptime(dates, '%m/%d/%Y')
    data = pd.read_csv('newairline.csv', parse_dates=['Log_Date'], index_col='Log_Date', date_parser=dateparse)
    z = []
    returnable = []
    z.append(data.loc[(data['FROM'] == origin) & (data['TO'] == destination)])
    for v in z:
        if v.empty:
            print("empty")
        else:
            # for Num, row in v.iterrows():
            #     returnable.append([row["count"]])
                #print(matplotlib.dates.datestr2num(row["Log_Date"]))
                #plt.scatter(matplotlib.dates.datestr2num(row["Log_Date"]), int(row["count"]))
                #plt.xlabel("date")
                #plt.ylabel("count")
            return v

    #plt.show()
    return z

