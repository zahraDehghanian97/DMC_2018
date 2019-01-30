import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# data = pd.read_csv("airline.csv")
dateparse = lambda dates: pd.datetime.strptime(dates, '%m/%d/%Y')
data = pd.read_csv('newairline.csv')#, parse_dates=['Log_Date'], index_col='Log_Date', date_parser=dateparse)
# print(data.head(10))

for i in range(75, 77):
    for j in range(75, 77):
        flag = False
        z = []

        # print(data.loc[(data['FROM']==i) & (data['TO']==j)])
        z.append(data.loc[(data['FROM'] == i) & (data['TO'] == j)])
        # if flag:
        #     print(row[1])
        #     if (row['FROM']==i and row['TO']==j):
        #         z.append(row)
        #     else:
        #         flag = True
        print(z)
        for v in z:
            if v.empty:
                print ("empty")
            else :
                for Num, row in v.iterrows():
                    print(matplotlib.dates.datestr2num(row["Log_Date"]))
                    plt.scatter(matplotlib.dates.datestr2num(row["Log_Date"]),int(row["count"]))
            #   z[k].loc[z['FROM']==i] &z[k].loc[z['TO']==j] )
                    plt.xlabel("date")
                    plt.ylabel("count")
                    
    #print(z)
plt.show()
# cnt = data["count"]
# print(cnt.head(10))
