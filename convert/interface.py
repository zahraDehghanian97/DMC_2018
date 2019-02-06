import testConvertor
import pandas as pd
import DataFinder as df
import arima
#data should be transofrm before running


testConvertor.cast()
z = pd.read_csv('newtest.csv', header=None)
flag = False
flag2 = False
for index,row in z.iterrows() :
    if flag and flag2 :
        print("here")
        print(row)
        print("here")
        origin = row[2]
        dest = row[3]
        date = row[1]
        returnable = df.data_finder(origin,dest)
        ts = returnable["count"]
        ts.to_csv('tsnew.csv')
        row[4]= arima.predict(date)
    else:
        if flag :
            flag2= True
        else:
            flag = True
