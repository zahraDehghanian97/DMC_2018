import Jalali as j
import pandas as pd

#data should be transofrm before running
def cast():
    z = pd.read_csv('test.csv', header=None)
    flag = False
    for Num, row in z.iterrows():
        if flag :
            row[0] = j.Persian(row[0]).gregorian_string("{}/{}/{}")
        else :
            flag = True
    print(z)
    z.to_csv(r'newtest.csv')