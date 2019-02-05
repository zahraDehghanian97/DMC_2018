import convert.Jalali as j
import pandas as pd

#data should be transofrm before running
z = pd.read_csv('airline.csv', header=None)
flag = False
for Num, row in z.iterrows():
    if flag :
        row[1] = j.Persian(row[1]).gregorian_string("{}/{}/{}")
    else :
        flag = True
print(z)
z.to_csv(r'newairline.csv')