import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dateparse = lambda dates: pd.datetime.strptime(dates, '%m/%d/%Y')
data = pd.read_csv('newairline.csv', parse_dates=['Log_Date'], index_col='Log_Date', date_parser=dateparse)
# print(data)
#print('size:', data.size)

# test = data.loc['3/21/2017':]
# print(test.size)




# processed = data.columns.tolist()

# processed_1 = data['Log_Date']
# processed_2 = data['count']
# processed=pd.concat(processed_1,processed_2)
# processed=pd.DataFrame(processed)

# processed = pd.DataFrame(processed)
processed = data['count']

# pd.to_datetime(processed.astype('int64'))

# max_a = df.A.max()
# min_a = df.A.min()
# min_norm = -1
# max_norm =1
# processed = pd.to_datetime(processed).astype('int64')
# df['NORMA'] = (df.A- min_a) *(max_norm - min_norm) / (max_a-min_a) + min_norm
# print(datetime.fromtimestamp(processed))


# print('dates:\n', processed)

# # normalization
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
#
scaled = scaler.fit_transform(processed)
#

features_set = []
labels = []

#

#
for i in range(24, 567252):
    features_set.append(scaled[i - 24:i, 0])
labels.append(scaled[i, 0])

features_set, labels = np.array(features_set), np.array(labels)

features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))
# print(features_set)


from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.models import Sequential

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(features_set, labels, epochs=1, batch_size=32)

test = data.loc['3/21/2017':]
print(test.size)
testing_processed = test.loc[:, 1:2].values
print(testing_processed)

total = pd.concat((data['count'], test['count']), axis=0)
test_inputs = total[len(total) - len(test) - 24:].values

test_inputs = test_inputs.reshape(-1, 1)
test_inputs = scaler.transform(test_inputs)

test_features = []

for i in range(24,328268):
    # print(test_inputs[i - 60:i, 0])

    test_features.append(test_inputs[i - 24:i, 0])
#
test_features = np.array(test_features)
test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))
#
predictions = model.predict(test_features)
predictions = scaler.inverse_transform(predictions)
#
plt.figure(figsize=(10, 6))
plt.plot(testing_processed, color='blue', label='Actual count')
plt.plot(predictions, color='red', label='Predicted count')
plt.title(' Prediction')
plt.xlabel('Date')
plt.ylabel('count')
plt.legend()
plt.show()
plt.savefig('res.png')
