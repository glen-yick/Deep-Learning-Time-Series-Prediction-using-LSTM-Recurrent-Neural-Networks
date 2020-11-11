import pandas as pd
import warnings
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels
import statsmodels.api as sm
import itertools
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARMAResults
# for LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential


# define a function to convert a vector of time series into a 2D matrix
def convertSeriesToMatrix(vectorSeries, sequence_length):
    matrix = []
    for i in range(len(vectorSeries)-sequence_length+1):
        matrix.append(vectorSeries[i:i+sequence_length])
    return matrix


# random seed
np.random.seed(1234)

vixdata = pd.read_csv("convertcsv.csv")
print(vixdata.columns)
filtered = vixdata[['Date', 'Spot price']]
print(filtered)
filtered.plot('Date', 'Spot price')
filtered.set_index('Date')

train_data = filtered[0:1939]
# test_data = filtered[1250:1700]

p = q = d = [0, 1, 2]
d = [0, 1]
combs = list(itertools.product(p, d, q))
train_data.set_index('Date')
# test_data.set_index('Date')


fig = plt.figure(figsize=(20, 8))
model = ARIMA(train_data['Spot price'], order=(1, 0, 1))
ax = plt.gca()
results = model.fit()
plt.plot(filtered['Spot price'].loc[0:1939])
plt.plot(results.fittedvalues, color='red')
ax.legend(['Real', 'Forecast'])

forecast = results.predict()
rmse = mean_squared_error(train_data['Spot price'], forecast) ** 0.5
plt.plot(forecast)
plt.show()

print(results.summary())
print(rmse)
df = pd.DataFrame((train_data['Spot price'] - forecast), columns=['error'])
df.to_csv("error_1_0_1.csv")
errordata = pd.read_csv("error_1_0_1.csv")
# CombinedCSV = pd.concat([errordata['error'], vixdata], axis=1)
# CombinedCSV.to_csv("merge.csv")

# LSTM part
# load the data
path_to_dataset = 'error_1_0_1.csv'
sequence_length = 20

# vector to store the time series
vector_vix = []
with open(path_to_dataset) as f:
    next(f) # skip the header row
    for line in f:
        fields = line.split(',')
        vector_vix.append(float(fields[1]))

# convert the vector to a 2D matrix
matrix_vix = convertSeriesToMatrix(vector_vix, sequence_length)

# shift all data by mean
matrix_vix = np.array(matrix_vix)
shifted_value = matrix_vix.mean()
matrix_vix -= shifted_value
print ("Data  shape: ", matrix_vix.shape)
print(matrix_vix.shape[0])
# split dataset: 90% for training and 10% for testing
train_row = int(round(0.9 * matrix_vix.shape[0]))
train_set = matrix_vix[:train_row, :]

# shuffle the training set (but do not shuffle the test set)
np.random.shuffle(train_set)
# the training set
X_train = train_set[:, :-1]
# the last column is the true value to compute the mean-squared-error loss
y_train = train_set[:, -1]
# the test set
X_test = matrix_vix[train_row:, :-1]
y_test = matrix_vix[train_row:, -1]

# the input to LSTM layer needs to have the shape of (number of samples, the dimension of each element)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# build the model
model = Sequential()
# layer 1: LSTM
model.add(LSTM(units=100, input_shape=(19, 1), return_sequences=True))
model.add(Dropout(0.2))
# layer 2: LSTM
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.2))
# layer 3: dense
# linear activation: a(x) = x
model.add(Dense(units=1))
model.add(Activation('linear'))
# compile the model
model.compile(loss="mse", optimizer="adam")
print(model.summary())


# train the model
model.fit(X_train, y_train, batch_size=512, epochs=50, validation_split=0.05, verbose=1)

# evaluate the result
# test_mse = model.evaluate(X_test, y_test, verbose=1)
# print ('\nThe mean squared error (MSE) on the test data set is %.3f over %d test samples.' % (test_mse, len(y_test)))

# get the predicted values
predicted_values = model.predict(X_test)
num_test_samples = len(predicted_values)
predicted_values = np.reshape(predicted_values, (num_test_samples,1))

# plot the results
fig = plt.figure()
plt.plot(y_test + shifted_value, label='real error')
plt.plot(predicted_values + shifted_value, label='predicted error')
plt.xlabel('Date')
plt.ylabel('error')
plt.legend()
plt.show()
fig.savefig('output_prediction.jpg', bbox_inches='tight')

# save the result into txt file
test_result = list(zip(predicted_values, y_test)) + shifted_value
np.savetxt('output_result.txt', test_result)

fig = plt.figure()
df = pd.DataFrame(forecast[1747:1939])
plt.plot(df + predicted_values + shifted_value, label='predicted VIX')
plt.plot(vixdata['Spot price'][1747:1939], label='real VIX')
plt.xlabel('Date')
plt.ylabel('VIX')
plt.legend()
plt.show()

rmse = mean_squared_error(pd.DataFrame(vixdata['Spot price'][1747:1939]), df + predicted_values + shifted_value) ** 0.5
print(rmse)
