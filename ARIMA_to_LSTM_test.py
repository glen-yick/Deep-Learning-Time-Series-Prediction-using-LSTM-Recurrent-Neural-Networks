import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
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
from sklearn.preprocessing import MinMaxScaler


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# load dataset
dataset = pd.read_csv('merge.csv', header=0, index_col=0)
dataset = dataset.drop(['Date', 'Date.1', 'Date.2', 'Date.3'], axis=1)
values = dataset.values
# integer encode direction
# encoder = LabelEncoder()
# values[:, 4] = encoder.fit_transform(values[:, 4])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
days = 20
features = 50
reframed = series_to_supervised(scaled, days, 1)
# drop columns we don't want to predict
# reframed.drop(reframed.columns.to_series()[51:100], axis=1, inplace=True)
print(reframed.shape)

# split into train and test sets
values = reframed.values
n_train_hours = 1744  # 0.9*1938
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
obs = days * features
train_X, train_y = train[:, :obs], train[:, -features]
test_X, test_y = test[:, :obs], test[:, -features]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], days, features))
test_X = test_X.reshape((test_X.shape[0], days, features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# build the model
model = Sequential()
# layer 1: LSTM
model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
# layer 2: LSTM
model.add(LSTM(256, return_sequences=False))
model.add(Dropout(0.2))
# layer 3: dense
# linear activation: a(x) = x
model.add(Dense(units=1))
model.add(Activation('linear'))
# compile the model
model.compile(loss="mse", optimizer="adam")
print(model.summary())

history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2)
# evaluate the result
test_mse = model.evaluate(test_X, test_y, verbose=1)
print ('\nThe mean squared error (MSE) on the test data set is %.3f over %d test samples.' % (test_mse, len(test_y)))

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], days*features))

yhat = np.concatenate((yhat, test_X[:, -(features-1):]), axis=1)
yhat = scaler.inverse_transform(yhat)
yhat = yhat[:, 0]

test_y = test_y.reshape((len(test_y), 1))
test_y = np.concatenate((test_y, test_X[:, -(features-1):]), axis=1)
test_y = scaler.inverse_transform(test_y)
test_y = test_y[:, 0]
# calculate RMSE
rmse = math.sqrt(mean_squared_error(test_y, yhat))
print('Test RMSE: %.3f' % rmse)
# print(yhat)
print(test_y)
plt.plot(test_y, label='real error')
plt.plot(yhat, label='predicted error')
plt.legend()
plt.show()
