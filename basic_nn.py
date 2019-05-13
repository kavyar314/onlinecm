import config

from keras.models import Sequential
from keras.layers import Dense, LSTM

def model(n_layers=config.n_layers):
	model = Sequential()
	model.add(Dense(100, activation='relu', input_shape=(100,)))

	for _ in range(n_layers-1):
		model.add(Dense(100, activation='relu'))

	model.add(Dense(50, activation='relu'))
	model.add(Dense(10, activation='linear'))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(loss='mean_squared_error', optimizer='sgd')

	return model


def lstm_model(n_units=512, n_layers=5):
	model = Sequential()
	model.add(LSTM(n_units, input_shape=(config.trun_len, config.trun_len)))

	for _ in range(n_layers-1):
		model.add(Dense(100, activation='relu'))

	model.add(Dense(50, activation='relu'))
	model.add(Dense(10, activation='linear'))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(loss='mean_squared_error', optimizer='sgd')

	return model