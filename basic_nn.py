import config
import simple_rnn

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, RNN

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

def rnn_model(n_emb=64, n_hidden=64):
	rnn_cell = simple_rnn.SimpleRNNCell(n_emb, n_hidden)
	x = keras.Input((config.trun_len, config.trun_len))
	layer = RNN(rnn_cell)
	emb = layer(x)
	fc_1 = Dense(32, activation='relu')(emb)
	y = Dense(1, activation='sigmoid')(fc_1)

	model = Model(inputs=[x], outputs=[y])
	model.compile(loss='binary_crossentropy', optimizer='sgd')

	return model
	'''
	model = Sequential()
	model.add(RNN(rnn_cell, input_shape=(config.trun_len, config.trun_len)))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='mean_squared_error', optimizer='sgd')
	'''
