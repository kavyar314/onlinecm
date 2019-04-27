import config

from keras.models import Sequential
from keras.layers import Dense

def model(n_layers=config.n_layers):
	model = Sequential()
	model.add(Dense(100, activation='relu', input_shape=(100,)))

	for _ in range(n_layers-1):
		model.add(Dense(100, activation='relu'))

	model.add(Dense(50, activation='relu'))
	model.add(Dense(10, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(loss='binary_crossentropy', optimizer='sgd')

	return model
