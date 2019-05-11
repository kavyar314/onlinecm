import data_loader
import lookup_table_oracle as o
import basic_nn as nn_model
import config

from keras.models import Sequential
import numpy as np

def train(dataset, verbose=False, n_samples=config.half_batch, epochs=config.n_gradient_updates, 
			n_heavy_hitters=config.n_heavy_hitters, decay=config.decay, n_before_update=config.time_between_train):
	params = {"n_samples": n_samples, "epochs": epochs, "n_heavy_hitters": n_heavy_hitters, "decay": decay, "n_before_update": n_before_update}
	if verbose:
		print("entered training module")
	raw, data_x = data_loader.load(dataset, verbose=verbose, amount=config.len_stream)
	raw_to_vec = dict(zip(raw, data_x))
	if verbose:
		print("data loaded and has length:", len(data_x), "with each part being: ", data_x[0].shape)
	oracle = o.lookup_table(n_heavy_hitters, decay)
	if dataset == 'aol':
		nn = nn_model.lstm_model()
	else:
		nn = nn_model.model()
	j = 0
	for i in range(len(data_x)):
		oracle.increment_count(raw[i])
		j = j+1
		if j % n_before_update == 0 and i != 0:
			print(i, j)
			# train
			#	collect data
				#	k positive examples (heavy hitters)
			positives, y_pos = oracle.sample_elements(hh=True, n_samples=n_samples)
				#	k randomly selected non-heavy hitters
			actual_n_samples = len(positives)
			if verbose: print(actual_n_samples)
			negatives, y_neg = oracle.sample_elements(hh=False, n_samples=actual_n_samples)
			full_training_x = np.array([np.array(raw_to_vec[x]) for x in positives + negatives])
			full_training_y = np.array(y_pos + y_neg)
			print(full_training_x.shape)
			if i // n_before_update == 0:
				initial_loss = nn.evaluate(full_training_x, full_training_y)
				loss_thres = 0.5 * initial_loss
				print(initial_loss, loss_thres, i)
			if full_training_x.shape[0] > 0:
				loss = nn.evaluate(full_training_x, full_training_y)
				print(loss, loss_thres, i)
				if loss < loss_thres:
					oracle.decay_n_heavy_hitters()
				oracle.flush()
				#	fit for n_gradient_updates epochs
				print(full_training_x.shape)
				nn.fit(full_training_x, full_training_y, batch_size=full_training_x.shape[0], epochs=epochs, verbose=2)
			else:
				print("no more samples on which to train")
			# reset j
			j = 0
	return nn, params

