import data_loader
import lookup_table_oracle as o
import basic_nn as nn_model
import config

from keras.models import Sequential
import numpy as np

def train(verbose=False, n_samples=config.half_batch, epochs=config.n_gradient_updates, 
			n_heavy_hitters=config.n_heavy_hitters, decay=config.decay, n_before_update=config.time_between_train):
	params = {"n_samples": n_samples, "epochs": epochs, "n_heavy_hitters": n_heavy_hitters, "decay": decay, "n_before_update": n_before_update}
	if verbose:
		print("entered training module")
	data_x = data_loader.load(verbose=verbose, amount=config.len_stream)
	if verbose:
		print("data loaded and has length:", len(data_x), "with each part being: ", data_x[0].shape)
	oracle = o.lookup_table(n_heavy_hitters, decay)
	nn = nn_model.model()
	j = 0
	for i in range(len(data_x)):
		d = data_x[i]
		oracle.increment_count(tuple(d))
		j = j+1
		if j % n_before_update == 0 and i != 0:
			# train
			#	collect data
				#	k positive examples (heavy hitters)
			positives, y_pos = oracle.sample_elements(hh=True, n_samples=n_samples)
				#	k randomly selected non-heavy hitters
			actual_n_samples = len(positives)
			if verbose: print(actual_n_samples)
			negatives, y_neg = oracle.sample_elements(hh=False, n_samples=actual_n_samples)
			oracle.decay_n_heavy_hitters()
			full_training_x = np.array([np.array(list(x)) for x in positives + negatives])
			full_training_y = np.array(y_pos + y_neg)
			#	fit for n_gradient_updates epochs
			if full_training_x.shape[0] > 0:
				nn.fit(full_training_x, full_training_y, batch_size=full_training_x.shape[0], epochs=epochs, verbose=2)
			else:
				print("no more samples on which to train")
			# reset j
			j = 0
	return nn, params

