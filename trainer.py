import data_loader
import lookup_table_oracle as o
import learned_oracle as nn_model

from keras.models import Sequential
import numpy as np

def train(verbose=False):
	data_x, data_y = data_loader.load('dataset_name')
	if verbose:
		print("data loaded and has dimensions:\nx: ", data_x.shape, "y:", data_y.shape)
	if config.len_stream is not None:
		used_data = data_x[:config.len_stream], data_y[:config.len_stream]
	else:
		used_data = data_x, data_y
	oracle = o.lookup_table()
	nn = nn_model.model()
	j = 0
	for i in range(used_data[0].size[0]):
		if oracle.contains(d):
			oracle.increment_count(d)
		else:
			oracle.add_element(d)
		j = j+1
		if j % config.time_between_train == 0 and i != 0:
			# train
			#	collect data
				#	k positive examples (heavy hitters)
			positives, y_pos = oracle.sample_elements(hh=True, n_samples=config.half_batch)
				#	k randomly selected non-heavy hitters
			negatives, y_neg = oracle.sample_elements(hh=False, n_samples=config.half_batch)
			oracle.decay_n_heavy_hitters()
			full_training_x, full_training_y = np.hstack((positives, negatives)), np.stack((y_pos, y_neg))
			#	fit for n_gradient_updates epochs
			nn.fit(full_training_x, full_training_y, batch_size=full_training_x.shape[0], epochs=config.n_gradient_updates, verbose=2)
			# reset j
			j = 0

