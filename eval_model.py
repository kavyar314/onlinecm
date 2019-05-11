import lookup_table_oracle
import data_loader
import config

import numpy as np

def evaluate_model(model, dataset, full=True, verbose=False):
	'''
	:param full: boolean. True if full stream, if False, then start evaluation from where training stream ended (config.len_stream)
	'''
	# make the lookup table
	table_for_eval = gen_lookup_table(dataset, full=full, verbose=verbose)
	hh_x, _ = table_for_eval.sample_elements(hh=True,n_samples=200)
	not_hh_x, _ = table_for_eval.sample_elements(hh=False,n_samples=20000)

	pred_y_hh = model.predict(np.array([list(hh_x[i]) for i in range(len(hh_x))]))
	pred_y_not_hh = model.predict(np.array([list(not_hh_x[i]) for i in range(len(not_hh_x))]))

	correct_hh = sum([p >= 0.5 for p in pred_y_hh])
	correct_not_hh = sum([p < 0.5 for p in pred_y_not_hh])

	return correct_hh[0], len(pred_y_hh), correct_not_hh[0], len(pred_y_not_hh)


def gen_lookup_table(dataset, full, verbose):
	oracle_table = lookup_table_oracle.lookup_table(decay=1, reservoir=False) # will keep track of all the elements

	if full:
		raw_data, data = data_loader.load(dataset, amount=None, verbose=verbose)
	else:
		raw_data, data = data_loader.load(dataset, amount=-1, start=config.len_stream, verbose=verbose)

	for d in raw_data:
		oracle_table.increment_count(tuple(d))
	return oracle_table, dict(zip(raw_data, data))
		