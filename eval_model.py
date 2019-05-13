import lookup_table_oracle
import data_loader
import config

import numpy as np

def evaluate_model(model, dataset, amount=4000, verbose=False):
	'''
	:param full: boolean. True if full stream, if False, then start evaluation from where training stream ended (config.len_stream)
	'''
	# make the lookup table
	table_for_eval, mapping = gen_lookup_table(dataset, amount=amount, verbose=verbose)
	hh_x, hh_y = table_for_eval.sample_elements(hh=True,n_samples=200)
	not_hh_x, not_hh_y = table_for_eval.sample_elements(hh=False,n_samples=20000)

	pred_y_hh = model.predict(np.array([list(mapping[hh_x[i]]) for i in range(len(hh_x))]))
	pred_y_not_hh = model.predict(np.array([list(mapping[not_hh_x[i]]) for i in range(len(not_hh_x))]))

	hh_within_ep = frac_within_epsilon(pred_y_hh, hh_y)
	not_hh_within_ep = frac_within_epsilon(pred_y_not_hh, not_hh_y)

	hh_k_hh = k_hh(pred_y_hh, hh_y, True)
	not_hh_k_hh = k_hh(pred_y_not_hh, not_hh_y, False)

	return hh_within_ep, not_hh_within_ep, hh_k_hh, not_hh_k_hh


def gen_lookup_table(dataset, amount, verbose):
	oracle_table = lookup_table_oracle.lookup_table(decay=1, reservoir=False) # will keep track of all the elements

	raw_data, data = data_loader.load(dataset, amount=amount, verbose=verbose)

	for d in raw_data:
		oracle_table.increment_count(d)
	return oracle_table, dict(zip(raw_data, data))
		
def frac_within_epsilon(pred_y, act_y, eps=0.001):
	return sum([(pred_y[p]-act_y[p]) <= eps for p in pred_y])/len(act_y)

def k_hh(pred_y, act_y, hh):
	one_over_k = sum(act_y)/len(act_y)
	if hh:
		return sum([p >= one_over_k for p in pred_y])/len(pred_y)
	else:
		return sum([p < one_over_k for p in pred_y])/len(pred_y)