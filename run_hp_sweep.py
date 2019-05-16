import trainer, eval_model, config

# n_heavy_hitters, time_between_train, f, n_gradient_updates
n_samples = [60]#[50, 60]
time_between_train = [200]#[100, 200, 500]
f_list = [0.99]#config.f_list
n_gradient_updates = [20]#[2, 4, 10, 20]
# n_layers = [4, 3]

outfile = "aol_parameter_sweep_rnn_log_longstream.csv"
# outfile

def hp_sweep():
	# for lay in n_layers:
	for h in n_samples[::-1]:
		for t in time_between_train[::-1]:
			for decay in f_list:
				for ep in n_gradient_updates:
					model, params = trainer.train('aol', n_samples=h, epochs=ep, n_heavy_hitters=h, decay=decay, n_before_update=t)#, n_layers=lay)
					print("completed:", params)
					[hh_within_e4, hh_within_e3, hh_within_e2],[not_hh_within_e4, not_hh_within_e3, not_hh_within_e2] = eval_model.evaluate_model(model, 'aol', eps_list=[0.1, 0.5, 1])
					write_string = "%d, %d, %02f, %d, %04f, %04f, %04f, %04f, %04f, %04f\n" % (h, t, decay, ep, hh_within_e4, not_hh_within_e4, hh_within_e3, not_hh_within_e3, hh_within_e2, not_hh_within_e2)
					with open(outfile, 'a') as f:
						f.write(write_string)


	# train(verbose=False, n_samples=config.half_batch, epochs=config.n_gradient_updates, 
	#		n_heavy_hitters=config.n_heavy_hitters, decay=config.decay, n_before_update=config.time_between_train)

if __name__ == '__main__':
	hp_sweep()