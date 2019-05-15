import trainer, eval_model, config

# n_heavy_hitters, time_between_train, f, n_gradient_updates
n_samples = [20, 40, 50, 60]
time_between_train = [200]
f_list = config.f_list
n_gradient_updates = [1, 2, 3, 4, 10, 20]
n_layers = [4, 3]

outfile = "aol_parameter_sweep_morespecific.csv"
# outfile

def hp_sweep():
	for lay in n_layers:
		for h in n_samples[::-1]:
			for t in time_between_train[::-1]:
				for decay in f_list:
					for ep in n_gradient_updates:
						model, params = trainer.train('aol', n_samples=h, epochs=ep, n_heavy_hitters=h, decay=decay, n_before_update=t, n_layers=lay)
						print("completed:", params)
						[hh_within_e4, hh_within_e3, hh_within_e2],[not_hh_within_e4, not_hh_within_e3, not_hh_within_e2] = eval_model.evaluate_model(model, 'aol', eps_list=[0.0001, 0.001, 0.01])
						write_string = "%d, %d, %d, %02f, %d, %04f, %04f, %04f, %04f, %04f, %04f\n" % (lay, h, t, decay, ep, hh_within_e4, not_hh_within_e4, hh_within_e3, not_hh_within_e3, hh_within_e2, not_hh_within_e2)
						with open(outfile, 'a') as f:
							f.write(write_string)


	# train(verbose=False, n_samples=config.half_batch, epochs=config.n_gradient_updates, 
	#		n_heavy_hitters=config.n_heavy_hitters, decay=config.decay, n_before_update=config.time_between_train)

if __name__ == '__main__':
	hp_sweep()