import trainer, eval_model, config

# n_heavy_hitters, time_between_train, f, n_gradient_updates
n_samples = [20, 40, 50, 60]
time_between_train = [100, 200, 500]
f_list = config.f_list
n_gradient_updates = [2, 4, 10, 20]
# n_layers = [4, 3]

outfile = "aol_parameter_sweep_classify.csv"
# outfile

def hp_sweep():
	# for lay in n_layers:
	for h in n_samples[::-1]:
		for t in time_between_train[::-1]:
			for decay in f_list:
				for ep in n_gradient_updates:
					model, params = trainer.train('aol', n_samples=h, epochs=ep, n_heavy_hitters=h, decay=decay, n_before_update=t)#, n_layers=lay)
					print("completed:", params)
					pp_n, ap_n, pn_n, an_n = eval_model.evaluate_model(model, 'aol')
					write_string = "%d, %d, %02f, %d, %04f, %04f\n" % (h, t, decay, ep, pp_n/ap_n, pn_n/an_n)
					with open(outfile, 'a') as f:
						f.write(write_string)


	# train(verbose=False, n_samples=config.half_batch, epochs=config.n_gradient_updates, 
	#		n_heavy_hitters=config.n_heavy_hitters, decay=config.decay, n_before_update=config.time_between_train)

if __name__ == '__main__':
	hp_sweep()