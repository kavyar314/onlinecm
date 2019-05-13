import trainer, eval_model, config

# n_heavy_hitters, time_between_train, f, n_gradient_updates
n_heavy_hitters = [20, 40, 50, 60]
time_between_train = [10, 20, 50, 75, 100]
f_list = config.f_list
n_gradient_updates = [1, 2, 3, 4, 10]

outfile = "parameter_sweep.csv"
# outfile

def hp_sweep():
	for h in n_heavy_hitters:
		for t in time_between_train:
			for decay in f_list:
				for ep in n_gradient_updates:
					model, params = trainer.train(n_samples=h, epochs=ep, n_heavy_hitters=h, decay=decay, n_before_update=t)
					print("completed:", params)
					pp_f, ap_f, pn_f, an_f = eval_model.evaluate_model(model, full=True)
					pp_n, ap_n, pn_n, an_n = eval_model.evaluate_model(model, full=False)
					write_string = "%d, %d, %02f, %d, %04f, %04f, %04f, %04f\n" % (h, t, decay, ep, pp_f/ap_f, pn_f/an_f, pp_n/ap_n, pn_n/an_n)
					with open(outfile, 'a') as f:
						f.write(write_string)


	# train(verbose=False, n_samples=config.half_batch, epochs=config.n_gradient_updates, 
	#		n_heavy_hitters=config.n_heavy_hitters, decay=config.decay, n_before_update=config.time_between_train)

if __name__ == '__main__':
	hp_sweep()