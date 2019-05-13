import trainer, eval_model

if __name__ == '__main__':
	model, params = trainer.train('aol')
	ep_hh, ep_not_hh, k_hh, k_not_hh = eval_model.evaluate_model(model, 'aol')
	write_string = "aol, %04f, %04f, %04f, %04f" % (ep_hh, ep_not_hh, k_hh, k_not_hh)
	with open('regression_results.csv', 'a') as f:
		f.write(write_string)