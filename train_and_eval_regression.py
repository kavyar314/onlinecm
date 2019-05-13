import trainer, eval_model

if __name__ == '__main__':
	model, params = trainer.train('aol')
	pp_f, ap_f, pn_f, an_f = eval_model.evaluate_model(model, 'aol')
	write_string = "aol, %d, %d, %d, %d" % (pp_f, ap_f, pn_f, an_f)
	with open('regression_results.csv', 'a') as f:
		f.write(write_string)