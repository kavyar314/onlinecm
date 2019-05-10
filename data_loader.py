import config

import random
import numpy as np
import os

def load(dataset_name='p_and_p', amount=config.len_stream, start=0, verbose=False):
	GLOVE_DIR = '.'
	embeddings_index = {}
	if verbose:
		print("generating embedding dictionary")
	with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
	    for line in f:
	        values = line.split()
	        word = values[0]
	        coefs = np.asarray(values[1:], dtype='float32')
	        embeddings_index[word] = coefs
	if verbose:
		print("embedding dictionary done; processing data")
	if dataset_name == 'p_and_p':
		text = open('P_and_P_plaintext.txt').read()
		words = [a.strip(',;.?,":') for a in text.split()]
		if amount is not None:
			used_words = words[start:amount]
		else:
			used_words = words
		embedded_words = []
		for word in used_words:
		    if word in embeddings_index.keys():
		        embedded_words.append(embeddings_index[word])
		del embeddings_index
		return embedded_words