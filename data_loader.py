import config

import random
import numpy as np
import os

def load(dataset_name, amount=config.len_stream, start=0, verbose=False):
	if dataset_name == 'p_and_p':
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
		return words, embedded_words
	if dataset_name == 'aol':
		if verbose:
			print("processing aol dataset")
		with open(config.aol_file, 'r') as f:
		    x = f.readlines()
		queries = [lin.split('\t')[1] for lin in x]
		if verbose:
			print("taking subset")
		if amount is not None:
			used_queries = queries[start:amount]
		else:
			used_queries = queries
		return used_queries[1:], queries_to_vec(used_queries[1:])


def queries_to_vec(queries, trun_len=config.trun_len):
	words = []
	for q in queries:
		rep = [ord(c) for c in q]
		word_len = len(rep)
		if len(rep) < trun_len:
			rep = rep + [0 for _ in range(trun_len-len(rep))]
		else:
			rep = rep[:trun_len]
		words.append(sequence_ize(rep, trun_len))
	return words

def sequence_ize(vec, word_len):
	return np.array([vec[:i] + [0 for _ in range(len(vec) - i)] for i in range(1, word_len+1)])