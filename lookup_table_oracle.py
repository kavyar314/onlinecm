import config
import random


class lookup_table():
	def __init__(self, n_heavy_hitters=config.n_heavy_hitters, decay=config.decay):
		self.table = {}
		self.n_heavy_hitters = n_heavy_hitters
		self.init_n_heavy_hitters = n_heavy_hitters
		self.decay = decay
		self.len_stream=0

	def contains(self, element):
		return element in self.table.keys()

	def decay_n_heavy_hitters(self):
		# this happens exactly when the model is trained
		# so we can also purge non-heavy hitters, I think
		if len(self.table.keys()) < self.init_n_heavy_hitters:
			print("haven't started decaying yet")
		else:
			self.n_heavy_hitters = int(self.decay * self.n_heavy_hitters)
			print("Now, there are %d heavy hitters" % self.n_heavy_hitters)

	def check_hh(self, element):
		'''
		note that, as implemented, this returns "None" if it's in No Man's Land between current HH threshold and original HH threshold
		'''
		decreasing_frequencies = sorted(list(self.table.values()))[::-1]
		if element in self.table.keys():
			# check more stuff
			if len(decreasing_frequencies) < self.n_heavy_hitters + 1:
				return True
			elif self.table[element] > decreasing_frequencies[self.n_heavy_hitters]:
				return True
			elif self.table[element] <= decreasing_frequencies[self.init_n_heavy_hitters]:
				return False
		else:
			return False

	def increment_count(self, element):
		if element in self.table.keys():
			self.table[element] += 1
		else:
			self.add_element()

	def add_element(self, element):
		self.len_stream += 1
		if element in self.table.keys():
			print("already present")
		else:
			light_el = [x for x in self.table.keys() if not self.check_hh(x) and self.check_hh(x) is not None]
			if len(light_el) < self.n_heavy_hitters:
				self.table[element] = 1
			elif random.random() < self.n_heavy_hitters/self.len_stream:
				remove = random.choice(light_el)
				# print("removing", remove, "to add", element)
				self.table.pop(remove)
				self.table[element] = 1
			else:
				pass

	def sample_elements(self, hh, n_samples):
		'''
		:param hh: boolean about whether to pick heavy hitters (True), non-heavy hitters (False), or sample uniformly from all elements(None)
		'''
		if hh is None:
			sample_list = list(self.table.keys())
			# stuff
		elif hh:
			sample_list = [x for x in self.table.keys() if self.check_hh(x)]
			# stuff
		elif not hh:
			sample_list = [x for x in self.table.keys() if not self.check_hh(x) and self.check_hh(x) is not None]
			# stuff
		num_samples = min([n_samples, len(sample_list)])
		sampled_elements = random.choices(sample_list, k=num_samples)
		labels = [1 if self.check_hh(x) else 0 for x in sampled_elements]
		return sampled_elements, labels

