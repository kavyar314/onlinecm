import config


class lookup_table():
	def __init__(n_heavy_hitters=config.n_heavy_hitters, decay=config.decay):
		self.table = {}
		self.n_heavy_hitters = n_heavy_hitters
		self.decay = decay

	def contains(self, element):
		return element in self.table.keys()

	# def decay_n_heavy_hitters(self):
	#	self.n_heavy_hitters = int(self.decay * self.n_heavy_hitters)

	 def get_hh_thres(self, n, m):
		return int(self.n_heavy_hitters * self.decay**(n//m))

	def check_hh(self, element, n, m):
		if element in self.table.keys():
			# check more stuff
			thres_index = self.get_hh_thres(n, m)
			if len(list(self.table.keys())) < thres_index:
				return True
			elif self.table[element] >= self.table.values().sort[thres_index]:
				return True
			else:
				return False
		else:
			return False

	def increment_count(self, element):
		if element in self.table.keys():
			self.table[element] += 1
		else:
			self.add_element()

	def add_element(self, element):
		if element in self.table.keys():
			print("already present")
		else:
			self.table[element] = 1

	def sample_elements(self, hh, n, m):
		'''
		:param hh: boolean about whether to pick heavy hitters (True), non-heavy hitters (False), or sample uniformly from all elements(None)
		'''
		if hh is None:
			sample_list = self.table.keys()
			# stuff
		elif hh:
			sample_list = [x for x in self.table.keys() if self.check_hh(x, n, m)]
			# stuff
		elif not hh:
			sample_list = [x for x in self.table.keys() if not self.check_hh(x, n, m)]
			# stuff
		sampled_elements = random.choices()

