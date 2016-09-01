from collections import defaultdict

class BasicFeatureSet(object):

	def __init__(self):
		self.feature_dict = defaultdict(int)

	def __len__(self):
		return len(self.feature_dict)

	def get_longest_keylength(self):
		return len(max(self.feature_dict, key = len))

	def __getitem__(self, ki):
		return self.feature_dict[ki]

	def add_feature(self, name, weight):
		self.feature_dict[name] += weight

	def show(self):
		kl = self.get_longest_keylength()
		print("".join(["{0:<%d}"%kl,"\t{1:}"]).format("feature","weight"))
		for k, v in sorted(self.feature_dict.items(), key=lambda x: x[1], reverse=True):
			print("".join(['{0:<%d}'%kl,'\t{1:}']).format(k,v))



