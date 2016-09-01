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
	
	def __str__(self):
		kl = self.get_longest_keylength()
		print("".join(["{0:<%d}"%kl,"\t{1:}"]).format("feature","weight"))
		for k, v in sorted(self.feature_dict.items(), key=lambda x: x[1], reverse=True):
			print("".join(['{0:<%d}'%kl,'\t{1:}']).format(k,v))


class MarkovFeatures(BasicFeatureSet):

	def __init__(self, sent, word_idx):
		BasicFeatureSet.__init__(self)
		self.sent = sent
		self.word_idx = word_idx
		if self.word_idx == 0:
			self.prev_word = self.prev_pos = self.prev_lemma = self.prev_entity = self.prev_event = "<s>"
		else:
			self.prev_word = self.sent["words"][word_idx-1]
			self.prev_lemma = self.sent["lemmas"][word_idx-1]
			self.prev_pos = self.sent["POSs"][word_idx-1]
			self.prev_entity = self.sent["entities"][word_idx-1]
			self.prev_event = self.sent["events"][word_idx-1]

	def generate(self):

		self.feature_dict["->".join([self.prev_word, self.sent["words"][word_idx]])] += 1 
