from collections import defaultdict

class wFeatures(object):

	def __init__(self, sent, word_idx=None):

		self.sent = sent  		  # sentence
		self.word_idx = word_idx  # word index: None to extract from all words, int index otherwise
		self.feature_dict = defaultdict(int)

	def add(self, name, weight):
		self.feature_dict[name] += weight

	def extract(self):

		if self.word_idx is None:
			wi = range(len(self.sent["words"]))
			print("wi=",wi)
		else:
			wi = range(self.word_idx,self.word_idx+1)

		for i in wi:

			self.__pidx = i - 1
			self.__nidx = i + 1

			if i == 0:  # if this is the 1st word in sentence
				self.prev_word = self.prev_pos = self.prev_lemma = self.prev_entity = self.prev_event = "<s>"
				self.prev2_word = self.prev2_pos = self.prev2_lemma = self.prev2_entity = self.prev2_event = None
			elif i == 1:
				self.prev_word = self.sent["words"][self.__pidx]
				self.prev_lemma = self.sent["lemmas"][self.__pidx]
				self.prev_pos = self.sent["POSs"][self.__pidx]
				self.prev_entity = self.sent["entities"][self.__pidx]
				self.prev_event = self.sent["events"][self.__pidx]
				self.prev2_word = self.prev2_pos = self.prev2_lemma = self.prev2_entity = self.prev2_event = "<s>"
			elif i == len(self.sent)-1:
				self.next_word = self.next_pos = self.next_lemma = self.next_entity = self.next_event = "<e>"
				self.next2_word = self.next2_pos = self.next2_lemma = self.next2_entity = self.next2_event = None
				self.prev_word = self.sent["words"][self.__pidx]
				self.prev_lemma = self.sent["lemmas"][self.__pidx]
				self.prev_pos = self.sent["POSs"][self.__pidx]
				self.prev_entity = self.sent["entities"][self.__pidx]
				self.prev_event = self.sent["events"][self.__pidx]
			elif i == len(self.sent)-2:
				self.next2_word = self.next2_pos = self.next2_lemma = self.next2_entity = self.next2_event = "<e>"
				self.next_word = self.sent["words"][self.__nidx]
				self.next_lemma = self.sent["lemmas"][self.__nidx]
				self.next_pos = self.sent["POSs"][self.__nidx]
				self.next_entity = self.sent["entities"][self.__nidx]
				self.next_event = self.sent["events"][self.__nidx]
				self.prev_word = self.sent["words"][self.__pidx]
				self.prev_lemma = self.sent["lemmas"][self.__pidx]
				self.prev_pos = self.sent["POSs"][self.__pidx]
				self.prev_entity = self.sent["entities"][self.__pidx]
				self.prev_event = self.sent["events"][self.__pidx]
			else:
				self.prev_word = self.sent["words"][self.__pidx]
				self.prev_lemma = self.sent["lemmas"][self.__pidx]
				self.prev_pos = self.sent["POSs"][self.__pidx]
				self.prev_entity = self.sent["entities"][self.__pidx]
				self.prev_event = self.sent["events"][self.__pidx]
				self.next_word = self.sent["words"][self.__nidx]
				self.next_lemma = self.sent["lemmas"][self.__nidx]
				self.next_pos = self.sent["POSs"][self.__nidx]
				self.next_entity = self.sent["entities"][self.__nidx]
				self.next_event = self.sent["events"][self.__nidx]

			self.feature_dict["->".join(["[-words-]"+self.prev_word, self.sent["words"][i]])] += 1
			self.feature_dict["->".join(["[-lemmas-]"+self.prev_lemma, self.sent["lemmas"][i]])] += 1
			self.feature_dict["->".join(["[-POSs-]"+self.prev_pos, self.sent["POSs"][i]])] += 1
			self.feature_dict["->".join(["[-entities-]"+self.prev_entity, self.sent["entities"][i]])] += 1
			self.feature_dict["->".join(["[-events-]"+self.prev_event, self.sent["events"][i]])] += 1
			# emission features
			self.feature_dict["+".join(["word+event#"+self.sent["words"][i], self.sent["events"][i]])] += 1

	def __len__(self):
		return len(self.feature_dict)
		
	def get_longest_keylength(self):
		return len(max(self.feature_dict, key = len))

	def __getitem__(self, ki):
		return self.feature_dict[ki]
	
	def __str__(self):
		kl = self.get_longest_keylength()
		st = "".join(["{0:<%d}"%kl,"\t{1:}"]).format("feature","weight")
		for k, v in sorted(self.feature_dict.items(), key=lambda x: x[1], reverse=True):
			st = "\n".join([st, "".join(['{0:<%d}'%kl,'\t{1:}']).format(k,v)])
		return st

	# def __add__(self, other):
	# 	return wFeatures(self.sent, self.word_idx, defaultdict(int,{**self.feature_dict, **other.feature_dict}))



s = wFeatures({"words":["dynamo", "floated", "in", "the", "sea."], "lemmas":["dynamo", "floated", "in", "the", "sea."],
"entities":["dynamo", "floated", "in", "the", "sea."], "events":["dynamo", "floated", "in", "the", "sea."],"POSs":["N", "fB", "V", "E", "P"]},2)
s.extract()

print(s)