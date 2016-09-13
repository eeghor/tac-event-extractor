from collections import defaultdict
from nltk.corpus import wordnet

"""
EXTRACT FEATURES FROM A WORD.
* can either extract all fetures or only Markov features needed for the Viterbi algorithm
"""

class wFeatures(object):

	def __init__(self, sent, what_features='all', word_idx=None, nomlex_dict=None):

		self.sent = sent  		  # sentence
		self.word_idx = word_idx  # word index: None to extract from all words, int index otherwise
		self.feature_dict = defaultdict(int)
		self.nomlex_dict = nomlex_dict
		self.what_features = what_features

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

			if self.what_features in ['all','markov']:
				self.feature_dict["word transition [{} -> {}]".format(self.prev_word, self.sent["words"][i])] += 1
				self.feature_dict["lemma transition [{} -> {}]".format(self.prev_lemma, self.sent["lemmas"][i])] += 1
				self.feature_dict["pos transition [{} -> {}]".format(self.prev_pos, self.sent["POSs"][i])] += 1
				self.feature_dict["entity transition [{} -> {}]".format(self.prev_entity, self.sent["entities"][i])] += 1
				self.feature_dict["event transition [{} -> {}]".format(self.prev_event, self.sent["events"][i])] += 1
				# emission features
				self.feature_dict["event word emission [{} -> {}]".format( self.sent["events"][i], self.sent["words"][i])] += 1

			if self.what_features in ['all']:
				# lemma synonyms and hypernyms from WordNet
				if wordnet.synsets(self.sent["lemmas"][i]):
				    for w in wordnet.synsets(self.sent["lemmas"][i].lower()):
				        for ln in w.lemma_names():
				            self.feature_dict["wordnet lemma synonym [{}]".format(ln)] += 1
				        for g in w.hypernyms():
				            for l in g.lemma_names():
				                self.feature_dict["wordnet lemma hypernym [{}]".format(l)] += 1		

				# sub-word features (applied to words)
				cc = 0
				for char in self.sent["words"][i]:
					cc += 1
					if cc == 1:
						self.feature_dict["first letter in word [{}]".format(self.sent["words"][i][0])] += 1
					if cc == 2:
						self.feature_dict["last 2 letters in word [{}]".format(self.sent["words"][i][:2])] += 1
					if cc == 3:
						self.feature_dict["last 3 letters in word [{}]".format(self.sent["words"][i][:3])] += 1

				self.feature_dict["word length [{}]".format(len(self.sent["words"][i]))] += 1

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
"entities":["dynamo", "floated", "in", "the", "sea."], "events":["O", "swim", "O", "O", "O"],"POSs":["N", "fB", "V", "E", "P"]},'markov',3)
s.extract()

print(s)