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



# class sFeatures(object):

# 	def __init__(self, sent, ftype='all'):
# 		self.sent = sent
# 		self.ftype = ftype
# 		self.sent_features = defaultdict(int)
		
# 	def extract(self):
# 		for i, word in enumerate(self.sent["words"]):
# 			word_features = wFeatures(self.sent, i)
# 			if self.ftype in ['all']:
# 				# transition features
# 				word_features.add("->".join(["[-words-]"+word_features.prev_word, self.sent["words"][i]]),1)
# 				word_features.add("->".join(["[-lemmas-]"+word_features.prev_lemma, self.sent["lemmas"][i]]),1)
# 				word_features.add("->".join(["[-POSs-]"+word_features.prev_pos, self.sent["POSs"][i]]),1)
# 				word_features.add("->".join(["[-entities-]"+word_features.prev_entity, self.sent["entities"][i]]),1)
# 				word_features.add("->".join(["[-events-]"+word_features.prev_event, self.sent["events"][i]]),1)
# 				# emission features
# 				word_features.add("+".join(["word+event#"+self.sent["words"][i], self.sent["events"][i]]),1)
# 		return word_features			




# def get_markov_features(sent, word_idx):

# 	f = wFeatures(sent, word_idx)

# 	f.add("->".join(["words#"+f.prev_word, sent["words"][word_idx]]),1)
# 	f.add("->".join(["lemmas#"+f.prev_lemma, sent["lemmas"][word_idx]]),1)
# 	f.add("->".join(["POSs#"+f.prev_pos, sent["POSs"][word_idx]]),1)
# 	f.add("->".join(["entities#"+f.prev_entity, sent["entities"][word_idx]]),1)
# 	f.add("->".join(["events#"+f.prev_event, sent["events"][word_idx]]),1)
# 	f.add("+".join(["word+event#"+sent["words"][word_idx], sent["events"][word_idx]]),1)

# 	return f

# def word_features(sent, word_idx):

# 	f = wFeatures(sent, word_idx)

# 	f.add("=".join(["word#", sent["words"][word_idx]]),1)
# 	f.add("=".join(["lemmas#", sent["lemmas"][word_idx]]),1)
# 	f.add("=".join(["POSs#", sent["POSs"][word_idx]]),1)
# 	f.add("=".join(["entities#", sent["entities"][word_idx]]),1)
# 	f.add("=".join(["events#", sent["events"][word_idx]]),1)

# 	g = wFeatures(sent, word_idx)
# 	g.add("index="+str(word_idx),1)

# 	return f+g

	# def get_features(self):

	# 	return self.feature_dict

# class wFeaturesExtractor(object):

# 	def __init__(self, sent, word_idx):
# 		self = BasicFeatureDict(sent)
# 		self.word_idx = word_idx
# 		if self.word_idx == 0:  # if this is the 1st word in sentence
# 			self.prev_word = self.prev_pos = self.prev_lemma = self.prev_entity = self.prev_event = "<s>"
# 		else:
# 			self.prev_word = self.sent["words"][self.__pidx]
# 			self.prev_lemma = self.sent["lemmas"][self.__pidx]
# 			self.prev_pos = self.sent["POSs"][self.__pidx]
# 			self.prev_entity = self.sent["entities"][self.__pidx]
# 			self.prev_event = self.sent["events"][self.__pidx]

	# def extract_emission_transition_features(self):

	# 	self.add("->".join([":words:"+self.prev_word, self.sent["words"][self.word_idx]]),1)
	# 	self.add("->".join([":lemmas:"+self.prev_lemma, self.sent["lemmas"][self.word_idx]]),1)

		# self.feature_dict["->".join([":words:"+self.prev_word, self.sent["words"][self.word_idx]])] += 1
		# self.feature_dict["->".join([":lemmas:"+self.prev_lemma, self.sent["lemmas"][self.word_idx]])] += 1
		# self.feature_dict["->".join([":POSs:"+self.prev_pos, self.sent["POSs"][self.word_idx]])] += 1
		# self.feature_dict["->".join([":entities:"+self.prev_entity, self.sent["entities"][self.word_idx]])] += 1
		# self.feature_dict["->".join([":events:"+self.prev_event, self.sent["events"][self.word_idx]])] += 1

		# self.feature_dict["+".join([":word+event:"+self.sent["words"][self.word_idx], self.sent["events"][self.word_idx]])] += 1



# class EmissionTransitionFeatures(BasicFeatureDict):

# 	def __init__(self, sent, word_idx):
# 		BasicFeatureDict.__init__(self, sent)	
# 		self.word_idx = word_idx
# 		if self.word_idx == 0:  # if this is the 1st word in sentence
# 			self.prev_word = self.prev_pos = self.prev_lemma = self.prev_entity = self.prev_event = "<s>"
# 		else:
# 			self.prev_word = self.sent["words"][self.__pidx]
# 			self.prev_lemma = self.sent["lemmas"][self.__pidx]
# 			self.prev_pos = self.sent["POSs"][self.__pidx]
# 			self.prev_entity = self.sent["entities"][self.__pidx]
# 			self.prev_event = self.sent["events"][self.__pidx]

# 	def generate(self):

# 		self.feature_dict["->".join([":words:"+self.prev_word, self.sent["words"][self.word_idx]])] += 1
# 		self.feature_dict["->".join([":lemmas:"+self.prev_lemma, self.sent["lemmas"][self.word_idx]])] += 1
# 		self.feature_dict["->".join([":POSs:"+self.prev_pos, self.sent["POSs"][self.word_idx]])] += 1
# 		self.feature_dict["->".join([":entities:"+self.prev_entity, self.sent["entities"][self.word_idx]])] += 1
# 		self.feature_dict["->".join([":events:"+self.prev_event, self.sent["events"][self.word_idx]])] += 1

# 		self.feature_dict["+".join([":word+event:"+self.sent["words"][self.word_idx], self.sent["events"][self.word_idx]])] += 1


s = wFeatures({"words":["dynamo", "floated", "in", "the", "sea."], "lemmas":["dynamo", "floated", "in", "the", "sea."],
"entities":["dynamo", "floated", "in", "the", "sea."], "events":["dynamo", "floated", "in", "the", "sea."],"POSs":["N", "fB", "V", "E", "P"]})
s.extract()

print(s)