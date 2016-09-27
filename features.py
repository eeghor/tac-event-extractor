from collections import defaultdict
from nltk.corpus import wordnet

"""
EXTRACT FEATURES FROM A WORD.
* can either extract all fetures or only Markov features needed for the Viterbi algorithm
"""

class FeatureFactory(object):

	def __init__(self, sent, word_idx, nomlex_dict=None):  # Nomlex dictionary is optional

		self.sentence = {k: ["START1","START2"] + v +["END1","END2"] for k,v, in sent.items()}   # sentence
		self.word_idx = word_idx  		
		self.nomlex_dict = nomlex_dict

		self.feature_dict = defaultdict(int)

	def add(self, name, weight):
		self.feature_dict[name] += weight

	def extract(self):  # extract features from a word

		i = self.word_idx + 2  # because now  we have START1, START2, word1, ..
		__pidx = i - 1  # previous index
		__nidx = i + 1  # next index

		# transition features
		self.add("(word-1):[{}]->(word)[{}]".format(self.sentence["words"][__pidx],self.sentence["words"][i]), 1)
		self.add("(lem-1):[{}]->(lem)[{}]".format(self.sentence["lemmas"][__pidx],self.sentence["lemmas"][i]), 1)
		self.add("(lem-2):[{}]->(lem-1)[{}]->(lem)[{}]".format(self.sentence["lemmas"][i-2], self.sentence["lemmas"][__pidx],self.sentence["lemmas"][i]), 1)
		self.add("(pos-1):[{}]->(pos)[{}]".format(self.sentence["POSs"][__pidx],self.sentence["POSs"][i]), 1)
		self.add("(ent-1):[{}]->(ent)[{}]".format(self.sentence["entities"][__pidx],self.sentence["entities"][i]), 1)
		self.add("(ev-1):[{}]->(ev)[{}]".format(self.sentence["events"][__pidx],self.sentence["events"][i]), 1)

		self.add("(word):[{}]->(word+1)[{}]".format(self.sentence["words"][i], self.sentence["words"][__nidx]), 1)
		self.add("(lem):[{}]->(lem+1)[{}]".format(self.sentence["lemmas"][i], self.sentence["lemmas"][__nidx]), 1)
		self.add("(pos):[{}]->(pos+1)[{}]".format(self.sentence["POSs"][i], self.sentence["POSs"][__nidx]), 1)
		self.add("(ent):[{}]->(ent+1)[{}]".format(self.sentence["entities"][i], self.sentence["entities"][__nidx]), 1)
		self.add("(ev):[{}]->(ev+1)[{}]".format(self.sentence["events"][i], self.sentence["events"][__nidx]), 1)

		# emission features: in HMM, likelihood that hidden event i generated the observed word i
		self.add("(ev):[{}]=>(word)[{}]".format(self.sentence["events"][i],self.sentence["words"][i]), 1)

		# lemma synonyms and hypernyms from WordNet
		if wordnet.synsets(self.sentence["lemmas"][i]):
		    for w in wordnet.synsets(self.sentence["lemmas"][i].lower()):
		        for ln in w.lemma_names():
		            self.add("(syn):[{}]".format(ln),1)
		        for g in w.hypernyms():
		            for l in g.lemma_names():
		            	self.add("(hyp):[{}]".format(l),1)	

		# last 2 letters in word
		if len(self.sentence["words"][i]) > 4:
			self.add("(word_end2))[{}]".format(self.sentence["words"][i][-2:]), 1)
		# is alphanumeric
		if ~self.sentence["words"][i].isalnum():
			self.add("(not_alphanumeric)", 1)
		# lemma
		self.add("(lemma)[{}]".format(self.sentence["lemmas"][i]), 1)
		# word length
		if len(self.sentence["words"][i]) < 3:
			self.add("(very_short_word)", 1)

		return self.feature_dict


s = FeatureFactory({"words":["dynamo", "floated", "in", "the", "sea."], "lemmas":["dynamo", "floated", "in", "the", "sea."],
"entities":["dynamo", "floated", "in", "the", "sea."], "events":["O", "swim", "O", "O", "O"],"POSs":["N", "fB", "V", "E", "P"]},0)
print(s.extract())
