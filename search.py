"""
this is a base class for search algorithms like Viterbi or Beam search;
"""
class SearchAlgorithm(object):

	def __init__(self, sentence, possible_labels, features_n_weights):

		# sentence is a dict like {"words": [word1, word2, ..], "events":[O,O,..]}
		# features_n_weights is a defaultdict of features like {f1: weight1,f2: weight2,..}

		self.sentence = sentence 
		self.labels = list(possible_labels)
		self.fw = features_n_weights
