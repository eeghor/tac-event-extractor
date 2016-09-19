"""
this is a base class for search algorithms like Viterbi or Beam search
"""
class SearchAlgorithm(object):

	START = ["START1","START2"]
	END = ["END1","END2"]

	def __init__(self, sentence, possible_labels, features_n_weights):

		# sentence is a dict like {"words": [word1, word2, ..], "events":[O,O,..]}
		# features_n_weights is a dict of features like {f1: weight1,f2: weight2,..}

		self.sentence = sentence
		self.labels = possible_labels
		self.assigned_labels = []  # the best sequence of labels to be found by algorithm
		self.features_n_weights = features_n_weights
		


