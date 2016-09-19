from search import SearchAlgorithm
from collections import defaultdict

class Viterbi(SearchAlgorithm):

	def __init__(self,sentence, possible_labels, features_n_weights,iterations):
		
		SearchAlgorithm.__init__(self,sentence, possible_labels, features_n_weights)
		self.iterations = iterations
		self._sent_count = 0

	def _label_words(self):

		scores = []  # list of dicts like [{lab1:12, lab2:0,...}, {lab1:33,..}], i-th word corresp. i-th dict

		for i, w in enumerate(self.sentence["words"]):

			i += 2

			for label in possible_labels:
				
				score_this_word = defaultdict(int)
				score_this_word[label] = features_n_weights["event i-1 -> event i: [{}]->[{}]".format(self.sentence["events"][i-1],self.sentence["events"][i])]+\
				features_n_weights["word i + event i: [{}]+[{}]".format(self.sentence["words"][i],self.sentence["events"][i])]

			scores.append(score_this_word)



			
