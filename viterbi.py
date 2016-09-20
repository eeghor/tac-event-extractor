from search import SearchAlgorithm
from collections import defaultdict
import numpy as np

class Viterbi(SearchAlgorithm):

	def __init__(self,sentence, possible_labels, fw):
		
		SearchAlgorithm.__init__(self,sentence, possible_labels, fw)
		self._NW = len(sentence)
		self._NL = len(possible_labels)
		self._vit_matrix = np.zeros(shape=(self._NL,self._NW +2))
		self._backpointer_matrix = np.zeros(shape=(self._NL,self._NW+1))
		self.possible_labels = possible_labels

	def _run_viterbi_algorithm(self):

		# fill in the 1st column
		for j, label in enumerate(possible_labels):
			self._vit_matrix[j, 0] = fw["(ev-1):[{}]->(ev)[{}]".format("START",self.sentence["events"][j]+\
										fw["(ev):[{}]=>(word)[{}]".format(self.sentence["events"][j],self.sentence["words"][0])
			self._backpointer_matrix[j,0] = -1
		# for all other columns, i.e. second word to the last word
		for i, w in enumerate(self.sentence["words"][1:],1):
			for j, label in enumerate(possible_labels):
				# introduce a temporary storage for scores: a list [(scores if coming from l1), (scores if coming from l2),...]
				scores = []
				# suppose label has been placed on word; then run through all the labels and calculate scores
				for l, lb in enumerate(possible_labels):
					scores.append(self._vit_matrix[l, i-1] +\
						fw["(ev-1):[{}]->(ev)[{}]".format(lb,label)+fw["(ev):[{}]=>(word)[{}]".format(label,w)
				# find maximum score and its index
				highest_score = max(scores)
				index_highest_score = scores.index(highest_score)
				self._vit_matrix[j, i] = highest_score
				self._backpointer_matrix[j,i] = index_highest_score
		# finally, the very last imaginary termination
		scores = []
		for j, label in enumerate(possible_labels):
			scores.append(self._vit_matrix[j, -1] +\
				fw["(ev-1):[{}]->(ev)[{}]".format(label,"END").format(label,w)

		highest_score = max(scores)
		index_highest_score = scores.index(highest_score)

		for 


			
