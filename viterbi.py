from search import SearchAlgorithm
from collections import defaultdict
import numpy as np

class Viterbi(SearchAlgorithm):

	def __init__(self,sentence, possible_labels, fw):
		
		SearchAlgorithm.__init__(self,sentence, possible_labels, fw)
		self._NW = len(sentence)
		self._NL = len(possible_labels)
		self._vit_matrix = np.zeros(shape=(self._NL,self._NW +2))
		self._backpointer_matrix = np.zeros(shape=(self._NL,self._NW))
		self.possible_labels = ["START"]+possible_labels+["END"]

	def _run_viterbi_algorithm(self):

		for j, label in enumerate(possible_labels, 1):
			self._vit_matrix[j-1, 0] = fw["word i + event i: [{}]+[{}]".format(self.sentence["words"][0],self.sentence["events"][j])+\
										fw["event i-1 -> event i: [{}]->[{}]".format(self.sentence["events"][j-1],self.sentence["events"][j])

		for i, w in enumerate(self.sentence["words"]):

			for j, label in enumerate(possible_labels):

				self._vit_matrix[j, 0] = fw["word i + event i: [{}]+[{}]".format(self.sentence["words"][i],self.sentence["events"][i])
				
				self._vit_matrix[j,i] = self._vit_matrix[,i-1] fw["event i-1 -> event i: [{}]->[{}]".format(self.sentence["events"][i-1],self.sentence["events"][i])]+\
										fw["word i + event i: [{}]+[{}]".format(self.sentence["words"][i],self.sentence["events"][i])]

			# by now we filled out the i-th column; now time to find




			
