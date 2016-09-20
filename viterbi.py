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
		self.possible_labels = possible_labels

	def _run_viterbi_algorithm(self):

		# fill in the 1st column
		for j, label in enumerate(possible_labels):
			self._vit_matrix[j, 0] = fw["(ev-1):[{}]->(ev)[{}]".format("START",self.sentence["events"][j]+\
										fw["(ev):[{}]=>(word)[{}]".format(self.sentence["events"][j],self.sentence["words"][0])
		# for all other columns, i.e. second word to the last word
		for i, w in enumerate(self.sentence["words"][1:],1):
			for j, label in enumerate(possible_labels):
				# introduce a temporary storage for scores: a list [(scores if coming from l1), (scores if coming from l2),...]
				scores = []
				# suppose label has been placed on word; then run through all the labels and calculate scores
				for 
				self._vit_matrix[j, 0] = fw["word i + event i: [{}]+[{}]".format(self.sentence["words"][i],self.sentence["events"][i])
				
				self._vit_matrix[j,i] = self._vit_matrix[,i-1] fw["event i-1 -> event i: [{}]->[{}]".format(self.sentence["events"][i-1],self.sentence["events"][i])]+\
										fw["word i + event i: [{}]+[{}]".format(self.sentence["words"][i],self.sentence["events"][i])]

			# by now we filled out the i-th column; now time to find




			
