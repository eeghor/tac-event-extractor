from search import SearchAlgorithm
from collections import defaultdict
import numpy as np

class Viterbi(SearchAlgorithm):

	def __init__(self,sentence, possible_labels, features_n_weights):
		
		SearchAlgorithm.__init__(self,sentence, possible_labels, features_n_weights)
		self._NW = len(self.sentence["words"])
		self._NL = len(self.labels)
		# viterbi matrix to store the highest scores for each label on a word
		self._vit_matrix = np.zeros(shape=(self._NL,self._NW))
		self._backpointer_matrix = np.zeros(shape=(self._NL,self._NW)).astype(int)
		self.paz = []
		#print(self._backpointer_matrix)

	def run(self):

		# fill in the 1st column
		for j, label in enumerate(self.labels):

			self._vit_matrix[j, 0] = self.fw["(ev-1):[{}]->(ev)[{}]".format("START",label)]+\
										self.fw["(ev):[{}]=>(word)[{}]".format(label,self.sentence["words"][0])]
			self._backpointer_matrix[j,0] = -1

		# for all other columns, i.e. from second word to the last word
		for i, w in enumerate(self.sentence["words"][1:],1):
			for j, label in enumerate(self.labels):
				# introduce a temporary storage for scores: a list [(scores if coming from l1), (scores if coming from l2),...]
				scores = []
				# suppose label has been placed on word; then run through all the labels and calculate scores
				for l, lb in enumerate(self.labels):
					scores.append(self._vit_matrix[l, i-1] +\
						self.fw["(ev-1):[{}]->(ev)[{}]".format(lb,label)]+self.fw["(ev):[{}]=>(word)[{}]".format(label,w)])
				# find maximum score and its index
				highest_score = max(scores)
				index_highest_score = scores.index(highest_score)
				self._vit_matrix[j, i] = highest_score
				self._backpointer_matrix[j,i] = index_highest_score
		# finally, the very last imaginary termination
		scores = [self._vit_matrix[j, -1] +
					self.fw["(ev-1):[{}]->(ev)[{}]".format(label,"END")] for j, label in enumerate(self.labels)]

		label_index = scores.index(max(scores))  # index of the highest scoring label at termination
		#print(self._backpointer_matrix)
		v = self._backpointer_matrix[label_index,self._NW-1]  # index of previous label best scoring for the label scoring highest at termination

		# go backwards through the word indices
		for i in range(self._NW-1, -1,-1):
			self.paz.append(self.labels[label_index])
			label_index = self._backpointer_matrix[label_index,i]

		self.paz.reverse()

		return self.paz

## test if the algorighm works
from collections import defaultdict

f = defaultdict(int, {"(ev-1):[O]->(ev)[O]": 1, "(ev-1):[Jump]->(ev)[O]": 2, "(ev-1):[O]->(ev)[Jump]": 3})
pz = Viterbi({"words":["Fly","is","jumping","."], "events":["O","O","Jump","O"]}, ["Jump","Sit","O"],f).run()
print(pz)




			
