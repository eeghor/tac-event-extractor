from search import SearchAlgorithm

class Beam(SearchAlgorithm):

	def __init__(self,sentence, possible_labels, features_n_weights,iterations,BEAM_SIZE):

		SearchAlgorithm.__init__(self,sentence, possible_labels, features_n_weights)
		self.iterations = iterations
		self.BEAM_SIZE = BEAM_SIZE
		