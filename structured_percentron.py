import json
import sys
import gensim
import pandas as pd
import os  # pathname manipulations
#from features import FeatureFactory
from collections import defaultdict
from nltk.corpus import wordnet
from viterbi import Viterbi
import time
#from scorer import Scores
import copy

class SPerceptron(object):

	def __init__(self):

		print("initialising structured perceptron...")
		#
		# -- get the current directory name --
		# __file__ is the pathname of the file from which the module was loaded; realpath returns canonical path of this pathname
		#  eliminating any symbolic links encountered in the path; dirname returns the directory name of pathname
		# 
		self.current_dir = os.path.dirname(os.path.realpath('__file__'))
		self.feature_file = "features_init.json"
		self.feature_file_path = self.current_dir+"/saved_features/"+self.feature_file
		self.train_file = sys.argv[1]  # training dataset
		self.load_features_flag = sys.argv[2]

		# upload the training dataset; it's a list of dicts, [{"words":[], "events":[]},...]
		with open(self.	train_file, "r") as f:
			self.training_set = json.load(f)
		print("loaded training set...")

		# -- load NOMLEX dictionary
		self.nomlex_file = self.current_dir+"/data/"+"nomlex_dict.json"
		with open(self.nomlex_file,"r") as f:
			nomlex_dict = json.load(f)
		print("loaded NOMLEX dictionary...")
		
		# -- load Word2Vec model pretrained on Google News
		# WARNING: this takes a lot of memory, 8 Gb is hadrly enough
		
		w2v_start_time = time.time()
		self.word2vec_file = self.current_dir+"/word2vec_pretrained/"+"GoogleNews-vectors-negative300.bin"	
		self.w2v_model = gensim.models.Word2Vec.load_word2vec_format(self.word2vec_file, binary=True)
		w2v_end_time = time.time()
		print("loaded pretrained Word2Vec model... elapsed time {} seconds".format(round(w2v_end_time - w2v_start_time),1))	

		# feature dictionary
		# self.fweights = defaultdict(int)

		self.rename_events()

		# make set of all events, e.g. {"attack": {"fire","kill"}, "transfer": {"deposit"}, ...}
		self.events_and_lemmas = defaultdict(set)

		for sent in self.training_set:
			for i, event_label in enumerate(sent["events"]):
				self.events_and_lemmas[event_label].add(sent["lemmas"][i].lower())

		self.event_names = set(self.events_and_lemmas.keys())
		print("we have {} events in this dataset (incl. non-event)".format(len(self.event_names)))

		# extract or load features

		assert self.load_features_flag in ["load_features","extract_features"], "you must provide flag load_features or extract_features in command line!"

		self.feature_names = set()


		if self.load_features_flag == "load_features":
			
			if os.path.exists(self.feature_file_path):
				with open(self.feature_file_path, "r") as f:
					self.fweights = json.load(f)
				print("loaded features from file {}".format(self.feature_file))
		elif self.load_features_flag == "extract_features":
		
			fet_start_time = time.time()
			if not os.path.exists(self.feature_file_path):
				os.makedirs(self.current_dir+"/saved_features/")
			else:
				os.remove(self.feature_file_path)
				print("deleted previous feature file...")
		
			for sent in self.training_set:
		
				for i, w in enumerate(sent["lemmas"]):
		
					self.feature_names |= set(self.create_features(sent, i, "train").keys())
		
			self.fweights = dict.fromkeys(self.feature_names, 0)

			

			with open(self.feature_file_path,"w+") as f:
				json.dump(self.fweights, f)
			
			fet_end_time = time.time()
			print("done extracting features from training set. elapsed time {} minutes".format(round((fet_end_time-fet_start_time)/60, 1)))
		
		self.nfeatures = len(self.fweights)
		print("have {} features in total".format(self.nfeatures))

	
	def rename_events(self):
	
		evmaps = {
			"I-Justice_Sue": "sue", 
			"I-Life_Injure": "injure",  
			"I-Contact_Contact": "contact",
			"I-Conflict_Attack": "attack",
			"I-Personnel_Start-Position": "start_position", 
			"I-Transaction_Transfer-Ownership": "change_ownership",
			"I-Justice_Charge-Indict": "charge",
			"I-Justice_Appeal": "appeal",  
			"I-Justice_Convict": "convict",
			"I-Justice_Arrest-Jail": "arrest",
			"I-Personnel_Nominate": "nominate",
			"I-Transaction_Transaction": "transaction",
			"I-Movement_Transport-Artifact,I-Transaction_Transfer-Ownership": "deliver_for_someone", 
			"I-Justice_Release-Parole": "parole", 
			"I-Contact_Broadcast": "broadcast",  
			"I-Transaction_Transfer-Money": "transfer_money",
			"I-Justice_Acquit": "free_from_charge",
			"I-Life_Divorce": "divorce", 
			"I-Justice_Extradite": "extradict", 
			"I-Justice_Trial-Hearing": "trial",
			"I-Business_Merge-Org": "merger", 
			"I-Justice_Execute": "execute",
			"I-Personnel_End-Position": "end_position",
			"I-Justice_Execute,I-Life_Die": "lethal_execution",  
			"I-Business_Declare-Bankruptcy": "bancrupcy", 
			"I-Conflict_Attack,I-Life_Injure": "attack_resulting_in_injury", 
			"I-Life_Be-Born": "beborn",   
			"I-Contact_Meet": "meet",
			"I-Conflict_Demonstrate": "protest",
			"I-Justice_Sentence": "sentence",  
			"I-Business_End-Org": "liquidation",
			"I-Justice_Fine": "fine", 
			"I-Contact_Correspondence": "correspondence",  
			"I-Personnel_Elect": "elect", 
			"I-Life_Marry": "marry", 
			"I-Manufacture_Artifact": "produce_something",   
			"I-Business_Start-Org": "new_business",  
			"I-Justice_Pardon": "pardon", 
			"I-Movement_Transport-Artifact": "transport_something",
			"I-Movement_Transport-Person": "transport_people",  
			"I-Conflict_Attack,I-Transaction_Transfer-Ownership": "violent_robbery", 
			"I-Life_Die": "die"}
	
		for sent in self.training_set:
				for i, event_label in enumerate(sent["events"]):
					if event_label in evmaps:
						sent["events"][i] = evmaps[event_label]
					else:
						sent["events"][i] = "O"
	
	def create_features(self, sent, word_idx, tt_flag):
	
		# add 2 extra tokens on each side of the words, lemmas and so on for the supplied sentence
		sentence = {k: ["BEFORE_START","START"] + v +["END","AFTER_END"] for k,v, in sent.items()} 
		i = word_idx + 2 # because now we have START1, START2, word1, ..
		_pidx = i - 1  # previous index
		_nidx = i + 1  # next index
		feature_dict = defaultdict(int)
	
		# transition features involving lemmas, POS and events
		feature_dict["(lem-1):[{}]->(lem)[{}]".format(sentence["lemmas"][_pidx].lower(),sentence["lemmas"][i].lower())] += 1
		feature_dict["(lem-2):[{}]->(lem-1)[{}]->(lem)[{}]".format(sentence["lemmas"][i-2].lower(), sentence["lemmas"][_pidx].lower(), sentence["lemmas"][i])] += 1
		feature_dict["(pos-2):[{}]->(pos-1)[{}]->(pos)[{}]".format(sentence["POSs"][i-2].lower(), sentence["POSs"][_pidx].lower(), sentence["POSs"][i])] += 1
		feature_dict["(pos-1):[{}]->(pos)[{}]".format(sentence["POSs"][_pidx].lower(), sentence["POSs"][i].lower())] += 1
		feature_dict["(ent-1):[{}]->(ent)[{}]".format(sentence["entities"][_pidx].lower(),sentence["entities"][i].lower())] += 1
		feature_dict["(ev-1):[{}]->(ev)[{}]".format(sentence["events"][_pidx],sentence["events"][i])] += 1
		feature_dict["(lem):[{}]->(lem+1)[{}]".format(sentence["lemmas"][i].lower(), sentence["lemmas"][_nidx].lower())] += 1
		feature_dict["(pos):[{}]->(pos+1)[{}]".format(sentence["POSs"][i].lower(), sentence["POSs"][_nidx].lower())] += 1
		feature_dict["(ent):[{}]->(ent+1)[{}]".format(sentence["entities"][i].lower(), sentence["entities"][_nidx].lower())] += 1
		feature_dict["(ev):[{}]->(ev+1)[{}]".format(sentence["events"][i], sentence["events"][_nidx])] += 1
	
		# emission features: in HMM, likelihood that hidden event i generated the observed lemma i
		feature_dict["(ev):[{}]=>(lem)[{}]".format(sentence["events"][i].lower(),sentence["lemmas"][i].lower())] += 1
	
		# lemma synonyms and hypernyms from WordNet
		if wordnet.synsets(sentence["lemmas"][i].lower()):
		    for w in wordnet.synsets(sentence["lemmas"][i].lower()):
		        for ln in w.lemma_names():
		            feature_dict["(syn):[{}]".format(ln)] += 1
		        for g in w.hypernyms():
		            for l in g.lemma_names():
		            	feature_dict["(hyp):[{}]".format(l)] += 1	
	
		# last 2 letters in word
		if len(sentence["words"][i]) > 4:
			feature_dict["(word_end2))[{}]".format(sentence["words"][i][-2:])] += 1
		
		# is alphanumeric
		if ~sentence["words"][i].isalnum():
			feature_dict["(not_alphanumeric)"] += 1
		# lemma
		feature_dict["(lemma)[{}]".format(sentence["lemmas"][i].lower())] += 1
		
		# entity
		feature_dict["(entity)[{}]".format(sentence["entities"][i])] += 1
		if sentence["entities"][i] == "O":
			feature_dict["(non_entity)"] += 1
		# POS
		feature_dict["(POS)[{}]".format(sentence["POSs"][i])] += 1
		# word length
		if len(sentence["words"][i]) < 3:
			feature_dict["(very_short_word)"] += 1
		# hyphen
		if "-" in sentence["words"][i]:
			feature_dict["hyphen"] += 1
	
		# Word2Vec features
		if tt_flag == "train":
			# only generate w2v features for the words that are labelled as events
			if sentence["events"][i] != "O" and sentence["lemmas"][i].isalnum():
				# take topn most similar words
				if sentence["lemmas"][i].lower() in self.w2v_model.vocab:

					for word, simil in self.w2v_model.most_similar(positive=[sentence["lemmas"][i].lower()], negative=[],topn=16):
						feature_dict["(w2v) simto {}".format(word)] += 1
	
		return feature_dict

	def predict(self):

		nvi = 50

		for i in range(nvi):

			predicted_labels_training_set = []

			print("starting viterbi run {}...".format(i))

			for j, sent in enumerate(self.training_set):

				predicted_labels_training_set.append(Viterbi(sent, self.event_names, self.fweights).run())

				tmp_sent = copy.deepcopy(sent)
				tmp_sent["events"] = predicted_labels_training_set[j]

				for i,w in enumerate(sent["words"]):
					# extract features from each word from the correcly labelled sentence..
					ff = create_features(sent, i, "train")
					# and the labelling by Viterbi
					ff_pr = create_features(tmp_sent, i, "train")

					if sent["events"][i] != tmp_sent["events"][i]:

						for k in ff_pr:
							fd[k] -= 1
						for g in ff:
							fd[g] += 1

			# now get scores for this Viterbi iteration
			training_labels = [st["events"] for st in self.training_set]
			# print("have {} training sentences and {} predicted ones".format(len(training_labels), len(predicted_labels_training_set)))
			Scores(training_labels, predicted_labels_training_set).show()
	

sp = SPerceptron()
sp.predict()

	# upload the training dataset; it's a list of dicts, [{"words":[], "events":[]},...]
	#with open(train_file, "r") as f:
	#	training_set = json.load(f)

