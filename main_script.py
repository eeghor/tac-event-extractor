import json
import sys
import pandas as pd
import os
from features import FeatureFactory
from collections import defaultdict
from viterbi import Viterbi
import time
from scorer import Scores
import copy

# __file__ is the pathname of the file from which the module was loaded, if it was loaded from a file
# Return the canonical path of the specified filename, eliminating any symbolic links encountered in the path
# return the directory name of pathname path
current_dir = os.path.dirname(os.path.realpath('__file__'))  

train_file = sys.argv[1]  # training dataset
nomlex_file = "nomlex_dict.json"

# upload the training dataset; it's a list of dicts, [{"words":[], "events":[]},...]
with open(train_file, "r") as f:
	training_set = json.load(f)

# change event names
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
"I-Justice_Execute,I-Life_Die", "lethal_execution",  
"I-Business_Declare-Bankruptcy", "bancrupcy", 
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
"I-Conflict_Attack,I-Transaction_Transfer-Ownership": "violent_robbery"  
"I-Life_Die": "die"}    

# let's find out what and how many event labels are there; 
# note that some words can have a few labels on them, separated by a coma

all_events = defaultdict(set)

double_labelled = defaultdict(set)

for sent in training_set:
# 	predicted_labels_training_set.append([])
	for i, event_labels in enumerate(sent["events"]):
		if len(event_labels.split(","))>1:
			double_labelled[event_labels].add(sent["lemmas"][i].lower())
		for event_label in event_labels.split(","):
			all_events[event_label].add(sent["lemmas"][i].lower())

all_event_labels = [k for k in all_events.keys()] + ["I-Justice_Execute,I-Life_Die",
						 "I-Movement_Transport-Artifact,I-Transaction_Transfer-Ownership",
						 "I-Conflict_Attack,I-Life_Injure",
						 "I-Conflict_Attack,I-Transaction_Transfer-Ownership"]

print(all_events)

print(all_event_labels)
print("found {} variations of multiple labels on the same word".format(len(double_labelled)))

# how many words relate each event
words_per_event = {e: len(all_events[e]) for e in all_events}

#print(words_per_event)
extras_ev = "up over off on down out in".split()
#all_event_labels = {event_label  for sent in training_set for event_labels in sent["events"] for event_label in event_labels.split(",")}

NUM_EVENT_LABELS = len(all_event_labels)
print("found {} event labels (incl. non-event) in the training dataset".format(NUM_EVENT_LABELS))
# create an event dictionary: {"event":[set of words labelled as this event],..}

# load NOMLEX dictionary
with open(nomlex_file, "r") as f:
	nomlex_dict = json.load(f)

# extract all features from all sentences
fd = defaultdict(int)

for sent in training_set:
	for i,w in enumerate(sent["words"]):
		ff = FeatureFactory(sent, i, nomlex_dict)
		fd.update(ff.extract())

for k in fd:
	fd[k] = 0

print("collected {} features".format(len(fd)))

print("unique features:{}".format(len(set(fd.keys()))))

Scores([["O","Attack","O","O","O","Business"],["O","Business","O","Business","O","Business"]],
	[["O","Business","Attack", "O","O","Business"],["O","Attack","Attack", "O","Attack","Business"]]).show()


nvi = 50


start_time = time.time()
for i in range(nvi):
	predicted_labels_training_set = []
#	print(predicted_labels_training_set)
	print("starting viterbi run {}...".format(i))
	for j, sent in enumerate(training_set):
		#sent_features = defaultdict(int)
		#print("original sentence:",sent["events"])
		#print("viterbi prediction:",Viterbi(sent, all_event_labels, fd).run())
		predicted_labels_training_set.append(Viterbi(sent, all_event_labels, fd).run())
		tmp_sent = copy.deepcopy(sent)
		tmp_sent["events"] = predicted_labels_training_set[j]
		# print("after viterbi:",predicted_labels_training_set)
		# print("predicted:",predicted_labels)
		# print("actual:",sent["events"])
		for i,w in enumerate(sent["words"]):
			ff = FeatureFactory(sent, i, nomlex_dict).extract()
			ff_pr = FeatureFactory(tmp_sent, i, nomlex_dict).extract()
			# print("features from word are",ff)
			if sent["events"][i] != tmp_sent["events"][i]:
				#print("actual label {} - predicted {}".format(sent["events"][i], predicted_labels_training_set[j][i]))
				#print("need to update weights")
				for k in ff_pr:
					fd[k] -= 1
				for g in ff:
					fd[g] += 1
			# else:
			# 	for k in ff_pr:
			# 		fd[k] -= 1
	#print(predicted_labels_training_set)
	# now get scores 
	training_labels = [st["events"] for st in training_set]
	print("have {} training sentences and {} predicted ones".format(len(training_labels), len(predicted_labels_training_set)))
	Scores(training_labels, predicted_labels_training_set).show()

	end_time = time.time()

print("elapsed time: {} minutes".format(round((end_time-start_time)/60.0),1))


