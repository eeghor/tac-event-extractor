import json
import sys
import pandas as pd
import os
from features import FeatureFactory
from collections import defaultdict
from viterbi import Viterbi
import time
#import pdb
from scorer import Scores

#pdb.set_trace()

current_dir = os.path.dirname(os.path.realpath('__file__'))

train_file = sys.argv[1]  # training dataset
nomlex_file = "nomlex_dict.json"

# upload the training dataset; it's a list of dicts, [{"words":[], "events":[]},...]
with open(train_file, "r") as f:
	training_set = json.load(f)

# let's find out what and how many event labels are there; 
# note that some words can have a few labels on them, separated by a coma

all_events = defaultdict(set)



for sent in training_set:
# 	predicted_labels_training_set.append([])
	for i, event_labels in enumerate(sent["events"]):
		for event_label in event_labels.split(","):
			all_events[event_label].add(sent["words"][i].lower())

all_event_labels = [k for k in all_events.keys()]

# how many words relate each event
words_per_event = {e: len(all_events[e]) for e in all_events}

#print(words_per_event)

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

Scores([["O","Attack","O","O","O","Business"],["O","Business","O","Business","O","Business"]],
	[["O","Business","Attack", "O","O","Business"],["O","Attack","Attack", "O","Attack","Business"]]).show()


nvi = 8


start_time = time.time()
for i in range(nvi):
	predicted_labels_training_set = []
	print(predicted_labels_training_set)
	print("starting viterbi run {}...".format(i))
	for j, sent in enumerate(training_set):
		#sent_features = defaultdict(int)
		predicted_labels_training_set.append(Viterbi(sent, all_event_labels, fd).run())
		# print("after viterbi:",predicted_labels_training_set)
		# print("predicted:",predicted_labels)
		# print("actual:",sent["events"])
		for i,w in enumerate(sent["words"]):
			ff = FeatureFactory(sent, i, nomlex_dict).extract()
			# print("features from word are",ff)
			if sent["events"][i] != predicted_labels_training_set[j][i]:
				for k in ff:
					fd[k] -= 1
			else:
				for k in ff:
					fd[k] +=1
	#print(predicted_labels_training_set)
	# now get scores 
	training_labels = [st["events"] for st in training_set]
	print("have {} training sentences and {} predicted ones".format(len(training_labels), len(predicted_labels_training_set)))
	Scores(training_labels, predicted_labels_training_set).show()

	end_time = time.time()

print("elapsed time: {} minutes".format(round((end_time-start_time)/60.0),1))


