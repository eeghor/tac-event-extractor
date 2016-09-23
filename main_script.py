import json
import sys
import pandas as pd
import os
from features import FeatureFactory
from collections import defaultdict
from viterbi import Viterbi
import time
#import pdb
from scorer import Scorer

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

print("collected {} features".format(len(fd)))

sco = Scorer(["O","Attack","O","O","O","Business"],["O","Business","Attack", "O","O","Business"])
print(sco._get_scores())

start_time = time.time()
print("starting viterbi run...")
for sent in training_set:
	sent_features = defaultdict(int)
	vi = Viterbi(sent, all_event_labels, fd)
	predicted_labels = vi._run_viterbi_algorithm()
	# print("predicted:",predicted_labels)
	# print("actual:",sent["events"])
	for i,w in enumerate(sent["words"]):
		ff = FeatureFactory(sent, i, nomlex_dict).extract()
		if sent["events"][i] != predicted_labels[i]:
			for k in ff:
				fd[k] -= 1
		else:
			for k in ff:
				fd[k] +=1
end_time = time.time()

print("elapsed time: {} minutes".format(round((end_time-start_time)/60.0),1))


