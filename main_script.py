import json
import sys
import pandas as pd
import os
from features import FeatureFactory
from collections import defaultdict
from viterbi import Viterbi
import time

current_dir = os.path.dirname(os.path.realpath('__file__'))

train_file = sys.argv[1]  # training dataset
nomlex_file = "nomlex_dict.json"

with open(train_file, "r") as f:
	train_df = json.load(f)

# let's find out what and how many event labels are there; note that some words can have a few labels on them, separated by a coma
EVENT_LABEL_SET = {event_label  for sent in train_df for event_labels in sent["events"] for event_label in event_labels.split(",")}
NUM_EVENT_LABELS = len(EVENT_LABEL_SET)
print("found {} event labels in the training dataset".format(NUM_EVENT_LABELS))

# load NOMLEX dictionary
with open(nomlex_file, "r") as f:
	nomlex_dict = json.load(f)

# extract all features from all sentences
fd = defaultdict(int)

for sent in train_df:
	for i,w in enumerate(sent["words"]):
		ff = FeatureFactory(sent, i, nomlex_dict)
		fd.update(ff.extract())

print("collected {} features".format(len(fd)))


start_time = time.time()

for sent in train_df:
	sent_features = defaultdict(int)
	vi = Viterbi(sent, EVENT_LABEL_SET, fd)
	predicted_labels = vi._run_viterbi_algorithm()
	for i,w in enumerate(sent["words"]):
		ff = FeatureFactory(sent, i, nomlex_dict)
		if sent["events"][i] != predicted_labels[i]:
			for k in ff:
				fd[k] -= 1
		else:
			for k in ff:
				fd[k] +=1
end_time = time.time()

print("spent {} sec".format(end_time-start_time))


