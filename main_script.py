import json
import sys
import pandas as pd
import os

current_dir = os.path.dirname(os.path.realpath('__file__'))

train_file = sys.argv[1]  # training dataset

with open(train_file, "r") as f:
	train_df = json.load(f)

# let's find out what and how many event labels are there; note that some words can have a few labels on them, separated by a coma
EVENT_LABEL_SET = {event_label  for sent in train_df for event_labels in sent["events"] for event_label in event_labels.split(",")}
NUM_EVENT_LABELS = len(EVENT_LABEL_SET)
print("found {} event labels in the training dataset".format(NUM_EVENT_LABELS))
print(EVENT_LABEL_SET)
