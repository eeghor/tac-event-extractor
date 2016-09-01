"""
 	SPLIT A CORPUS INTO THE TRAINING AND TESTING SETS
	-------------------------------------------------
 	how to use: python3 [corpus_file.JSON] [% training]
"""

import sys
import json
import random

corpus_file = sys.argv[1]
pc_train = sys.argv[2]

with open(corpus_file,"r") as f:
	data = json.load(f)

NSENS = len(data)
print("corpus contains %d sentences" % NSENS)

random.seed(555)

NSAMPLES = int(NSENS*float(pc_train)/100)
print("randomly sampling %d indexes for training set..." % NSAMPLES)
indexes_train = random.sample(range(NSENS), NSAMPLES)

corpus_training = [data[i] for i in indexes_train]
print("created a training corpus with %d sentences..." % len(corpus_training))
corpus_testing = [data[i] for i, _ in enumerate(data) if i not in indexes_train]
print("created a testing corpus with %d sentences" % len(corpus_testing))

# names of the  training and testing files
training_file = corpus_file.split(".")[0] + "-training-" + str(round(float(pc_train),1)) + ".json"
testing_file = corpus_file.split(".")[0] + "-testing-" + str(round(100-float(pc_train),1)) + ".json"

with open(training_file, "w") as f:
	json.dump(corpus_training, f)

with open(testing_file, "w") as f:
	json.dump(corpus_testing, f)



