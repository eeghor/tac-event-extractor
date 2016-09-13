import json
import sys
import pandas as pd
import os

current_dir = os.path.dirname(os.path.realpath('__file__'))

train_file = sys.argv[1]  # training dataset

with open(train_file, "r") as f:
	train_df = json.load(f)

with open(os.path.join(current_dir, "gazetteers/kaggle_us_babynames.csv"),"r") as f:
	kaggle_babynames_df = pd.read_csv(f, usecols=["Name","Gender"])

print(kaggle_babynames_df.head(3))

print("found {} unique names in the Kaggle US babynames dataset".format(kaggle_babynames_df["Name"].nunique()))