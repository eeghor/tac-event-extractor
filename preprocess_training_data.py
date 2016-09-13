import json
import sys
import pandas as pd
import os
import dawg  # Directed Acyclic Word Graph
import numpy as np

current_dir = os.path.dirname(os.path.realpath('__file__'))

train_file = sys.argv[1]  # training dataset

with open(train_file, "r") as f:
	train_df = json.load(f)

with open(os.path.join(current_dir, "gazetteers/kaggle_us_babynames.csv"),"r") as f:
	kaggle_babynames_df = pd.read_csv(f, usecols=["Name","Gender","Count"])

print(kaggle_babynames_df.head(4))

print("found {} unique names in the Kaggle US babynames dataset".format(kaggle_babynames_df["Name"].nunique()))


# print(kaggle_babynames_df.groupby('Name','Gender').sum())

df1 = pd.DataFrame(kaggle_babynames_df.groupby(['Name','Gender']).sum()).reset_index()

df1.columns = ["Name","Gender","Count"]
total_records = df1["Count"].sum()

print(total_records)
df1.sort(columns="Count",ascending=False,inplace=True)
df1["Pct"] = df1["Count"].apply(lambda _: np.round(_*100/total_records, 5))

print(df1.head(5))
print(df1.tail())

print(df1["Pct"].quantile(q=[0.5, 0.95]))

# create a DAWG
completion_dawg = dawg.CompletionDAWG(kaggle_babynames_df["Name"].str.lower())

# check the training dataset for names and replace any identified names with the token NAME

NSENT = len(train_df)
print("have {} sentences in training dataset".format(NSENT))

for i, sent in enumerate(train_df):
	sent["ptags"] = [["NAME"] if w.lower() in completion_dawg else [] for w in sent["words"]]

with open("new_ds.json","w") as f:
	json.dump(train_df,f)


