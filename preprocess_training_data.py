import json
import sys
import pandas as pd
import os
import dawg  # Directed Acyclic Word Graph
import numpy as np
import re
from collections import defaultdict

current_dir = os.path.dirname(os.path.realpath('__file__'))

train_file = sys.argv[1]  # training dataset

with open(train_file, "r") as f:
	train_df = json.load(f)

with open(os.path.join(current_dir, "gazetteers/kaggle_us_babynames.csv"),"r") as f:
	kaggle_babynames_df = pd.read_csv(f, usecols=["Name","Gender","Count"])

print("found {} unique names in the Kaggle US babynames dataset".format(kaggle_babynames_df["Name"].nunique()))


# extract currency names and abbreviations and put them in sets

with open(os.path.join(current_dir, "gazetteers/currencies.txt"),"r") as f:
	currencies = pd.read_csv(f, sep="-", names=["Abbr","Country"])

currency_names = set()
currency_abbr = set()

for row in currencies.iterrows():
	currency_abbr.add(row[1]["Abbr"].lower())
	currency_names.add(row[1]["Country"].split()[-1].lower())

print("got a list of {} currency abbreviations".format(len(currency_abbr)))
print("got a list of {} currency names".format(len(currency_names)))



df1 = pd.DataFrame(kaggle_babynames_df.groupby(['Name','Gender']).sum()).reset_index()

df1.columns = ["Name","Gender","Count"]
total_records = df1["Count"].sum()

#print(total_records)
df1.sort_values(by="Count",ascending=False,inplace=True)
df1["Pct"] = df1["Count"].apply(lambda _: np.round(_*100/total_records, 5))

print(df1.head(5))
print(df1.tail(5))

quantile09 = df1["Pct"].quantile(0.97)
idx_rare = df1["Pct"] < quantile09
#print(idx_rare)
idx_common = df1["Pct"] >= quantile09

rare_names = df1[idx_rare]["Name"].tolist()
common_names = df1[idx_common]["Name"].tolist()

print("{} rare names".format(len(rare_names)))
print("{} common names".format(len(common_names)))

print(common_names[:10])

# create a DAWG
rare_names_dawg = dawg.CompletionDAWG(rare_names)
common_names_dawg = dawg.CompletionDAWG(common_names)

# check the training dataset for names and replace any identified names with the token NAME

NSENT = len(train_df)
print("have {} sentences in training dataset".format(NSENT))

WORDS_AND_EVENTS = defaultdict(lambda: defaultdict(int))

p=re.compile("(\d\d\d\d)")

NCURR = 0

for i, sent in enumerate(train_df):
	sent["ptags"] = [["RARE_NAME"] if w.capitalize() in rare_names_dawg else 
					["COMMON_NAME"] if w.capitalize() in common_names_dawg else [] for w in sent["words"]]
	for j, w in enumerate(sent["words"]):
		if not w.isalnum():
			sent["ptags"][j].append("SYMBOL")
		if len(w) < 3:
			sent["ptags"][j].append("TOO_SHORT")
		if len(w) > 19:
			sent["ptags"][j].append("TOO_LONG")
		if p.match(w):
			sent["ptags"][j].append("POSSIBLY_YEAR")
		if w in currency_abbr | currency_names:
			NCURR += 1
			sent["ptags"][j].append("POSSIBLY_CURRENCY")
		if sent["events"][j] == "O":
			WORDS_AND_EVENTS[w.lower()]["nonevent"] += 1
		else:
			WORDS_AND_EVENTS[w.lower()]["event"] += 1

print("labelled {} suspected currency words".format(NCURR))
#print(WORDS_AND_EVENTS)
for i, sent in enumerate(train_df):
	for j, w in enumerate(sent["words"]):
		sent["ptags"][j].append("E"+str(WORDS_AND_EVENTS[w]["event"]))
		sent["ptags"][j].append("N"+str(WORDS_AND_EVENTS[w]["nonevent"]))

with open("new_ds.json","w") as f:
	json.dump(train_df,f)


