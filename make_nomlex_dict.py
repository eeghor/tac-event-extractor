from collections import defaultdict
import sys
import re
import json

try:
	nomlex_file = sys.argv[1]
except IndexError:
	print("Usage: python3 make_nomlex_dict.py [NOMLEX FILE]")
	sys.exit(1)  # exit with the abnormal termination exit status "1"

"""
Below is a sample Nomlex file entry. We want to only extract the noun ("abandonment") and the corresponding verb ("abandon") to put them in a dictionary,
{"abandonment": "abandon"}

(NOM       :ORTH "abandonment"  :PLURAL *NONE*
                                :VERB "abandon"
                                :NOM-TYPE ((VERB-NOM))
                                :VERB-SUBJ ((DET-POSS))
                                :SUBJ-ATTRIBUTE ((COMMUNICATOR))
                                :VERB-SUBC ((NOM-NP :OBJECT ((DET-POSS)
                                                             (PP-OF)))
                                            (NOM-NP-PP :OBJECT ((DET-POSS)
                                                                (PP-OF))
                                                       :PVAL ("for" "to"))
                                            (NOM-NP-TO-INF-OC :OBJECT ((DET-POSS)
                                                                       (PP-OF)))
                                            (NOM-NP-AS-NP :OBJECT ((DET-POSS)
                                                                   (PP-OF)))))
"""

# regular expression to find the noun
# note that some words may have hyphens
noun_pat = re.compile('\(NOM\s+:ORTH\s+\"([\w\-]+)\"',re.IGNORECASE)
verb_pat = re.compile(':VERB\s+\"([\w\-]+)\"',re.IGNORECASE)

found_nouns = []
found_verbs = []

mverbs = 0
mnouns = 0

with open(nomlex_file,"r") as f:
	for line in f:
		nouns_in_line = noun_pat.findall(line)
		verbs_in_line = verb_pat.findall(line)
		if len(nouns_in_line) == 1:
			found_nouns.append(nouns_in_line[0])
		if len(verbs_in_line) == 1:
			found_verbs.append(verbs_in_line[0])

print("total found nouns:",len(found_nouns))
print("total found verbs:",len(found_verbs))

if len(found_nouns) != len(found_verbs):
	raise IndexError("The number of extracted nouns is not the same as the number of extracted verbs!")

nomlex_dict = dict(zip(found_nouns, found_verbs))

with open("nomlex_dict.json","w") as f:
  json.dump(nomlex_dict, f)
