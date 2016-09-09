# Event extraction using the Structured Perceptron algorithm

*  Suppose there is a sentence _Cat eats a cake._ We labelled this sentence as *E E NE NE* where *NE* stands for "non-event" and *E* is for "event". This is almost correct except for the label for word "cat" which is supposed to be a *NE*. 
*  For each word in this sentence we take all features related to this word and **increase** their weights by 1 if we labelled this word correctly. If we labelled the word incorrectly, all features related to this word have their weights **decreased** by 1.
