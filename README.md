# Event extraction using the Structured Perceptron algorithm

* What is a "positive example": suppose the word "fired" has the correct event label *attack*; if this word has any other label, event label or anything else but not "*attack*, it is a *negative* example*

*  Suppose there is a sentence _Cat eats a cake._ We labelled this sentence as *E E NE NE* where NE stands for "non-event" and E is for "event". This is almost correct except for the label for word "cat" which is supposed to be a NE. 
