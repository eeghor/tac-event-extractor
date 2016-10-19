# Event extraction using the Structured Perceptron algorithm

## Preprocessing
### Non-alphanumeric tokens
* All tokens that contain not only letters but also something else or no letters at all will be labelled as **non-events** in case they have never been labelled as events in the training set
* Numbers will always be **non-events** although someone did label some numbers as events in the training set
### Tokens with multiple labels
There is a number of tokens that have multiple labels. Specifically, we have found the following in the training set:

* **I-Justice\_Sentence,I-Justice\_Sentence** sentence

* **I-Conflict\_Attack,I-Life\_Die** slaying, attempt, kill, massacre, death, the, blow, killer, genocide, killing, murder, slaughter, manslaughter

* **I-Contact\_Meet,I-Movement\_Transport-Person** visit

* **I-Personnel\_End-Position,I-Personnel\_End-Position** former

* **I-Contact\_Meet,I-Justice\_Trial-Hearing** tell, plead, testimony

* **I-Justice\_Execute,I-Life\_Die** penalty, execute, lethal, death, punishment, to, put, capital, execution, hang

* **I-Personnel\_End-Position,I-Personnel\_End-Position,I-Personnel\_End-Position** work

* **I-Movement\_Transport-Artifact,I-Transaction\_Transfer-Money,I-Transaction\_Transfer-Ownership** trafficking, smuggling

* **I-Conflict\_Attack,I-Transaction\_Transaction** robbery

* **I-Justice\_Execute,I-Justice\_Sentence** penalty

* **I-Transaction\_Transfer-Money,I-Transaction\_Transfer-Ownership** sale, purchase, run, buyer, buy, donate, sell

* **I-Contact\_Correspondence,I-Transaction\_Transfer-Ownership** receive

* **I-Movement\_Transport-Artifact,I-Transaction\_Transfer-Ownership** smuggle, ship, transfer, smuggling, trafficking, pick, receive, smugglee, supply, smuggler

* **I-Transaction\_Transfer-Money,I-Transaction\_Transfer-Money** pay

* **I-Conflict\_Attack,I-Life\_Injure** over, hurt, abuse, attack, knee, assault, run, rape, shot, injure, wound, injured, cap

* **I-Contact\_Meet,I-Justice\_Arrest-Jail** apprehension

* **I-Contact\_Broadcast,I-Justice\_Trial-Hearing** rule

* **I-Conflict\_Attack,I-Transaction\_Transfer-Ownership** hijacking, seize, rob, robbery, burglary

* **I-Justice\_Extradite,I-Movement\_Transport-Person** deport, extradition, extradite

* **I-Justice\_Charge-Indict,I-Justice\_Charge-Indict** charge

* **I-Conflict\_Attack,I-Transaction\_Transfer-Money** robbery

* **I-Justice\_Fine,I-Transaction\_Transfer-Money** fine

* **I-Movement\_Transport-Artifact,I-Transaction\_Transaction** supply

#### What to do about these labels
* If we have the same label duplicated like **I-Justice\_Sentence,I-Justice\_Sentence** we simply aim to label the corresponding token once and consider the predicted label correct if it matches a single instance of the duplicated label
* We consider the relatively widespread multiple labels as separate labels. Specifically, **I-Justice\_Execute,I-Life\_Die** means something like *killed according to justice*. Then **I-Conflict\_Attack,I-Transaction\_Transfer-Ownership** means *assault that involves taking someone’s property*. Also, **I-Conflict\_Attack,I-Life\_Die** is clearly *violent death* and **I-Conflict\_Attack,I-Life\_Injure** is *assault that results in injusries*.


## Training and testing data
We have both the training and testing datasets as JSON files in the **data** directory. The main script first loads the training data and does some preprocessing.

## Preprocessing
* Find all names in the text and replace them with a token **NAME**. To find the names, we use the US babyname dataset from Kaggle as well as the name gazeteers from GATE.

## Viterbi Algorithm
*  Suppose there is a sentence _Cat eats a cake._ We labelled this sentence as *E E NE NE* where *NE* stands for "non-event" and *E* is for "event". This is almost correct except for the label for word "cat" which is supposed to be a *NE*. 
*  For each word in this sentence we take all features related to this word and **increase** their weights by 1 if we labelled this word correctly. If we labelled the word incorrectly, all features related to this word have their weights **decreased** by 1.

## Viterbi algorithm

A very good description of this algorithm can be found in 
**Jurafsky, Daniel, and James H. Martin. 2008. Speech and Language Processing. Second Edition. Prentice Hall** or even better, in the unpublished third edition [here][1]

[1]:	https://web.stanford.edu/~jurafsky/slp3/