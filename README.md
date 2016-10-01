# Event extraction using the Structured Perceptron algorithm

## Preprocessing
### Tokens with multiple labels
There is a number of tokens that have multiple labels. Specifically, we have found the following in the training set:

* **I-Justice_Sentence,I-Justice_Sentence** sentence

* **I-Conflict_Attack,I-Life_Die** slaying, attempt, kill, massacre, death, the, blow, killer, genocide, killing, murder, slaughter, manslaughter 

* **I-Contact_Meet,I-Movement_Transport-Person** visit

* **I-Personnel_End-Position,I-Personnel_End-Position** former

* **I-Contact_Meet,I-Justice_Trial-Hearing** tell, plead, testimony

* **I-Justice_Execute,I-Life_Die** penalty, execute, lethal, death, punishment, to, put, capital, execution, hang

* **I-Personnel_End-Position,I-Personnel_End-Position,I-Personnel_End-Position** work

* **I-Movement_Transport-Artifact,I-Transaction_Transfer-Money,I-Transaction_Transfer-Ownership** trafficking, smuggling

* **I-Conflict_Attack,I-Transaction_Transaction** robbery

* **I-Justice_Execute,I-Justice_Sentence** penalty

* **I-Transaction_Transfer-Money,I-Transaction_Transfer-Ownership** sale, purchase, run, buyer, buy, donate, sell

* **I-Contact_Correspondence,I-Transaction_Transfer-Ownership** receive

* **I-Movement_Transport-Artifact,I-Transaction_Transfer-Ownership** smuggle, ship, transfer, smuggling, trafficking, pick, receive, smugglee, supply, smuggler

* **I-Transaction_Transfer-Money,I-Transaction_Transfer-Money** pay

* **I-Conflict_Attack,I-Life_Injure** over, hurt, abuse, attack, knee, assault, run, rape, shot, injure, wound, injured, cap

* **I-Contact_Meet,I-Justice_Arrest-Jail** apprehension

* **I-Contact_Broadcast,I-Justice_Trial-Hearing** rule

* **I-Conflict_Attack,I-Transaction_Transfer-Ownership** hijacking, seize, rob, robbery, burglary

* **I-Justice_Extradite,I-Movement_Transport-Person** deport, extradition, extradite

* **I-Justice_Charge-Indict,I-Justice_Charge-Indict** charge

* **I-Conflict_Attack,I-Transaction_Transfer-Money** robbery

* **I-Justice_Fine,I-Transaction_Transfer-Money** fine

* **I-Movement_Transport-Artifact,I-Transaction_Transaction** supply

#### What to do about these labels
* If we have the same label duplicated like **I-Justice_Sentence,I-Justice_Sentence** we simply aim to label the corresponding token once and consider the predicted label correct if it matches a single instance of the duplicated label


## Training and testing data
We have both the training and testing datasets as JSON files in the **data** directory. The main script first loads the training data and does some preprocessing.

## Preprocessing
* Find all names in the text and replace them with a token **NAME**. To find the names, we use the US babyname dataset from Kaggle as well as the name gazeteers from GATE.

*  Suppose there is a sentence _Cat eats a cake._ We labelled this sentence as *E E NE NE* where *NE* stands for "non-event" and *E* is for "event". This is almost correct except for the label for word "cat" which is supposed to be a *NE*. 
*  For each word in this sentence we take all features related to this word and **increase** their weights by 1 if we labelled this word correctly. If we labelled the word incorrectly, all features related to this word have their weights **decreased** by 1.

## Viterbi algorithm

A very good description of this algorithm can be found in 
**Jurafsky, Daniel, and James H. Martin. 2000. Speech and Language Processing. Prentice Hall.**