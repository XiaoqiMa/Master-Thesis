The subgroup discovery toolkit for Orange implements three algorithms for subgroup discovery: SD, CN2-SD and Apriori-SD, two visualization methods: the BAR and the ROC visualization and six evaluation measures for subgroup discovery.

It is distributed free under GPL and can be downloaded from this web page.


https://docs.biolab.si//3/visual-programming/widgets/model/cn2ruleinduction.html

CN2 Rule Induction
Induce rules from data using CN2 algorithm.

Inputs

Data: input dataset

Preprocessor: preprocessing method(s)

Outputs

Learner: CN2 learning algorithm

CN2 Rule Classifier: trained model

The CN2 algorithm is a classification technique designed for the efficient induction of simple, comprehensible rules of form “if cond then predict class”, even in domains where noise may be present.

CN2 Rule Induction works only for classification.


https://blog.biolab.si/tag/rules/

import Orange
data = Orange.data.Table('titanic')
learner = Orange.classification.CN2Learner()
classifier = learner(data)
