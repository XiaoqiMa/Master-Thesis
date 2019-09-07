Decision trees

### Cluster-grouping: from subgroup discovery to clustering (Albrecht Zimmermann · Luc De Raedt)

- To the best of our knowledge, all rule-based approaches to classification mine local patterns in some way and build classifiers from them. Decision trees solve the problem of finding the **optimal splitting pattern** by essentially limiting the number of conditions to one. (Here, we focus on a divisive approach, which bears some similarities to decision tree induction, in which clusters are **repeatedly divided into** sub-clusters according to some criterion)

- An additional work that has to be mentioned is that of Blockeel et al. (1998). In their approach, a decision tree is constructed, with tests in first order logic in the splitting nodes. While the measure used is intra-class variance, to induce similarity of numerical features, CU could be used instead. The main difference lies in the fact that TIC describes clusters by conjunctions formed by along branches of the tree but not in each splitting node

### A Brief Overview of Rule Learning  (Johannes Furnkranz1 and Tomas Kliegr)

decision trees: predictive rule learning

subgroup discovery: descriptive rule learning

**RF and Gradient boosting trees**

error = bias + varianceerror = bias + variance

- Boosting is based on **weak** learners (high bias, low variance). In terms of decision trees, weak learners are shallow trees, sometimes even as small as decision stumps (trees with two leaves). Boosting reduces error mainly by reducing bias (and also to some extent variance, by aggregating the output from many models).
- On the other hand, Random Forest uses as you said **fully grown decision trees** (low bias, high variance). It tackles the error reduction task in the opposite way: by reducing variance. The trees are made uncorrelated to maximize the decrease in variance, but the algorithm cannot reduce bias (which is slightly higher than the bias of an individual tree in the forest). Hence the need for large, unpruned trees, so that the bias is initially as low as possible.

Please note that unlike Boosting (which is sequential), RF grows trees in **parallel**. The term `iterative` that you used is thus inappropriate.

### Finding Patterns of Attrition using Decision Trees: A Preliminary Study (**Wendy Osborn**1, Mandy Moser)

- A decision tree is one technique for modeling the class rules that exist in a set of records. Each path from the root to a leaf node represents a rule
- Many decision trees have been proposed in the literature [2][3][4]. The main difference between decision tree induction strategies is in their attribute selection methods. 
- Therefore, pruning strategies need to be explored. At this point the best option is pessimistic pruning [1], which terminates the production of certain rules when a majority of records with a certain class label exist in a subset of records,