### 2019-04-29

[CN2 rule](<https://docs.biolab.si//3/visual-programming/widgets/model/cn2ruleinduction.html>)

1. Name under which the learner appears in other widgets. The default name is *CN2 Rule Induction*.
2. *Rule ordering*:
   - **Ordered**: induce ordered rules (decision list). Rule conditions are found and the majority class is assigned in the rule head.
   - **Unordered**: induce unordered rules (rule set). Learn rules for each class individually, in regard to the original learning data.
3. *Covering algorithm*:
   - **Exclusive**: after covering a learning instance, remove it from further consideration.
   - **Weighted**: after covering a learning instance, decrease its weight (multiplication by *gamma*) and in-turn decrease its impact on further iterations of the algorithm.
4. *Rule search*:
   - **Evaluation measure**: select a heuristic to evaluate found hypotheses:
     - [Entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)) (measure of unpredictability of content)
     - [Laplace Accuracy](https://en.wikipedia.org/wiki/Laplace's_method)
     - Weighted Relative Accuracy
   - **Beam width**; remember the best rule found thus far and monitor a fixed number of alternatives (the beam).
5. *Rule filtering*:
   - **Minimum rule coverage**: found rules must cover at least the minimum required number of covered examples. Unordered rules must cover this many target class examples.
   - **Maximum rule length**: found rules may combine at most the maximum allowed number of selectors (conditions).
   - **Default alpha**: significance testing to prune out most specialised (less frequently applicable) rules in regard to the initial distribution of classes.
   - **Parent alpha**: significance testing to prune out most specialised (less frequently applicable) rules in regard to the parent class distribution.
6. Tick ‘Apply Automatically’ to auto-communicate changes to other widgets and to immediately train the classifier if learning data is connected. Alternatively, press ‘Apply‘ after configuration.



```python
# https://docs.biolab.si//3/data-mining-library/reference/classification.html?highlight=rule#Orange.classification.rules.CN2Learner
class Orange.classification.rules.CN2SDLearner(preprocessors=None, base_rules=None)[source]¶

class Orange.classification.rules.CN2SDUnorderedLearner(preprocessors=None, base_rules=None)[source]

#Notes

"A weighted covering algorithm is applied, in which subsequently induced rules also represent interesting and sufficiently large subgroups of the population. Covered positive examples are not deleted from the learning set, rather their weight is reduced.

"The algorithm demonstrates how classification rule learning (predictive induction) can be adapted to subgroup discovery, a task at the intersection of predictive and descriptive induction."
```



#### subgroup discovery

- Rule Learning
  ∗ An approach to predictive induction (or supervised learning),
  aimed at constructing a set of rules to be used for classification
  and/or prediction
- Association Rule Learning
  ∗ A form of descriptive induction (non-classificatory induction or
  unsupervised learning), aimed at the discovery of individual rules
  which define interesting patterns in data.
- **Subgroup Discovery**
  ∗ A Task at the Intersection of Predictive and Descriptive Induction
  ∗ Definition: **Given a population of individuals and a property of**
  **those individuals we are interested in, find population subgroups**
  **that are statistically ‘most interesting’**, e.g., are as large as
  possible and have the most unusual statistical (distributional)
  characteristics with respect to the property of interest