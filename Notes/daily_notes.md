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

#### Explanation

**An explanation usually relates the feature values of an instance to its model prediction in a humanly understandable way**

**Properties of Individual Explanations**

- **Accuracy**: How well does an explanation predict unseen data? High accuracy is especially important if the explanation is used for predictions in place of the machine learning model. Low accuracy can be fine if the accuracy of the machine learning model is also low, and if the goal is to explain what the black box model does. In this case, only fidelity is important.
- **Fidelity**: How well does the explanation approximate the prediction of the black box model? High fidelity is one of the most important properties of an explanation, because an explanation with low fidelity is useless to explain the machine learning model. Accuracy and fidelity are closely related. If the black box model has high accuracy and the explanation has high fidelity, the explanation also has high accuracy. Some explanations offer only local fidelity, meaning the explanation only approximates well to the model prediction for a subset of the data (e.g. [local surrogate models](https://christophm.github.io/interpretable-ml-book/lime.html#lime)) or even for only an individual data instance (e.g. [Shapley Values](https://christophm.github.io/interpretable-ml-book/shapley.html#shapley)).
- **Consistency**: How much does an explanation differ between models that have been trained on the same task and that produce similar predictions? For example, I train a support vector machine and a linear regression model on the same task and both produce very similar predictions. I compute explanations using a method of my choice and analyze how different the explanations are. If the explanations are very similar, the explanations are highly consistent. I find this property somewhat tricky, since the two models could use different features, but get similar predictions (also called [“Rashomon Effect”](https://en.wikipedia.org/wiki/Rashomon_effect)). In this case a high consistency is not desirable because the explanations have to be very different. High consistency is desirable if the models really rely on similar relationships.
- **Stability**: How similar are the explanations for similar instances? While consistency compares explanations between models, stability compares explanations between similar instances for a fixed model. High stability means that slight variations in the features of an instance do not substantially change the explanation (unless these slight variations also strongly change the prediction). A lack of stability can be the result of a high variance of the explanation method. In other words, the explanation method is strongly affected by slight changes of the feature values of the instance to be explained. A lack of stability can also be caused by non-deterministic components of the explanation method, such as a data sampling step, like the [local surrogate method](https://christophm.github.io/interpretable-ml-book/lime.html#lime) uses. High stability is always desirable.
- **Comprehensibility**: How well do humans understand the explanations? This looks just like one more property among many, but it is the elephant in the room. Difficult to define and measure, but extremely important to get right. Many people agree that comprehensibility depends on the audience. Ideas for measuring comprehensibility include measuring the size of the explanation (number of features with a non-zero weight in a linear model, number of decision rules, …) or testing how well people can predict the behavior of the machine learning model from the explanations. The comprehensibility of the features used in the explanation should also be considered. A complex transformation of features might be less comprehensible than the original features.
- **Certainty**: Does the explanation reflect the certainty of the machine learning model? Many machine learning models only give predictions without a statement about the models confidence that the prediction is correct. If the model predicts a 4% probability of cancer for one patient, is it as certain as the 4% probability that another patient, with different feature values, received? An explanation that includes the model’s certainty is very useful.
- **Degree of Importance**: How well does the explanation reflect the importance of features or parts of the explanation? For example, if a decision rule is generated as an explanation for an individual prediction, is it clear which of the conditions of the rule was the most important?
- **Novelty**: Does the explanation reflect whether a data instance to be explained comes from a “new” region far removed from the distribution of training data? In such cases, the model may be inaccurate and the explanation may be useless. The concept of novelty is related to the concept of certainty. The higher the novelty, the more likely it is that the model will have low certainty due to lack of data.
- **Representativeness**: How many instances does an explanation cover? Explanations can cover the entire model (e.g. interpretation of weights in a linear regression model) or represent only an individual prediction (e.g. [Shapley Values](https://christophm.github.io/interpretable-ml-book/shapley.html#shapley)).

