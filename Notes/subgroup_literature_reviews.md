[TOC]

### 1. Fast Subgroup Discovery for Continuous Target Concepts

- Introduction: We first focus on pruning strategies for reducing the search space utilizing optimistic estimate functions for obtaining up- per bounds for the possible quality of the discovered patterns.
- Subgroup discovery: Specifially, these interesting subgroups should have the most unusual (distributional) characteristics with respect to the concept of interest given by the target variable 
- Quality function for binary variables: ![image-20190506200906581](/Users/xiaoqi/Library/Application Support/typora-user-images/image-20190506200906581.png)

qWRACC (weighted relative accuracy) trades off the increase in the target share p vs. the generality (n) of the subgroup

- while continuous target variables require averages/aggregations of values, e.g., the mean. As equivalents to the quality functions for binary targets dis- cussed above, we consider the functions Continuous Piatetsky-Shapiro (qCPS), Contin- uous LIFT (qCLIFT), and Continuous Weighted Relative Accuracy (qCWRACC)
- ![image-20190506201006865](/Users/xiaoqi/Library/Application Support/typora-user-images/image-20190506201006865.png)

- optimistic estimate for each quality function
- Conclusion: In this paper, we propose novel formalizations of tight optimistic estimates for numeric
  quality functions. In contrast, SD-Map* is applicable for binary, categorical, and continuous target concepts.



### 2. Subgroup Discovery – Advanced Review

- Introduction: Standard subgroup discovery approaches commonly focus on a single target concept as the property of interest (66; 80; 60), while the quality func- tion framework also enables multi-target concepts
- Interestingness: “interestingness is defined as distributional unusual- ness with respect to a certain property of interest”
- **Binary and Nominal Target Quality Functions**
  ![image-20190506201730229](/Users/xiaoqi/Library/Application Support/typora-user-images/image-20190506201730229.png)

- **Numeric Target Quality Functions**: The target shares tP, t0 of the subgroup and the general population, are re- placed by the mean values of the target variable mP,m0, respectively
  ![image-20190506201747586](/Users/xiaoqi/Library/Application Support/typora-user-images/image-20190506201747586.png)

- **Multi-Target Quality Functions**: A more general framework for multi-target quality functions is given by exceptional model mining (81): It tries to identify interesting patterns with respect to a local model derived from a set of attributes. The interestingness can be defined, e.g., by a signifi- cant deviation from a model that is derived from the total population or the respective complement set of instances within the population
  - e. g., as measured by the Pearson correlation coefficient
- **Beam search**: For heuristic approaches, commonly a beam search (89) strategy is used because of its efficiency. The search starts with a list of subgroup hypotheses of size w (correspond- ing to the beam width), which may be initially empty. The w subgroup hypotheses contained in the beam are then expanded iteratively, and only the best w expanded subgroups are kept implementing a hill-climbing greedy search

### 3. An overview on subgroup discovery: foundations and applications

- Subgroup discovery is a data mining technique which extracts interesting rules with respect to a target variable. subgroup discovery is somewhere halfway between supervised and unsupervised learning
- Definition: In subgroup discovery, we assume we are given a so-called population of individuals (objects, customer, ...) and a property of those individuals we are interested in. The task of subgroup discovery is then to discover the subgroups of the population that are statistically “most interesting”, i.e. are as large as possible and have the most unusual statistical (distributional) characteristics with respect to the property of interest.
- Main elements in a subgroup discovery
  - Type ofthe target variable
    - Binary analysis. The variables have only two values (True or False), and the task is focused on providing interesting subgroups for each of the possible values
    - Nominal analysis. The target variable can take an undetermined number of values, but the philosophy for the analysis is similar to the binary, to find subgroups for each value.
    - Numeric analysis. This type is the most complex because the variable can be studied different ways such as dividing the variable in two ranges with respect to the average, discretisising the target variable in a determined number of intervals [91], or searching for significant deviations of the mean among others
  - Description language
  - Quality measures
  - Search strategy

- Quality measures: **complexity, generality, precision, and interest**

  - generality: 

    - Coverage: It measures the percentage of examples covered on average [85]. This can be computed as
    - Support: It measures the frequency of correctly classified examples covered by the rule [85]. This can be computed as:

  - precision:

    - Confidence: It measures the relative frequency of examples satisfying the complete rule among those satisfying only the antecedent. This can be computed with different expres- sions, e.g
    - Precision measure Qg: It measures the tradeoff of a subgroup between the number of examples classified perfectly and the unusualness of their distribution [70]. This can be computed as

  - interest: 

    - Interest: It measures the interest of a rule determined by the antecedent and consequent [93]. It can be computed as
    - Novelty: This measure is able to detect unusual subgroups [108]. It can be computed as:
    - Significance: This measure indicates the significance of a finding, if measured by the likelihood ratio of a rule

  - hybrid: because subgroup discovery attempts to obtain a tradeoff between generality, interest and precision in the results obtained. The different quality measures used can be found below

    - Sensitivity: This measure is the proportion of actual matches that have been classified correctly [70]. It can be computed as

    - Specificity: It measures the proportion of negative cases incorrectly classified [70]. It can be computed as

    - Unusualness: This measure is defined as the *weighted relative accuracy* of a rule [81]. It can be computed as:

      ![image-20190506203314782](/Users/xiaoqi/Library/Application Support/typora-user-images/image-20190506203314782.png)

- Conclusions: 
  - An important problem to address is to determine which quality measures are more adapted both to evaluating the subgroups discovered and to guiding the search process
  - Another issue to be dealt with in more depth is the scalability of the subgroup discovery algorithms

### Exceptional Model Mining 

- author: Arno Knobbe1, Ad Feelders2, and Dennis Leman

- objective: We have stated that the objective is to find subgroups where a model fitted
  to the subgroup is substantially different from that same model fitted to the entire database.
- In regular subgroup discovery, only the y attribute is used, which is typically binary. Well-known examples of quality measures for binary targets are frequency, confidence, χ2, and novelty
- Subgroup discovery [3] is a data mining framework aimed at discovering patterns that satisfy a number of user-specified inductive constraints. Constrains are: 
  - an interestingness constraint ϕ(p) ≥ t
  - minimum support threshold n ≥ minsup
  - complexity of the pattern p
- Model class:
  - correlation model
    - Absolute difference between correlations (ϕabs)
    - Entropy (ϕent)
    - Significance of correlation difference (ϕscd): As a quality measure we take 1 minus the computed p-value so that ϕscd ∈ [0, 1], and higher values indicate a more interesting subgroup
  - regression model
    - Significance of Slope Difference (ϕssd): If n + ¯n ≥ 40 the t-statistic is quite accurate, so we should be confident to use it unless we are analysing a very small dataset
  - Classification Models
    - Logistic Regression.
    - DTM classifier (Decision Table Majority)
    - BDeu score: The BDeu score ϕBDeu is a measure from Bayesian theory [2] and is used to estimate the performance of a classifier on a subgroup, with a penalty for small contingencies that may lead to overfitting
    - Hellinger (ϕHel). : this measure is aimed at producing subgroups for which the conditional distribution of y is substantially different from its conditional distri- bution in the overall database

