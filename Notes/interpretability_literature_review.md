[TOC]

### A Unified Approach to Interpreting Model Predictions (Scott 2017)

- abstract: SHAP assigns each feature an importance value for a particular prediction. there is a unique solution in this class with a set of desirable properties.
- introduction: However, the growing availability of big data has increased the benefits of using complex models, so bringing to the forefront the trade-off between accuracy and interpretability of a model’s output. (game theory results guaranteeing a unique solution apply)
- Additive Feature Attribution Methods
  - we must use a simpler explanation model, which we define as any interpretable approximation of the original model
  - ![image-20190708124938937](/Users/xiaoqi/Library/Application Support/typora-user-images/image-20190708124938937.png)
  - LIME: The LIME method interprets individual model predictions based on locally approximating the model around a given prediction
  - DeepLIFT: DeepLIFT was recently proposed as a recursive prediction explanation method for deep learning [8, 7]. It attributes to each input xi a value C∆xi∆y that represents the effect of that input being set to a reference value as opposed to its original value. 
  - Classic Shapley Value Estimation
  - ![image-20190708125204644](/Users/xiaoqi/Library/Application Support/typora-user-images/image-20190708125204644.png)
- Model-Agnostic Approximations
  - Kernel SHAP (Linear LIME + Shapley values)
- Model-Specific Approximations
  - Linear SHAP
  - Max SHAP
  - Deep SHAP (DeepLIFT + Shapley values)



### Learning Important Features Through Propagating Activation Differences Avanti

- abstract: DeepLIFT (Deep Learning Important FeaTures), a method for decomposing the output prediction of a neural network on a specific input by backpropagating the contributions of all neurons in the network to every feature of the input. DeepLIFT

- introduction: In contrast to most gradient-based methods, using a difference-from-reference allows DeepLIFT to propagate an importance signal even in situations where the gradi- ent is zero and avoids artifacts caused by discontinuities in the gradient. 
- DeepLIFT assigns contribution scores C∆xi∆t to ∆xi
  - ![image-20190708162457023](/Users/xiaoqi/Library/Application Support/typora-user-images/image-20190708162457023.png)

- Reference: When formulating the DeepLIFT rules described in Sec- tion 3.5, we assume that the reference of a neuron is its activation on the reference input. The choice of a reference input is critical for obtaining insightful results from DeepLIFT. measuring differences against?”. For MNIST, we use a ref- erence input of all-zeros as this is the background of the im- ages.
- Briefly, the Shapely values measure the average marginal effect of including an input over all possible orderings in which inputs can be included. If we define “including” an input as setting it to its actual value instead of its reference value, DeepLIFT can be thought of as a fast approximation of the Shapely values.



### An Efficient Explanation of Individual Classifications using Game Theory

- In our example, the contributions of the three feature values can be interpreted as follows. The
  prior probability of a Titanic passenger’s survival is 32% and the model predicts a 67% chance of
  survival. The fact that this passenger was female is the sole and largest contributor to the increased
  chance of survival. Being a passenger from the third class and an adult both speak against survival,
  the latter only slightly. The actual class label for this instance is ”yes”, so the classification is
  correct. This is a trivial example, but providing the end-user with such an explanation on top of a
  prediction, makes the prediction easier to understand and to trust. The latter is crucial in situations where important and sensitive decisions are made
- So, *why do we even need a general explanation method?* It is not difficult to think of a rea-
  sonable scenario where a general explanation method would be useful. For example, imagine a
  user using a classifier and a corresponding explanation method. At some point the model might
  be replaced with a better performing model of a different type, which usually means that the explanation method also has to be modified or replaced. The user then has to invest time and effort
  into adapting to the new explanation method. This can be avoided by using a general explanation
  method. Overall, a good general explanation method reduces the dependence between the user-end
  and the underlying machine learning methods, which makes work with machine learning models
  more user-friendly. This is especially desirable in commercial applications and applications of ma-
  chine learning in fields outside of machine learning, such as medicine, marketing, etc. An effective
  and efficient general explanation method would also be a useful tool for comparing how a model
  predicts different instances and how different models predict the same instance
- there exist two other general explanation methods for explaining a model’s prediction: the work by Robnik-Sˇ ikonja and Kononenko (2008) and the work by Lemaire et al. (2008): **A feature value’s contribution is defined as the difference between the model’s initial prediction and its average prediction across perturbations of the corresponding feature.**
- 

$$
\phi_{j}(v a l)=\sum_{S \subseteq\left\{x_{1}, \ldots, x_{p}\right\} \backslash\left\{x_{j}\right\}} \frac{|S| !(p-|S|-1) !}{p !}\left(\operatorname{val}\left(S \cup\left\{x_{j}\right\}\right)-\operatorname{val}(S)\right)
$$


$$
\sum_{j=1}^{p} \phi_{j}=\hat{f}(x)-E_{X}(\hat{f}(X))
$$

$$
\phi_{j}(v+w) = \phi_{j}(v) + \phi_{j}(w)
\\ where (v+w)(S) = v(S) + w(S)
$$

$$
\begin{aligned} f\left(h_{x}\left(z^{\prime}\right)\right) &=E\left[f(z) | z_{S}\right] \\ &=E_{z_{\overline{S}} | z_{S}}[f(z)] \\ & \approx E_{z_{\overline{S}}}[f(z)] \\ & \approx f\left(\left[z_{S}, E\left[z_{\overline{S}}\right]\right]\right) \end{aligned}
$$

### “Why Should I Trust You? (Marco 2016)

- abstract: 
  - Understanding the reasons behind predictions is, however, quite important in assessing trust, which is fundamental if one plans to take action based on a prediction, or when choosing whether to deploy a new model. 
  - In this work, we propose LIME, a novel explanation technique that explains the predictions of any classifier in an in- terpretable and faithful manner
  - We show the utility of explanations via novel experiments, both simulated and with human subjects, on various scenarios that require trust: deciding if one should trust a prediction, choosing between models, improving an untrustworthy classifier, and identifying why a classifier should not be trusted
- Introduction:
  - how much the human understands a model’s behaviour, as opposed to seeing it as a black box
  - Inspecting individual predictions and their explanations is a worthwhile solution, in addition to such metrics. 
  - LIME, an algorithm that can explain the predictions of any classifier or regressor in a faithful way, by approximating it locally with an interpretable model.
  - By“explaining a prediction”, we mean presenting textual or visual artifacts that **provide qualitative understanding of the relationship** between the instance’s components (e.g. words in text, patches in an image) and the model’s prediction. 
  - Sneeze and headache **are portrayed as** contributing to the “flu” prediction, while “no fatigue” is evidence against it.
  - provide qualitative understanding between the input variables and the response.  **layman**
  - Any choice of interpretable representations and G will
    have some inherent drawbacks. First, while the underlying model can be treated as a black-box, certain interpretable representations will not be powerful enough to explain certain behaviors. For example, a model that predicts sepia-toned images to be retro cannot be explained by presence of absence of super pixels. Second, our choice of G (sparse linear models) means that if the underlying model is highly non-linear even in the locality of the prediction, there may not be a faithful explanation. However,
- Experiment:
  - we generate explanations and compute the fraction of these gold features that are recovered by the explanations. 
  - We observe that the greedy approach is **comparable** to parzen on logistic regression, but is substantially worse on decision trees since ...
- Conclusion: We proposed LIME, a modular and extensible ap- proach to faithfully explain the predictions of any model in an interpretable manner.



```python
"""Explains predictions on tabular (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    def __data_inverse(self,
                       data_row,
                       num_samples):
        """Generates a neighborhood around a prediction.

        For numerical features, perturb them by sampling from a Normal(0,1) and
        doing the inverse operation of mean-centering and scaling, according to
        the means and stds in the training data. For categorical features,
        perturb by sampling according to the training distribution, and making
        a binary feature that is 1 when the value is the same as the instance
        being explained.

        Args:
            data_row: 1d numpy array, corresponding to a row
            num_samples: size of the neighborhood to learn the linear model

        Returns:
            A tuple (data, inverse), where:
                data: dense num_samples * K matrix, where categorical features
                are encoded with either 0 (not equal to the corresponding value
                in data_row) or 1. The first row is the original instance.
                inverse: same as data, except the categorical features are not
                binary, but categorical (as the original data)
        """
      data = self.random_state.normal(
                0, 1, num_samples * num_cols).reshape(
                num_samples, num_cols)
      if self.sample_around_instance:
          data = data * scale + instance_sample
      else:
          data = data * scale + mean
          
 """Generates a neighborhood around a prediction.

        Generates neighborhood data by randomly removing words from
        the instance, and predicting with the classifier. Uses cosine distance
        to compute distances between original and perturbed instances.
```



### Explaining prediction models and individual predictions with feature contributions (Erik 2014)

- abstract: Its advantage over existing general methods is that all subsets of input features are perturbed, so interactions and redundancies between features are taken into account
- Introduction:
  - For additive models, a prediction is the sum of individualmarginal effects, whichmakes such visualizations a tool for graphical computation of predictions—a nomogram
  - A prediction is explained by assigning to each feature a number which denote its influence
  - If the situational importance is positive, then the feature has a positive contribution (increases
  - we proposed a general method for computing the situational importance for classifi- cation and, separately, regression models that dealt with the aforementioned shortcomings of existing general explanation methods.  **essential background** from
  - We considered two improvements that increase the efficiency of the approximation algorithm. First, the approximation algorithm is a form of Monte Carlo integration.
- Experiment:
  - First, we preform a detailed analysis of running times across several well-known real-world data sets and artificial data sets using several different types ofmachine learning models.
  - We also included several well-known regression and classification data sets: autoMpg, bodyfat, concrete, elevators, fishcatch, fruitfly, hous- ing, machinecpu, pollution, stock, wine, and wisconsin(regression), anneal, breastCancerLJ, hepatitis, iris, monks1, monks2, monks3, mushroom, nursery, soybean, and zoo
  - We used the following procedure to measure the benefits of ...
- Conclusion:
  - By design, the method perturbs all subsets of features to deal with the shortcomings of other existing general methods that do not properly take into account interactions between features.
  - We also proposed two enhancements to the sampling algorithm (quasi-random and adap-
    tive sampling) that reduce the running time of the algorithm.



### Model-Agnostic Interpretability of Machine Learning (Marco 2016)

- abstract: 
  - Understanding why machine learning models behave the way they do empowers both system designers and end-users in many ways: in mode selection, feature engineering, in order to trust and act upon the predictions, and in more intuitive user interfaces.
  - these approaches provide crucial flexibility in the choice of models, explanations, and representations, improving debugging, comparison, and interfaces for a variety of users and models
- Introduction
  - is of utmost importance for
  - the model’s utility may decrease due to cognitive overhead. In contrast, if one uses model-agnostic explanations, switching the underlying model for a new one is trivial, while the way in which the explanations are presented is maintained.
  - Global explanation, however, are often either not interpretable, or too simplistic to represent the original model. LIME’s focus on explaining individual predictions allows more accurate explanations while retaining model flexibility
- Conclusion: 
  - We argued that model-agnostic explanation systems provide a generic framework for interpretability that allows for flexibility in the choice of models, representations, and the user expertise

### Permutation importance: a corrected feature importance measure (André 2010)

- abstract: 
  - In life sciences, interpretability of machine learning models is as important as their prediction accuracy
  - Recently, it has been observed that RF models are biased in such a way that categorical variables with a large number of categories are preferred
- Introduction:
  - the most frequently applied methods for quantifying feature importance are linear models and decision trees.
  - Decision trees are suitable for finding non-linear prediction rules
    that are also interpretable, although their instability and lack of smoothness have been a cause of concern (Hastie et al., 2001)
  - The author of RF proposes two measures for feature ranking, the variable importance (VI) and Gini importance (GI)
  - The method normalizes the biased measure based on a permutation test and returns significance P-values for each feature.
  - we argue that our method can be used in combination with any learning method that provides feature ranking, because it assigns significance P-values to each variable, which improves model interpretability.
  - The VI of a feature is computed as the average decrease in model accuracy on the OOB samples when the values of the respective feature are randomly permuted. The GI uses the decrease of Gini index (impurity) after a node split as a measure of feature relevance.

### A peek into the black box: exploring classifiers by randomization

- Introduction:

  - In summary, the main contributions of this paper are...

  - an empirical investigation is presented in which the proposed algorithm is used to analyze 15 different classifiers generated from two synthetic datasets with known structure and 26 datasets from the UCI Machine Learning Repository (Bache and Lichman 2014).

    Bache K, Lichman M (2014) UCI machine learning repository. http://archive.ics.uci.edu/ml

  - ![image-20190903103441250](/Users/xiaoqi/Library/Application Support/typora-user-images/image-20190903103441250.png)

  

- Experiments
  - a number of experiments were performed using
  - Experimental setup: 1. classifiers 2. experimental procedure 3. synthetic datasets 4. UCI datasets
  - Experimental results: 1. synthetic datasets 2. UCI datasets 3. stability 

### Auditing Black-box Models by Obscuring Features

- introduction: 
  - The output of our audit will be a ranking of features based on their influence on the outcome (this goal will be developed more precisely in Section 2).
  - that the presence or absence of the feature alone, and its impact on the model’s accuracy, should determine its importance
  - Any decrease in accuracy can then be attributed to the feature we eliminated.
- Experiments: 
  - we will consider experiments on the following five data sets, chosen to balance easy replicability with demonstration on domains where these techniques are of practical interest.
  - **Adult Income and German Credit data**. We will consider two data sets from the UC Irvine machine learning repository2 that are of frequent use in the algorithmic fairness literature, comparing fairness (obscurity, in our context) to accuracy. The first of those is the Adult Income data set which consists of 48,842 people, each with 14 descriptive attributes from the US census and a classification of that person as making more or less than $50,000 per year. We use the training / test split given in the original data. The second is the German Credit data set consisting of 1000 people, each with 20 descriptive attributes and a classification as having good or bad credit. We use a random two-thirds / one-third training / test split on this data and the two data sets described below.

### Datasets

![image-20190903104422495](/Users/xiaoqi/Library/Application Support/typora-user-images/image-20190903104422495.png)

![image-20190903105723117](/Users/xiaoqi/Library/Application Support/typora-user-images/image-20190903105723117.png)

![image-20190903105834338](/Users/xiaoqi/Library/Application Support/typora-user-images/image-20190903105834338.png)

- Ideally these would be linked to domain experts who have agreed to be employed to evaluate methods when applied to their domain of expertise. Just as there are now large open repositories for problems in classification, regression, and reinforcement learning [Blake and Merz, 1998, Brockman et al., 2016, Vanschoren et al., 2014], we advocate for the creation of repositories that contain problems corresponding to real-world tasks in which human-input is required. Creating such repositories will be more challenging than creating collections of standard machine learning datasets because they must include a system for human assessment, but with the availablity of crowdsourcing tools these technical challenges can be surmounted.
- Datasets We carry out quantitative evaluation on three classification settings: (i) MNIST digit recognition, (ii) benchmark UCI datasets [13] and (iii) Propublica’s COMPAS Recidivism Risk Score datasets.1 In addition, we provide some qualitative results on CIFAR10 [10] in the supplement (§A.5). The COMPAS data consists of demographic features labeled with criminal *recidivism* (“relapse”) risk scores produced by a private company’s proprietary algorithm, currently used in the Criminal Justice System to aid in bail granting decisions. Propublica’s study showing racial-biased scores sparked a flurry of interest in the COMPAS algorithm both in the media and in the fairness in machine learning community [25, 7]. Details on data pre-processing for all datasets are provided in the supplement.

### Comprehensible Classification Models (Alex A. Freitas)

- Clearly, the comprehensibility of a model depends strongly (and subjectively)on the actual “contents” of the model, i.e. the attributes in the decision tree, the attribute-value conditions in classification rules, the parent attributes of each attribute in a Bayesian network classifier, etc.
- To cite from [20]: “Humans by nature are mentally opposed to too simplistic representations of complex relations.”
- We favor instead a more principled approach for coping with the accuracy- comprehensibility trade-off, such as multi-objective optimization based on Pareto dominance [25], [38] or lexicographic optimization [25], [42].



### Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead (Cynthia Rudin)

- abstract: but trying to explain black box models, rather than creating models that are inter- pretable in the first place, is likely to perpetuate bad practice and can potentially cause great harm to society.
- Introduction:
  - The lack of transparency and accountability of predictive models can have (and has already had) severe consequences;
  - where a second (post hoc) model is created to explain the first black box model.
- Key issues with explainable ML
  - an explanation is a separate model that is supposed to replicate most of the behaviour of a black box (for example, ‘the black box says that people who have been delinquent on current credit are more likely to default on a new loan’). 
  - **It is a myth that there is necessarily a trade-off between accuracy and interpretability.** There is a widespread belief that more complex models are more accurate, meaning that a complicated black box is necessary for top predictive performance. However, this is often not true, particularly when the data are structured, with a good repre- sentation in terms of naturally meaningful features. 
  - blind belief in the myth of the accuracy–interpretability trade-off.
  - Worse, toolkits of ML algorithms offer little in the way of useful interfaces for interpretable ML methods.
- **Explainable ML methods provide explanations that are not faith- ful to what the original model computes**
  -  Explanations must be wrong. They cannot have perfect fidelity with respect to the original model. 
  - l. Rather than producing explanations that are faithful to the original model, they show trends in how predictions are related to the features. Calling these ‘summaries of predictions’, ‘summary statistics’ or ‘trends’ rather than ‘explanations’ would be less misleading.
  - In the next section, we completely switch gears. We will discuss reasons why so many people appear to advocate for black box mod- els with separate explanation models, rather than inherently inter- pretable models
  - a possible counter- factual explanation might be ‘your loan application was denied, but if you had $1,000 less debt, you would have qualified for the loan’.
- **Black box models seem to uncover ‘hidden patterns’.** 
  - The fact that many scientists have difficulty constructing interpretable models may be fueling the belief that black boxes have the ability to uncover subtle hidden patterns in the data about which the user was not pre- viously aware. 
  - This proposal will not solve all problems, but it could at least rule out companies selling recidivism prediction models, possibly credit scoring models and other kinds of models where we can construct accurate yet interpretable alternatives. 
  - As it turns out, for classification of natural images, domain experts often direct our attention to different parts of the image and explain why these parts of the image were important in their reasoning process.