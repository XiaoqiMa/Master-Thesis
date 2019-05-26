[TOC]

#### The Importance of Human Interpretable Machine Learning [online]

[Explainable Artificial Intelligence](<https://towardsdatascience.com/human-interpretable-machine-learning-part-1-the-need-and-importance-of-model-interpretation-2ed758f5f476>)

A **machine learning model** by itself consists of an algorithm which tries to learn latent patterns and relationships from data without hard-coding fixed rules. Hence, explaining how a model works to the business always poses its own set of challenges. There are some domains in the industry especially in the world of finance **like insurance or banking** where data scientists often end up having to use more traditional machine learning models (linear or tree-based). The reason being that model interpretability is very important for the business to explain each and every decision being taken by the model. However, this often leads to a **sacrifice in performance**. This is where complex models like ensembles and neural networks typically give us better and more accurate performance (since true relationships are rarely linear in nature). We, however, end up being unable to have proper interpretations for model decisions. To address and talk about these gaps, I will be writing a series of articles where we will explore some of these challenges in-depth about explainable artificial intelligence (XAI) and human interpretable machine learning.

*However, the harsh reality is that without a reasonable understanding of how machine learning models or the data science pipeline works, real-world projects rarely succeed*

However, being humans, logic and reasoning is something we adhere to for most of our decisions. Hence, the paradigm shift towards artificial intelligence (AI) making decisions will no doubt be questioned. There are a lot of real-world scenarios where biased models might have really **adverse effects**. This includes [*predicting potential criminals*](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing)*,* [*judicial sentencing risk scores*](https://www.propublica.org/article/making-algorithms-accountable), *credit scoring, fraud detection, health assessment, loan lending, self-driving* and many more where model understanding and interpretation is of **utmost** importance. 

When a model predicts or finds our insights, it takes certain decisions and choices. Model interpretation tries to **understand and explain these decisions taken by the response function** i.e., the what, why and how. The key to model interpretation is **transparency**, the ability to question, and the ease of understanding model decisions by humans

**specific criteria**

- **Intrinsic or post hoc?** Intrinsic interpretability is all about leveraging a machine learning model which is intrinsically interpretable in nature (like linear models, parametric models or tree based models). Post hoc interpretability means selecting and training a black box model (ensemble methods or neural networks) and applying interpretability methods after the training (feature importance, partial dependency plots). We will focus more on post hoc model interpretable methods in our series of articles.
- **Model-specific or model-agnostic?** Model-specific interpretation tools are very specific to intrinsic model interpretation methods which depend purely on the capabilities and features on a per-model basis. This can be coefficients, p-values, AIC scores pertaining to a regression model, rules from a decision tree and so on. Model-agnostic tools are more relevant to post hoc methods and can be used on any machine learning model. These agnostic methods usually operate by analyzing (and perturbations of inputs) feature input and output pairs. By definition, these methods do not have access to any model internals like weights, constraints or assumptions.
- **Local or global?** This classification of interpretation talks about if the interpretation method explains a single prediction or the entire model behavior? Or if the scope is somewhere in between? We will talk more about global and local interpretations soon



**Metrics**

- **Supervised Learning — Classification:** For classification problems, our main objective is to predict a discrete categorical response variable. The confusion matrix is extremely useful here from which we can derive a whole bunch of useful metrics including accuracy, precision, recall, F1-Score as depicted in the following example.
- **Unsupervised Learning — Clustering:** For unsupervised learning problems based on clustering we can use metrics like the [silhouette coefficient](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html), [homogeneity, completeness, V-measure](http://scikit-learn.org/stable/modules/clustering.html#homogeneity-completeness-and-v-measure) and the [Calinski-Harabaz index](http://scikit-learn.org/stable/modules/clustering.html#calinski-harabaz-index).



#### Predictive modeling

[Predictive modeling](<https://www.oreilly.com/ideas/predictive-modeling-striking-a-balance-between-accuracy-and-interpretability>)

- Try surrogate models for explaining black box models: **Surrogate models are interpretable models used as a proxy to explain black box models**. For example, fit a black box model to your training data. Then train a single decision tree on the original training data, but instead of using the actual target in the training data, use the predictions of the more complex algorithm as the target for this single decision tree. This single decision tree will likely be a more interpretable proxy you can use to explain the more complex logic of the black box model.

- Try **variable importance measures** and **partial dependency plots** for explaining black box models: Variable importance measures are available for models like gradient boosted ensembles, neural networks, and random forests. Partial dependency plots are an interpretation tool that visually depicts complex interactions between important variables in a model. Depending on your organization’s validation requirements or regulator, variable importance measures and partial dependency plots may be an acceptable documentation tool for black box models used as part of your overall analytics infrastructure

#### [Explainable ML](<https://towardsdatascience.com/explainable-artificial-intelligence-part-2-model-interpretation-strategies-75d4afa6b739>)

Hence to reinforce our motivation, we need model interpretation such that, we are able to account for **fairness** (unbiasedness/non-discriminative), **accountability** (**reliable results**) and **transparency**(being able to query and validate predictive decisions) of a predictive model.

TO READ: [Interpreting predictive models with Skater: Unboxing model opacity](<https://www.oreilly.com/ideas/interpreting-predictive-models-with-skater-unboxing-model-opacity>) 

- LIME :
  - Choose your instance of interest for which you want to have an explanation of the predictions of your black box model.
  - Perturb your dataset and get the black box predictions for these new points.
  - Weight the new samples by their proximity to the instance of interest.
  - Fit a weighted, interpretable (surrogate) model on the dataset with the variations.
  - Explain prediction by interpreting the local model.
- SHAP: Its novel components include: the identification of a new class of additive feature importance measures, and theoretical results showing there is a unique solution in this class with a set of desirable properties. Typically, SHAP values try to explain the output of a model (function) as a sum of the effects of each feature being introduced into a conditional expectation. Importantly, for non-linear functions the order in which features are introduced matters.