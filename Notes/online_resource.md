https://towardsdatascience.com/interpretable-machine-learning-1dec0f2f3e6b

1. Permutation Importance
What features does a model think are important? Which features might have a greater impact on the model predictions than the others? This concept is called feature importance and Permutation Importance is a technique used widely for calculating feature importance. It helps us to see when our model produces counterintuitive results, and it helps to show the others when our model is working as we’d hope.

Permutation Importance works for many scikit-learn estimators. The idea is simple: Randomly permutate or shuffle a single column in the validation dataset leaving all the other columns intact. A feature is considered “important” if the model’s accuracy drops a lot and causes an increase in error. On the other hand, a feature is considered ‘unimportant’ if shuffling its values don’t affect the model’s accuracy.


Permutation Importance is calculated using the **ELI5** library. ELI5 is a Python library which allows to visualize and debug various Machine Learning models using unified API. It has built-in support for several ML frameworks and provides a way to explain black-box models.

```
import eli5
from eli5.sklearn import PermutationImportance
perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())

```


Interpretation

The features at the top are most important and at the bottom, the least. For this example, goals scored was the most important feature.
The number after the ± measures how performance varied from one-reshuffling to the next.
Some weights are negative. This is because in those cases predictions on the shuffled data were found to be more accurate than the real data.

2. Partial Dependence Plots
The partial dependence plot (short PDP or PD plot) shows the marginal effect one or two features have on the predicted outcome of a machine learning model( J. H. Friedman 2001). PDPs show how a feature affects predictions. PDP can show the relationship between the target and the selected features via 1D or 2D plots.

The library to be used for plotting PDPs is called python partial dependence plot toolbox or simply PDPbox.

```
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots
# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=my_model, dataset=val_X, model_features=feature_names, feature='Goal Scored')
# plot it
pdp.pdp_plot(pdp_goals, 'Goal Scored')
plt.show()

```

3. SHAP Values
SHAP which stands for SHapley Additive exPlanation, helps to break down a prediction to show the impact of each feature. It is based on Shapley values, a technique used in game theory to determine how much each player in a collaborative game has contributed to its success¹. Normally, getting the trade-off between accuracy and interpretability just right can be a difficult balancing act but SHAP values can deliver both.

SHAP values are calculated using the Shap library which can be installed easily from PyPI or conda.



Machine Learning doesn’t have to be a black box anymore. What use is a good model if we cannot explain the results to others. Interpretability is as important as creating a model. To achieve wider acceptance among the population, it is crucial that Machine learning systems are able to provide satisfactory explanations for their decisions. 

**As Albert Einstein said,” If you can’t explain it simply, you don’t understand it well enough”.**



<https://github.com/pbiecek/DALEX/>

Machine Learning models are widely used and have various applications in classification or regression tasks. Due to increasing computational power, availability of new data sources and new methods, ML models are more and more complex. Models created with techniques like boosting, bagging of neural networks are true black boxes. It is hard to trace the link between input variables and model outcomes. They are use because of high performance, but lack of interpretability is one of their weakest sides.

Today we are surrounded by complex predictive algorithms used for decision making. Machine learning models are used in health care, politics, education, judiciary and many other areas. Black box predictive models have far larger influence on our lives than physical robots. Yet, applications of such models are left unregulated despite many examples of their potential harmfulness. See *Weapons of Math Destruction* by Cathy O'Neil for an excellent overview of potential problems.

It's clear that we need to control algorithms that may affect us. Such control is in our civic rights. Here we propose three requirements that any predictive model should fulfill.

- **Prediction's justifications**. For every prediction of a model one should be able to understand which variables affect the prediction and how strongly. Variable attribution to final prediction.
- **Prediction's speculations**. For every prediction of a model one should be able to understand how the model prediction would change if input variables were changed. Hypothesizing about what-if scenarios.
- **Prediction's validations** For every prediction of a model one should be able to verify how strong are evidences that confirm this particular prediction.

There are two ways to comply with these requirements. One is to use only models that fulfill these conditions by design. White-box models like linear regression or decision trees. In many cases the price for transparency is lower performance. The other way is to use approximated explainers – techniques that find only approximated answers, but work for any black box model. Here we present such techniques.



https://appsilon.com/please-explain-black-box/

**Feature importance analysis** offers first good insights into what the model is learning and what factors might be important. However, this technique can be unreliable if features are correlated. It can provide good insights only if model variables are interpretable. For many [GBMs](https://towardsdatascience.com/boosting-algorithm-gbm-97737c63daa3) libraries it’s fairly easy to generate [feature importance plots](https://www.r-bloggers.com/variable-importance-plot-and-variable-selection/).

No wonder that when in 2016 [**LIME** (Local Interpretable Model-Interpretable Explanations) ](https://arxiv.org/abs/1602.04938)paper was presented at NIPS conference it had a huge impact. The idea behind LIME is to locally approximate a black-box model with an easier to understand white-box model constructed on interpretable input data. It has proven great results providing [interpretation for image classification](https://www.oreilly.com/learning/introduction-to-local-interpretable-model-agnostic-explanations-lime) and [text](https://christophm.github.io/interpretable-ml-book/lime.html#lime-for-text). However, for tabular data, it’s difficult to find interpretable features and their local interpretation might be misleading.

Another promising idea is [SHAP (Shapley Additive Explanations)](https://arxiv.org/abs/1705.07874). It’s based on game theory. It assumes that features are players, models are coalitions and Shapley values tell how to fairly distribute the “payout” among the features. This technique distributes the effects fairly, is easy to use and offers visually compelling implementation.



<https://www.kdnuggets.com/2018/06/human-interpretable-machine-learning-need-importance-model-interpretation.html>

<https://towardsdatascience.com/human-interpretable-machine-learning-part-1-the-need-and-importance-of-model-interpretation-2ed758f5f476>

**The field of Machine Learning has gone through some phenomenal changes over the last decade.** 

There are some domains in the industry especially in the world of finance like insurance or banking where data scientists often end up having to use more traditional machine learning models (linear or tree-based). The reason being that model interpretability is very important for the business to explain each and every decision being taken by the model. However, this often leads to a sacrifice in performance. This is where complex models like ensembles and neural networks typically give us better and more accurate performance (since true relationships are rarely linear in nature). We, however, end up being unable to have proper interpretations for model decisions

**human interpretable machine learning**

**Interpretability** also popularly known as **human-interpretable interpretations (HII)** of a machine learning model is the extent to which a human (including non-experts in machine learning) can understand the choices taken by models in their decision-making process (the how, why and what).

**The Importance of Model Interpretation**

When tackling machine learning problems, data scientists often have a tendency to fixate on model performance metrics like accuracy, precision and recall and so on (This is important no doubt!). This is also prevalent in most online competitions around data science and machine learning. However, metrics only tell a part of the story of a model’s predictive decisions. Over time, the performance might change due to model concept drift caused by various factors in the environment. Hence, it is of **paramount** importance to understand what drives a model to take certain decisions.

**Global Interpretations**

This is all about trying to understand **“How does the model make predictions?”** and **“How do subsets of the model influence model decisions?”.** To comprehend and interpret the whole model at once, we need global interpretability. Global interpretability is all about being able to explain and understand model decisions based on conditional interactions between the dependent (response) variable(s) and the independent (predictor) features on the complete dataset. Trying to understand feature interactions and importances is always a good step towards understanding global interpretation. Of course, visualizing features after more than two or three dimensions becomes quite difficult when trying to analyze interactions. Hence, often looking at modular parts and subsets of features, which might influence model predictions on a global knowledge, helps. Complete knowledge of the model structure, assumptions and constraints are needed for a global interpretation.

**Local Interpretations**

This is all about trying to understand **“Why did the model make specific decisions for a single instance?”** and **“Why did the model make specific decisions for a group of instances?”**. For local interpretability, we do not care about the inherent structure or assumptions of a model and we treat it as a black box. For understanding prediction decisions for a single datapoint, we focus specifically on that datapoint and look at a local subregion in our feature space around that point, and try to understand model decisions for that point based on this local region. Local data distributions and feature spaces might behave completely different and give more accurate explanations as opposed to global interpretations. The Local Interpretable Model-Agnostic Explanation (LIME) framework is an excellent method which can be used for model-agnostic local interpretation. We can use a combination of global and local interpretations to explain model decisions for a group of instances.

**The major reasons for doing this is to really unbox the ‘black-box’ model’s behavior and prediction decisions. This in turn helps us understand how our model works better**















