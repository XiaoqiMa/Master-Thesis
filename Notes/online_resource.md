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



As Albert Einstein said,” If you can’t explain it simply, you don’t understand it well enough”.



















