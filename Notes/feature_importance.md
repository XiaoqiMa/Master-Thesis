https://xgboost.readthedocs.io/en/latest/python/python_api.html?highlight=xgbclassifier#xgboost.XGBClassifier

Get feature importance of each feature. **Importance_type** can be defined as:

- ‘weight’: the number of times a feature is used to split the data across all trees.
- ‘gain’: the average gain across all splits the feature is used in.
- ‘cover’: the average coverage across all splits the feature is used in.
- ‘total_gain’: the total gain across all splits the feature is used in.
- ‘total_cover’: the total coverage across all splits the feature is used in.

Feature importance is defined only for tree boosters

Feature importance is only defined when the decision tree model is chosen as base learner (booster=gbtree). It is not defined for other base learner types, such as linear learners (booster=gblinear).



https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier

- **importance_type** (*string**,* *optional* *(**default='split'**)*) – The type of feature importance to be filled into `feature_importances_`. If ‘split’, result contains numbers of times the feature is used in a model. If ‘gain’, result contains total gains of splits which use the feature.

[Beware Default Random Forest Importances](https://explained.ai/rf-importance/index.html)

### Default feature importance mechanism

The most common mechanism to compute feature importances, and the one used in scikit-learn's [RandomForestClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) and [RandomForestRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html), is the *mean decrease in impurity* (or *gini importance*) mechanism (check out the [Stack Overflow conversation](https://stackoverflow.com/questions/15810339/how-are-feature-importances-in-randomforestclassifier-determined)). The mean decrease in impurity importance of a feature is computed by measuring how effective the feature is at reducing uncertainty (classifiers) or variance (regressors) when creating decision trees within RFs. The problem is that this mechanism, while fast, does not always give an accurate picture of importance. Breiman and Cutler, the inventors of RFs, [indicate](https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#varimp) that this method of “*adding up the gini decreases for each individual variable over all trees in the forest gives a **fast** variable importance that is **often very consistent** with the permutation importance measure.*” (Emphasis ours and we'll get to permutation importance shortly.)

We've known for years that **this common mechanism for computing feature importance is biased**; i.e. it tends to inflate the importance of continuous or high-cardinality categorical variables For example, in 2007 Strobl *et al* pointed out in [Bias in random forest variable importance measures: Illustrations, sources and a solution](https://link.springer.com/article/10.1186%2F1471-2105-8-25) that “*the variable importance measures of Breiman's original Random Forest method ... are not reliable in situations where potential predictor variables vary in their scale of measurement or their number of categories*.” That's unfortunate because not having to normalize or otherwise futz with predictor variables for Random Forests is very convenient.

### Permutation importance

Breiman and Cutler also described *permutation importance*, which measures the importance of a feature as follows. Record a baseline accuracy (classifier) or R2 score (regressor) by passing a validation set or the out-of-bag (OOB) samples through the Random Forest. Permute the column values of a single predictor feature and then pass all test samples back through the Random Forest and recompute the accuracy or R2. The importance of that feature is the difference between the baseline and the drop in overall accuracy or R2 caused by permuting the column. The permutation mechanism is much more computationally expensive than the mean decrease in impurity mechanism, but the results are more reliable. The permutation importance strategy does not require retraining the model after permuting each column; we just have to re-run the perturbed test samples through the already-trained model.

```python
def permutation_importances(rf, X_train, y_train, metric):
baseline = metric(rf, X_train, y_train)
imp = []
for col in X_train.columns:
save = X_train[col].copy()
X_train[col] = np.random.permutation(X_train[col])
m = metric(rf, X_train, y_train)
X_train[col] = save
imp.append(baseline - m)
return np.array(imp)
```

Permutation importance is pretty efficient and generally works well, but Strobl *et al* show that “*permutation importance over-estimates the importance of correlated predictor variables.*” in [Conditional variable importance for random forests](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-9-307). It's unclear just how big the bias towards correlated predictor variables is, but there's a way to check.

### Drop-column importance

If we ignore the computation cost of retraining the model, we can get the most accurate feature importance using a brute force *drop-column importance* mechanism. The idea is to get a baseline performance score as with permutation importance but then drop a column entirely, retrain the model, and recompute the performance score. The importance value of a feature is the difference between the baseline and the score from the model missing that feature. This strategy answers the question of how important a feature is to overall model performance even more directly than the permutation importance strategy.

If we had infinite computing power, the drop-column mechanism would be the default for all RF implementations because it gives us a “ground truth” for feature importance. We can mitigate the cost by using a subset of the training data, but drop-column importance is still extremely expensive to compute because of repeated model training. Nonetheless, it's an excellent technique to know about and is a way to test our permutation importance implementation. The importance values could be different between the two strategies, but the order of feature importances should be roughly the same.

```python
def dropcol_importances(rf, X_train, y_train):
    rf_ = clone(rf)
    rf_.random_state = 999
    rf_.fit(X_train, y_train)
    baseline = rf_.oob_score_
    imp = []
    for col in X_train.columns:
        X = X_train.drop(col, axis=1)
        rf_ = clone(rf)
        rf_.random_state = 999
        rf_.fit(X, y_train)
        o = rf_.oob_score_
        imp.append(baseline - o)
    imp = np.array(imp)
    I = pd.DataFrame(
            data={'Feature':X_train.columns,
                  'Importance':imp})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=True)
    return I
```



### Dealing with collinear features

From these experiments, it's safe to conclude that permutation importance (and mean-decrease-in-impurity importance) computed on random forest models spreads importance across collinear variables. The amount of sharing appears to be a function of how much noise there is in between the two. We do not give evidence that correlated, rather than duplicated and noisy variables, behave in the same way. On the other hand, one can imagine that longitude and latitude are correlated in some way and could be combined into a single feature. Presumably this would show twice the importance of the individual features.



### Summary

The takeaway from this article is that the most popular RF implementation in Python (scikit) and R's RF default importance strategy do not give reliable feature importances when “*... potential predictor variables vary in their scale of measurement or their number of categories*.” (Strobl *et al*). Rather than figuring out whether your data set conforms to one that gets accurate results, simply use permutation importance. You can either use our Python implementation ([rfpimp](https://github.com/parrt/random-forest-importances/tree/master/src) via `pip`) or, if using R, make sure to use `importance=T` in the Random Forest constructor then `type=1` in R's `importance()` function.

Finally, we'd like to recommend the use of permutation, or even drop-column, importance strategies for all machine learning models rather than trying to interpret internal model parameters as proxies for feature importances.

