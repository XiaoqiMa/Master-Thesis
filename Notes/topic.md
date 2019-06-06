Regarding you master thesis topic:
The idea is to explore the effect of a specific variable in a classification or regression model through Subgroup Discovery. We would do this by creating variation/perturbation of some training or test data, i.e., we flip for example the value of the variable under consideration and the look at the outcome. Then we analyze these changes with subgroup discovery. 



- datasets?
- explore a variable or groups of variables?
- direct influence or indirect influence?
- remove variable or change variable values? obscure the dataset?
- performance measure? balanced error rate?
- subgroup discovery use libraries? 
- subgroup discovery --> search algorithms, pruning



```python
# You could use the Shannon entropy as a measure of balance
def balance(seq):
    from collections import Counter
    from numpy import log

    n = len(seq)
    classes = [(clas,float(count)) for clas,count in Counter(seq).iteritems()]
    k = len(classes)

    H = -sum([ (count/n) * log((count/n)) for clas,count in classes]) #shannon entropy
    return H/log(k)
```

1. **Use precision and recall to focus on small positive class —** When the positive class is smaller and the ability to detect correctly positive samples is our main focus (correct detection of negatives examples is less important to the problem) we should use precision and recall.
2. **Use ROC when both classes detection is equally important —** When we want to give equal weight to both classes prediction ability we should look at the ROC curve.
3. **Use ROC when the positives are the majority or switch the labels and use precision and recall —** When the positive class is larger we shoulwd probably use the ROC metrics because the precision and recall would reflect mostly the ability of prediction of the positive class and not the negative class which will naturally be harder to detect due to the smaller number of samples. If the negative class (the minority in this case) is more important, we can switch the labels and use precision and recall (As we saw in the examples above — switching the labels can change everything).

[Tactics to combat Imbalanced data](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/)