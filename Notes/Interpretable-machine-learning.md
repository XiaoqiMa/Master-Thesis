



**Model-specific or model-agnostic?** Model-specific interpretation tools are limited to specific model classes. The interpretation of regression weights in a linear model is a model-specific interpretation, since – by definition – the interpretation of intrinsically interpretable models is always model-specific. Tools that only work for the interpretation of e.g. neural networks are model-specific. Model-agnostic tools can be used on any machine learning model and are applied after the model has been trained (post hoc). These agnostic methods usually work by analyzing feature input and output pairs. By definition, these methods cannot have access to model internals such as weights or structural information.

**Local or global?** Does the interpretation method explain an individual prediction or the entire model behavior? Or is the scope somewhere in between? Read more about the scope criterion in the next section.

Local explanations can therefore be more accurate than global explanations. This book presents methods that can make individual predictions more interpretable in the [section on model-agnostic methods](https://christophm.github.io/interpretable-ml-book/agnostic.html#agnostic).



### Chapter 5 Model-Agnostic Methods

The great advantage of model-agnostic interpretation methods over model-specific ones is their flexibility. 



#### Partial Dependence Plot (PDP)

The partial dependence plot (short PDP or PD plot) shows the marginal effect one or two features have on the predicted outcome of a machine learning model. A partial dependence plot can show whether the relationship **between the target and a feature** is linear, monotonous or more complex

The partial dependence plot is a global method: The method considers all instances and gives a statement about the **global relationship of a feature with the predicted outcome**.

- PDP is a global method

#### Individual Conditional Expectation (ICE)

Individual Conditional Expectation (ICE) plots display one line per instance that shows how the instance’s prediction changes when a feature changes.

The partial dependence plot for the average effect of a feature is a global method because it does not focus on specific instances, but on an overall average. The equivalent to a PDP for individual data instances is called individual conditional expectation (ICE) plot

- A PDP is the average of the lines of an ICE plot

- Individual conditional expectation curves are **even more intuitive to understand** than partial dependence plots. One line represents the predictions for one instance if we vary the feature of interest.

  Unlike partial dependence plots, ICE curves can **uncover heterogeneous relationships**.

- Disadvantage: ICE curves **can only display one feature** meaningfully, because two features would require the drawing of several overlaying surfaces and you would not see anything in the plot.

#### Feature Importance

The importance of a feature is the increase in the prediction error of the model after we permuted the feature’s values, which breaks the relationship between the feature and the true outcome.

A feature is “important” if shuffling its values increases the model error, because in this case the model relied on the feature for the prediction. A feature is “unimportant” if shuffling its values leaves the model error unchanged, because in this case the model ignored the feature for the prediction.

In the end, you need to decide whether you want to know how much the model relies on each feature for making predictions (-> training data) or how much the feature contributes to the performance of the model on unseen data (-> test data).

- Advantages:

  - **Nice interpretation**: Feature importance is the increase in model error when the feature’s information is destroyed.

    Feature importance provides a **highly compressed, global insight** into the model’s behavior.

    A positive aspect of using the error ratio instead of the error difference is that the feature importance measurements are **comparable across different problems**.

  - Permutation feature importance **does not require retraining the model**.

- Disadvantages:

  - It is very **unclear whether you should use training or test data** to compute the feature importance.
  - The permutation feature importance depends on shuffling the feature, which adds randomness to the measurement. When the permutation is repeated, the **results might vary greatly**. Repeating the permutation and averaging the importance measures over repetitions stabilizes the measure, but increases the time of computation.



#### Local Surrogate (LIME)[<https://github.com/marcotcr/lime>]

Local surrogate models are interpretable models that are used to explain individual predictions of black box machine learning models. Local interpretable model-agnostic explanations (LIME)[37](https://christophm.github.io/interpretable-ml-book/lime.html#fn37) is a paper in which the authors propose a concrete implementation of local surrogate models. Surrogate models are trained to approximate the predictions of the underlying black box model. Instead of training a global surrogate model, LIME focuses on training local surrogate models to explain individual predictions.

- How do you get the variations of the data? This depends on the type of data, which can be either text, image or tabular data. For text and images, the solution is to turn single words or super-pixels on or off. In the case of tabular data, LIME creates new samples by perturbing each feature individually, drawing from a normal distribution with mean and standard deviation taken from the feature.

- From the figure it becomes clear that it is easier to interpret categorical features than numerical features. One solution is to categorize the numerical features into bins.

- Advantages: LIME is one of the few methods that **works for tabular data, text and images**.

  The **fidelity measure** (how well the interpretable model approximates the black box predictions) gives us a good idea of how reliable the interpretable model is in explaining the black box predictions in the neighborhood of the data instance of interest.

#### Shapley Values

The Shapley value, coined by Shapley (1953)[39](https://christophm.github.io/interpretable-ml-book/shapley.html#fn39), is a method for assigning payouts to players depending on their contribution to the total payout. Players cooperate in a coalition and receive a certain profit from this cooperation.

The average prediction for all apartments is €310,000. How much has each feature value contributed to the prediction compared to the average prediction?

- The contribution is the difference between the feature effect minus the average effect

- Advantages: The difference between the prediction and the average prediction is **fairly distributed** among the feature values of the instance – the Efficiency property of Shapley values. This property distinguishes the Shapley value from other methods such as [LIME](https://christophm.github.io/interpretable-ml-book/lime.html#lime). LIME does not guarantee that the prediction is fairly distributed among the features. The Shapley value might be the only method to deliver a full explanation.
- Disadvantages: 
  - The Shapley value requires **a lot of computing time**.
  - The Shapley value **can be misinterpreted**: The interpretation of the Shapley value is: Given the current set of feature values, the contribution of a feature value to the difference between the actual prediction and the mean prediction is the estimated Shapley value.
  - The Shapley value returns a simple value per feature, but **no prediction model** like LIME. This means it cannot be used to make statements about changes in prediction for changes in the input, such as: “If I were to earn €300 more a year, my credit score would increase by 5 points.”
  - Another disadvantage is that **you need access to the data** if you want to calculate the Shapley value for a new data instance