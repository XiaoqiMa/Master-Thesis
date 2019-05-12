



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