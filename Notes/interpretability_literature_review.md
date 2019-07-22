[TOC]

### A Unified Approach to Interpreting Model Predictions

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