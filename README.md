# Generative_AI_Notes
### General Definitions that might Help
1. Parametric models vs. non-parametric models:- 
  1.1 Parametric models are models that make assumptions about the underlying form of the data distribution. These models are defined by a set of parameters, and once those parameters are estimated, the entire model is specified. The data distribution is typically summarized by a small, fixed number of parameters.
Ex:- Linear Regression, Logistic Regression, Naive Bayes

   1.2 Non-parametric models do not assume any specific form for the data distribution. Instead, they are more flexible and can adapt to the structure of the data, often requiring more data to build accurate models. The number of parameters grows with the size of the dataset, making them more complex.
Ex:- kNN, Decision Trees, SVM

## Learning
We can think of any observed data (D) as a finite set of samples from an underlying distribution p_data.

Any generative model's goal is to approximate this data distribution given access to dataset D. 
The hope is that if we can learn a good generative model, we can use the learned model for downstream inference. 

Our primary focus is to focus on a probability distribution that can be described using a finite set of parameters that can summarize all the information about dataset D. 

# Reference:- https://deepgenerativemodels.github.io/notes/index.html
