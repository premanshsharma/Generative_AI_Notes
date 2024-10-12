# Autoregressive Models

## Basic Definitions

1. **Autoregressive**:
   - Autoregressive models use previous observations to predict the current value. For example, in time series, the current value \(x_t\) might depend on previous values \(x_{t-1}, x_{t-2}, \ldots, x_{t-p}\).

2. **Bayesian Network**:
   - A Bayesian Network is a graphical model that represents a set of variables and their conditional dependencies using a directed acyclic graph (DAG). It helps in modeling the joint probability distribution of variables.

---

## Introduction

Autoregressive models are important for analyzing and predicting time series data. We work with a dataset \(D\) containing \(n\)-dimensional data points \(x\), where each \(x\) can be either 0 or 1.

---

## Representation of Autoregressive Models

A Bayesian Network without independence assumptions follows the autoregressive property. For the \(i\)th random variable, the distribution depends on all previous variables:

- **Mathematical Representation**: 
  - \(P(x_i | x_{<i}) = P(x_i | x_1, x_2, \ldots, x_{i-1})\)

If we use a tabular format for storing probabilities, the space complexity is:

- **Space Complexity**: 
  - \(2^{(n-1)} - 1\)

This exponential growth makes tabular representations impractical for large \(n\).

---

### Fully-Visible Sigmoid Belief Network (FVSBN)

To model the probability of each \(x_i\), we define a function that maps previous variables to the mean of the distribution:

- **Mean Function**: 
  - \(f_i(x_1, x_2, \ldots, x_{i-1}) = \sigma(\alpha_i^0 + \alpha_i^1 x_1 + \ldots + \alpha_i^{i-1} x_{i-1})\)

Where \(\sigma\) is the sigmoid function. The total number of parameters in this model is:

- **Total Parameters**: 
  - \(O(n^2)\)

This is more efficient than the tabular representation.

---

### Neural Autoregressive Density Estimator (NADE)

To increase the model's expressiveness, we can use multi-layer perceptrons (MLPs):

- **Mean Function with MLP**: 
  - \(h_i = \sigma(A_i x_{<i} + c_i)\)

Where \(A_i\) is a weight matrix and \(c_i\) is a bias vector. The total number of parameters is dominated by the matrices:

- **Total Parameters**: 
  - \(O(d \cdot n^2)\)

---

### Parameter Sharing in NADE

In NADE, parameters are shared across conditionals, which reduces the total number of parameters:

- **Shared Parameters**: 
  - \(O(nd)\)

This sharing allows for efficient computation of hidden unit activations.

---

### Extensions to NADE

- **RNADE**: This algorithm models generative models for real-valued data, using a mixture of \(K\) Gaussians for conditionals.
- **EoNADE**: This allows training multiple NADE models with different variable orderings, capturing complex dependencies.

---

This README provides an overview of autoregressive models and their extensions, suitable for readers who want to understand the concepts clearly.
