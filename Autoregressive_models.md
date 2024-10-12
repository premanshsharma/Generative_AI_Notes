# Autoregressive Models

## Basic Definitions

1. **Autoregressive**:
   - The term "autoregressive" refers to a type of model where the value at the current time step is predicted based on the values from previous time steps. This is mathematically represented as:

   \[
   x_t = f(x_{t-1}, x_{t-2}, \ldots, x_{t-p}) + \epsilon_t
   \]

   where \(x_t\) is the value at time \(t\), \(f\) is a function capturing the relationship, \(p\) is the number of previous observations used for prediction, and \(\epsilon_t\) is the error term.

2. **Bayesian Network**:
   - A Bayesian Network is a directed acyclic graph (DAG) representing a set of variables and their conditional dependencies. If we have random variables \(X_1, X_2, \ldots, X_n\), the joint probability distribution can be expressed as:

   \[
   P(X_1, X_2, \ldots, X_n) = \prod_{i=1}^n P(X_i | \text{Parents}(X_i))
   \]

   This allows us to model complex relationships efficiently.

---

## Introduction

Autoregressive models are crucial for analyzing and predicting time-series data. In this context, we work with a dataset \(D\) consisting of \(n\)-dimensional data points \(x\), where \(x \in \{0, 1\}^n\).

---

## Representation of Autoregressive Models

A Bayesian Network without conditional independence assumptions adheres to the autoregressive property. For the \(i\)th random variable, the distribution depends on all preceding random variables:

\[
P(x_i | x_{<i}) = P(x_i | x_1, x_2, \ldots, x_{i-1})
\]

If we store the probabilities in a tabular format, the space complexity becomes:

\[
\text{Space Complexity} = 2^{(n-1)} - 1
\]

This exponential growth makes tabular representations impractical for large \(n\).

---

### Fully-Visible Sigmoid Belief Network (FVSBN)

To model the probability distribution of each \(x_i\), we can define a function that maps the preceding variables to the mean of the distribution:

\[
f_i(x_1, x_2, \ldots, x_{i-1}) = \sigma(\alpha_i^0 + \alpha_i^1 x_1 + \ldots + \alpha_i^{i-1} x_{i-1})
\]

Where:
- \(\sigma\) is the sigmoid function:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

- \(\theta_i = \{\alpha_i^0, \alpha_i^1, \ldots, \alpha_i^{i-1}\}\) are the parameters of the mean function.

The total number of parameters for the entire model is:

\[
\text{Total Parameters} = \sum_{i=1}^n i = O(n^2)
\]

This is significantly more efficient than the exponential complexity of the tabular case.

![fvsbn](https://github.com/user-attachments/assets/1b67204d-584a-44dc-ab8a-1d0bb6a851ff)

---

### Neural Autoregressive Density Estimator (NADE)

To enhance expressiveness, we can utilize **multi-layer perceptrons (MLPs)**. The mean function for variable \(x_i\) can then be expressed as:

\[
h_i = \sigma(A_i x_{<i} + c_i)
\]
\[
f_i(x_1, x_2, \ldots, x_{i-1}) = \sigma(\alpha_i h_i + b_i)
\]

Where:
- \(h_i\) are the activations of the hidden layer for the MLP.
- \(A_i \in \mathbb{R}^{d \times (i-1)}\) is a weight matrix, \(c_i \in \mathbb{R}^d\) is a bias vector, and \(\alpha_i \in \mathbb{R}^d\) is the parameter vector.

The total number of parameters is dominated by the matrices \(A_i\) and is given by:

\[
\text{Total Parameters} = O(d \cdot n^2)
\]

![nade](https://github.com/user-attachments/assets/8fa897c6-5d26-4c66-bb38-cfcd216607d3)

---

### Parameter Sharing in NADE

The **Neural Autoregressive Density Estimator (NADE)** improves efficiency by sharing parameters across the different conditionals. This means that the same weight matrix \(W\) and bias vector \(c\) are used for all conditionals, leading to:

\[
h_i = \sigma(W \cdot x_{<i} + c)
\]

The shared parameters reduce the total number of parameters to:

\[
\text{Total Parameters} = O(nd)
\]

This leads to significant improvements in both memory efficiency and computational speed. The activations can be computed recursively as follows:

1. For the first activation:

   \[
   a_1 = c
   \]

2. For subsequent activations:

   \[
   a_{i+1} = a_i + W[.,i] x_i
   \]

---

### Extensions to NADE

- **RNADE**: The RNADE algorithm extends NADE to generative modeling over real-valued data. Here, we model each conditional distribution as a mixture of \(K\) Gaussians:

   \[
   P(x_i | x_{<i}) = \sum_{k=1}^K w_k \cdot \mathcal{N}(\mu_{i,k}, \sigma_{i,k}^2)
   \]

- **EoNADE**: The EoNADE algorithm allows for training an ensemble of NADE models with different orderings, enhancing flexibility and performance.

By allowing variable ordering, we can capture more intricate dependencies within the data, leading to improved generative capabilities.

---

This README provides a detailed mathematical understanding of autoregressive models and their extensions, making it suitable for users looking to delve deeper into this topic.
