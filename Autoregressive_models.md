# Autoregressive models
## Definitions that might help
1. Autoregressive:-
- The term autoregressive originates from the literature on time-series models where observations from the previous time-steps are used to predict the value at the current time step.
3. Bayesian Network
- It is a graphical model that represents a set of variables and their conditional dependencies using a directed acyclic graph (DAG). It is used to model probabilistic relationships among variables and allows for efficient representation and inference of joint probability distributions.
  
## Introduction
- These are statistical models used for analyzing and predicting time series data.
- We are given dataset **D** of **n** dimensional datapoints **x**. for simplicity, we will consider **x ∈ {00, 1} ^ n**.
## Representation
- A Bayesian network that makes no conditional independence assumptions is said to obey the autoregressive property.
- The distribution for the ith random variable depends on the values of all the preceding random variables in the chosen ordering x1,x2,…,xi−1
- If we allow for every conditional p(xi|x<i) to be specified in a tabular form, then such a representation is fully general and can represent any possible distribution over n random variables. However, the space complexity for such a representation grows exponentially with n and will be equals to 2^(n-1)-1. Hence, a tabular representation for the conditionals is impractical for learning the joint distribution factorized via chain rule.
### fully-visible sigmoid belief network
- To get the probability distribution of every xi we can have a function that maps the preceding random variables x1,......, xi-1 to the mean of this distribution.
- In the simplest case, we can specify the function as a linear combination of the input elements followed by a sigmoid non-linearity (to restrict the output to lie between 0 and 1). This gives us the formulation of a fully visible sigmoid belief network (FVSBN).
- fi(x1,x2,…,xi−1)=σ(α(i)0+α(i)1x1+…+α(i)i−1xi−1)
- where σ denotes the sigmoid function and θi={α(i)0,α(i)1,…,α(i)i−1} denote the parameters of the mean function. The conditional for variable i requires i parameters, and hence the total number of parameters in the model is given by ∑ni=1i=O(n^2). Note that the number of parameters is much fewer than the exponential complexity of the tabular case.
![fvsbn](https://github.com/user-attachments/assets/1b67204d-584a-44dc-ab8a-1d0bb6a851ff)
### Neural Autoregressive Density Estimator 
- To increase the expressiveness of the autoregressive generative model we can use multi-layer perceptrons.
- For example, consider the case of a neural network with 1 hidden layer. The mean function for variable i can be expressed as hi=σ(Aix<i+ci)fi(x1,x2,…,xi−1)=σ(α(i)hi+bi) where hi∈R^d denotes the hidden layer activations for the MLP and θi={Ai∈Rd×(i−1),ci∈Rd,α(i)∈Rd,bi∈R} are the set of parameters for the mean function μi(⋅). The total number of parameters in this model is dominated by the matrices Ai and given by O(d*n^2)
![nade](https://github.com/user-attachments/assets/8fa897c6-5d26-4c66-bb38-cfcd216607d3)
- The Neural Autoregressive Density Estimator (NADE) provides an alternate MLP-based parameterization that is more statistically and computationally efficient than the vanilla approach. In NADE, parameters are shared across the functions used for evaluating the conditionals. - In particular, the hidden layer activations are specified as hi=σ(W.<ix<i+c)fi(x1,x2,…,xi−1)=σ(α(i)hi+bi) where θ={W∈Rd×n,c∈Rd,{α(i)∈Rd}ni=1,{bi∈R}ni=1}is the full set of parameters for the mean functions f1(⋅),f2(⋅),…,fn(⋅).
- The weight matrix W and the bias vector c are shared across the conditionals. Sharing parameters offers two benefits: The total number of parameters gets reduced from O(d*n^2) to O(nd). The hidden unit activations can be evaluated in O(nd) time via the following recursive strategy: hi=σ(ai)ai+1=ai+W[.,i]xi with the base case given by a1=c.
### Extensions to NADE
- The **RNADE** algorithm extends NADE to learn generative models over real-valued data. Here, the conditionals are modeled via a continuous distribution such as an equi-weighted mixture of K Gaussians. Instead of learning a mean function, we know learn the means μi,1,μi,2,…,μi,K and variances Σi,1,Σi,2,…,Σi,K of the K Gaussians for every conditional. For statistical and computational efficiency, a single function gi:Ri−1→R2K outputs all the means and variances of the K Gaussians for the i-th conditional distribution
- Notice that NADE requires specifying a single, fixed ordering of the variables. The choice of ordering can lead to different models. The EoNADE algorithm allows training an ensemble of NADE models with different orderings.
## Learning and Inference
