# Generative_AI_Notes

## Reference:- 

https://deepgenerativemodels.github.io/notes/index.html

## Basic Definitions that might Help

1. Parametric models vs. non-parametric models:-
   
  1.1 Parametric Models:
   
    1.1.1 Assumption: Assume a specific distribution (e.g., normal, linear).
    
    1.1.2 Parameters: Fixed number of parameters.
    
    1.1.3 Advantages: Simple, fast, works with small datasets.
    
    1.1.4 Disadvantages: Inflexible, can perform poorly if assumptions are wrong.
    
    1.1.5 Examples: Linear regression, logistic regression, Naive Bayes.

  2.2 Non-Parametric Models:
  
    1.2.1 Assumption: No assumption about data distribution.
    
    1.2.2 Parameters: The number of parameters grows with data size.
    
    1.2.3 Advantages: Flexible, handles complex data patterns.
    
    1.2.4 Disadvantages: Requires more data, computationally expensive.
    
    1.2.5 Examples: K-Nearest Neighbors, decision trees, kernel density estimation.

2. Inference:-
   
  2.1  It is the process of drawing conclusions or generating new data points based on a learned model. 

3. Discriminative Model:-

  3.1 It is a type of model that focuses on modeling the boundary or decision rule that separates different classes in the data. 
  
## Introduction
- We can think of any observed data (D) as a finite set of samples from an underlying distribution p_data.
- Any generative model's goal is to approximate this data distribution given access to dataset D. 
The hope is that if we can learn a good generative model, we can use the learned model for downstream inference. 
- Our primary focus is to focus on a probability distribution that can be described using a finite set of parameters that can summarize all the information about dataset D. 
- As compared to non-parametric models, parametric models work better with large dataset but are limited in the family of distribution they can represent. 
- Our goal is to get a model such that it minimizes the distance between the model distribution and data distribution
- given dataset of images D, and the goal is to learn the parameters of a generative model θ
 within a model M such that the model distribution pθ is close to the data distribution data with distance function d(⋅)

  minθ∈Md(pdata,pθ)

- An image with 700 x 1400 pixels with 3 channels has a total of 10 ^ 800000 images and the largest public image dataset is 15 million images so it is impossible to learn from such a small dataset.

## Inference
- The generative model has to learn joint distribution over the entire data.
- Inference Queries:-
  - Density estimation: Given a datapoint x, what is the probability assigned by the model, i.e., pθ(x)
  - Sampling: How can we generate novel data from the model distribution, i.e., xnew∼pθ(x)
  - Unsupervised representation learning: How can we learn meaningful feature representations for a data point x
