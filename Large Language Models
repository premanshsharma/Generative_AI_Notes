# Large Language Models(LLM)

1. Architecture ---------------------- Most of acdemia
2. Training algorithm/loss ----------- Most of acdemia
3. Data------------------------------- 
4. Evaluation
5. Systems
 

## Language Modeling
- LM:- probability distribution of over sequences of tokens/words p(x1, x2, ......., xL) -> gives sematic  knowledge
- LMs are generative models : X<sub>1:L</sub> ~ p(x1, ...., xL)
- Autoregressive(AR) Language models:- chain rule of probability
  - p(x1, .........xL) = p(x1)p(x2|x1)p(x3|x1, x2)...... = $$\prod_{i=1}^{n} a_i$$ p(xi|x1:i-1)
  - Task:- predict next word. 
  - Steps:-
    1. tokenize
    2. forward
    3. predict probability of next token
    4. sample ---------------------------------------------- Inference
    5. detokenize ------------------------------------------ Inference
