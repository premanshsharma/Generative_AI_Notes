# Questions to ask
- what is speculative decoding
-----------------------------------------

- main paper:- https://arxiv.org/pdf/2401.07851
# Problem with Large Language Models (LLMs):
  - one token is predicted, it is given input to llm and whole calculations are done again and again autoregresively because of which it takes a lot of time in inference on one token and computational power is also wasted
  - LLMs are slow because they generate text one token at a time, causing high inference latency.
```formula
Require: Language model Mq, input sequence x1,...,xt, and target sequence length T;

1: initialize n ← t
2: while n < T do
    3: Set qn+1 ← Mq(x | x<n+1)
    4: Sample xn+1 ∼ qn+1
    5: n ← n+1
6: end while

```
# Solution: Speculative Decoding:
Speculative Decoding is a new method to speed up LLM inference.
It drafts several future tokens efficiently and verifies them in parallel, reducing latency compared to the traditional method.
- How It Works:
In each step, speculative decoding drafts tokens ahead of time and checks them all together.
Only the correctly predicted tokens are kept.
- Key Observations:
Some tokens are easy to predict with fewer resources.
Memory operations (reading and writing LLM parameters) cause most of the delay.
- Research Focus:
The paper focuses on improving drafter selection (choosing how to generate the speculative tokens) and verification strategies (how to check the tokens).
