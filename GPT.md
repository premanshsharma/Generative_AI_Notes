- GPT - Generative Pretrained Transformer
- It is used for generating and understanding human-like text.
- It is based on Transformer and processes language using layers of attention mechanisms and mathematical functions.

- During Training:
  - The model sees a sequence of words and computes logits for each possible next word.
  - The logits are transformed into probabilities using the softmax function.
  - The model compares the predicted probabilities with the true word (from the training data) and computes the loss using the negative log likelihood (NLL), derived from MLE.
  - The model updates its parameters to reduce the NLL loss. This process helps the model improve its logits generation over time, so the predicted probabilities are closer to the true probabilities.
- During Inference (Prediction):
  - The model takes an input sequence, generates logits, and applies softmax to get a probability distribution over the vocabulary for the next word.
  - The word with the highest probability is chosen (or sampled) as the predicted next word.
 
Here’s how GPT generates text autoregressively:

Start with an Input Sequence:

The input could be a single word, a sentence, or a longer text sequence. The model takes this input and processes it.
Self-Attention with Causal Masking:

GPT uses self-attention to allow each word (or token) to attend to previous tokens in the sequence.
Causal masking (or autoregressive masking) is applied to ensure that the model can only "see" tokens that come before the current token, preventing it from looking at future tokens.
This masking ensures that when predicting the next word, GPT doesn't "cheat" by seeing the ground truth for the future words.
Logit Generation:

GPT generates logits (raw, unnormalized scores) for every token in the vocabulary, which represent the model’s "confidence" in each word being the next token.
Softmax to Convert Logits to Probabilities:

These logits are passed through a softmax function, which turns them into a probability distribution over the entire vocabulary. Each word in the vocabulary gets assigned a probability based on how likely it is to be the next word.
Sample or Select the Next Word:

Based on the probability distribution, the model either:
Selects the word with the highest probability (greedy decoding), or
Samples from the probability distribution (stochastic decoding, which introduces randomness and helps generate more diverse text).
Update the Input with the Predicted Word:

The predicted word is then added to the input sequence. The model now has a longer input: the original input plus the newly generated word.
Repeat the Process:

The model repeats this process, predicting the next word based on the expanded input sequence (which now includes all previously generated words), and continues generating words one by one until it produces a complete output.


Example of Autoregressive Text Generation in GPT
Let’s walk through an example:

Initial Input: Suppose the input is the word "The".
The model generates logits for all possible next words in the vocabulary (e.g., "cat", "dog", "sky", etc.).
After applying softmax, the word with the highest probability is selected or sampled, say, "cat".
Updated Input: Now the input becomes "The cat".
The model again generates logits for the next word. Let’s say it predicts "is".
Further Predictions: Now the input is "The cat is".
The model predicts the next word, perhaps "sleeping".
Continue: This process continues until the model predicts an end-of-sequence token (or a stopping criterion is met).


# Work flow of GPT
1. Input Text and Tokenization
You start with an input sequence of text. For example, let's say you provide the input "The cat is".
This text input is tokenized into smaller units (tokens). In GPT, tokenization is typically done using Byte Pair Encoding (BPE), which splits words into subword units. For example, "The cat is" might be tokenized into individual tokens like ["The", "cat", "is"].
2. Embedding Layer
Each token is mapped to a word embedding (a high-dimensional vector) using a learned embedding matrix. This gives each token a numerical representation that the model can work with. The embeddings are of a fixed size (e.g., 768-dimensional for GPT-2 small).

For example, after embedding:

"The" → 
𝑒
1
e 
1
​
  (an embedding vector)
"cat" → 
𝑒
2
e 
2
​
 
"is" → 
𝑒
3
e 
3
​
 
Positional encodings are then added to these embeddings to provide information about the position of each token in the sequence, since the Transformer architecture doesn't have any inherent sense of order.

3. Self-Attention Layer (with Causal Masking)
After embedding, GPT applies the self-attention mechanism (specifically masked self-attention in GPT).

The self-attention mechanism computes how each token in the input sequence relates to other tokens, allowing the model to capture dependencies between words. It does this by:

Creating queries (Q), keys (K), and values (V) from the token embeddings. These are computed as linear projections of the embedding vectors.
Calculating an attention score using the dot product of queries and keys, then applying a softmax to get a weighted sum of values. This operation allows each token to attend to other tokens in the sequence.
In GPT, causal masking is applied, which means that the model can only attend to tokens that came before it (or itself) in the sequence. This is crucial for autoregressive generation, as the model shouldn’t see future tokens during prediction.

For example, when processing the token "is", the model can only attend to "The" and "cat", but not to any future tokens (because they haven't been generated yet).
4. Feedforward Network
After the self-attention step, the resulting attention output for each token is passed through a feedforward neural network.

The feedforward network consists of two fully connected (dense) layers with a non-linearity (like GELU or ReLU) in between. This helps the model learn more complex patterns and relationships in the data.

For each token in the sequence, the feedforward network processes its attended output.

5. Logits Generation (Output of the Decoder)
The final output of the feedforward network for each token is a vector that represents that token's contextual meaning based on the attention mechanism and the previous tokens.

The last step is to compute logits—raw scores for the next possible token:

GPT projects the output of the feedforward network to the size of the vocabulary using a linear layer. This produces a vector of logits, one for each possible token in the vocabulary.
For example, after processing the input "The cat is", the model produces a set of logits for the next token (e.g., it might assign high logits to "sleeping", "running", "on", etc.).

6. Softmax for Next Token Prediction
These logits are then passed through a softmax function, which converts the logits into a probability distribution over the vocabulary. The softmax ensures that the output is a set of probabilities that sum to 1.

For instance, after softmax:

The probability of "sleeping" might be 0.6.
The probability of "on" might be 0.3.
The probability of "happy" might be 0.05, and so on.
7. Selecting or Sampling the Next Token
GPT then selects or samples the next token from this probability distribution:
If the model is set to greedy decoding, it will pick the token with the highest probability (e.g., "sleeping").
If the model is using sampling (often combined with techniques like temperature or top-k sampling), it might randomly select a token based on the probability distribution, which introduces more diversity into the text.
8. Autoregressive Process: Repeat the Steps
Once the next token (e.g., "sleeping") is generated, it is appended to the input sequence.
The input becomes "The cat is sleeping", and the model repeats the process to predict the next token.
This autoregressive process continues, with the model predicting one token at a time based on the previously generated tokens, until the model produces an end-of-sequence token or reaches a stopping criterion.
