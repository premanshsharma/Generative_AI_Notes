# Transformer Architecture

-  let input sequence length be '**t**'
1. **Encoder-Decoder Structure**:  
   A transformer consists of two main parts: an encoder and a decoder. The encoder processes the input sequence, and the decoder generates the output sequence based on the encoder's output.

2. **Embedding Layer**:  
   - Before the input is fed to the encoder, we pass it through an **embedding layer**, which converts the tokens into continuous vector representations (embeddings). These embeddings serve as the input for the transformer.
   - **d<sub>embedding</sub>** is the size of these embeddings, for example, d<sub>embedding</sub> = 512, and each token in the input sequence will be represented by a 512-dimension vector. 

```math
Input Embedding Shape=(t,d_{(embedding)})
```
4. **Positional Encoding**:  
   Since transformers do not inherently have a sense of token order (unlike RNNs or CNNs), we add **positional encoding** to the embeddings to give the model information about the position of each token in the sequence. The positional encoding values are derived from **sine** and **cosine** functions because they provide a smooth and periodic pattern that captures relative position information across different sequence lengths. 
```math
PE_{(pos, 2i)} = \sin \left( \frac{pos}{10000^{2i/d}} \right)
```
```math
PE_{(pos, 2i+1)} = \cos \left( \frac{pos}{10000^{2i/d}} \right)
```
```math
\text{Position Encoding shape} = (t,d_{(embedding)})
```
- We add positional encoding of dimension

```math
\text{Input}_{\text{final}} = \text{Embedding}(x) + \text{Positional Encoding}
```
- final dimensions of the input layer
```math
\text{Input Embedding Shape} = (t,d_{(embedding)})
```
5. **Encoder Layer**:  
   The encoder has multiple layers, each consisting of two main components:
   - **Self-attention**: The encoder attends to the input sequence to create a contextualized representation of each token.
   - **Feedforward neural network**: After self-attention, the output is passed through a position-wise feedforward network (usually consisting of a fully connected layer with ReLU activation).
   Each encoder layer also includes layer normalization and residual connections.

6. **Attention Mechanism**:  
   The core of the transformer is the **attention mechanism**, specifically **scaled dot-product attention**. This involves three key components:
   - **Query (Q)**: Represents what we are looking for (a certain relationship or feature).
   - **Key (K)**: Contains the encoded features or summary of the knowledge.
   - **Value (V)**: Contains the detailed information corresponding to the keys.
```math
\text{Shape of Q, K, V} = (t,d_{(embedding)})
```
   The attention mechanism works as follows:
   - Compute the dot product of **Query** and **Key** to get a similarity score that tells us how much attention to give to each value.
   - Scale the result by dividing it by the square root of the dimension of the keys, which helps in stabilizing the gradients.
   - Apply the **softmax** function to these scores to convert them into a probability distribution.
   - Multiply the result by the **Value** to get a weighted sum, where tokens with higher attention scores contribute more to the final output.
     
```math
\text{score} = QK^T
```
```math
\text{scaled score} = \frac{QK^T}{\sqrt{d_embedding}}
```
```math
\text{attention weights} = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right)
```
6. **Multi-head Attention**:  
   Instead of performing a single attention operation, transformers use **multi-head attention**. This means that multiple attention mechanisms (heads) run in parallel, each learning different aspects of the relationships between tokens. The results from all attention heads are concatenated and passed through a linear layer to combine the information.

```math
\text{output} = \text{attention weights} \times V
```
```math
\text{Multi-head Output} = \text{concat}(head_1, head_2, ..., head_h)W^O
```

7. **Normalization and Residual Connection**:  
   After the attention layer, the output is passed through a **Layer Normalization** and is added to the input of the attention layer (a residual connection). This helps stabilize training and facilitates gradient flow.
```math
\text{output}_{\text{norm}} = \text{LayerNorm}(x_{\text{input}} + \text{output}_{\text{attention}})
```

8. **Decoder Layer**:  
   The decoder also has several layers, and each layer consists of:
   - **Masked self-attention**: Similar to the encoder's attention, but with masking to ensure the decoder can only attend to previous tokens and not future ones (important for autoregressive tasks like language generation).
   - **Encoder-decoder attention**: In this step, the decoder attends to the encoder’s output, which allows the decoder to use the context from the entire input sequence when generating the output.

   In the decoder, we compute attention between the decoder's current state and the encoder's output. We apply **Query** (Q) and **Key-Value** (K, V) transformations on the encoder's output (X_encoder) and decoder's current state (X_decoder) to establish this relationship.
```math
Q = X_{\text{decoder}}W_Q, \quad K = X_{\text{encoder}}W_K, \quad V = X_{\text{encoder}}W_V
```

9. **Final Output Layer**:  
   The final output of the transformer consists of logits, which are unnormalized predictions for each token in the vocabulary. These logits are passed through a **softmax** function to convert them into a probability distribution, from which we can predict the next token or generate the output sequence.

```math
\text{probability} = \text{softmax}(\text{logits})
```

## Key Points to Note:
- Positional Encoding allows the model to understand the order of tokens in the sequence, which is important for NLP tasks.
- Self-attention is used in both the encoder and decoder to allow each token to focus on other tokens in the sequence, capturing relationships.
- Masked Self-Attention in the decoder ensures that the model doesn't "cheat" by peeking at future tokens when generating outputs.
- Multi-head Attention enables the model to learn different relationships in parallel.
- The final logits are used to generate a distribution over all possible tokens, typically used in tasks like translation, text generation, etc.


# BERT
- it is an encoder only model.
- 
