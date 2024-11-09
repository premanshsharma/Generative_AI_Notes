```math
PE_{(pos, 2i)} = \sin \left( \frac{pos}{10000^{2i/d}} \right)

```

# Transformer Architecture

1. **Encoder-Decoder Structure**:  
   A transformer consists of two main parts: an encoder and a decoder. The encoder processes the input sequence, and the decoder generates the output sequence based on the encoder's output.

2. **Embedding Layer**:  
   Before the input is fed to the encoder, we pass it through an **embedding layer**, which converts the tokens into continuous vector representations (embeddings). These embeddings serve as the input for the transformer.

3. **Positional Encoding**:  
   Since transformers do not inherently have a sense of token order (unlike RNNs or CNNs), we add **positional encoding** to the embeddings to give the model information about the position of each token in the sequence. The positional encoding values are derived from **sine** and **cosine** functions because they provide a smooth and periodic pattern that captures relative position information across different sequence lengths.

4. **Encoder Layer**:  
   The encoder has multiple layers, each consisting of two main components:
   - **Self-attention**: The encoder attends to the input sequence to create a contextualized representation of each token.
   - **Feedforward neural network**: After self-attention, the output is passed through a position-wise feedforward network (usually consisting of a fully connected layer with ReLU activation).
   Each encoder layer also includes layer normalization and residual connections.

5. **Attention Mechanism**:  
   The core of the transformer is the **attention mechanism**, specifically **scaled dot-product attention**. This involves three key components:
   - **Query (Q)**: Represents what we are looking for (a certain relationship or feature).
   - **Key (K)**: Contains the encoded features or summary of the knowledge.
   - **Value (V)**: Contains the detailed information corresponding to the keys.

   The attention mechanism works as follows:
   - Compute the dot product of **Query** and **Key** to get a similarity score that tells us how much attention to give to each value.
   - Scale the result by dividing it by the square root of the dimension of the keys, which helps in stabilizing the gradients.
   - Apply the **softmax** function to these scores to convert them into a probability distribution.
   - Multiply the result by the **Value** to get a weighted sum, where tokens with higher attention scores contribute more to the final output.

6. **Multi-head Attention**:  
   Instead of performing a single attention operation, transformers use **multi-head attention**. This means that multiple attention mechanisms (heads) run in parallel, each learning different aspects of the relationships between tokens. The results from all attention heads are concatenated and passed through a linear layer to combine the information.

7. **Normalization and Residual Connection**:  
   After the attention layer, the output is passed through a **Layer Normalization** and is added to the input of the attention layer (a residual connection). This helps stabilize training and facilitates gradient flow.

8. **Decoder Layer**:  
   The decoder also has several layers, and each layer consists of:
   - **Masked self-attention**: Similar to the encoder's attention, but with masking to ensure the decoder can only attend to previous tokens and not future ones (important for autoregressive tasks like language generation).
   - **Encoder-decoder attention**: In this step, the decoder attends to the encoder’s output, which allows the decoder to use the context from the entire input sequence when generating the output.

   In the decoder, we compute attention between the decoder's current state and the encoder's output. We apply **Query** (Q) and **Key-Value** (K, V) transformations on the encoder's output (X_encoder) and decoder's current state (X_decoder) to establish this relationship.

9. **Final Output Layer**:  
   The final output of the transformer consists of logits, which are unnormalized predictions for each token in the vocabulary. These logits are passed through a **softmax** function to convert them into a probability distribution, from which we can predict the next token or generate the output sequence.









# 1. Transformer
- In transformer we have first have encoder layer and tehn we have decoder layer.
- Before encoder we have an embedding layer which gives us the encoded vector of our input
- We add positional encoder to our previous encoding to get a sense of position of every token. positional encoding values are taken from sin and cosin function because they provide a vide range of values. 
- Whatever we input to the transformer layer encoder extracts the information from input and gives it to decoder layer.
- Based on output of encoder layer, decoder layer generates the output.
- First we need to under stand Attention layer
- In attention layer first we have Query, keys and values.
  - Query is what are we looking for, keys are the keywords or lets say summarised information of our knowledge, and values are detialed knowledge.
  - First we multiply Q with K to get values that tells us how much attention should we give to each value. 
  - we divide our Q*K with root of key vectors dimentions to get better smooth gradients.
  - we pass this Q*K through softmax to get their probabilities then multiply with V. If Q*K represent less value that means we have to give less attention to that parrticular value. so when multiplied with V final encoded vector will have values for every token in our input representing how much attention we should give to our input.
- we take multiple attention heads in parallel to get different encoded vectors with different attention.
- Output_attention = concatnate(atention1, attention2, .....)
- normalise the output attention = normalized layer = LayerNorm(x_input+output_attention)
- Now we flatten our layer, now this is new input and we pass this to our encoder again.
- Like this we do multiple times
- To extract information of given token we need to give attention to previous and future tokens also.

- Decoder layer
- Now output of encoding layer is input of decoder, now we have add masking M to Q*V/dimension^0.5. this M is -ve of infinite for future values so that decoder gives attention to previous values only.
- other attention processes are same
- now we do Q = Xdecoder attWQ, V = XencoderWV, K = XencoderWK, where Xdecoder att is the output of masked self-attention from previous step, Xencoder is the output of encoder
- this gives a sense of relation between encoder output and current decoder output.
- stack many decoder layers

- Now in final ouptut layer we have a vector for all the tokens in our direcctory. these are called logits. these are output of transformer, when we pass these logits through softmax function we get probability of occurance of a given token. this is how transofrmer works
