# Definitions for references
- **Vocabulary** refers to the set of unique words or tokens the model recognizes and uses in its processing tasks. **Vocab size** is the number of unique tokens
- **Logits**
  - Definition: Logits are the raw, unnormalized scores output by a model before applying any activation function.
  - Use in Classification: Typically used in classification tasks, where each class has its own logit value.
  - Mathematical Transformation:
    - For binary classification: Use the sigmoid function to convert the logit into a probability
    - For multiclass classification: Use the softmax function to convert logits into probabilities and the sum of exponentials ensures probabilities.
    - Cross-Entropy Loss: Logits are passed to loss functions like cross-entropy directly for better numerical stability.
  - Example:
  - Logits for 3 classes: [2.3,−1.7,0.5]
  - After softmax transformation, probabilities might look like: [0.76,0.05,0.19]
  - Here, the first class is predicted with 76% probability.

# Mind Map
- Transformer Architecture
- Training
  - Loss Function:- Cross-Entropy Loss
  - Optimizer:- Adam Optimizer, Learning Rate Schedule
  - Backpropagation
- Application
  - Machine Translation
  - Text Summarization
  - Text Generation
  - Language Modeling
  - Speech Recognition
  - Vision Tasks (e.g., ViT)
- Variants
  - BERT (Bidirectional Encoder Representations from Transformers)
    - Encoder-only architecture
    - Pre-training (Masked Language Modeling, Next sentence prediction)
  - GPT (Generative Pre trained Transformer)
    - Decoder only architecture
    - Causal language modeling
  - T5 (Text to text transfer transformer)
    - Encoder-Decoder architecture
    - Unified text-to-text framework
  - ViT (Vision Transformer)
    - Adapting transformer for vision task
   
# Transformer Architecture
## Mind Map
```math
Attention(Q, K, V) = softmax (\frac{Q.K^T}{\sqrt{dk}}).V
```
- Input
  - Tokenization
  - Embedding
- Encoder
  - Multi-Head Self-Attention
    - Scaled Dot-Product Attention 
      - Queries, Keys, Values
      -  Attention Scores: Attention Formula
      - Softmax Multiple Heads to capture different relationships
    - Multi Head Attention
      - Concatenating the attention of multiple heads
      - Linear transformations per head
  - Add and Normalize
    - Residual Connections
    - Layer Normalization
  - Feed-Forward Network
    - Two fully connected layers
  - Position Encoding
    - Adding positional information to tokens
    - Sinusoidal or learned
- Decoder
  - Masked Multi-Head Self-Attention
    - Similar to the encoder, but with masking to prevent future tokens from attending
  - Encoder-Decoder Attention
    - Attention between the encoder's output and the decoder's input
  - Add and Normalize
    - Residual connections and normalization
  - Feed-Forward Network
    - Same as in the encoder
- Output Layer
  - Linear Transformation
  - Softmax

## Input
- This layer converts discrete tokens i.e. words, subwords, or characters into continuous vectors
that the model can process. 
### Mindmap
- Tokenization
- Embedings
- Positional Encoding
### Tokenization
- The input text is first split into tokens
- These tokens are then mapped to integer indices based on vocabulary
- Example:-
  - Sentence:- "The cat sat on the mat,"
  - Tokens: [The, cat, sat, on, the, mat]
  - Mapping from vocab: {"The": 0, "cat": 1, "sat": 2, "on": 3, "the": 4, "mat": 5}
  - Tokenized sequence: [0, 1, 2, 3, 4, 5]
### Embedings
- Here we convert these token indices into dense vectors that the Transformer can process.
- **Embedding Matrix E** has a shape V x d<sub>model</sub>
  - V is the size of vocabulary
  - d<sub>model</sub> is the dimensionality of the embedding vectors (the vector space in which each token will be represented)
- **E** is learned during training and contains a vector for each token in vocab.
- For each token t<sub>i</sub> there is a corresponding embedding vector e<sub>i</sub> that is fetched from the matrix:
  - e<sub>i</sub>=**E**[t<sub>i</sub>]
- Input sentence in form of embedding vector looks like **X<sub>embed</sub> = [e1, e2, ........, en]**
### Positional Encoding
- Transformers don’t have a built-in sense of the order of words like RNNs do, because they process all words simultaneously (not sequentially). So, we give each word some extra information to indicate its position in the sequence. This is called positional encoding. It helps the model understand where each word is located in a sentence.
- These encodings are learned or fixed values (e.g., sinusoidal functions) that help the model understand the order of words in the sequence.
- **Mathematical Explanation:**
  -  We use sine and cosine functions to assign different positions a unique encoding.
  -  For each position **pos** in a sentence and each embedding dimension i, the positional encoding is calculated as:
    - PE<sub>pos, 2i</sub> = sin(pos/10000<sup>2i/d<sub>model</sub></sup>)
    - PE<sub>pos, 2i+1</sub> = cos(pos/10000<sup>2i/d<sub>model</sub></sup>)
    - where
      - pos is the position of the word in the sequence.
      - i is the dimension index.
      - d<sub>model</sub> is the size of the word embedding.
      - Shape of PE is n x d<sub>model</sub>
    - The sine function is used for even indices, and the cosine function is used for odd indices.
    - This encoding produces a vector for each position in the sequence. The sine and cosine functions with different frequencies allow the model to distinguish between different positions and capture the relative distances between tokens.
    - **Why Sinusoidal?**
      - The use of sine and cosine functions allows the positional encoding to generalize well to sequences longer than those seen during training. The periodicity of these functions ensures that the model can infer positions it has never encountered before based on the patterns in the encodings.
- X<sub>input</sub> = X<sub>embed</sub> + PE

### **Final Input to transformer** = X<sub>input</sub> of shape n x d<sub>model</sub>

## Encoder
- The encoder in the Transformer architecture is responsible for processing the input sequence and creating a meaningful representation that captures the relationships between tokens.
- Each encoder layer consists of two main components: multi-head self-attention and a feed-forward neural network. These components are repeated in a stack of identical layers (usually 6 to 12), with each layer refining the representation of the input sequence.
### Mind Map
- Multi-Head Self-Attention
  - Scaled Dot-Product Attention 
    - Queries, Keys, Values
    -  Attention Scores: Attention Formula
    - Softmax Multiple Heads to capture different relationships
  - Multi Head Attention
    - Concatenating the attention of multiple heads
    - Linear transformations per head
- Add and Normalize
  - Residual Connections
  - Layer Normalization
- Feed-Forward Network
  - Two fully connected layers
- Stacking Encoder
### 1. Self-Attention Mechanism
- It allows the model to weigh the importance of different words in a sentence when encoding a
  given the word, irrespective of their position.
- The self-attention mechanism computes a weighted sum of all other word representations in
  the sequence allowing the model to focus on important words, not just nearby ones.
- **Goal:-** Calculated weighted sum of values for each word in the sequence which tells us how
 much each word should influence the processing of the current word.
#### 1.1 Step1:- Linear Projections of (Q, K, V)
- ##### 1.1.1 Understanding meaning of Query(Q), Key(K), Value(V) Example:-
    - Imagine you're in a classroom, and the teacher asks a question.
    - The teacher is trying to figure out which students have the right knowledge to answer that question.
      - **Query (Q):** The query is the teacher's question. 
        - It represents what information the teacher is looking for. 
        - In this case, it's the thing you want to focus on (like a word in a sentence).
      - **Key (K):** The key is like each student's set of notes. 
        - Each student (or word) has their notes (key), and the teacher looks at them to figure out if that student might have the answer. 
        - The teacher compares the question (query) to the notes (key) to see how relevant each student's knowledge is to the question.
      - **Value (V):** The value is the actual information each student knows. 
        - Once the teacher identifies which students have relevant notes (using the key), they go to those students and listen to their answers (the value).
- ##### 1.1.2 How Q, K, and V work together
1. **Dot Product of Q and K**
- Q = X<sub>input</sub>W<sub>Q</sub>, V = X<sub>input</sub>W<sub>V</sub>, K = X<sub>input</sub>W<sub>K</sub>
- The query vector for the current word is compared to the key vectors of all words in the sequence using a dot product.
  - score(Q, K) = Q.K<sup>T</sup>
  - This produces a similarity score for each pair (query, key), which tells us how much attention the current word should pay to every other word.
2. **Scaling**
- The scores are divided by the square root of the dimensionality of the key vectors,**(d<sub>k</sub>)<sup>(1/2)</sup>**, to ensure more stable gradients.
- This prevents the dot products from becoming too large when the dimensions of Q and K are high, leading to more balanced attention scores.
(\frac{Q.K^T}{\sqrt{d<sub>k</sub>}})
3. **Softmax**
-  The scaled scores are passed through a softmax function to convert them into probabilities.
-  These probabilities represent how much attention should be paid to each word in the sequence.
-  This step normalizes the scores so that they sum to 1.
4. **Weighted Sum of V**
- The attention probabilities (from the softmax) are then applied to the value vectors.
- The model computes a weighted sum of the values, where the weight for each value is the attention score.
- This results in a new representation of the word that incorporates information from other relevant words in the sequence.
```math
Attention(Q, K, V) = softmax (\frac{Q.K^T}{\sqrt{dk}}).V
```
### 1.2 Step 2. Multi Head Attention
  - Instead of performing a single attention function, the Transformer employs multi-head attention. This allows the model to capture different types of relationships between words by splitting the Query, Key, and Value vectors into multiple heads, processing them in parallel, and then concatenating the results.
- Each head can learn different aspects of the sentence, enabling the model to attend to various features simultaneously.
- Steps:-
    1. Linear Projections
      - For each head, we create different versions of Q, K, and V by multiplying them with learned weight matrices:
        - **(Q<sub>head<sub>i</sub></sub>) = QW<sub>i</sub><sup>(Q)</sup>**
        - **(K<sub>head<sub>i</sub></sub>) = KW<sub>i</sub><sup>(K)</sup>**
        - **(V<sub>head<sub>i</sub></sub>) = VW<sub>i</sub><sup>(V)</sup>**
        - where QW<sub>i</sub><sup>(Q)</sup>**, KW<sub>i</sub><sup>(K)</sup>**, VW<sub>i</sub><sup>(V)</sup>** are the weight matrices for the i-th head (these are learned during training).
    2. Compute Attention for each head
      - Attention<sub>i</sub) = softmax((Q<sub>head<sub>i</sub></sub>)(K<sub>head<sub>i</sub></sub>)<sup>T</sup>/(d<sub>k</sub>)<sup>(1/2)</sup>)(V<sub>head<sub>i</sub></sub>)
         - Each head focuses on different aspects of the input because of the different weight matrices.
    3. Final Linear Projection:-
         - Output:- **Multi_Head_Attention(Q, K, V) = Concat(Attention1, Attention2, .........., Attentionh)W<sup>O</sup>**
         - Where h is the number of heads. This gives us a combined representation of all the different heads.
         - Finally, we apply one more linear transformation (with a weight matrix W<sup>O</sup>)
 to the concatenated results to produce the final output of the multi-head attention
### Step 3. Residual Connection and Layer Normalization
- After multi-head attention, a residual connection is applied by adding the input of the layer back to the output of the attention mechanism, followed by layer normalization:
- **Output<sub>att</sub>=LayerNorm(X<sub>input</sub> + MultiHead(Q,K,V))**
- This ensures that gradients flow through the network efficiently during backpropagation, and normalization helps stabilize training.
- Each sub-layer (like self-attention or the feed-forward network) is followed by layer normalization and uses residual connections (or skip connections). Residual connections help in gradient flow during backpropagation, preventing vanishing gradients and allowing the model to go deeper. It makes outputs stay balanced by normalizing the values so they have a mean of 0 and a standard deviation of 1.
- Residual connections (also called skip connections) help the model avoid problems with vanishing gradients. Instead of completely replacing the input to a layer with its output, the model adds the input to the output. This lets the model retain some information from earlier layers.
### Feed-Forward Neural Networks(FNN)
- After computing attention, the model applies a simple neural network (called a feed-forward network) to each word independently. This helps the model process the word’s representation in a non-linear way.
- FNN is applied independently to each token. 
- The feed-forward network is just two linear transformations with a ReLU activation in between. For each word, the output is calculated as:
  - **Step 1 Apply FNN**
    - **FNN output => FFN(x) = max(0, xW1+b1)W2+b2** where max is ReLU activation function. 
  - **Step 2 Apply Residual Connection and Layer Normalization**
    - **Output<sub>fnn</sub> = LayerNorm(Output<sub>att</sub> + FNN output)**
### Stacking Encoders
- The Transformer architecture consists of multiple encoder and decoder layers, typically stacked on top of each other (e.g., 6 layers in the original paper).
- The outputs of one encoder layer become the inputs to the next encoder layer. This process is repeated for each of the N encoder layers (usually 6 to 12). Each layer refines the representations by capturing increasingly abstract relationships between tokens.
- The final output from the last encoder layer is passed to the decoder (if used, as in sequence-to-sequence tasks like translation) or directly used for tasks like classification or text generation.
- Encoder: Each encoder layer consists of a self-attention mechanism followed by a feed-forward network.
- Decoder: The decoder layers are similar to the encoder layers, but they include an additional attention mechanism that attends to the output of the encoder.

## Decoder:- 
### Mind Map:-
- Masked Multi-Head Self-Attention
  - Similar to the encoder, but with masking to prevent future tokens from attending
- Encoder-Decoder Attention
  - Attention between the encoder's output and the decoder's input
- Add and Normalize
  - Residual connections and normalization
- Feed-Forward Network
  - Same as in the encoder
### 1. Masked Multi Head seal attention (output of encoder)
#### Step 1 Linear Projections (Queries, Keys, Values):-
- Like in the encoder, the decoder’s self-attention starts by generating query, key, and value vectors from the decoder input (which is the previously generated tokens):
- Q = X<sub>decoder</sub>W<sub>Q</sub>, V = X<sub>decoder</sub>W<sub>V</sub>, K = X<sub>decoder</sub>W<sub>K</sub>
#### Step 2 Masked Scaled Dot-Product Attention:-
- The major difference in the decoder’s self-attention is the masking. This mask ensures that the model only attends to earlier positions in the sequence, preventing "cheating" by looking at future tokens when generating a sequence.
```math
Masked Attention(Q, K, V) = softmax (\frac{Q.K^T}{\sqrt{dk}} + M).V
```
- M is a mask that sets the scores of future tokens to −∞, ensuring the softmax output for those positions is effectively zero.
- This masking ensures that for any token position t, the model only attends to tokens from positions 1 to t, making the self-attention causal.
#### Step 3 Multi-Head Attention
- ** Masked Multi_Head_Attention(Q, K, V) = Concat(Attention1, Attention2, .........., Attentionh)W<sup>O</sup>**
- **Output<sub>masked att</sub>=LayerNorm(X<sub>decoder</sub> + Masked MultiHead(Q,K,V))**
### 2. Multi-Head Attention (over the encoder outputs)
- The second attention mechanism in the decoder is a multi-head attention layer that allows the decoder to focus on relevant parts of the encoder’s output. This is how the decoder “attends” to the input sequence.
#### Step 1 Linear Projections (Queries, Keys, Values):-
- Like in the encoder, the decoder’s self-attention starts by generating query, key, and value vectors from the decoder input (which is the previously generated tokens):
- Q = X<sub>decoder att</sub>W<sub>Q</sub>, V = X<sub>encoder</sub>W<sub>V</sub>, K = X<sub>encoder</sub>W<sub>K</sub>
- where X<sub>decoder att</sub> is the output of masked self-attention from previous step
- X<sub>encoder</sub> is the output of encoder
#### Step 2 Scaled Dot-Product Attention:-
- The attention mechanism works similarly to the encoder, except the decoder’s queries are now attending to the encoder’s keys and values:
```math
Attention(Q, K, V) = softmax (\frac{Q.K^T}{\sqrt{dk}} + M).V
```
#### Step 3 Multi-Head Attention
- Again, multi-head attention is applied to allow the decoder to focus on different parts of the encoder’s output in parallel:
- ** Multi_Head_Attention(Q, K, V) = Concat(Attention1, Attention2, .........., Attentionh)W<sup>O</sup>**
- **Output<sub>enc-dec att</sub>=LayerNorm(X<sub>decoder att</sub> + Masked MultiHead(Q,K,V))**
#### Step 4 Residual Connection and Layer Normalization
- A residual connection is applied, and the output is passed through layer normalization:
- **Output<sub>enc-dec att</sub> = LayerNorm(Output<sub>decoder att</sub> + MultiHead(Q, K, V))** 
### 3. Feed Forward Neural Network
- FFN(x) = max(0, xW1+b1)W2+b2 where max is ReLU activation function.
- **Step 1 Apply FNN**
  - **FNN output = FFN(Output<sub>enc-dec att</sub>)**  
- **Step 2 Apply Residual Connection and Layer Normalization**
  - **Output<sub>fnn</sub> = LayerNorm(Output<sub>enc-decatt</sub> + FNN output)**
### 4. Stacking Decoders
- Just like the encoder, the decoder consists of a stack of N identical decoder layers (usually 6 to 12 layers). The output of one decoder layer becomes the input to the next, progressively refining the representation of the generated tokens while attending to the input sequence.

## Output Layer
### Mindmap
- Linear Transformation
- Softmax
### Final Linear Layer and softmax
Once the output passes through all decoder layers, it is projected to the vocabulary size using a final linear layer:

    - **_Z_ = Output<sub>decoder</sub>W<sub>vocab</sub> + b<sub>vocab</sub>**
- where
  - _Z_ is the raw logits (unnormalized probabilities) for each token in the vocab
  - W<sub>vocab</sub> is the learned weight matrix that maps the decoder output to the vocab.

- Finally, the softmax function is applied to produce a probability distribution over the next token in the sequence:
  
      ** y'<sub>t</sub>=softmax(_Z_)**
      - where  y'<sub>t</sub> represents the probability distribution over the next token at time step t.

## Transformer Workflow
### 1. Encoder:-
- Takes an input sequence (e.g., a sentence) and converts each word into an embedding. It then applies positional encoding and passes the embeddings through multiple layers of self-attention and feed-forward networks.
### 2. Decoder:- 
- Takes the output of the encoder along with a target sequence (e.g., in machine translation, this could be the partially generated translated sentence). The decoder attends both to itself (through self-attention) and to the encoder outputs (through encoder-decoder attention) to generate the next word in the output sequence.

## Advantages of Transformer:-
1. **Parallelization:** Unlike recurrent networks (RNNs, LSTMs), Transformers process the entire sequence in parallel, which allows for faster training.
2. **Long-Range Dependencies:** Self-attention enables the model to capture relationships between distant words, which is difficult for RNNs to achieve effectively.
3. **Scalability:** Transformers scale well with data and computational resources, making them suitable for large-scale tasks like training language models with billions of parameters.

## Applications of Transformers
- Natural Language Processing: Tasks like machine translation (e.g., Google Translate), text summarization, and sentiment analysis.
- Language Models: Pre-trained models like GPT, BERT, and T5 are all based on the Transformer architecture.
- Computer Vision: Vision Transformers (ViT) have adapted the architecture for image classification tasks.
- Speech Recognition and Music Generation: Transformers are being explored in other modalities as well, including audio and music.
