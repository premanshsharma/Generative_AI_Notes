# Mind Map
- Encoder-Decoder Architecture
  - Encoder (Used for tasks like text understanding, classification)
  - Decoder (Used for tasks like text generation)
- Components of Transformer
  - Self-Attention Mechanism
    - Query (Q)
    - Key (K)
    - Value (V)
    - Attention Formula:
```math
Attention(Q, K, V) = softmax (\frac{Q.K^T}{\sqrt{dk}}).V
```
## Key components of the Transformer Architecture
![image](https://github.com/user-attachments/assets/d9938997-fdcc-4960-a7a3-ad12b04f6fa9)

### 1. Self-Attention Mechanism
- It allows the model to weigh the importance of different words in a sentence when encoding a
  given word, irrespective of their position.
- The self-attention mechanism computes a weighted sum of all other word representations in
  the sequence allowing the model to focus on important words, not just nearby ones.
- **Goal:-** Calculated weighted sum of values for each word in the sequence which tells us how
 much each word should influence the processing of the current word.  
- We can think of attention as performing a fuzzy lookup in a key-value store.
  - #### Lookup table:-
    - In the look-up table, we have a table of **Keys that map to values and the Query matches
      one of the keys, returning its value**.
    -![image](https://github.com/user-attachments/assets/f8a3bf84-8105-4b2f-8077-f005f59b0051)
  - #### Attention:-
    -  The query matches all keys softly, to a weight between 0 and 1.
      The Keys' values are multiplied by weights and summed.
    -![image](https://github.com/user-attachments/assets/7da15bfd-be78-48d0-8e17-88eaf0e53d20)
  - #### Purpose of Query(Q), Key(K), Value(V) Example:-
    - Imagine you're in a classroom, and the teacher asks a question.
      The teacher is trying to figure out which students have the right knowledge to answer
      that question.

          **Query (Q):** The query is the teacher's question. 
          It represents what information the teacher is looking for. 
          In this case, it's the thing you want to focus on (like a word in a sentence).
          
          **Key (K):** The key is like each student's set of notes. 
          Each student (or word) has their notes (key), 
          and the teacher looks at them to figure out if that student might have the answer. 
          The teacher compares the question (query) to the notes (key) to see how relevant each student's 
          knowledge is to the question.
          
          **Value (V):** The value is the actual information each student knows. 
          Once the teacher identifies which students have relevant notes (using the key), 
          they go to those students and listen to their answers (the value).
  - #### How Q, K, and V work together
      1. **Dot Product of Q and K**
          -  The query vector for the current word is compared to the key vectors of
            all words in the sequence using a dot product.
          - This produces a similarity score for each pair (query, key),
             which tells us how much attention the current word should pay to every other word.
             -![image](https://github.com/user-attachments/assets/aba07666-944f-4c6f-9d9f-90b3115c1082)

      3. **Scaling**
          - The scores are divided by the square root of the dimensionality of the key vectors,
            **(d<sub>k</sub>)<sup>(1/2)</sup>**, to ensure more stable gradients.
          - This prevents the dot products from becoming too large when the dimensions of Q and K are high,
            leading to more balanced attention scores.
            -![image](https://github.com/user-attachments/assets/3f1d9f66-17af-4cbe-ba69-8e8744baa522)
      5. **Softmax**
          -  The scaled scores are passed through a softmax function to convert them into probabilities.
          -  These probabilities represent how much attention should be paid to each word in the sequence.
          -  This step normalizes the scores so that they sum to 1.
      7. **Weighted Sum of V**
          - The attention probabilities (from the softmax) are then applied to the value vectors.
          - The model computes a weighted sum of the values, where the weight for each value is the attention score.
          - This results in a new representation of the word that incorporates information from other relevant words in the sequence.
          -![image](https://github.com/user-attachments/assets/963278c1-639d-40d8-8ee7-9899b500259d)
### 2. Multi Head Attention
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
    3. Concatenate the Results:-
         - Concat(Attention1, Attention2, .........., Attentionh) Where h is the number of heads. This gives us a combined representation of all the different heads.
    4. Final Linear Projection:-
         - Output = Concat(Attention1, Attention2, .........., Attentionh)W<sup>O</sup>
         - Finally, we apply one more linear transformation (with a weight matrix W<sup>O</sup>)
 to the concatenated results to produce the final output of the multi-head attention
### 3. Positional Encoding
- Transformers don’t have a built-in sense of the order of words like RNNs do, because they process all words simultaneously (not sequentially). So, we give each word some extra information to indicate its position in the sequence. This is called positional encoding. It helps the model understand where each word is located in a sentence.
- These encodings are learned or fixed values (e.g., sinusoidal functions) that help the model understand the order of words in the sequence.
- **Mathematical Explanation:**
  -  We use sine and cosine functions to assign different positions a unique encoding.
  -  For each position **pos** in a sentence and each dimension i of the model, the positional encoding is calculated as:
    - PE<sub>pos, 2i</sub> = sin(pos/10000<sup>2i/d<sub>model</sub></sup>)
    - PE<sub>pos, 2i+1</sub> = cos(pos/10000<sup>2i/d<sub>model</sub></sup>)
    - where
      - pos is the position of the word in the sequence.
      - i is the dimension index.
      - d<sub>model</sub> is the size of the word embedding.
    - The sine function is used for even indices, and the cosine function is used for odd indices.
    - The idea is that these encodings will give the model a way to understand the order and relative positions of words.
### 4. Feed-Forward Neural Networks
- After computing attention, the model applies a simple neural network (called a feed-forward network) to each word independently. This helps the model process the word’s representation in a non-linear way.
- The feed-forward network is just two linear transformations with a ReLU activation in between. For each word, the output is calculated as:
- FFN(x) = max(0, xW1+b1)W2+b2 where max is ReLU activation function. 
### 5. Layer Normalization and Residual Connections
- Each sub-layer (like self-attention or the feed-forward network) is followed by layer normalization and uses residual connections (or skip connections). Residual connections help in gradient flow during backpropagation, preventing vanishing gradients and allowing the model to go deeper. It makes outputs stay balanced by normalizing the values so they have a mean of 0 and standard deviation of 1.
- Residual connections (also called skip connections) help the model avoid problems with vanishing gradients. Instead of completely replacing the input to a layer with its output, the model adds the input to the output. This lets the model retain some information from earlier layers.
### 6. Stacking Layers
- The Transformer architecture consists of multiple encoder and decoder layers, typically stacked on top of each other (e.g., 6 layers in the original paper).
- Encoder: Each encoder layer consists of a self-attention mechanism followed by a feed-forward network.
- Decoder: The decoder layers are similar to the encoder layers, but they include an additional attention mechanism that attends to the output of the encoder.

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

# To look:-
masking of input in transformer layer:- for encoder, we make every this -infinite of all the indexes greater than current, if we don't do that it works like bidirectional attention so for encoding do this and for decoding have bidirectional attention. 
