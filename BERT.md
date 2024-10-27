- BERT is Bidirectional Encoder Representations from Transformers
- BERT is based on the transformer architecture, but it is only composed of encoder layers
# BERT Architecture
## 1. Input Representation
- Token Embeddings
  - Each word in the input sequence is tokenized into WordPiece tokens.
  - The token embeddings are taken from a pre-trained vocabulary that BERT uses, where each token is mapped to a fixed-size embedding vector.
- Segment Embeddings
  - BERT can handle two different sequences (such as in question-answering tasks where you have a question and a passage). To differentiate between the two, BERT uses segment embeddings:
    - Tokens from the first sentence/sequence are assigned a segment embedding E<sub>A</sub>.
    - Tokens from the second sentence/sequence are assigned a segment embedding E<sub>B</sub>.
- Position Embeddings
  - Since transformers don’t inherently understand word order, BERT adds position embeddings to capture the order of tokens. This positional information is critical for processing sequential data like text.
- **Input Embedding** = Token Embedding + Segment Embedding + Positioin Embedding
## 2. Pre traning objectives of BERT
### 2.1 Masked Language Modeling (MLM)
- The core idea of MLM is to randomly mask some percentage of the tokens in the input and then train the model to predict those masked tokens. This allows BERT to learn a deep, bidirectional understanding of language.
- For instance, consider the sentence: "The cat sat on the mat." If the word "cat" is masked, the input might look like:
  - Input: "The [MASK] sat on the mat."
  - Target: "cat"
- The model tries to predict the masked word ("cat") based on the context provided by the other words ("The", "sat", "on", "the", "mat").
### 2.2 Next Sentence Prediction (NSP)
- To capture the relationship between sentences, BERT is also trained on the NSP task. During training, BERT is given pairs of sentences, and the model must predict whether the second sentence logically follows the first one.
- Example of a positive pair:
  - Sentence 1: "I love playing football."
  - Sentence 2: "It’s my favorite sport."
- Example of a negative pair:
  - Sentence 1: "I love playing football."
  - Sentence 2: "The sky is blue."
- BERT uses this task to understand sentence relationships, which is useful for tasks like question-answering and text classification.
## 3 BERT Fine - tuning for Downstream Tasks
- After pre-training, BERT can be fine-tuned on specific NLP tasks by adding a task-specific layer on top of the pre-trained model. The pre-training tasks (MLM and NSP) help the model learn a general understanding of language, and then fine-tuning adjusts this understanding for specific use cases.
### 3.1 Text Classification
- For tasks like sentiment analysis, BERT is fine-tuned by adding a classifier on top of the transformer encoders. Specifically, a classification token [CLS] is added at the beginning of the input sequence, and the final hidden state corresponding to this token is used to classify the input text.
- Input: "[CLS] I love this movie. [SEP]"
- Output: Sentiment label (e.g., positive, negative)
### 3.2 Question Answering
- For question-answering tasks, BERT is fine-tuned by predicting the start and end positions of the answer within the passage.
- Input: "[CLS] What is the capital of France? [SEP] The capital of France is Paris. [SEP]"
- Output: Start token: "Paris", End token: "Paris"
### 3.3 Named Entity Recognition (NER)
- For NER, each token in the input sequence is classified into categories (e.g., person, location, organization). BERT processes each token and predicts its entity class.
- Input: "Barack Obama was born in Hawaii."
- Output: "Barack Obama" → PERSON, "Hawaii" → LOCATION
# Applications
1. Question Answering (QA)
2. Text Classification
3. Named Entity Recognition (NER)
4. Natural Language Inference (NLI)
5. Sentence Pair Task
6. Text Summarization and Translation

## 1. Question Answering (QA)
- In QA tasks, BERT predicts the start and end positions of the answer span within a passage. It has been highly effective in tasks like the SQuAD (Stanford Question Answering Dataset).
- Task: Given a question and a passage, identify the span of text in the passage that contains the answer.
- **Changes Needed:**
  - Input: Concatenate the question and the passage into a single sequence, separated by the [SEP] token.
    - Input: [CLS] Question [SEP] Passage [SEP]
    - Output: Add two classification layers on top of BERT’s final hidden states to predict the start and end tokens of the answer span.
- **Mathematically**
  - let **H** be output from BERT (shape: (sequence_length, higgen_size))
  - use two separate linear layers (W_start, W_end) to predict the start and end positions.
    - Start logits: start_logit = W_start * H
    - End Logits: end_logits = W_end * H
  - Loss: The model is trained using the cross-entropy loss for both the start and end positions.
    - Loss = (cross_entropy(start_logits, true_start) + cross_entropy(end_logits, true_end))/2


# Strengths of BERT
- Bidirectional Context: BERT’s ability to consider both left and right contexts makes it highly effective for NLP tasks.
- Transfer Learning: BERT can be fine-tuned on a wide variety of tasks, making it extremely versatile.
- State-of-the-Art Performance: BERT has set new benchmarks for many NLP tasks, such as question answering, sentiment analysis, and named entity recognition.
# Limitations of BERT
- Computationally Expensive: BERT’s large size (110M or 340M parameters) makes it resource-intensive to train and fine-tune.
- Input Length Limitation: BERT can process sequences of up to 512 tokens at once due to its fixed input size. This can be a limitation when working with longer texts, requiring truncation or splitting of the input, which can lead to loss of context.

