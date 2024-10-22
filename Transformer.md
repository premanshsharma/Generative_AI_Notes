## Key components of the Transformer Architecture

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
  
  
