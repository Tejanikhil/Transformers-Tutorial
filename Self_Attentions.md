# Self Attention Mechanism

This file dives into the self-attention mechanism.
- Generally, the self-attention mechanism is used to extract contextual information.

## Overview

As we already know in math, the relevancy of two vectors is defined by the dot product, which is the projection of the 1st vector onto the 2nd vector. In other words, the dot product between the two vectors is a quantitative representation of how similar the two vectors are.

The self-attention mechanism is formulated using the concept of projecting one vector (embedding) onto another vector (other embeddings). In other words, it measures how a word is related to the remaining words in the sentence.

Let's say the embeddings of the input sequence before applying self-attention are Ea, Eb, Ec, Ed.

## Terminologies Alert!!!

- **Query (Q)**: The word for which we are calculating the attention score.
- **Key (K)**: Represents the word with which we are attending the query.
- **Value (V)**: The word/vector that is weighted by the attention scores to form the final output.

## Example

- **Query = Ea**: Finding the attention scores for Ea.
- **Key = Eb**: We are trying to find the attention score of Ea with respect to Eb (how relevant Ea is to Eb).
- Similarly, we will calculate the attention score of Ea with all other words (Ea, Eb, Ec, Ed), resulting in scores a1, a2, a3, a4.
- **Value for query Ea**: 
  ```Ea = a1.Ea + a2.Eb + a3.Ec + a4.Ed```

## Note

While calculating the value, we normalize the attention scores to avoid complex computation:
- **Normalized Scores**:
  ```Ni = Softmax(ai/sqrt(dimensions(ai)))```
- Therefore, the value for query Ea is:
  ```C_a = N1.Ea + N2.Eb + N3.Ec + N4.Ed```

The contextual embeddings \( C \) are:
```C = [C_a, C_b, C_c, C_d]```

```selfattention(Ea, Eb, Ec, Ed) = C_a, C_b, C_c, C_d```

## Matrix Format

Let's put all this in a matrix format:

- **Embedding Dimensions**: \( d_k \)
- Ea : Row vector of dimension(\( 1 x dk \))
- ```X = [Ea; Eb; Ec; Ed] \) -> Dimensions (4 x dk)```

Initially,
```
Q = X_i (4 x dk)
K = X_i (4 x dk)
V = X_i (4 x dk)
```

- **Attention Weights (A)**:
  ```A = softmax(Q.K'/sqrt(dk))```

- **Weighted Embeddings (Contextual Embeddings)**:
 ```Contextual Embeddings = A.V = C -> Dimensions (4 x dk)```

```selfattention(E) = C```

## Masked Self Attention 
- The concept of masking comes in the decoding mechanism.
- While decoding, the decoder predicts the tokens one after the other in an autoregressive manner. The mathematical way to forumalate is this self attention and is called as masking.
- Lets say the output sequence is x y z, and the decoder state is just [BOS] x y
  Now, while caluclating self attention it masks the attention scores of z and also the words with z -> makes it as zero, this is called as masking
  ```Attentionscores(Anytoken, z) = 0 => Masking```
