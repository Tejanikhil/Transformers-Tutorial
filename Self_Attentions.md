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
  \[
  Value for query Ea = a1*Ea + a2*Eb + a3*Ec + a4*Ed
  \]

## Note

While calculating the value, we normalize the attention scores to avoid complex computation:
- **Normalized Scores**:
  \[
  Ni = Softmax(ai/sqrt(dimensions(ai)))
  \]
- Therefore, the value for query Ea is:
  \[
  C_a = N1*Ea + N2*Eb + N3*Ec + N4*Ed
  \]

The contextual embeddings \( C \) are:
\[
C = [C_a, C_b, C_c, C_d]
\]

\[
\text{selfattention}(Ea, Eb, Ec, Ed) = C_a, C_b, C_c, C_d
\]

## Matrix Format

Let's put all this in a matrix format:

- **Embedding Dimensions**: \( d_k \)
- \( Ea \): Row vector of dimension \( d_k \) (\( 1 \times d_k \))
- \( X = [Ea; Eb; Ec; Ed] \): Dimension \( 4 \times d_k \)

Initially,
\[
Q = X_i (4 x dk)
K = X_i (4 x dk) \\
V = X_i (4 x dk)
\]

- **Attention Weights (A)**:
  \[
  A = softmax(Q.K'/sqrt(dk))
  \]

- **Weighted Embeddings (Contextual Embeddings)**:
  \[
  Contextual Embeddings = A.V = C -> Dimensions (4 x dk)
  \]

\[
C = [C_a, C_b, C_c, C_d]
\]
